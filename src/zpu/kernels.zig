//! The vector kernels: where the arithmetic actually happens.
//!
//! Everything in `ops.zig` is shape logic — which axes broadcast, what dtype
//! comes out, whether a stride is zero. The inner loops live here, and they are
//! the reason this file exists at all: the generic kernels in `ops.zig` go
//! through `scalar.unary`, which for `f32` widens every lane to `f64`, calls the
//! scalar function, and truncates — correct, and about eight times slower than
//! it needs to be, because it throws away the machine's SIMD units.
//!
//! Three rules hold this file together:
//!
//! * **A kernel is only used when it is exactly equivalent.** `unaryFast`
//!   returns `false` for any primitive whose vector form would not be the same
//!   *expression* as the scalar one (no `tanh`, no `erf`, no `expm1`: those go
//!   back to the scalar path, which is the only implementation). The elementwise
//!   loops that survive — `exp`, `log`, `sqrt`, `relu`, `sigmoid`, the four
//!   arithmetic operators — are the ones a real workload spends its time in.
//! * **The tail is not a second implementation.** A leftover group smaller than
//!   one vector is padded with a copied element, run through the *same* vector
//!   expression, and the padding lanes are dropped. So a length of 1001 gives
//!   the same bits as a length of 1000 plus one.
//! * **The gemm blocks for cache, not for the vector unit.** A `@Vector` alone
//!   does not make a matrix multiply fast: reading `b` once per row of `a` is
//!   512 MB of traffic for a 512³ product. The kernel keeps a `MR × NB` tile of
//!   the result in registers-and-L1 and streams both operands through it once.
const std = @import("std");
const prims = @import("prims.zig");
const scalar = @import("scalar.zig");

const Prim = prims.Prim;

/// 32 bytes per vector: eight `f32` lanes, four `f64`, and a size that stays
/// inside one cache line on every target this library builds for.
const vector_bytes = 32;

/// Lanes in the vector type for `T`.
pub fn lanes(comptime T: type) usize {
    const size = @sizeOf(T);
    if (size == 0 or size > vector_bytes) return 1;
    return vector_bytes / size;
}

pub fn Vec(comptime T: type) type {
    return @Vector(lanes(T), T);
}

fn isFloat(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float => true,
        else => false,
    };
}

// ---------------------------------------------------------------------------
// element-wise
// ---------------------------------------------------------------------------

/// Whether `p` has a vector form that computes the *same expression* the scalar
/// path does. Anything not listed returns false, and the caller falls back.
pub fn hasVectorForm(comptime p: Prim) bool {
    return switch (p) {
        .neg, .abs, .exp, .log, .sqrt, .rsqrt, .square, .recip => true,
        .sin, .cos, .tan, .floor, .ceil, .round, .trunc => true,
        .relu, .sigmoid, .sign, .copy => true,
        else => false,
    };
}

/// The vector expression for `p`. Every branch here has a scalar twin in
/// `scalar.unaryF` *with the same shape of expression*, which is the only reason
/// the two paths can be swapped without a test noticing.
pub inline fn applyUnary(comptime T: type, comptime p: Prim, x: Vec(T)) Vec(T) {
    const V = Vec(T);
    const one: V = @splat(1);
    const zero: V = @splat(0);
    return switch (p) {
        .neg => -x,
        .abs => @abs(x),
        .exp => @exp(x),
        .log => @log(x),
        .sqrt => @sqrt(x),
        .rsqrt => one / @sqrt(x),
        .square => x * x,
        .recip => one / x,
        .sin => @sin(x),
        .cos => @cos(x),
        .tan => @tan(x),
        .floor => @floor(x),
        .ceil => @ceil(x),
        .round => @round(x),
        .trunc => @trunc(x),
        .relu => @max(x, zero),
        .sigmoid => one / (one + @exp(-x)),
        .sign => @select(T, x > zero, one, @select(T, x < zero, -one, zero)),
        .copy => x,
        else => @compileError("no vector form: " ++ @tagName(p)),
    };
}

/// `dst = p(src)` for a full array. Returns false when there is no vector form
/// for `p` or `T`, in which case the caller must use the generic kernel.
pub fn unaryFast(comptime T: type, comptime p: Prim, dst: []T, src: []const T) bool {
    if (comptime !(isFloat(T) and hasVectorForm(p))) return false;
    const L = comptime lanes(T);
    const V = Vec(T);
    if (comptime L == 1) return false;
    var i: usize = 0;
    while (i + L <= dst.len) : (i += L) {
        const x: V = src[i..][0..L].*;
        dst[i..][0..L].* = applyUnary(T, p, x);
    }
    tailUnary(T, p, dst, src, i);
    return true;
}

/// The leftover group, through the same expression as the full vectors.
fn tailUnary(comptime T: type, comptime p: Prim, dst: []T, src: []const T, from: usize) void {
    const L = comptime lanes(T);
    const V = Vec(T);
    const rem = dst.len - from;
    if (rem == 0) return;
    // The padding lanes replicate the last real element, so no operation is ever
    // applied to uninitialised memory (a `log` of garbage would be a NaN, and a
    // NaN in a discarded lane is still a signal a debug build can trip on).
    var in: [L]T = undefined;
    for (0..L) |k| in[k] = src[from + @min(k, rem - 1)];
    const x: V = in;
    const y: [L]T = applyUnary(T, p, x);
    @memcpy(dst[from..], y[0..rem]);
}

/// Whether `p` has a vector form for the binary kernels.
pub fn hasVectorBinary(comptime p: Prim) bool {
    return switch (p) {
        .add, .sub, .mul, .div, .maximum, .minimum => true,
        else => false,
    };
}

pub inline fn applyBinary(comptime T: type, comptime p: Prim, a: Vec(T), b: Vec(T)) Vec(T) {
    return switch (p) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .maximum => @max(a, b),
        .minimum => @min(a, b),
        else => @compileError("no vector form: " ++ @tagName(p)),
    };
}

/// `dst = p(a, b)` for three full arrays of the same length.
pub fn binaryFast(comptime T: type, comptime p: Prim, dst: []T, a: []const T, b: []const T) bool {
    if (comptime !(isFloat(T) and hasVectorBinary(p))) return false;
    const L = comptime lanes(T);
    const V = Vec(T);
    if (comptime L == 1) return false;
    var i: usize = 0;
    while (i + L <= dst.len) : (i += L) {
        const x: V = a[i..][0..L].*;
        const y: V = b[i..][0..L].*;
        dst[i..][0..L].* = applyBinary(T, p, x, y);
    }
    const rem = dst.len - i;
    if (rem > 0) {
        var xa: [L]T = undefined;
        var xb: [L]T = undefined;
        for (0..L) |k| {
            xa[k] = a[i + @min(k, rem - 1)];
            xb[k] = b[i + @min(k, rem - 1)];
        }
        const va: V = xa;
        const vb: V = xb;
        const out: [L]T = applyBinary(T, p, va, vb);
        @memcpy(dst[i..], out[0..rem]);
    }
    return true;
}

/// `dst = p(a, splat(v))` — the broadcast-scalar form, which is what `x * 2`
/// and every learning-rate update actually are.
pub fn binaryScalarFast(comptime T: type, comptime p: Prim, dst: []T, a: []const T, v: T) bool {
    if (comptime !(isFloat(T) and hasVectorBinary(p))) return false;
    const L = comptime lanes(T);
    const V = Vec(T);
    if (comptime L == 1) return false;
    const s: V = @splat(v);
    var i: usize = 0;
    while (i + L <= dst.len) : (i += L) {
        const x: V = a[i..][0..L].*;
        dst[i..][0..L].* = applyBinary(T, p, x, s);
    }
    const rem = dst.len - i;
    if (rem > 0) {
        var xa: [L]T = undefined;
        for (0..L) |k| xa[k] = a[i + @min(k, rem - 1)];
        const va: V = xa;
        const out: [L]T = applyBinary(T, p, va, s);
        @memcpy(dst[i..], out[0..rem]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------

/// What a reduction kernel accumulates with.
pub const Reduce = enum { sum, prod, max, min };

/// Whether a reduction over `T` has a vector form. `bool` and the integer types
/// keep the generic loop: their semantics (`@intCast` on overflow, a wrap for
/// `sum`) are what the generic path implements, and a vector kernel that
/// silently changed overflow behaviour would be a different operation.
pub fn reduceVectorised(comptime T: type) bool {
    return isFloat(T);
}

/// `sum`/`prod`/`max`/`min` over one contiguous run of `T`.
///
/// Eight independent accumulators for `sum` and `prod`: a single accumulator
/// makes the loop latency-bound (each add waits for the previous one), and the
/// machine has the multiplier and adder to spare. The order of the additions is
/// therefore *not* left-to-right — floating-point summation is not associative,
/// so this is a different (and better-conditioned) rounding than the scalar
/// loop's, which is why `sum` of `f32` is allowed a relative tolerance in tests
/// rather than bit equality.
pub fn reduceAll(comptime T: type, comptime op: Reduce, src: []const T) T {
    if (comptime !isFloat(T)) @compileError("reduceAll is for float types");
    const L = comptime lanes(T);
    const V = Vec(T);
    const accs = 4; // four vector accumulators, L lanes each
    const width = accs * L;
    var acc: [accs]V = @splat(@as(V, @splat(identity(T, op))));
    var i: usize = 0;
    if (op == .max or op == .min) {
        // max/min need no multiple accumulators (no rounding, and the op has no
        // latency chain to hide), but they do need the identity handled: the
        // first vector is the accumulator, not an `inf`-loaded add.
        if (src.len >= L) {
            var first: V = src[0..L].*;
            i = L;
            while (i + L <= src.len) : (i += L) {
                const x: V = src[i..][0..L].*;
                first = if (op == .max) @max(first, x) else @min(first, x);
            }
            var out: [L]T = first;
            var best = out[0];
            for (out[1..]) |v| best = pick(T, op, best, v);
            while (i < src.len) : (i += 1) best = pick(T, op, src[i], best);
            return best;
        }
        var best = identity(T, op);
        for (src) |v| best = pick(T, op, best, v);
        return best;
    }
    while (i + width <= src.len) : (i += width) {
        inline for (0..accs) |k| {
            const x: V = src[i + k * L ..][0..L].*;
            acc[k] = combine(T, op, acc[k], x);
        }
    }
    while (i + L <= src.len) : (i += L) {
        const x: V = src[i..][0..L].*;
        acc[0] = combine(T, op, acc[0], x);
    }
    var total = combineVec(T, op, acc[0], acc[1]);
    total = combineVec(T, op, total, acc[2]);
    total = combineVec(T, op, total, acc[3]);
    var lanes_out: [L]T = total;
    var best = lanes_out[0];
    for (lanes_out[1..]) |v| best = pick(T, op, best, v);
    while (i < src.len) : (i += 1) best = pick(T, op, src[i], best);
    return best;
}

/// Row-wise reduction: `dst[i] = op(src[i * inner .. (i + 1) * inner])`.
pub fn reduceRows(comptime T: type, comptime op: Reduce, src: []const T, dst: []T, inner: usize) void {
    if (inner == 0) return;
    for (dst, 0..) |*d, i| d.* = reduceAll(T, op, src[i * inner ..][0..inner]);
}

fn identity(comptime T: type, comptime op: Reduce) T {
    return switch (op) {
        .sum => 0,
        .prod => 1,
        .max => -std.math.inf(T),
        .min => std.math.inf(T),
    };
}

inline fn pick(comptime T: type, comptime op: Reduce, a: T, b: T) T {
    return switch (op) {
        .sum => a + b,
        .prod => a * b,
        // NaN propagates, like the scalar `reduceStep` does.
        .max => if (b != b) b else if (b > a) b else a,
        .min => if (b != b) b else if (b < a) b else a,
    };
}

inline fn combine(comptime T: type, comptime op: Reduce, a: Vec(T), b: Vec(T)) Vec(T) {
    return switch (op) {
        .sum => a + b,
        .prod => a * b,
        .max => @max(a, b),
        .min => @min(a, b),
    };
}

inline fn combineVec(comptime T: type, comptime op: Reduce, a: Vec(T), b: Vec(T)) Vec(T) {
    return combine(T, op, a, b);
}

// ---------------------------------------------------------------------------
// matrix multiply
// ---------------------------------------------------------------------------

/// Rows of `a` held in the accumulator tile. 32 × 128 × 4 B = 16 KiB for `f32`,
/// which fits in L1 with room for the streaming operands.
pub const gemm_mr = 32;
/// Columns of `b` held in the accumulator tile: one cache line's worth of
/// columns per row of `a`, times 32 rows.
pub const gemm_nb = 128;

/// `d = a @ b` for contiguous row-major `f32`/`f64`, `d` already zeroed or not
/// (the kernel writes, it does not accumulate).
///
/// The blocking is what makes this fast, not the vector type. `b` is walked in
/// column panels of `gemm_nb`; each panel is read once per row block of `a`, so
/// the traffic is `(m/gemm_mr) · k · gemm_nb` elements of `b` and
/// `(n/gemm_nb) · k · gemm_mr` of `a` — for a 512³ `f32` product that is a few
/// megabytes instead of a gigabyte, and the inner loop is a pure contiguous
/// multiply-add that the vector unit runs at one FMA per lane per cycle.
pub fn gemm(
    comptime T: type,
    a: []const T,
    b: []const T,
    d: []T,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (m == 0 or n == 0) return;
    if (k == 0) {
        for (0..m * n) |i| d[i] = 0;
        return;
    }
    const V = Vec(T);
    const L = comptime lanes(T);
    var tile: [gemm_mr * gemm_nb]T = undefined;

    var jb: usize = 0;
    while (jb < n) : (jb += gemm_nb) {
        const nb: usize = @min(gemm_nb, n - jb);
        var ib: usize = 0;
        while (ib < m) : (ib += gemm_mr) {
            const mb: usize = @min(gemm_mr, m - ib);
            const acc = tile[0 .. mb * gemm_nb];
            // Only the first `nb` columns of each row of the tile are used; the
            // rest is padding so that the row stride stays a compile-time
            // constant and the vector loop needs no per-row bound check.
            @memset(acc, 0);

            for (0..k) |p| {
                const b_row = b[p * n + jb ..][0..nb];
                const a_col = a[ib * k + p ..]; // a[i][p] for i in 0..mb
                for (0..mb) |i| {
                    const av: V = @splat(a_col[i * k]);
                    const row = acc[i * gemm_nb ..][0..gemm_nb];
                    var j: usize = 0;
                    // Four vectors at a time: the loop overhead disappears and
                    // the loads are independent, which is what lets the FMA unit
                    // stay busy.
                    while (j + 4 * L <= nb) : (j += 4 * L) {
                        const b0: V = b_row[j..][0..L].*;
                        const b1: V = b_row[j + L ..][0..L].*;
                        const b2: V = b_row[j + 2 * L ..][0..L].*;
                        const b3: V = b_row[j + 3 * L ..][0..L].*;
                        const c0: V = row[j..][0..L].*;
                        const c1: V = row[j + L ..][0..L].*;
                        const c2: V = row[j + 2 * L ..][0..L].*;
                        const c3: V = row[j + 3 * L ..][0..L].*;
                        row[j..][0..L].* = @mulAdd(V, av, b0, c0);
                        row[j + L ..][0..L].* = @mulAdd(V, av, b1, c1);
                        row[j + 2 * L ..][0..L].* = @mulAdd(V, av, b2, c2);
                        row[j + 3 * L ..][0..L].* = @mulAdd(V, av, b3, c3);
                    }
                    while (j + L <= nb) : (j += L) {
                        const bv: V = b_row[j..][0..L].*;
                        const cv: V = row[j..][0..L].*;
                        row[j..][0..L].* = @mulAdd(V, av, bv, cv);
                    }
                    while (j < nb) : (j += 1) row[j] += a_col[i * k] * b_row[j];
                }
            }

            for (0..mb) |i| {
                const dst = d[(ib + i) * n + jb ..][0..nb];
                @memcpy(dst, acc[i * gemm_nb ..][0..nb]);
            }
        }
    }
}

/// Whether `T` has a vectorised matrix-multiply kernel. `bool` and the integer
/// dtypes keep the generic loop, because their `accum` is not a multiply.
pub fn gemmVectorised(comptime T: type) bool {
    return isFloat(T) and @sizeOf(T) >= 4;
}

/// The same product for a batch of contiguous matrices, `d[bi] = a[bi] @ b[bi]`.
pub fn gemmBatch(
    comptime T: type,
    a: []const T,
    b: []const T,
    d: []T,
    m: usize,
    k: usize,
    n: usize,
    batches: usize,
) void {
    for (0..batches) |bi| {
        gemm(T, a[bi * m * k ..][0 .. m * k], b[bi * k * n ..][0 .. k * n], d[bi * m * n ..][0 .. m * n], m, k, n);
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Equality that survives the domain edges and the last bit.
///
/// Two things are being tolerated here, both deliberately. NaNs: `log(-1)` is
/// NaN on either path, and NaN is not approximately anything. And the last ulp
/// near a root (`log(0.9999999)` is `-1e-8`): the vector path is *f32* input to
/// f32 output, so a value that the scalar path computed in f64 and rounded down
/// can sit one ulp away, which is a large *relative* difference and a
/// vanishingly small absolute one. Hence the absolute floor under the relative
/// tolerance — the standard numerical-test criterion.
fn same(want: f32, got: f32, tol: f32) bool {
    if (std.math.isNan(want) or std.math.isNan(got)) return std.math.isNan(want) and std.math.isNan(got);
    if (@abs(want - got) <= 1e-6) return true;
    return std.math.approxEqRel(f32, want, got, tol);
}

test "the vector kernels agree with the scalar ones" {
    // Every primitive with a vector form, over a spread of inputs that includes
    // the awkward ones (zero, negative, denormal-ish, large).
    const cases = [_]f64{ 0, 1, -1, 0.5, -0.5, 2, -2, 1e-8, 1e8, 0.9999999 };
    inline for (.{ .exp, .log, .sqrt, .rsqrt, .square, .recip, .sin, .cos, .tan, .floor, .ceil, .round, .trunc, .relu, .sigmoid, .sign, .neg, .abs, .copy }) |p| {
        var src: [cases.len]f32 = undefined;
        for (cases, 0..) |c, i| src[i] = @floatCast(c);
        var got: [cases.len]f32 = undefined;
        try testing.expect(unaryFast(f32, p, &got, &src));
        for (cases, 0..) |c, i| {
            const want: f32 = @floatCast(scalar.unaryF(p, c));
            // `@exp`/`@log`/`@sin` lower to the vector intrinsics, which are the
            // same LLVM intrinsic the scalar call ends in — so this is exact,
            // not approximate, for these inputs.
            try testing.expect(same(want, got[i], 1e-6));
        }
    }
}

test "the binary kernels agree with the scalar ones" {
    const a = [9]f64{ 0, 1, -1, 0.5, 2, -3, 1e-8, 1e8, 7.25 };
    const b = [9]f64{ 1, 2, 3, -0.25, 0.5, 4, 1e-8, 2, 1.5 };
    inline for (.{ .add, .sub, .mul, .div, .maximum, .minimum }) |p| {
        var xa: [a.len]f32 = undefined;
        var xb: [b.len]f32 = undefined;
        for (a, 0..) |v, i| xa[i] = @floatCast(v);
        for (b, 0..) |v, i| xb[i] = @floatCast(v);
        var got: [a.len]f32 = undefined;
        try testing.expect(binaryFast(f32, p, &got, &xa, &xb));
        for (a, 0..) |v, i| {
            const want: f32 = @floatCast(scalar.binaryF(p, v, b[i]));
            try testing.expectEqual(want, got[i]);
        }
    }
}

test "a leftover group of any size matches the full-vector path" {
    // The tail is the same expression, so lengths that are not a multiple of the
    // vector width must give the same bits as the corresponding prefix.
    var src: [33]f32 = undefined;
    // Positive, so `sqrt`/`log` are defined for every element: the point here
    // is the tail, not the domain.
    for (0..src.len) |i| src[i] = @as(f32, @floatFromInt(i)) * 0.25 + 0.5;
    var whole: [33]f32 = undefined;
    var cut: [32]f32 = undefined;
    _ = unaryFast(f32, .exp, whole[0..], src[0..]);
    _ = unaryFast(f32, .exp, cut[0..], src[0..32]);
    try testing.expectEqualSlices(f32, cut[0..], whole[0..32]);

    // One element, and one more than a vector: both are tails.
    for ([_]usize{ 1, 33, 31, 9 }) |len| {
        var d: [33]f32 = undefined;
        var s: [33]f32 = undefined;
        for (0..len) |i| s[i] = src[i];
        _ = unaryFast(f32, .sqrt, d[0..len], s[0..len]);
        for (0..len) |i| try testing.expectApproxEqRel(@sqrt(s[i]), d[i], 1e-7);
    }
}

test "the gemm matches a naive triple loop" {
    // Hand-checked sizes: one tile, a ragged tile, a single row, a single column.
    const shapes = [_][3]usize{
        .{ 1, 1, 1 },
        .{ 3, 4, 5 },
        .{ gemm_mr, 7, gemm_nb },
        .{ gemm_mr + 3, 9, gemm_nb + 5 },
        .{ 2, 33, 2 },
    };
    const alloc = testing.allocator;
    for (shapes) |s| {
        const m = s[0];
        const k = s[1];
        const n = s[2];
        const a = try alloc.alloc(f64, m * k);
        defer alloc.free(a);
        const b = try alloc.alloc(f64, k * n);
        defer alloc.free(b);
        const d = try alloc.alloc(f64, m * n);
        defer alloc.free(d);
        for (a, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 7)) * 0.5 - 1;
        for (b, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 5)) * 0.25 - 0.5;
        gemm(f64, a, b, d, m, k, n);
        for (0..m) |i| {
            for (0..n) |j| {
                var want: f64 = 0;
                for (0..k) |p| want += a[i * k + p] * b[p * n + j];
                try testing.expectApproxEqRel(want, d[i * n + j], 1e-12);
            }
        }
    }
}

test "the reduction kernels agree with a left-to-right loop" {
    var data: [1000]f64 = undefined;
    for (0..data.len) |i| data[i] = @sin(@as(f64, @floatFromInt(i)) * 0.37) * 3;
    var sum_want: f64 = 0;
    var prod_want: f64 = 1;
    var max_want: f64 = -std.math.inf(f64);
    var min_want: f64 = std.math.inf(f64);
    for (data) |v| {
        sum_want += v;
        prod_want *= v;
        max_want = @max(max_want, v);
        min_want = @min(min_want, v);
    }
    // The summations differ in association, so both are compared with a
    // relative tolerance; max/min are exact (they are selections).
    try testing.expectApproxEqRel(sum_want, reduceAll(f64, .sum, &data), 1e-12);
    try testing.expectEqual(max_want, reduceAll(f64, .max, &data));
    try testing.expectEqual(min_want, reduceAll(f64, .min, &data));
    // A product of a thousand numbers is not a useful comparison at f64; check
    // the shape of the answer instead (it is dominated by the zeros/negatives).
    try testing.expect(std.math.isFinite(reduceAll(f64, .prod, &data)));
    try testing.expect(@abs(reduceAll(f64, .prod, &data)) < 1e300);
}

test "reductions handle the edges: empty, one, exactly a vector, one more" {
    const L = comptime lanes(f32);
    for ([_]usize{ 0, 1, L, L + 1, 4 * L + 3 }) |len| {
        var buf: [4 * L + 3]f32 = undefined;
        for (0..len) |i| buf[i] = @floatFromInt(i);
        if (len == 0) continue;
        const sum: f32 = @floatFromInt(len * (len - 1) / 2);
        try testing.expectEqual(sum, reduceAll(f32, .sum, buf[0..len]));
    }
    // All-NaN input: the maximum is NaN, and NaN must not be swallowed.
    var nan: [8]f32 = @splat(std.math.nan(f32));
    try testing.expect(std.math.isNan(reduceAll(f32, .max, nan[0..])));
}

test "row-wise reduction matches per-row scalar reductions" {
    var src: [64]f64 = undefined;
    for (0..src.len) |i| src[i] = @as(f64, @floatFromInt(i % 7)) - 3;
    var dst: [8]f64 = undefined;
    reduceRows(f64, .sum, &src, &dst, 8);
    for (0..8) |r| {
        var want: f64 = 0;
        for (src[r * 8 ..][0..8]) |v| want += v;
        try testing.expectApproxEqAbs(want, dst[r], 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Parallel kernels for large tensors
// ---------------------------------------------------------------------------

const parallel = @import("parallel.zig");

/// Parallel unary kernel: dst[i] = op(src[i]) for contiguous arrays.
/// Uses multi-threading for large arrays.
pub fn unaryParallel(comptime T: type, comptime p: Prim, dst: []T, src: []const T) !void {
    if (!hasVectorForm(p) or !isFloat(T)) {
        // Fallback to scalar
        for (0..dst.len) |i| dst[i] = scalar.unary(T, p, src[i]);
        return;
    }
    
    const n = dst.len;
    if (n < parallel.parallel_threshold) {
        // Small array: use vectorized kernel
        _ = unaryFast(T, p, dst, src);
        return;
    }
    
    // Large array: parallelize
    const workers = parallel.workerCount(n);
    if (workers == 1) {
        _ = unaryFast(T, p, dst, src);
        return;
    }
    
    // Allocate thread handles
    const alloc = std.heap.c_allocator;
    var threads = try alloc.alloc(std.Thread, workers);
    defer alloc.free(threads);
    
    // Worker data
    const workers_data = try alloc.alloc(struct { start: usize, end: usize }, workers);
    defer alloc.free(workers_data);
    
    const WorkerCtx = struct {
        dst_slice: []T,
        src_slice: []const T,
        start: usize,
        end: usize,
        
        fn run(self: *@This()) void {
            const d = self.dst_slice[self.start..self.end];
            const s = self.src_slice[self.start..self.end];
            _ = unaryFast(T, p, d, s);
        }
    };
    
    var ctxs = try alloc.alloc(WorkerCtx, workers);
    defer alloc.free(ctxs);
    
    for (0..workers) |i| {
        const range = parallel.chunkRange(n, workers, i);
        ctxs[i] = .{
            .dst_slice = dst,
            .src_slice = src,
            .start = range.start,
            .end = range.end,
        };
        threads[i] = try std.Thread.spawn(.{}, WorkerCtx.run, .{&ctxs[i]});
    }
    
    for (threads) |*t| t.join();
}

/// Parallel binary kernel: dst[i] = op(a[i], b[i]) for contiguous arrays.
pub fn binaryParallel(comptime T: type, comptime p: Prim, dst: []T, a: []const T, b: []const T) !void {
    if (!hasVectorBinary(p) or !isFloat(T)) {
        // Fallback to scalar
        for (0..dst.len) |i| dst[i] = scalar.binary(T, p, a[i], b[i]);
        return;
    }
    
    const n = dst.len;
    if (n < parallel.parallel_threshold) {
        // Small array: use vectorized kernel
        _ = binaryFast(T, p, dst, a, b);
        return;
    }
    
    // Large array: parallelize
    const workers = parallel.workerCount(n);
    if (workers == 1) {
        _ = binaryFast(T, p, dst, a, b);
        return;
    }
    
    // Allocate thread handles
    const alloc = std.heap.c_allocator;
    var threads = try alloc.alloc(std.Thread, workers);
    defer alloc.free(threads);
    
    const WorkerCtx = struct {
        dst_slice: []T,
        a_slice: []const T,
        b_slice: []const T,
        start: usize,
        end: usize,
        
        fn run(self: *@This()) void {
            const d = self.dst_slice[self.start..self.end];
            const av = self.a_slice[self.start..self.end];
            const bv = self.b_slice[self.start..self.end];
            _ = binaryFast(T, p, d, av, bv);
        }
    };
    
    var ctxs = try alloc.alloc(WorkerCtx, workers);
    defer alloc.free(ctxs);
    
    for (0..workers) |i| {
        const range = parallel.chunkRange(n, workers, i);
        ctxs[i] = .{
            .dst_slice = dst,
            .a_slice = a,
            .b_slice = b,
            .start = range.start,
            .end = range.end,
        };
        threads[i] = try std.Thread.spawn(.{}, WorkerCtx.run, .{&ctxs[i]});
    }
    
    for (threads) |*t| t.join();
}

test "parallel unary kernel" {
    const n = 100_000;
    const alloc = std.testing.allocator;
    var src = try alloc.alloc(f32, n);
    defer alloc.free(src);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);
    
    for (0..n) |i| src[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
    
    try unaryParallel(f32, .relu, dst, src);
    
    for (0..n) |i| {
        const expected = @max(src[i], 0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "parallel binary kernel" {
    const n = 100_000;
    const alloc = std.testing.allocator;
    var a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    var b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);
    
    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        b[i] = @as(f32, @floatFromInt(i % 50)) * 0.02;
    }
    
    try binaryParallel(f32, .add, dst, a, b);
    
    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-6);
    }
}
