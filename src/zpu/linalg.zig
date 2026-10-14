//! Dense linear algebra on the last two axes (batched everywhere).
//!
//! The routines record dedicated primitives (`solve`, `inv`, `det`, `cholesky`)
//! so that autodiff can use analytic rules (`dA^-1 = -A^-1 dA A^-1`, ...) rather
//! than differentiating an elimination loop.
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");
const prims = @import("prims.zig");
const context = @import("context.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = tmod.Shape;
pub const Error = tmod.Error;
pub const releaseTemp = tmod.releaseTemp;

const Layout = struct {
    batch: usize,
    n: usize,
    /// columns of the right hand side (1 for a vector)
    m: usize,
    vec_b: bool,
};

fn floatOf(a: DType, b: DType) DType {
    const p = DType.promote(a, b);
    return if (p.isFloat()) p else .f32;
}

/// Validates `a` (square) against `b` (vector or matrix) and describes the batch.
pub fn matLayout(a: Tensor, b: Tensor) Error!Layout {
    if (a.rank() < 2 or b.rank() < 1) return Error.ShapeMismatch;
    const n = a.shape.dims[a.rank() - 2];
    if (a.shape.dims[a.rank() - 1] != n) return Error.ShapeMismatch;
    const batch = a.numel() / (n * n);
    if (b.rank() == a.rank() - 1) {
        if (b.shape.dims[b.rank() - 1] != n) return Error.ShapeMismatch;
        if (b.numel() / n != batch) return Error.ShapeMismatch;
        return .{ .batch = batch, .n = n, .m = 1, .vec_b = true };
    }
    if (b.rank() != a.rank()) return Error.ShapeMismatch;
    if (b.shape.dims[b.rank() - 2] != n) return Error.ShapeMismatch;
    const m = b.shape.dims[b.rank() - 1];
    if (b.numel() / (n * m) != batch) return Error.ShapeMismatch;
    return .{ .batch = batch, .n = n, .m = m, .vec_b = false };
}

fn loadMat(src: Tensor, dst: []f64) void {
    for (dst, 0..) |*v, i| v.* = src.readF64(i);
}

fn loadVec(src: Tensor, dst: []f64) void {
    for (dst, 0..) |*v, i| v.* = src.readF64(i);
}

/// LU factorisation with partial pivoting; returns the sign of the permutation.
fn luFactorize(m: []f64, n: usize, piv: []usize) f64 {
    var sign: f64 = 1;
    var k: usize = 0;
    while (k < n) : (k += 1) {
        var p = k;
        var best = @abs(m[k * n + k]);
        var i = k + 1;
        while (i < n) : (i += 1) {
            const v = @abs(m[i * n + k]);
            if (v > best) {
                best = v;
                p = i;
            }
        }
        piv[k] = p;
        if (p != k) {
            sign = -sign;
            var j: usize = 0;
            while (j < n) : (j += 1) {
                const t = m[k * n + j];
                m[k * n + j] = m[p * n + j];
                m[p * n + j] = t;
            }
        }
        const d = m[k * n + k];
        if (d != 0) {
            i = k + 1;
            while (i < n) : (i += 1) {
                const f = m[i * n + k] / d;
                m[i * n + k] = f;
                var j = k + 1;
                while (j < n) : (j += 1) {
                    m[i * n + j] -= f * m[k * n + j];
                }
            }
        }
    }
    return sign;
}

fn luSolveInPlace(m: []const f64, n: usize, piv: []const usize, rhs: []f64, mcols: usize) void {
    // apply the permutation
    var k: usize = 0;
    while (k < n) : (k += 1) {
        const p = piv[k];
        if (p != k) {
            var j: usize = 0;
            while (j < mcols) : (j += 1) {
                const t = rhs[k * mcols + j];
                rhs[k * mcols + j] = rhs[p * mcols + j];
                rhs[p * mcols + j] = t;
            }
        }
    }
    // forward substitution (unit lower triangular)
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < mcols) : (j += 1) {
            var acc = rhs[i * mcols + j];
            var l: usize = 0;
            while (l < i) : (l += 1) acc -= m[i * n + l] * rhs[l * mcols + j];
            rhs[i * mcols + j] = acc;
        }
    }
    // back substitution
    var ii = n;
    while (ii > 0) {
        ii -= 1;
        var j: usize = 0;
        while (j < mcols) : (j += 1) {
            var acc = rhs[ii * mcols + j];
            var l = ii + 1;
            while (l < n) : (l += 1) acc -= m[ii * n + l] * rhs[l * mcols + j];
            const d = m[ii * n + ii];
            rhs[ii * mcols + j] = if (d == 0) std.math.nan(f64) else acc / d;
        }
    }
}

/// Solves `a x = b` (`b` a vector or a matrix, batched over the leading axes).
pub fn solve(a: Tensor, b: Tensor) Error!Tensor {
    const L = try matLayout(a, b);
    var out = try Tensor.empty(b.shape, floatOf(a.dt, b.dt));
    out.batch_depth = b.batch_depth;
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const bc = try b.toContiguous();
    defer releaseTemp(bc);
    const alloc = context.allocatorRef();
    const mat = try alloc.alloc(f64, L.n * L.n);
    defer alloc.free(mat);
    const piv = try alloc.alloc(usize, L.n);
    defer alloc.free(piv);
    const rhs = try alloc.alloc(f64, L.n * L.m);
    defer alloc.free(rhs);
    for (0..L.batch) |bi| {
        loadMat(slicePat(ac, bi, L.n * L.n), mat);
        var j: usize = 0;
        const bcols = if (L.vec_b) 1 else L.m;
        while (j < bcols) : (j += 1) {
            var i: usize = 0;
            while (i < L.n) : (i += 1) {
                rhs[i * bcols + j] = bc.readF64((bi * L.n + i) * bcols + j);
            }
        }
        _ = luFactorize(mat, L.n, piv);
        luSolveInPlace(mat, L.n, piv, rhs, bcols);
        j = 0;
        while (j < bcols) : (j += 1) {
            var i: usize = 0;
            while (i < L.n) : (i += 1) {
                out.writeF64((bi * L.n + i) * bcols + j, rhs[i * bcols + j]);
            }
        }
    }
    return ops.finish(.solve, &.{ a, b }, out, .{});
}

fn slicePat(t: Tensor, batch: usize, block: usize) Tensor {
    var v = t;
    v.offset = t.offset + batch * block;
    v.shape = Shape.init(&.{block}) catch unreachable;
    v.strides = (Shape{ .len = 1, .dims = .{ 1, 0, 0, 0, 0, 0, 0, 0 } }).stridesContiguous();
    return v;
}

/// Inverse of a square (batched) matrix.
pub fn inv(a: Tensor) Error!Tensor {
    const n = try squareDim(a);
    const batch = a.numel() / (n * n);
    var out = try Tensor.empty(a.shape, floatOf(a.dt, a.dt));
    out.batch_depth = a.batch_depth;
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const alloc = context.allocatorRef();
    const mat = try alloc.alloc(f64, n * n);
    defer alloc.free(mat);
    const piv = try alloc.alloc(usize, n);
    defer alloc.free(piv);
    const rhs = try alloc.alloc(f64, n * n);
    defer alloc.free(rhs);
    for (0..batch) |bi| {
        loadMat(slicePat(ac, bi, n * n), mat);
        for (0..n * n) |i| rhs[i] = 0;
        for (0..n) |i| rhs[i * n + i] = 1;
        _ = luFactorize(mat, n, piv);
        luSolveInPlace(mat, n, piv, rhs, n);
        for (0..n * n) |i| out.writeF64(bi * n * n + i, rhs[i]);
    }
    return ops.finish(.inv, &.{a}, out, .{});
}

/// Determinant of a square (batched) matrix.
pub fn det(a: Tensor) Error!Tensor {
    const n = try squareDim(a);
    const batch = a.numel() / (n * n);
    var dims: [prims.MAX_RANK]usize = @splat(0);
    for (0..a.rank() - 2) |i| dims[i] = a.shape.dims[i];
    var out = try Tensor.empty(Shape.fromParts(dims, a.rank() - 2), floatOf(a.dt, a.dt));
    out.batch_depth = a.batch_depth;
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const alloc = context.allocatorRef();
    const mat = try alloc.alloc(f64, n * n);
    defer alloc.free(mat);
    const piv = try alloc.alloc(usize, n);
    defer alloc.free(piv);
    for (0..batch) |bi| {
        loadMat(slicePat(ac, bi, n * n), mat);
        const sign = luFactorize(mat, n, piv);
        var d: f64 = sign;
        for (0..n) |i| d *= mat[i * n + i];
        out.writeF64(bi, d);
    }
    return ops.finish(.det, &.{a}, out, .{});
}

fn squareDim(a: Tensor) Error!usize {
    if (a.rank() < 2) return Error.ShapeMismatch;
    const n = a.shape.dims[a.rank() - 2];
    if (a.shape.dims[a.rank() - 1] != n) return Error.ShapeMismatch;
    return n;
}

/// Cholesky factorisation of a symmetric positive definite matrix.
/// By default the lower factor `L` (`a = L Lᵀ`) is returned; the upper factor is
/// stored when `upper` is true.
pub fn cholesky(a: Tensor, upper: bool) Error!Tensor {
    const n = try squareDim(a);
    const batch = a.numel() / (n * n);
    var out = try Tensor.zeros(a.shape, floatOf(a.dt, a.dt));
    out.batch_depth = a.batch_depth;
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const alloc = context.allocatorRef();
    const mat = try alloc.alloc(f64, n * n);
    defer alloc.free(mat);
    const l = try alloc.alloc(f64, n * n);
    defer alloc.free(l);
    for (0..batch) |bi| {
        loadMat(slicePat(ac, bi, n * n), mat);
        for (0..n * n) |i| l[i] = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j <= i) : (j += 1) {
                var sum = mat[i * n + j];
                var k: usize = 0;
                while (k < j) : (k += 1) sum -= l[i * n + k] * l[j * n + k];
                if (i == j) {
                    if (sum <= 0) return Error.NotPositiveDefinite;
                    l[i * n + j] = @sqrt(sum);
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        for (0..n) |r| {
            for (0..n) |c| {
                const v = if (upper) l[c * n + r] else l[r * n + c];
                out.writeF64(bi * n * n + r * n + c, v);
            }
        }
    }
    var attrs = prims.Attrs{};
    attrs = attrs.withUpper(upper);
    return ops.finish(.cholesky, &.{a}, out, attrs);
}

/// Solves `a x = b` for triangular `a`.
pub fn triangularSolve(a: Tensor, b: Tensor, upper: bool) Error!Tensor {
    const L = try matLayout(a, b);
    const n = L.n;
    var out = try Tensor.empty(b.shape, floatOf(a.dt, b.dt));
    out.batch_depth = b.batch_depth;
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const bc = try b.toContiguous();
    defer releaseTemp(bc);
    const alloc = context.allocatorRef();
    const mat = try alloc.alloc(f64, n * n);
    defer alloc.free(mat);
    const rhs = try alloc.alloc(f64, n * L.m);
    defer alloc.free(rhs);
    for (0..L.batch) |bi| {
        loadMat(slicePat(ac, bi, n * n), mat);
        for (0..n * L.m) |i| rhs[i] = bc.readF64(bi * n * L.m + i);
        triSolve(mat, n, rhs, L.m, upper);
        for (0..n * L.m) |i| out.writeF64(bi * n * L.m + i, rhs[i]);
    }
    var attrs = prims.Attrs{};
    attrs = attrs.withUpper(upper);
    return ops.finish(.triangular_solve, &.{ a, b }, out, attrs);
}

fn triSolve(m: []const f64, n: usize, rhs: []f64, mcols: usize, upper: bool) void {
    if (upper) {
        var ii = n;
        while (ii > 0) {
            ii -= 1;
            var j: usize = 0;
            while (j < mcols) : (j += 1) {
                var acc = rhs[ii * mcols + j];
                var l = ii + 1;
                while (l < n) : (l += 1) acc -= m[ii * n + l] * rhs[l * mcols + j];
                rhs[ii * mcols + j] = acc / m[ii * n + ii];
            }
        }
    } else {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < mcols) : (j += 1) {
                var acc = rhs[i * mcols + j];
                var l: usize = 0;
                while (l < i) : (l += 1) acc -= m[i * n + l] * rhs[l * mcols + j];
                rhs[i * mcols + j] = acc / m[i * n + i];
            }
        }
    }
}

/// Dense LU factorisation (exposed for reuse; not differentiated).
pub const Lu = struct {
    lu: Tensor,
    pivots: []usize,
    alloc: std.mem.Allocator,

    pub fn deinit(self: *Lu) void {
        self.alloc.free(self.pivots);
        var l = self.lu;
        l.deinit();
    }
};

pub fn luFactor(a: Tensor) Error!Lu {
    const n = try squareDim(a);
    const batch = a.numel() / (n * n);
    var out = try Tensor.empty(a.shape, floatOf(a.dt, a.dt));
    out.batch_depth = a.batch_depth;
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const alloc = context.allocatorRef();
    const pivots = try alloc.alloc(usize, n * batch);
    errdefer alloc.free(pivots);
    const mat = try alloc.alloc(f64, n * n);
    defer alloc.free(mat);
    for (0..batch) |bi| {
        loadMat(slicePat(ac, bi, n * n), mat);
        _ = luFactorize(mat, n, pivots[bi * n ..][0..n]);
        for (0..n * n) |i| out.writeF64(bi * n * n + i, mat[i]);
    }
    return .{ .lu = out, .pivots = pivots, .alloc = alloc };
}

/// Solves using a precomputed `luFactor` result.
pub fn luSolve(luf: *const Lu, b: Tensor) Error!Tensor {
    const a = luf.lu;
    const n = try squareDim(a);
    const L = try matLayout(a, b);
    var out = try Tensor.empty(b.shape, floatOf(a.dt, b.dt));
    const bc = try b.toContiguous();
    defer releaseTemp(bc);
    const alloc = context.allocatorRef();
    const mat = try alloc.alloc(f64, n * n);
    defer alloc.free(mat);
    const rhs = try alloc.alloc(f64, n * L.m);
    defer alloc.free(rhs);
    for (0..L.batch) |bi| {
        loadMat(slicePat(a, bi, n * n), mat);
        for (0..n * L.m) |i| rhs[i] = bc.readF64(bi * n * L.m + i);
        luSolveInPlace(mat, n, luf.pivots[bi * n ..][0..n], rhs, L.m);
        for (0..n * L.m) |i| out.writeF64(bi * n * L.m + i, rhs[i]);
    }
    return out;
}

/// Vector dot product along the last axis.
pub fn dot(a: Tensor, b: Tensor) Error!Tensor {
    const p = try ops.mul(a, b);
    defer releaseTemp(p);
    return ops.sum(p, -1, false);
}

/// Matrix product (alias of `ops.matmul`, kept for numpy familiarity).
pub const matmul = ops.matmul;

/// Dot product of two 1-D vectors, returning a 0-d tensor.
pub fn vdot(a: Tensor, b: Tensor) Error!Tensor {
    const p = try ops.mul(a, b);
    defer releaseTemp(p);
    return ops.sumAll(p);
}

/// Euclidean norm along `axis`.
pub fn norm(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    const sq = try ops.square(x);
    defer releaseTemp(sq);
    const s = try ops.sum(sq, axis, keepdims);
    defer releaseTemp(s);
    return ops.sqrt(s);
}

/// Frobenius norm (all payload axes).
pub fn normAll(x: Tensor) Error!Tensor {
    return norm(x, ops.AXIS_NONE, false);
}

/// Sum along the main diagonal of the last two axes.
pub fn trace(x: Tensor) Error!Tensor {
    const d = try diag(x);
    defer releaseTemp(d);
    return ops.sum(d, -1, false);
}

/// Extracts the main diagonal of the last two axes; the result has rank+1 with
/// the diagonal as the new last axis.
pub fn diag(x: Tensor) Error!Tensor {
    if (x.rank() < 2) return Error.ShapeMismatch;
    const n = @min(x.shape.dims[x.rank() - 2], x.shape.dims[x.rank() - 1]);
    const batch = x.numel() / (x.shape.dims[x.rank() - 2] * x.shape.dims[x.rank() - 1]);
    const rows = x.shape.dims[x.rank() - 2];
    var dims: [prims.MAX_RANK]usize = @splat(0);
    for (0..x.rank() - 2) |i| dims[i] = x.shape.dims[i];
    dims[x.rank() - 2] = n;
    var out = try Tensor.empty(Shape.fromParts(dims, x.rank() - 1), x.dt);
    out.batch_depth = x.batch_depth;
    const xc = try x.toContiguous();
    defer releaseTemp(xc);
    for (0..batch) |bi| {
        for (0..n) |i| out.writeF64(bi * n + i, xc.readF64(bi * rows * x.shape.dims[x.rank() - 1] + i * x.shape.dims[x.rank() - 1] + i));
    }
    return ops.finish(.slice, &.{x}, out, .{});
}

/// Outer product of two 1-D tensors: `a ⊗ b`.
pub fn outer(a: Tensor, b: Tensor) Error!Tensor {
    const ae = try ops.expandDims(a, -1);
    defer releaseTemp(ae);
    return ops.mul(ae, b);
}

/// `a @ b` where the last axis of `a` and the first axis of `b` (payload
/// relative) contract — useful for einsum-style code.
pub fn contract(a: Tensor, b: Tensor) Error!Tensor {
    return ops.matmul(a, b);
}

test "solve, inv, det and cholesky" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    const S = Shape.initComptime;
    const T = Tensor;

    const a = try T.fromArray(f64, &.{ 4, 3, 6, 3 }, S(&.{ 2, 2 }));
    const b = try T.fromArray(f64, &.{ 10, 12 }, S(&.{2}));
    const x = try solve(a, b);
    // 4x + 3y = 10 ; 6x + 3y = 12 -> x = 1, y = 2
    try std.testing.expectApproxEqAbs(@as(f64, 1), x.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), x.readF64(1), 1e-12);

    const ai = try inv(a);
    const identity = try ops.matmul(a, ai);
    try std.testing.expectApproxEqAbs(@as(f64, 1), identity.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), identity.readF64(1), 1e-12);

    const d = try det(a);
    try std.testing.expectApproxEqAbs(@as(f64, -6), d.item(), 1e-12);

    const spd = try T.fromArray(f64, &.{ 4, 2, 2, 3 }, S(&.{ 2, 2 }));
    const l = try cholesky(spd, false);
    const ll = try ops.matmul(l, try ops.transpose(l, &.{ 1, 0 }));
    try std.testing.expectApproxEqAbs(@as(f64, 4), ll.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), ll.readF64(1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), ll.readF64(3), 1e-12);

    const tr = try T.fromArray(f64, &.{ 1, 0, 0, 2 }, S(&.{ 2, 2 }));
    const bb = try T.fromArray(f64, &.{ 3, 4 }, S(&.{2}));
    const ts = try triangularSolve(tr, bb, false);
    try std.testing.expectApproxEqAbs(@as(f64, 3), ts.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), ts.readF64(1), 1e-12);
}
