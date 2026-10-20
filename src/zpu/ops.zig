//! The eager op engine.
//!
//! Every public function in this file
//!
//! 1. materialises a *new* result tensor (or a view of an input),
//! 2. hands it to `finish`, which records it on the active tape (`autodiff`)
//!    and/or the active graph recorder (`jit`), and
//! 3. returns it.
//!
//! Because recording is centralised here, the whole public surface is
//! automatically differentiable, vmappable and jit-able.
//!
//! Performance optimizations:
//! - SIMD vectorization via kernels.zig (32-byte vectors)
//! - Multi-threaded parallel execution for large tensors via parallel.zig
//! - Tiled GEMM with cache-friendly access patterns
const std = @import("std");
const dtype_mod = @import("dtype.zig");
const shape_mod = @import("shape.zig");
const context = @import("context.zig");
const prims = @import("prims.zig");
const tmod = @import("tensor.zig");
const scalar = @import("scalar.zig");
const kernels = @import("kernels.zig");
const tape_mod = @import("tape.zig");
const graph_mod = @import("graph.zig");
const parallel = @import("parallel.zig");

pub const DType = dtype_mod.DType;
pub const Shape = shape_mod.Shape;
pub const Tensor = tmod.Tensor;
pub const Range = tmod.Range;
pub const Buffer = tmod.Buffer;
pub const Prim = prims.Prim;
pub const Attrs = prims.Attrs;
pub const Error = tmod.Error;
pub const MAX_RANK = prims.MAX_RANK;
pub const AXIS_NONE = prims.AXIS_NONE;
pub const AXIS_LAST = prims.AXIS_LAST;

pub const releaseTemp = tmod.releaseTemp;

// =============================================================================
// Recording
// =============================================================================

/// Records `out` on the active tape and graph and propagates `requires_grad`.
pub fn finish(prim: Prim, inputs: []const Tensor, out: Tensor, attrs: Attrs) Error!Tensor {
    var o = out;
    for (inputs) |i| {
        if (i.requires_grad) {
            o.requires_grad = true;
            break;
        }
    }
    if (o.requires_grad and prims.isPredicate(prim)) o.requires_grad = false;
    try tape_mod.record(prim, inputs, o, attrs);
    graph_mod.record(prim, inputs, o, attrs);
    return o;
}

/// `finish` under a name that reads better for hand written results.
pub fn sealResult(prim: Prim, inputs: []const Tensor, out: Tensor, attrs: Attrs) Error!Tensor {
    return finish(prim, inputs, out, attrs);
}

pub fn sealOut(prim: Prim, inputs: []const Tensor, out: *Tensor, attrs: Attrs) void {
    out.* = finish(prim, inputs, out.*, attrs) catch out.*;
}

/// Releases a temporary created inside an op unless a scope owns it.
pub fn release(t: Tensor) void {
    releaseTemp(t);
}

pub const isPredicate = prims.isPredicate;
pub const isFloatOnly = prims.isFloatOnly;

// =============================================================================
// Element-wise kernels
// =============================================================================

/// Iterates the *memory offsets* of a (possibly broadcast) tensor in row major
/// order.
const Offsets = struct {
    dims: [MAX_RANK]usize = @splat(0),
    st: [MAX_RANK]isize = @splat(0),
    idx: [MAX_RANK]usize = @splat(0),
    rank: usize = 0,
    n: usize = 0,
    count: usize = 0,
    off: isize = 0,

    fn init(shape: Shape, strides: [MAX_RANK]isize) Offsets {
        var o = Offsets{ .rank = shape.rank(), .n = shape.numel() };
        o.dims = shape.dims;
        o.st = strides;
        return o;
    }

    fn next(self: *Offsets) ?isize {
        if (self.count >= self.n) return null;
        const o = self.off;
        self.count += 1;
        var a = self.rank;
        while (a > 0) {
            a -= 1;
            self.idx[a] += 1;
            self.off += self.st[a];
            if (self.idx[a] < self.dims[a]) return o;
            self.off -= @as(isize, @intCast(self.dims[a])) * self.st[a];
            self.idx[a] = 0;
        }
        return o;
    }
};

/// Strides that broadcast `t` onto `out_shape` (right aligned, 0 on the axes
/// that are repeated).
fn broadcastStrides(t: Tensor, out_shape: Shape) [MAX_RANK]isize {
    var st: [MAX_RANK]isize = @splat(0);
    const ro = out_shape.len;
    const rt = t.rank();
    for (0..rt) |i| {
        const od = ro - rt + i;
        st[od] = if (out_shape.dims[od] == t.shape.dims[i]) t.strides[i] else 0;
    }
    return st;
}

fn iterFor(t: Tensor, out_shape: Shape) Offsets {
    return Offsets.init(out_shape, broadcastStrides(t, out_shape));
}

/// The buffer index of a memory offset from a view's origin.
///
/// The offsets the iterators walk are *signed*, because a view may have negative
/// strides (`flip`, and the reverse slices that come out of it) and its origin is
/// its first logical element. The sum is therefore only known to be a valid index
/// *after* the addition, which is why this is one helper rather than a cast at
/// every use site.
inline fn at(t: Tensor, off: isize) usize {
    const base: isize = @intCast(t.offset);
    return @intCast(base + off);
}

fn readAtF64(t: Tensor, off: isize) f64 {
    const i = at(t, off);
    return switch (t.dt) {
        inline else => |d| switch (d) {
            .bool => if (t.buf.data(bool)[i]) 1 else 0,
            inline else => scalar.toF64(t.buf.data(DType.Elt(d))[i]),
        },
    };
}

fn unaryKernel(comptime T: type, comptime p: Prim, a: Tensor, out: Tensor) void {
    const d = out.flatSliceMut(T).?;
    if (a.isContiguous() and a.shape.eql(out.shape)) {
        const src = a.buf.data(T);
        const in = src[a.offset..][0..d.len];
        // The vector kernel when this primitive has an exact one (see
        // `kernels.hasVectorForm`), the scalar loop otherwise. Nothing here
        // decides *what* to compute — only how many lanes at a time.
        if (kernels.unaryFast(T, p, d, in)) return;
        for (0..d.len) |j| d[j] = scalar.unary(T, p, in[j]);
        return;
    }
    const src = a.buf.data(T);
    var it = iterFor(a, out.shape);
    var j: usize = 0;
    while (it.next()) |o| : (j += 1) {
        d[j] = scalar.unary(T, p, src[at(a, o)]);
    }
}

fn unaryKernelF64(comptime p: Prim, a: Tensor, out: Tensor) void {
    var it = iterFor(a, out.shape);
    var j: usize = 0;
    while (it.next()) |o| : (j += 1) {
        out.writeF64(j, scalar.unaryF(p, readAtF64(a, o)));
    }
}

fn predKernel1(comptime p: Prim, a: Tensor, out: Tensor) void {
    const d = out.flatSliceMut(bool).?;
    var it = iterFor(a, out.shape);
    var j: usize = 0;
    while (it.next()) |o| : (j += 1) {
        d[j] = scalar.pred1F(p, readAtF64(a, o));
    }
}

fn binaryKernel(comptime T: type, comptime p: Prim, a: Tensor, b: Tensor, out: Tensor) void {
    const d = out.flatSliceMut(T).?;
    const da = a.buf.data(T);
    const db = b.buf.data(T);
    const fast_a = a.isContiguous() and a.shape.eql(out.shape);
    const fast_b = b.isContiguous() and b.shape.eql(out.shape);
    // Three shapes cover almost every real call: two arrays of the output's own
    // shape, and an array with a broadcast scalar. Both go to the vector
    // kernels; everything else keeps the general (stride-walking) loop below.
    if (fast_a and fast_b) {
        if (kernels.binaryFast(T, p, d, da[a.offset..][0..d.len], db[b.offset..][0..d.len])) return;
    } else if (fast_a and b.numel() == 1) {
        const s = scalar.toF64(db[b.offset]);
        const v: T = scalar.fromF64(T, s);
        if (kernels.binaryScalarFast(T, p, d, da[a.offset..][0..d.len], v)) return;
    } else if (fast_b and a.numel() == 1) {
        const s = scalar.toF64(da[a.offset]);
        const v: T = scalar.fromF64(T, s);
        // Only the commuting operators may swap the operands.
        switch (p) {
            .add, .mul, .maximum, .minimum => if (kernels.binaryScalarFast(T, p, d, db[b.offset..][0..d.len], v)) return,
            else => {},
        }
    }
    var ia = iterFor(a, out.shape);
    var ib = iterFor(b, out.shape);
    const n = out.numel();
    for (0..n) |j| {
        const va = if (fast_a) da[a.offset + j] else da[at(a, ia.next().?)];
        const vb = if (fast_b) db[b.offset + j] else db[at(b, ib.next().?)];
        d[j] = scalar.binary(T, p, va, vb);
    }
}

fn binaryKernelF64(comptime p: Prim, a: Tensor, b: Tensor, out: Tensor) void {
    var ia = iterFor(a, out.shape);
    var ib = iterFor(b, out.shape);
    const n = out.numel();
    for (0..n) |j| {
        const va = readAtF64(a, ia.next().?);
        const vb = readAtF64(b, ib.next().?);
        out.writeF64(j, scalar.binaryF(p, va, vb));
    }
}

fn predKernel2(comptime p: Prim, a: Tensor, b: Tensor, out: Tensor) void {
    const d = out.flatSliceMut(bool).?;
    var ia = iterFor(a, out.shape);
    var ib = iterFor(b, out.shape);
    const n = out.numel();
    for (0..n) |j| {
        const va = readAtF64(a, ia.next().?);
        const vb = readAtF64(b, ib.next().?);
        d[j] = scalar.pred2F(p, va, vb);
    }
}

fn unaryOutDType(p: Prim, a: DType) DType {
    if (prims.isPredicate(p)) return .bool;
    if (p == .sign) return a;
    if (isFloatOnly(p)) return if (a.isFloat()) a else .f32;
    return a;
}

fn binaryOutDType(p: Prim, a: DType, b: DType) Error!DType {
    if (prims.isPredicate(p)) return .bool;
    const pr = DType.promote(a, b);
    if (pr == .bool) return Error.DTypeMismatch;
    switch (p) {
        .div => return if (pr.isFloat()) pr else .f32,
        .atan2, .hypot => return if (pr.isFloat()) pr else .f32,
        .pow => return pr,
        else => return pr,
    }
}

pub fn broadcastPair(a: Tensor, b: Tensor) Error!Shape {
    return Shape.broadcastShapeRanked(a.shape, b.shape, @max(a.batch_depth, b.batch_depth));
}

pub fn outDepth(inputs: []const Tensor) u8 {
    var d: u8 = 0;
    for (inputs) |t| d = @max(d, t.batch_depth);
    return d;
}

fn mapUnary(comptime p: Prim, a: Tensor) Error!Tensor {
    const dt = unaryOutDType(p, a.dt);
    if (comptime prims.isPredicate(p)) {
        var out = try Tensor.empty(a.shape, .bool);
        predKernel1(p, a, out);
        out.batch_depth = a.batch_depth;
        return finish(p, &.{a}, out, .{});
    }
    var out = try Tensor.empty(a.shape, dt);
    out.batch_depth = a.batch_depth;
    if (dt == a.dt) {
        switch (a.dt) {
            inline else => |d| unaryKernel(DType.Elt(d), p, a, out),
        }
    } else {
        unaryKernelF64(p, a, out);
    }
    return finish(p, &.{a}, out, .{});
}

fn mapBinary(comptime p: Prim, a: Tensor, b: Tensor) Error!Tensor {
    const shape = try broadcastPair(a, b);
    const dt = try binaryOutDType(p, a.dt, b.dt);
    var out = try Tensor.empty(shape, dt);
    out.batch_depth = outDepth(&.{ a, b });
    if (comptime prims.isPredicate(p)) {
        predKernel2(p, a, b, out);
    } else if (a.dt == b.dt and a.dt == dt) {
        switch (a.dt) {
            .bool => binaryKernelF64(p, a, b, out),
            inline else => |d| binaryKernel(DType.Elt(d), p, a, b, out),
        }
    } else {
        binaryKernelF64(p, a, b, out);
    }
    return finish(p, &.{ a, b }, out, .{});
}

fn mapScalar(comptime p: Prim, a: Tensor, v: f64) Error!Tensor {
    // Scalars adapt to the tensor's dtype (`f32 + f64 scalar` stays f32), except
    // for `div`/`pow` which always produce a float.
    const dt: DType = switch (p) {
        .div, .pow => if (a.dt.isFloat()) a.dt else .f32,
        else => a.dt,
    };
    const s = try Tensor.fromScalarF64(v, dt);
    defer releaseTemp(s);
    return mapBinary(p, a, s);
}

// =============================================================================
// Element-wise API
// =============================================================================

pub fn neg(a: Tensor) Error!Tensor {
    return mapUnary(.neg, a);
}
pub fn abs(a: Tensor) Error!Tensor {
    return mapUnary(.abs, a);
}
pub fn exp(a: Tensor) Error!Tensor {
    return mapUnary(.exp, a);
}
pub fn expm1(a: Tensor) Error!Tensor {
    return mapUnary(.expm1, a);
}
pub fn log(a: Tensor) Error!Tensor {
    return mapUnary(.log, a);
}
pub fn log1p(a: Tensor) Error!Tensor {
    return mapUnary(.log1p, a);
}
pub fn sqrt(a: Tensor) Error!Tensor {
    return mapUnary(.sqrt, a);
}
pub fn rsqrt(a: Tensor) Error!Tensor {
    return mapUnary(.rsqrt, a);
}
pub fn square(a: Tensor) Error!Tensor {
    return mapBinary(.mul, a, a);
}
pub fn recip(a: Tensor) Error!Tensor {
    return mapUnary(.recip, a);
}
pub fn cbrt(a: Tensor) Error!Tensor {
    return mapUnary(.cbrt, a);
}
pub fn sin(a: Tensor) Error!Tensor {
    return mapUnary(.sin, a);
}
pub fn cos(a: Tensor) Error!Tensor {
    return mapUnary(.cos, a);
}
pub fn tan(a: Tensor) Error!Tensor {
    return mapUnary(.tan, a);
}
pub fn asin(a: Tensor) Error!Tensor {
    return mapUnary(.asin, a);
}
pub fn acos(a: Tensor) Error!Tensor {
    return mapUnary(.acos, a);
}
pub fn atan(a: Tensor) Error!Tensor {
    return mapUnary(.atan, a);
}
pub fn sinh(a: Tensor) Error!Tensor {
    return mapUnary(.sinh, a);
}
pub fn cosh(a: Tensor) Error!Tensor {
    return mapUnary(.cosh, a);
}
pub fn tanh(a: Tensor) Error!Tensor {
    return mapUnary(.tanh, a);
}
pub fn asinh(a: Tensor) Error!Tensor {
    return mapUnary(.asinh, a);
}
pub fn acosh(a: Tensor) Error!Tensor {
    return mapUnary(.acosh, a);
}
pub fn atanh(a: Tensor) Error!Tensor {
    return mapUnary(.atanh, a);
}
pub fn erf(a: Tensor) Error!Tensor {
    return mapUnary(.erf, a);
}
pub fn sigmoid(a: Tensor) Error!Tensor {
    return mapUnary(.sigmoid, a);
}
pub fn softplus(a: Tensor) Error!Tensor {
    return mapUnary(.softplus, a);
}
pub fn relu(a: Tensor) Error!Tensor {
    return mapUnary(.relu, a);
}
pub fn floor(a: Tensor) Error!Tensor {
    return mapUnary(.floor, a);
}
pub fn ceil(a: Tensor) Error!Tensor {
    return mapUnary(.ceil, a);
}
pub fn round(a: Tensor) Error!Tensor {
    return mapUnary(.round, a);
}
pub fn trunc(a: Tensor) Error!Tensor {
    return mapUnary(.trunc, a);
}
pub fn sign(a: Tensor) Error!Tensor {
    return mapUnary(.sign, a);
}
pub fn isnan(a: Tensor) Error!Tensor {
    return mapUnary(.isnan, a);
}
pub fn isinf(a: Tensor) Error!Tensor {
    return mapUnary(.isinf, a);
}
pub fn isfinite(a: Tensor) Error!Tensor {
    return mapUnary(.isfinite, a);
}

pub fn add(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.add, a, b);
}
pub fn sub(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.sub, a, b);
}
pub fn mul(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.mul, a, b);
}
pub fn div(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.div, a, b);
}
pub fn pow(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.pow, a, b);
}
pub fn maximum(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.maximum, a, b);
}
pub fn minimum(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.minimum, a, b);
}
pub fn atan2(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.atan2, a, b);
}
pub fn hypot(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.hypot, a, b);
}
pub fn mod(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.mod_op, a, b);
}
pub fn eq(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.eq, a, b);
}
pub fn ne(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.ne, a, b);
}
pub fn lt(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.lt, a, b);
}
pub fn le(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.le, a, b);
}
pub fn gt(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.gt, a, b);
}
pub fn ge(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.ge, a, b);
}
pub fn logicalAnd(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.logical_and, a, b);
}
pub fn logicalOr(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.logical_or, a, b);
}
pub fn logicalXor(a: Tensor, b: Tensor) Error!Tensor {
    return mapBinary(.logical_xor, a, b);
}

/// `not a`, as a predicate (`a == 0` for numeric inputs).
pub fn logicalNot(a: Tensor) Error!Tensor {
    const zero = try Tensor.fromScalarF64(0, a.dt);
    defer releaseTemp(zero);
    return eq(a, zero);
}

pub fn addScalar(a: Tensor, v: f64) Error!Tensor {
    return mapScalar(.add, a, v);
}
pub fn subScalar(a: Tensor, v: f64) Error!Tensor {
    return mapScalar(.sub, a, v);
}
pub fn mulScalar(a: Tensor, v: f64) Error!Tensor {
    return mapScalar(.mul, a, v);
}
pub fn divScalar(a: Tensor, v: f64) Error!Tensor {
    return mapScalar(.div, a, v);
}

/// A rank-0 tensor holding `v` in a dtype compatible with `like`.
/// A scalar with the dtype of `like`, so that integer arithmetic stays
/// integral (`jnp.int64(1) * 2` is `int64`, like NumPy/JAX). Predicates have no
/// arithmetic, so a bool tensor falls back to `f32`.
pub fn scalarLike(like: Tensor, v: f64) Error!Tensor {
    return Tensor.fromScalarF64(v, if (like.dt == .bool) .f32 else like.dt);
}

pub fn zerosLike(x: Tensor) Error!Tensor {
    return Tensor.zeros(x.shape, x.dt);
}
pub fn onesLike(x: Tensor) Error!Tensor {
    return Tensor.ones(x.shape, x.dt);
}
pub const zerosLikeT = zerosLike;
pub const onesLikeT = onesLike;

pub fn zeros(shape: Shape, dt: DType) Error!Tensor {
    return Tensor.zeros(shape, dt);
}
pub fn ones(shape: Shape, dt: DType) Error!Tensor {
    return Tensor.ones(shape, dt);
}
pub fn full(shape: Shape, dt: DType, v: f64) Error!Tensor {
    return Tensor.full(shape, dt, v);
}
pub fn arange(n: usize, dt: DType) Error!Tensor {
    return Tensor.arange(n, dt);
}
pub fn linspace(lo: f64, hi: f64, n: usize, dt: DType) Error!Tensor {
    return Tensor.linspace(lo, hi, n, dt);
}
pub fn eye(n: usize, dt: DType) Error!Tensor {
    return Tensor.eye(n, dt);
}

/// `where(cond, a, b)`; `cond` is a boolean tensor, `a`/`b` are broadcast.
pub fn where(cond: Tensor, a: Tensor, b: Tensor) Error!Tensor {
    var shape = try broadcastPair(a, b);
    shape = try Shape.broadcastShapeRanked(shape, cond.shape, @max(shape.rank(), cond.shape.rank()));
    const dt = DType.promote(a.dt, b.dt);
    var out = try Tensor.empty(shape, dt);
    out.batch_depth = outDepth(&.{ a, b, cond });
    var ic = iterFor(cond, out.shape);
    var ia = iterFor(a, out.shape);
    var ib = iterFor(b, out.shape);
    const n = out.numel();
    for (0..n) |j| {
        // every operand iterator must advance in lockstep with the output
        const c = readAtF64(cond, ic.next().?) != 0;
        const av = readAtF64(a, ia.next().?);
        const bv = readAtF64(b, ib.next().?);
        out.writeF64(j, if (c) av else bv);
    }
    return finish(.where, &.{ cond, a, b }, out, .{});
}

/// `min(max(x, lo), hi)`.
pub fn clamp(x: Tensor, lo: f64, hi: f64) Error!Tensor {
    const lo_t = try scalarLike(x, lo);
    defer releaseTemp(lo_t);
    const hi_t = try scalarLike(x, hi);
    defer releaseTemp(hi_t);
    const up = try maximum(x, lo_t);
    defer releaseTemp(up);
    return minimum(up, hi_t);
}

/// `a + t * (b - a)`.
pub fn lerp(a: Tensor, b: Tensor, t: f64) Error!Tensor {
    const d = try sub(b, a);
    defer releaseTemp(d);
    const td = try mulScalar(d, t);
    defer releaseTemp(td);
    return add(a, td);
}

// =============================================================================
// Reductions
// =============================================================================

const ReduceKind = enum { sum, prod, max, min, any, all };

fn primOfReduce(comptime kind: ReduceKind) Prim {
    return switch (kind) {
        .sum => .sum,
        .prod => .prod_op,
        .max => .max_reduce,
        .min => .min_reduce,
        .any => .any_reduce,
        .all => .all_reduce,
    };
}

/// The real axes a reduction with `axis` touches.
fn axesMaskOf(x: Tensor, axis: i32) Error![MAX_RANK]bool {
    var mask: [MAX_RANK]bool = @splat(false);
    const depth: usize = x.batch_depth;
    if (axis == AXIS_NONE) {
        for (depth..x.rank()) |i| mask[i] = true;
        return mask;
    }
    var a: isize = axis;
    if (a >= 0) a += @intCast(depth) else a += @intCast(x.rank());
    if (a < 0 or a >= @as(isize, @intCast(x.rank()))) return Error.InvalidAxis;
    mask[@intCast(a)] = true;
    return mask;
}

fn reduceOutShape(x: Tensor, mask: [MAX_RANK]bool, keepdims: bool) Shape {
    var s = Shape{};
    var n: usize = 0;
    for (0..x.rank()) |i| {
        if (mask[i]) {
            if (keepdims) {
                s.dims[n] = 1;
                n += 1;
            }
        } else {
            s.dims[n] = x.shape.dims[i];
            n += 1;
        }
    }
    s.len = @intCast(n);
    return s;
}

fn reduceIdentity(comptime T: type, comptime kind: ReduceKind) T {
    return switch (kind) {
        .sum, .prod => if (T == bool)
            @compileError("bool reduce")
        else if (@typeInfo(T) == .float)
            (if (kind == .sum) @as(T, 0.0) else @as(T, 1.0))
        else
            (if (kind == .sum) @as(T, 0) else @as(T, 1)),
        .max => switch (@typeInfo(T)) {
            .float => -std.math.inf(T),
            .int => std.math.minInt(T),
            else => @compileError("bad max type"),
        },
        .min => switch (@typeInfo(T)) {
            .float => std.math.inf(T),
            .int => std.math.maxInt(T),
            else => @compileError("bad min type"),
        },
        .any => false,
        .all => true,
    };
}

fn reduceStep(comptime T: type, comptime kind: ReduceKind, acc: T, v: T) T {
    switch (kind) {
        .sum => return if (@typeInfo(T) == .int) acc +% v else acc + v,
        .prod => return if (@typeInfo(T) == .int) acc *% v else acc * v,
        .max => {
            if (v != v) return v;
            return if (v > acc) v else acc;
        },
        .min => {
            if (v != v) return v;
            return if (v < acc) v else acc;
        },
        .any => return acc or v,
        .all => return acc and v,
    }
}

fn reduceKernel(
    comptime T: type,
    comptime kind: ReduceKind,
    xc: Tensor,
    out: Tensor,
    mask: [MAX_RANK]bool,
    keepdims: bool,
) void {
    const d = out.flatSliceMut(T).?;
    for (d) |*v| v.* = reduceIdentity(T, kind);
    const ost = out.shape.stridesContiguous();
    var vst: [MAX_RANK]isize = @splat(0);
    if (keepdims) {
        for (0..xc.rank()) |i| vst[i] = if (mask[i]) 0 else ost[i];
    } else {
        var oi: usize = 0;
        for (0..xc.rank()) |i| {
            if (mask[i]) {
                vst[i] = 0;
            } else {
                vst[i] = ost[oi];
                oi += 1;
            }
        }
    }
    const src = xc.buf.data(T);
    // The fast case: the reduced axes are exactly the trailing ones, so the
    // input is `outer` contiguous runs of `inner` and the output is one value
    // per run in the same order. That is `sum(x, -1)`, `mean(x, -1)`,
    // `sum(x)`, every softmax and every layer norm — and it makes the reduction
    // a vector kernel instead of a stride walk.
    if (comptime kernels.reduceVectorised(T)) {
        if (trailingMask(xc, mask)) |split| {
            const outer = split.outer;
            const inner = split.inner;
            if (inner * outer == xc.numel() and outer == out.numel()) {
                kernels.reduceRows(T, switch (kind) {
                    .sum => .sum,
                    .prod => .prod,
                    .max => .max,
                    .min => .min,
                    .any, .all => unreachable,
                }, src[xc.offset..][0 .. outer * inner], d, inner);
                return;
            }
        }
    }
    var it = Offsets.init(xc.shape, vst);
    var j: usize = 0;
    while (it.next()) |o| : (j += 1) {
        const dst: usize = @intCast(o);
        d[dst] = reduceStep(T, kind, d[dst], src[xc.offset + j]);
    }
}

/// Splits `(outer, inner)` when `mask` reduces exactly the trailing axes of
/// `x`'s contiguous shape — the condition the row-wise reduction kernel needs.
fn trailingMask(x: Tensor, mask: [MAX_RANK]bool) ?struct { outer: usize, inner: usize } {
    var reduced_axes: usize = 0;
    var i = x.rank();
    while (i > 0) : (i -= 1) {
        if (!mask[i - 1]) break;
        reduced_axes += 1;
    }
    if (reduced_axes == 0) return null;
    // Everything before the first reduced axis must be kept.
    for (0..x.rank() - reduced_axes) |k| {
        if (mask[k]) return null;
    }
    var outer: usize = 1;
    var inner: usize = 1;
    for (0..x.rank() - reduced_axes) |k| outer *= x.shape.dims[k];
    for (x.rank() - reduced_axes..x.rank()) |k| inner *= x.shape.dims[k];
    return .{ .outer = outer, .inner = inner };
}

/// `any`/`all` on boolean inputs.
fn reduceBoolKernel(
    comptime kind: ReduceKind,
    xc: Tensor,
    out: Tensor,
    mask: [MAX_RANK]bool,
    keepdims: bool,
) void {
    const d = out.flatSliceMut(bool).?;
    @memset(d, kind == .all);
    const ost = out.shape.stridesContiguous();
    var vst: [MAX_RANK]isize = @splat(0);
    if (keepdims) {
        for (0..xc.rank()) |i| vst[i] = if (mask[i]) 0 else ost[i];
    } else {
        var oi: usize = 0;
        for (0..xc.rank()) |i| {
            if (mask[i]) {
                vst[i] = 0;
            } else {
                vst[i] = ost[oi];
                oi += 1;
            }
        }
    }
    const src = xc.buf.data(bool);
    var it = Offsets.init(xc.shape, vst);
    var j: usize = 0;
    while (it.next()) |o| : (j += 1) {
        const dst: usize = @intCast(o);
        const v = src[xc.offset + j];
        d[dst] = if (kind == .all) (d[dst] and v) else (d[dst] or v);
    }
}

fn reduceOutDType(comptime kind: ReduceKind, in: DType) Error!DType {
    return switch (kind) {
        .any, .all => .bool,
        else => if (in == .bool) Error.DTypeMismatch else in,
    };
}

fn reduceOp(comptime kind: ReduceKind, x0: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    // Handle empty tensors: return identity element
    if (x0.numel() == 0) {
        const mask = try axesMaskOf(x0, axis);
        const out_shape = reduceOutShape(x0, mask, keepdims);
        const odt = try reduceOutDType(kind, x0.dt);
        const identity: f64 = switch (kind) {
            .sum => 0.0,
            .prod => 1.0,
            .max => -std.math.inf(f64),
            .min => std.math.inf(f64),
            .any => 0.0,
            .all => 1.0,
        };
        var out = try Tensor.full(out_shape, odt, identity);
        out.batch_depth = x0.batch_depth;
        const attrs = (Attrs{ .axis = axis }).withKeepDims(keepdims);
        return finish(primOfReduce(kind), &.{x0}, out, attrs);
    }
    
    var x = x0;
    var pred_tmp: ?Tensor = null;
    defer if (pred_tmp) |p| releaseTemp(p);
    if (comptime (kind == .any or kind == .all)) {
        if (x.dt != .bool) {
            const zero = try Tensor.fromScalarF64(0, x.dt);
            defer releaseTemp(zero);
            const pred = try ne(x, zero);
            pred_tmp = pred;
            x = pred;
        }
    }
    const mask = try axesMaskOf(x, axis);
    const out_shape = reduceOutShape(x, mask, keepdims);
    const odt = try reduceOutDType(kind, x.dt);
    var out = try Tensor.empty(out_shape, odt);
    out.batch_depth = x.batch_depth;
    const xc = try x.toContiguous();
    defer releaseTemp(xc);
    if (comptime (kind == .any or kind == .all)) {
        reduceBoolKernel(kind, xc, out, mask, keepdims);
    } else {
        switch (x.dt) {
            .bool => return Error.DTypeMismatch,
            inline else => |dt| reduceKernel(DType.Elt(dt), kind, xc, out, mask, keepdims),
        }
    }
    const attrs = (Attrs{ .axis = axis }).withKeepDims(keepdims);
    return finish(primOfReduce(kind), &.{x0}, out, attrs);
}

pub fn sum(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    return reduceOp(.sum, x, axis, keepdims);
}
pub fn prod(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    return reduceOp(.prod, x, axis, keepdims);
}
pub fn max(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    return reduceOp(.max, x, axis, keepdims);
}
pub fn min(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    return reduceOp(.min, x, axis, keepdims);
}
pub fn all(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    return reduceOp(.all, x, axis, keepdims);
}
pub fn any(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    return reduceOp(.any, x, axis, keepdims);
}

pub fn sumAll(x: Tensor) Error!Tensor {
    return reduceOp(.sum, x, AXIS_NONE, false);
}

pub fn mean(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    const mask = try axesMaskOf(x, axis);
    var count: usize = 1;
    for (0..x.rank()) |i| {
        if (mask[i]) count *= x.shape.dims[i];
    }
    const fdt: DType = if (x.dt.isFloat()) x.dt else .f32;
    const s = try reduceOp(.sum, x, axis, keepdims);
    defer releaseTemp(s);
    const sf = if (s.dt == fdt) s else try cast(s, fdt);
    defer if (sf.buf != s.buf) releaseTemp(sf);
    const c = try Tensor.fromScalarF64(1.0 / @as(f64, @floatFromInt(count)), fdt);
    defer releaseTemp(c);
    return mul(sf, c);
}

pub fn meanAll(x: Tensor) Error!Tensor {
    return mean(x, AXIS_NONE, false);
}

pub fn variance(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    const m = try mean(x, axis, true);
    defer releaseTemp(m);
    const d = try sub(x, m);
    defer releaseTemp(d);
    const sq = try square(d);
    defer releaseTemp(sq);
    const v = try mean(sq, axis, keepdims);
    return v;
}

pub fn stddev(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    const v = try variance(x, axis, keepdims);
    defer releaseTemp(v);
    return sqrt(v);
}

fn argReduce(x: Tensor, axis: i32, want_max: bool) Error!Tensor {
    const mask = try axesMaskOf(x, axis);
    const out_shape = reduceOutShape(x, mask, false);
    var out = try Tensor.empty(out_shape, .i64);
    out.batch_depth = x.batch_depth;
    const xc = try x.toContiguous();
    defer releaseTemp(xc);
    const alloc = context.allocatorRef();
    const n_out = out.numel();
    var best = try alloc.alloc(f64, n_out);
    defer alloc.free(best);
    var bidx = try alloc.alloc(i64, n_out);
    defer alloc.free(bidx);
    var seen = try alloc.alloc(usize, n_out);
    defer alloc.free(seen);
    for (0..n_out) |i| {
        best[i] = 0;
        bidx[i] = 0;
        seen[i] = 0;
    }
    const ost = out_shape.stridesContiguous();
    var vst: [MAX_RANK]isize = @splat(0);
    var oi: usize = 0;
    for (0..xc.rank()) |i| {
        if (mask[i]) {
            vst[i] = 0;
        } else {
            vst[i] = ost[oi];
            oi += 1;
        }
    }
    var it = Offsets.init(xc.shape, vst);
    var j: usize = 0;
    while (it.next()) |o| : (j += 1) {
        const dst: usize = @intCast(o);
        const v = xc.readF64(j);
        const better = if (seen[dst] == 0) true else (if (want_max) v > best[dst] else v < best[dst]);
        if (better) {
            best[dst] = v;
            bidx[dst] = @intCast(seen[dst]);
        }
        seen[dst] += 1;
    }
    for (0..n_out) |i| out.writeF64(i, @floatFromInt(bidx[i]));
    return finish(if (want_max) .argmax else .argmin, &.{x}, out, .{ .axis = axis });
}

pub fn argmax(x: Tensor, axis: i32) Error!Tensor {
    return argReduce(x, axis, true);
}
pub fn argmin(x: Tensor, axis: i32) Error!Tensor {
    return argReduce(x, axis, false);
}

/// Numerically stable softmax along `axis`.
pub fn softmax(x: Tensor, axis: i32) Error!Tensor {
    const m = try reduceOp(.max, x, axis, true);
    defer releaseTemp(m);
    const s = try sub(x, m);
    defer releaseTemp(s);
    const e = try exp(s);
    defer releaseTemp(e);
    const z = try reduceOp(.sum, e, axis, true);
    defer releaseTemp(z);
    return div(e, z);
}

pub fn logSoftmax(x: Tensor, axis: i32) Error!Tensor {
    const m = try reduceOp(.max, x, axis, true);
    defer releaseTemp(m);
    const s = try sub(x, m);
    defer releaseTemp(s);
    const e = try exp(s);
    defer releaseTemp(e);
    const z = try reduceOp(.sum, e, axis, true);
    defer releaseTemp(z);
    const lz = try log(z);
    defer releaseTemp(lz);
    return sub(s, lz);
}

// =============================================================================
// Linear algebra
// =============================================================================

fn gemmTyped(
    comptime T: type,
    a: Tensor,
    b: Tensor,
    out: Tensor,
    m: usize,
    k: usize,
    n: usize,
    batches: usize,
) void {
    const x = a.buf.data(T);
    const y = b.buf.data(T);
    const d = out.flatSliceMut(T).?;
    if (comptime kernels.gemmVectorised(T)) {
        // The blocked, vectorised kernel: the same product, with `b` streamed
        // once per row block instead of once per row.
        kernels.gemmBatch(T, x[a.offset..], y[b.offset..], d, m, k, n, batches);
        return;
    }
    for (0..batches) |bi| {
        const ao = a.offset + bi * m * k;
        const bo = b.offset + bi * k * n;
        const do = bi * m * n;
        for (0..m) |i| {
            for (0..n) |j| d[do + i * n + j] = 0;
            for (0..k) |p| {
                const av = x[ao + i * k + p];
                if (av == 0) continue;
                const bbase = bo + p * n;
                for (0..n) |j| d[do + i * n + j] = reduceStep(T, .sum, d[do + i * n + j], accum(T, av, y[bbase + j]));
            }
        }
    }
}

fn accum(comptime T: type, a: T, b: T) T {
    if (comptime isBoolType(T)) return a;
    return a * b;
}

fn isBoolType(comptime T: type) bool {
    return T == bool;
}

/// `a @ b` with NumPy semantics: the last two axes are the matrix axes and the
/// rest are batched (both operands broadcast along those axes).
pub fn matmul(a0: Tensor, b0: Tensor) Error!Tensor {
    if (a0.rank() == 0 or b0.rank() == 0) return Error.ShapeMismatch;
    // A `vmap` batch axis leads an operand's shape and is carried through
    // unchanged, so it takes no part in the matrix/batch shape logic below
    // (which only ever sees payload axes).
    const depth = @max(a0.batch_depth, b0.batch_depth);
    const fl_a = a0.batch_depth < depth;
    var a = if (fl_a) try withVmapDepth(a0, depth) else a0;
    defer if (fl_a) releaseTemp(a);
    const fl_b = b0.batch_depth < depth;
    var b = if (fl_b) try withVmapDepth(b0, depth) else b0;
    defer if (fl_b) releaseTemp(b);
    const prep_a = a.payloadRank() == 1;
    if (prep_a) a = try expandDims(a, 0);
    defer if (prep_a) releaseTemp(a);
    const prep_b = b.payloadRank() == 1;
    if (prep_b) b = try expandDims(b, 1);
    defer if (prep_b) releaseTemp(b);
    const ar = a.rank();
    const br = b.rank();
    const m = a.shape.dims[ar - 2];
    const k = a.shape.dims[ar - 1];
    const k2 = b.shape.dims[br - 2];
    const n = b.shape.dims[br - 1];
    if (k != k2) return Error.ShapeMismatch;

    // batch shapes: everything except the last two axes, right aligned
    var adims: [MAX_RANK]usize = @splat(1);
    var bdims: [MAX_RANK]usize = @splat(1);
    for (0..ar - 2 - depth) |i| adims[MAX_RANK - (ar - 2 - depth) + i] = a.shape.dims[depth + i];
    for (0..br - 2 - depth) |i| bdims[MAX_RANK - (br - 2 - depth) + i] = b.shape.dims[depth + i];
    var cdims: [MAX_RANK]usize = @splat(1);
    var brank: usize = 0;
    const nb = @max(ar - 2 - depth, br - 2 - depth); // payload batch axes only
    for (0..nb) |i| {
        const idx = MAX_RANK - nb + i;
        if (adims[idx] == bdims[idx]) {
            cdims[idx] = adims[idx];
        } else if (adims[idx] == 1) {
            cdims[idx] = bdims[idx];
        } else if (bdims[idx] == 1) {
            cdims[idx] = adims[idx];
        } else {
            return Error.ShapeMismatch;
        }
    }
    brank = nb;

    var out_dims: [MAX_RANK]usize = @splat(0);
    const out_rank = brank + 2 + depth;
    if (out_rank > MAX_RANK) return Error.RankTooLarge;
    // leading vmap axes come from an operand that has them (vmap gives every
    // mapped operand the same leading axes)
    const lead = if (a0.batch_depth == depth) a0 else b0;
    for (0..depth) |i| out_dims[i] = lead.shape.dims[i];
    for (0..brank) |i| out_dims[depth + i] = cdims[MAX_RANK - brank + i];
    out_dims[out_rank - 2] = m;
    out_dims[out_rank - 1] = n;
    var out = try Tensor.empty(Shape.fromParts(out_dims, out_rank), try binaryOutDType(.mul, a.dt, b.dt));
    out.batch_depth = depth;

    // the gemm kernels walk raw memory, so strided operands are materialised
    const ac = try a.toContiguous();
    defer releaseTemp(ac);
    const bc = try b.toContiguous();
    defer releaseTemp(bc);
    const a2 = blk2: {
        var t = ac;
        t.batch_depth = a.batch_depth;
        break :blk2 t;
    };
    const b2 = blk2: {
        var t = bc;
        t.batch_depth = b.batch_depth;
        break :blk2 t;
    };

    // batch strides with broadcasting (zero strides repeat a matrix)
    // Flat list of batch axes for the gemm kernels: the `vmap` axes first (they
    // lead the buffer), then the operands' own batched payload axes. A stride of
    // 0 means "broadcast along this axis".
    const ast = a2.shape.stridesContiguous();
    const bst = b2.shape.stridesContiguous();
    var gdims: [MAX_RANK]usize = @splat(0);
    var acs: [MAX_RANK]isize = @splat(0);
    var bcs: [MAX_RANK]isize = @splat(0);
    // the kernels read the dims right-aligned, the strides left-aligned
    const slot = MAX_RANK - depth - nb;
    for (0..depth) |i| {
        const dim = lead.shape.dims[i];
        gdims[slot + i] = dim;
        acs[i] = if (a.shape.dims[i] == dim) ast[i] else 0;
        bcs[i] = if (b.shape.dims[i] == dim) bst[i] else 0;
    }
    const a_nba = ar - 2 - depth;
    const b_nba = br - 2 - depth;
    for (0..nb) |i| {
        const dim = cdims[MAX_RANK - nb + i];
        gdims[slot + depth + i] = dim;
        const aj = @as(isize, @intCast(i)) - @as(isize, @intCast(nb - a_nba));
        const bj = @as(isize, @intCast(i)) - @as(isize, @intCast(nb - b_nba));
        acs[depth + i] = if (aj >= 0 and adims[MAX_RANK - a_nba + @as(usize, @intCast(aj))] == dim)
            ast[@as(usize, @intCast(aj)) + depth]
        else
            0;
        bcs[depth + i] = if (bj >= 0 and bdims[MAX_RANK - b_nba + @as(usize, @intCast(bj))] == dim)
            bst[@as(usize, @intCast(bj)) + depth]
        else
            0;
    }
    const nbatch = depth + nb;
    const adt = a.dt;
    const bdt = b.dt;
    if (adt == bdt and adt != .bool and adt != .f16) {
        switch (adt) {
            .bool, .f16 => gemmF64(a2, b2, out, m, k, n, acs, bcs, gdims, nbatch),
            inline else => |dt| {
                if (nbatch == 0) {
                    gemmTyped(DType.Elt(dt), a2, b2, out, m, k, n, 1);
                } else {
                    gemmBatched(DType.Elt(dt), a2, b2, out, m, k, n, acs, bcs, gdims, nbatch);
                }
            },
        }
    } else {
        gemmF64(a2, b2, out, m, k, n, acs, bcs, gdims, nbatch);
    }
    // a 1-D operand was promoted to a matrix; squeeze the axes that the
    // promotion added so the result keeps NumPy's shapes:
    // (k,)@(k,n)->(n,), (m,k)@(k,)->(m,), 1-D@1-D->scalar. Squeezing (rather
    // than reshaping) keeps any leading `vmap` axes and their batching intact.
    var result = out;
    if (prep_a) {
        const s = try squeeze(result, -2);
        releaseTemp(result);
        result = s;
    }
    if (prep_b) {
        const s = try squeeze(result, -1);
        releaseTemp(result);
        result = s;
    }
    return finish(.matmul, &.{ a0, b0 }, result, .{});
}

/// Inserts leading size-1 axes so that `t` carries `depth` `vmap` axes; the
/// added axes are stride-0 views, so nothing is copied. Ops that index operands
/// per mapped element (matmul is the only one) use this to give every operand
/// the same leading axes.
fn withVmapDepth(t: Tensor, depth: u8) Error!Tensor {
    if (t.batch_depth >= depth) return t;
    const extra: usize = depth - t.batch_depth;
    if (t.rank() + extra > MAX_RANK) return Error.RankTooLarge;
    var dims: [MAX_RANK]usize = @splat(1);
    var strides: [MAX_RANK]isize = @splat(0);
    for (0..t.rank()) |i| {
        dims[extra + i] = t.shape.dims[i];
        strides[extra + i] = t.strides[i];
    }
    var out = Tensor.viewOf(t, Shape.fromParts(dims, t.rank() + extra), strides, t.offset);
    out.batch_depth = depth;
    return out;
}

fn gemmBatched(
    comptime T: type,
    a: Tensor,
    b: Tensor,
    out: Tensor,
    m: usize,
    k: usize,
    n: usize,
    acs: [MAX_RANK]isize,
    bcs: [MAX_RANK]isize,
    cdims: [MAX_RANK]usize,
    brank: usize,
) void {
    const x = a.buf.data(T);
    const y = b.buf.data(T);
    const d = out.flatSliceMut(T).?;
    var idx: [MAX_RANK]usize = @splat(0);
    var aoff: isize = 0;
    var boff: isize = 0;
    var batch: usize = 0;
    const total: usize = blk: {
        var c: usize = 1;
        for (0..brank) |i| c *= cdims[MAX_RANK - brank + i];
        break :blk c;
    };
    while (batch < total) : (batch += 1) {
        const ao = a.offset + @as(usize, @intCast(aoff));
        const bo = b.offset + @as(usize, @intCast(boff));
        const do = batch * m * n;
        if (comptime kernels.gemmVectorised(T)) {
            kernels.gemm(T, x[ao..][0 .. m * k], y[bo..][0 .. k * n], d[do..][0 .. m * n], m, k, n);
        } else {
            for (0..m) |i| {
                for (0..n) |j| d[do + i * n + j] = 0;
                for (0..k) |p| {
                    const av = x[ao + i * k + p];
                    for (0..n) |j| {
                        d[do + i * n + j] = reduceStep(T, .sum, d[do + i * n + j], accum(T, av, y[bo + p * n + j]));
                    }
                }
            }
        }
        // advance the batch index
        var ax = brank;
        while (ax > 0) {
            ax -= 1;
            const dim = cdims[MAX_RANK - brank + ax];
            idx[ax] += 1;
            aoff += acs[ax];
            boff += bcs[ax];
            if (idx[ax] < dim) break;
            aoff -= @as(isize, @intCast(dim)) * acs[ax];
            boff -= @as(isize, @intCast(dim)) * bcs[ax];
            idx[ax] = 0;
        }
    }
}

fn gemmF64(
    a: Tensor,
    b: Tensor,
    out: Tensor,
    m: usize,
    k: usize,
    n: usize,
    acs: [MAX_RANK]isize,
    bcs: [MAX_RANK]isize,
    cdims: [MAX_RANK]usize,
    brank: usize,
) void {
    var idx: [MAX_RANK]usize = @splat(0);
    var aoff: isize = 0;
    var boff: isize = 0;
    var batch: usize = 0;
    const total: usize = blk: {
        var c: usize = 1;
        for (0..brank) |i| c *= cdims[MAX_RANK - brank + i];
        break :blk c;
    };
    while (batch < total) : (batch += 1) {
        const ao = a.offset + @as(usize, @intCast(aoff));
        const bo = b.offset + @as(usize, @intCast(boff));
        const do = batch * m * n;
        for (0..m) |i| {
            for (0..n) |j| out.writeF64(do + i * n + j, 0);
            for (0..k) |p| {
                const av = readAtF64(a, @intCast(ao + i * k + p - a.offset));
                for (0..n) |j| {
                    const bv = readAtF64(b, @intCast(bo + p * n + j - b.offset));
                    const cur = out.readLogical(out.numel(), do + i * n + j);
                    out.writeF64(do + i * n + j, cur + av * bv);
                }
            }
        }
        var ax = brank;
        while (ax > 0) {
            ax -= 1;
            const dim = cdims[MAX_RANK - brank + ax];
            idx[ax] += 1;
            aoff += acs[ax];
            boff += bcs[ax];
            if (idx[ax] < dim) break;
            aoff -= @as(isize, @intCast(dim)) * acs[ax];
            boff -= @as(isize, @intCast(dim)) * bcs[ax];
            idx[ax] = 0;
        }
    }
}

// =============================================================================
// Shape / movement ops
// =============================================================================

fn shapeFromUsizes(dims: []const usize) Error!Shape {
    return Shape.init(dims);
}

pub fn reshape(x: Tensor, requested: Shape) Error!Tensor {
    var req = requested;
    if (requested.rank() + x.batch_depth == x.rank() and x.batch_depth > 0) {
        var dims: [MAX_RANK]usize = @splat(0);
        for (0..x.batch_depth) |i| dims[i] = x.shape.dims[i];
        for (0..requested.rank()) |i| dims[x.batch_depth + i] = requested.dims[i];
        req = Shape.fromParts(dims, x.rank());
    }
    const target = try reshapeTarget(x, req);
    if (target.rank() < x.batch_depth) return Error.ShapeMismatch;
    if (x.isContiguous()) {
        var t = x;
        t.shape = target;
        t.strides = target.stridesContiguous();
        t.id = @import("tensor.zig").Tensor.viewOf(x, target, target.stridesContiguous(), x.offset).id;
        var out = t;
        out = try finish(.reshape, &.{x}, out, .{ .shape = target.dims, .shape_len = @intCast(target.rank()) });
        return out;
    }
    // Non-contiguous: materialise, then view the copy. The view has to own a
    // reference of its own (and a fresh id): `c` is a temporary, and without a
    // tracker installed its `releaseTemp` below really does free the buffer.
    const c = try x.toContiguous();
    defer releaseTemp(c);
    var t = c;
    t.shape = target;
    t.strides = target.stridesContiguous();
    t.batch_depth = x.batch_depth;
    t.id = @import("tensor.zig").Tensor.viewOf(c, target, target.stridesContiguous(), 0).id;
    var out = t;
    out = try finish(.reshape, &.{x}, out, .{ .shape = target.dims, .shape_len = @intCast(target.rank()) });
    return out;
}

/// Verifies that `requested` is compatible with `x` and resolves `-1`.
pub fn reshapeTarget(x: Tensor, requested: Shape) Error!Shape {
    var target = requested;
    const n = x.numel();
    var known: usize = 1;
    var unknown: ?usize = null;
    for (0..target.rank()) |i| {
        if (target.dims[i] == 0) return Error.InvalidShape;
        if (target.dims[i] == std.math.maxInt(usize)) {
            if (unknown != null) return Error.InvalidShape;
            unknown = i;
        } else {
            known *= target.dims[i];
        }
    }
    if (unknown) |u| {
        if (known == 0 or n % known != 0) return Error.ShapeMismatch;
        target.dims[u] = n / known;
    } else if (known != n) {
        return Error.ShapeMismatch;
    }
    return target;
}

/// Flattens the payload axes from `axis` onwards into a single axis.
pub fn flattenFrom(x: Tensor, axis: usize) Error!Tensor {
    const r = x.rank();
    if (axis > r) return Error.InvalidAxis;
    var dims: [MAX_RANK]usize = @splat(0);
    var n: usize = 0;
    for (0..axis) |i| {
        dims[n] = x.shape.dims[i];
        n += 1;
    }
    var prod_: usize = 1;
    for (axis..r) |i| prod_ *= x.shape.dims[i];
    dims[n] = prod_;
    n += 1;
    return reshape(x, Shape.fromParts(dims, n));
}

pub fn flatten(x: Tensor) Error!Tensor {
    const n = x.numel();
    const s = if (x.batch_depth == 0) Shape{ .len = 1, .dims = .{ n, 0, 0, 0, 0, 0, 0, 0 } } else try Shape.init(&.{n});
    return reshape(x, s);
}

pub fn transpose(x: Tensor, perm_payload: []const usize) Error!Tensor {
    const r = x.rank();
    var pbuf: [MAX_RANK]usize = undefined;
    var perm_user: []const usize = perm_payload;
    if (perm_payload.len != r) {
        if (perm_payload.len + x.batch_depth != r) return Error.ShapeMismatch;
        for (0..x.batch_depth) |i| pbuf[i] = i;
        for (perm_payload, 0..) |p, i| pbuf[x.batch_depth + i] = p + x.batch_depth;
        perm_user = pbuf[0..r];
    }
    var out = x;
    var seen: [MAX_RANK]bool = @splat(false);
    for (0..r) |i| {
        const p = perm_user[i];
        if (p >= r or seen[p]) return Error.InvalidShape;
        seen[p] = true;
    }
    for (0..r) |i| out.strides[i] = x.strides[perm_user[i]];
    for (0..r) |i| out.shape.dims[i] = x.shape.dims[perm_user[i]];
    out.id = @import("tensor.zig").Tensor.viewOf(x, out.shape, out.strides, x.offset).id;
    for (perm_user, 0..) |p, i| out.shape.dims[i] = x.shape.dims[p];
    var attrs = Attrs{};
    attrs.nsl = @intCast(r);
    for (0..r) |i| attrs.perm[i] = @intCast(perm_user[i]);
    return finish(.transpose, &.{x}, out, attrs);
}

pub fn swapAxes(x: Tensor, a: isize, b: isize) Error!Tensor {
    const r = x.rank();
    const ia = try normalizeAxisSigned(r, a);
    const ib = try normalizeAxisSigned(r, b);
    var perm: [MAX_RANK]usize = @splat(0);
    for (0..r) |i| perm[i] = i;
    perm[ia] = ib;
    perm[ib] = ia;
    return transpose(x, perm[0..r]);
}

/// The real axis index of a payload relative axis argument.
///
/// Positive axes count from the first *payload* axis (so `axis = 0` means "the
/// first non-batch axis" under `vmap`), negative axes count from the end.
pub fn realAxis(x: Tensor, axis: i32) Error!usize {
    var a: isize = axis;
    if (a >= 0) a += @intCast(x.batch_depth) else a += @intCast(x.rank());
    if (a < 0 or a >= @as(isize, @intCast(x.rank()))) return Error.InvalidAxis;
    return @intCast(a);
}

fn normalizeAxisSigned(rank: usize, axis: isize) Error!usize {
    const rr: isize = @intCast(rank);
    const a = if (axis < 0) axis + rr else axis;
    if (a < 0 or a >= rr) return Error.InvalidAxis;
    return @intCast(a);
}

pub fn broadcastTo(x: Tensor, requested: Shape) Error!Tensor {
    const target = try broadcastTarget(x, requested);
    var out = x;
    const depth = x.batch_depth;
    // batch axes keep their strides, payload axes are right aligned
    for (0..target.rank()) |i| {
        const from_last = target.rank() - 1 - i;
        const x_from_last: ?usize = if (from_last + x.rank() >= target.rank()) from_last else null;
        if (from_last < depth) {
            out.strides[target.rank() - 1 - from_last] = x.strides[x.rank() - 1 - from_last];
        } else if (x_from_last) |xf| {
            const xi = x.rank() - 1 - xf;
            out.strides[target.rank() - 1 - from_last] = if (x.shape.dims[xi] == 1) 0 else x.strides[xi];
        }
    }
    out.shape = target;
    if (target.rank() > x.rank()) {
        // the extra leading payload axes are size 1: stride 0
        for (0..target.rank() - x.rank()) |i| out.strides[depth + i] = 0;
    }
    out.id = @import("tensor.zig").Tensor.viewOf(x, target, out.strides, x.offset).id;
    return finish(.broadcast, &.{x}, out, .{ .shape = target.dims, .shape_len = @intCast(target.rank()) });
}

pub fn broadcastTarget(x: Tensor, requested: Shape) Error!Shape {
    if (requested.rank() != x.rank()) {
        // allow the payload to be right-aligned inside `requested`
        if (requested.rank() < x.rank()) return Error.ShapeMismatch;
        for (0..x.rank()) |i| {
            const ri = requested.rank() - x.rank() + i;
            const rd = requested.dims[ri];
            const xd = x.shape.dims[i];
            if (rd != xd and xd != 1 and rd != 1) return Error.ShapeMismatch;
        }
        return requested;
    }
    for (0..x.rank()) |i| {
        const rd = requested.dims[i];
        const xd = x.shape.dims[i];
        if (rd != xd and xd != 1 and rd != 1) return Error.ShapeMismatch;
    }
    return requested;
}

pub fn expandDims(x: Tensor, axis: i32) Error!Tensor {
    const payload = x.rank() - x.batch_depth;
    var a: isize = axis;
    if (a < 0) a += @intCast(payload + 1);
    if (a < 0 or a > @as(isize, @intCast(payload))) return Error.InvalidAxis;
    const pos: usize = x.batch_depth + @as(usize, @intCast(a));
    const target = try x.shape.withInserted(pos);
    var out = x;
    out.shape = target;
    var i = x.rank();
    while (i > pos) : (i -= 1) out.strides[i] = x.strides[i - 1];
    out.strides[pos] = 0;
    out.id = @import("tensor.zig").Tensor.viewOf(x, target, out.strides, x.offset).id;
    return finish(.expand_dims, &.{x}, out, .{ .axis = axis });
}

pub fn squeeze(x: Tensor, axis: i32) Error!Tensor {
    const payload = x.rank() - x.batch_depth;
    var a: isize = axis;
    if (a < 0) a += @intCast(payload);
    if (a < 0 or a >= @as(isize, @intCast(payload))) return Error.InvalidAxis;
    const pos: usize = x.batch_depth + @as(usize, @intCast(a));
    if (x.shape.dims[pos] != 1) return Error.ShapeMismatch;
    const target = try x.shape.without(pos);
    var out = x;
    out.shape = target;
    var i = pos;
    while (i < target.rank()) : (i += 1) out.strides[i] = x.strides[i + 1];
    out.strides[target.rank()] = 0;
    out.id = @import("tensor.zig").Tensor.viewOf(x, target, out.strides, x.offset).id;
    return finish(.squeeze, &.{x}, out, .{ .axis = axis });
}

const ResolvedRange = struct { start: usize, len: usize, step: isize };

fn resolveRange(r: Range, dim: usize) Error!ResolvedRange {
    if (r.step == 0) return Error.InvalidShape;
    const d: i64 = @intCast(dim);
    const step = r.step;
    var start = r.start;
    if (start < 0) start += d;
    if (step > 0) {
        if (start < 0) start = 0;
        if (start > d) start = d;
        var end: i64 = if (r.end) |e| e else d;
        if (end < 0) end += d;
        if (end > d) end = d;
        if (end < start) end = start;
        const len = @divTrunc(end - start + step - 1, step);
        return .{ .start = @intCast(start), .len = @intCast(len), .step = step };
    } else {
        if (start < 0) start = 0;
        if (start > d - 1) start = d - 1;
        var end: i64 = if (r.end) |e| e else -1;
        if (end < 0) end += d;
        if (end < -1) end = -1;
        if (end > start) end = start;
        const len = @divTrunc(start - end + (-step) - 1, -step);
        return .{ .start = @intCast(start), .len = @intCast(len), .step = step };
    }
}

/// A strided view following `ranges` (one range per axis of `x`).
pub fn sliceView(x: Tensor, ranges_user: []const Range) Error!Tensor {
    var rbuf: [MAX_RANK]Range = undefined;
    const ranges = try fullRanges(x, ranges_user, &rbuf);
    if (ranges.len != x.rank()) return Error.ShapeMismatch;
    var out = x;
    // Signed throughout: slicing a flipped (or otherwise negatively strided)
    // tensor moves the offset *backwards*, so a usize accumulator would trap
    // instead of producing the view. The result is still a valid element index.
    var off: isize = @intCast(x.offset);
    for (0..x.rank()) |i| {
        const rr = try resolveRange(ranges[i], x.shape.dims[i]);
        out.shape.dims[i] = rr.len;
        out.strides[i] = x.strides[i] * rr.step;
        off += @as(isize, @intCast(rr.start)) * x.strides[i];
    }
    out.offset = @intCast(off);
    out.id = @import("tensor.zig").Tensor.viewOf(x, out.shape, out.strides, out.offset).id;
    return out;
}

pub fn sliceRanges(x: Tensor, ranges_user: []const Range) Error!Tensor {
    var out = try sliceView(x, ranges_user);
    out = try finish(.slice, &.{x}, out, sliceAttrs(x, ranges_user));
    return out;
}

/// Ranges for a *batched* tensor may be given payload relative: the batch axes
/// are then filled in as "everything".
fn fullRanges(x: Tensor, ranges_user: []const Range, buf: *[MAX_RANK]Range) Error![]const Range {
    if (ranges_user.len == x.rank()) return ranges_user;
    if (ranges_user.len + x.batch_depth != x.rank()) return Error.ShapeMismatch;
    for (0..x.batch_depth) |i| buf[i] = Range.all();
    for (ranges_user, 0..) |r, i| buf[x.batch_depth + i] = r;
    return buf[0..x.rank()];
}

fn sliceAttrs(x: Tensor, ranges_user: []const Range) Attrs {
    var rbuf: [MAX_RANK]Range = undefined;
    const ranges = fullRanges(x, ranges_user, &rbuf) catch return Attrs{};
    var attrs = Attrs{};
    attrs.nsl = @intCast(x.rank());
    for (0..x.rank()) |i| {
        const rr = resolveRange(ranges[i], x.shape.dims[i]) catch return attrs;
        attrs.starts[i] = @intCast(rr.start);
        attrs.lens[i] = rr.len;
        attrs.steps[i] = rr.step;
    }
    return attrs;
}

/// A single-axis slice, the workhorse of the backward pass.
pub fn narrow(x: Tensor, axis: i32, start: usize, len: usize) Error!Tensor {
    const a = try realAxis(x, axis);
    var ranges: [MAX_RANK]Range = @splat(Range.all());
    ranges[a] = .{ .start = @intCast(start), .end = @intCast(start + len) };
    return sliceRanges(x, ranges[0..x.rank()]);
}

/// Copies `src` into `dst` element-wise (logical order); shapes must match.
pub fn copyInto(dst: Tensor, src: Tensor) Error!void {
    if (!src.shape.eql(dst.shape)) return Error.ShapeMismatch;
    const n = src.numel();
    var it = Offsets.init(dst.shape, dst.strides);
    for (0..n) |j| {
        const o = it.next().?;
        writeAtF64(dst, at(dst, o), src.readLogical(n, j));
    }
}

fn writeAtF64(t: Tensor, off: usize, v: f64) void {
    switch (t.dt) {
        inline else => |d| switch (d) {
            .bool => t.buf.dataMut(bool)[off] = v != 0,
            inline else => t.buf.dataMut(DType.Elt(d))[off] = scalar.fromF64(DType.Elt(d), v),
        },
    }
}

pub fn concat(a: Tensor, b: Tensor, axis: i32) Error!Tensor {
    if (a.rank() != b.rank()) return Error.ShapeMismatch;
    const ax = try realAxis(a, axis);
    var dims = a.shape.dims;
    dims[ax] = a.shape.dims[ax] + b.shape.dims[ax];
    for (0..a.rank()) |i| {
        if (i == ax) continue;
        if (a.shape.dims[i] != b.shape.dims[i]) return Error.ShapeMismatch;
    }
    var out = try Tensor.empty(Shape.fromParts(dims, a.rank()), DType.promote(a.dt, b.dt));
    out.batch_depth = @max(a.batch_depth, b.batch_depth);
    var ra: [MAX_RANK]Range = @splat(Range.all());
    ra[ax] = .{ .start = 0, .end = @intCast(a.shape.dims[ax]) };
    const va = try sliceView(out, ra[0..a.rank()]);
    defer releaseTemp(va);
    try copyInto(va, a);
    var rb: [MAX_RANK]Range = @splat(Range.all());
    rb[ax] = .{ .start = @intCast(a.shape.dims[ax]), .end = @intCast(dims[ax]) };
    const vb = try sliceView(out, rb[0..a.rank()]);
    defer releaseTemp(vb);
    try copyInto(vb, b);
    return finish(.concat_op, &.{ a, b }, out, .{ .axis = axis });
}

pub fn stack(a: Tensor, b: Tensor, axis: i32) Error!Tensor {
    const ae = try expandDims(a, axis);
    defer releaseTemp(ae);
    const be = try expandDims(b, axis);
    defer releaseTemp(be);
    return concat(ae, be, axis);
}

pub fn take(x: Tensor, indices: Tensor, axis: i32) Error!Tensor {
    const ax = try normalizeAxisSigned(x.rank(), axis);
    if (x.batch_depth > 0 and indices.batch_depth > 0) return Error.NotSupported;
    const outer: usize = blk: {
        var c: usize = 1;
        for (0..ax) |i| c *= x.shape.dims[i];
        break :blk c;
    };
    const inner: usize = blk: {
        var c: usize = 1;
        for (ax + 1..x.rank()) |i| c *= x.shape.dims[i];
        break :blk c;
    };
    const sdim = x.shape.dims[ax];
    const k = indices.numel();
    var dims: [MAX_RANK]usize = @splat(0);
    var n: usize = 0;
    for (0..ax) |i| {
        dims[n] = x.shape.dims[i];
        n += 1;
    }
    for (0..indices.rank()) |i| {
        dims[n] = indices.shape.dims[i];
        n += 1;
    }
    for (ax + 1..x.rank()) |i| {
        dims[n] = x.shape.dims[i];
        n += 1;
    }
    const dt = DType.promote(x.dt, indices.dt);
    var out = try Tensor.empty(Shape.fromParts(dims, n), x.dt);
    out.batch_depth = @max(x.batch_depth, indices.batch_depth);
    const xc = try x.toContiguous();
    defer releaseTemp(xc);
    const ic = try indices.toContiguous();
    defer releaseTemp(ic);
    const alloc = context.allocatorRef();
    const idx = try alloc.alloc(usize, k);
    defer alloc.free(idx);
    for (0..k) |t| {
        const v = ic.readF64(t);
        var iv: i64 = @intFromFloat(v);
        if (iv < 0) iv += @intCast(sdim);
        if (iv < 0 or iv >= @as(i64, @intCast(sdim))) return Error.IndexOutOfRange;
        idx[t] = @intCast(iv);
    }
    const block = k * inner;
    for (0..outer) |o| {
        for (0..k) |t| {
            for (0..inner) |i| {
                const dst = o * block + t * inner + i;
                const src_i = (o * sdim + idx[t]) * inner + i;
                out.writeF64(dst, xc.readF64(src_i));
            }
        }
    }
    _ = dt;
    return finish(.take, &.{ x, indices }, out, .{ .axis = axis });
}

/// `out[x] += updates` at `indices` along `axis` (the adjoint of `take`).
pub fn scatterAdd(x: Tensor, indices: Tensor, updates: Tensor, axis: i32) Error!Tensor {
    const ax = try normalizeAxisSigned(x.rank(), axis);
    if (x.batch_depth > 0 and indices.batch_depth > 0) return Error.NotSupported;
    const outer: usize = blk: {
        var c: usize = 1;
        for (0..ax) |i| c *= x.shape.dims[i];
        break :blk c;
    };
    const inner: usize = blk: {
        var c: usize = 1;
        for (ax + 1..x.rank()) |i| c *= x.shape.dims[i];
        break :blk c;
    };
    const sdim = x.shape.dims[ax];
    const k = indices.numel();
    var out = try Tensor.zeros(x.shape, x.dt);
    out.batch_depth = x.batch_depth;
    const ic = try indices.toContiguous();
    defer releaseTemp(ic);
    const uc = try updates.toContiguous();
    defer releaseTemp(uc);
    const alloc = context.allocatorRef();
    const idx = try alloc.alloc(usize, k);
    defer alloc.free(idx);
    for (0..k) |t| {
        const v = ic.readF64(t);
        var iv: i64 = @intFromFloat(v);
        if (iv < 0) iv += @intCast(sdim);
        if (iv < 0 or iv >= @as(i64, @intCast(sdim))) return Error.IndexOutOfRange;
        idx[t] = @intCast(iv);
    }
    const block = k * inner;
    for (0..outer) |o| {
        for (0..k) |t| {
            for (0..inner) |i| {
                const src = o * block + t * inner + i;
                const dst = (o * sdim + idx[t]) * inner + i;
                out.writeF64(dst, out.readF64(dst) + uc.readF64(src));
            }
        }
    }
    return finish(.scatter_add, &.{ x, indices, updates }, out, .{ .axis = axis });
}

/// Dynamic slice: extracts a slice from `operand` starting at `start_indices`.
/// Unlike regular slicing, the start indices are tensors (can be computed at runtime).
/// JAX equivalent: jax.lax.dynamic_slice
pub fn dynamicSlice(operand: Tensor, start_indices: []const Tensor, slice_sizes: []const usize) Error!Tensor {
    if (start_indices.len != operand.rank()) return Error.ShapeMismatch;
    if (slice_sizes.len != operand.rank()) return Error.ShapeMismatch;
    
    // Extract start positions from tensors
    var starts: [MAX_RANK]usize = undefined;
    for (start_indices, 0..) |idx_tensor, i| {
        const idx_val = @as(i64, @intFromFloat(idx_tensor.readF64(0)));
        const normalized_idx = if (idx_val < 0) idx_val + @as(i64, @intCast(operand.shape.dims[i])) else idx_val;
        // Clamp to valid range
        const clamped = @max(0, @min(normalized_idx, @as(i64, @intCast(operand.shape.dims[i])) - @as(i64, @intCast(slice_sizes[i]))));
        starts[i] = @intCast(clamped);
    }
    
    // Create output tensor
    var out = try Tensor.empty(try Shape.init(slice_sizes), operand.dt);
    out.batch_depth = operand.batch_depth;
    
    // Copy data
    const op_c = try operand.toContiguous();
    defer releaseTemp(op_c);
    
    // Iterate over output and copy from operand
    var out_idx: [MAX_RANK]usize = @splat(0);
    for (0..out.numel()) |flat_idx| {
        // Convert flat index to multi-dimensional
        var temp = flat_idx;
        for (0..out.rank()) |dim| {
            const stride = if (dim + 1 < out.rank()) blk: {
                var s: usize = 1;
                for (dim + 1..out.rank()) |d| s *= out.shape.dims[d];
                break :blk s;
            } else 1;
            out_idx[dim] = temp / stride;
            temp %= stride;
        }
        
        // Compute source index in operand
        var src_flat: usize = 0;
        for (0..operand.rank()) |dim| {
            var stride: usize = 1;
            for (dim + 1..operand.rank()) |d| stride *= operand.shape.dims[d];
            src_flat += (starts[dim] + out_idx[dim]) * stride;
        }
        
        out.writeF64(flat_idx, op_c.readF64(src_flat));
    }
    
    return out;
}

/// Dynamic update slice: updates a slice of `operand` at `start_indices` with `update`.
/// JAX equivalent: jax.lax.dynamic_update_slice
pub fn dynamicUpdateSlice(operand: Tensor, update: Tensor, start_indices: []const Tensor) Error!Tensor {
    if (start_indices.len != operand.rank()) return Error.ShapeMismatch;
    if (update.rank() != operand.rank()) return Error.ShapeMismatch;
    
    // Extract start positions
    var starts: [MAX_RANK]usize = undefined;
    for (start_indices, 0..) |idx_tensor, i| {
        const idx_val = @as(i64, @intFromFloat(idx_tensor.readF64(0)));
        const normalized_idx = if (idx_val < 0) idx_val + @as(i64, @intCast(operand.shape.dims[i])) else idx_val;
        // Clamp to valid range
        const clamped = @max(0, @min(normalized_idx, @as(i64, @intCast(operand.shape.dims[i])) - @as(i64, @intCast(update.shape.dims[i]))));
        starts[i] = @intCast(clamped);
    }
    
    // Clone operand
    var out = try operand.clone();
    
    // Copy update data into output
    const upd_c = try update.toContiguous();
    defer releaseTemp(upd_c);
    
    var upd_idx: [MAX_RANK]usize = @splat(0);
    for (0..update.numel()) |flat_idx| {
        // Convert flat index to multi-dimensional
        var temp = flat_idx;
        for (0..update.rank()) |dim| {
            const stride = if (dim + 1 < update.rank()) blk: {
                var s: usize = 1;
                for (dim + 1..update.rank()) |d| s *= update.shape.dims[d];
                break :blk s;
            } else 1;
            upd_idx[dim] = temp / stride;
            temp %= stride;
        }
        
        // Compute destination index in output
        var dst_flat: usize = 0;
        for (0..operand.rank()) |dim| {
            var stride: usize = 1;
            for (dim + 1..operand.rank()) |d| stride *= operand.shape.dims[d];
            dst_flat += (starts[dim] + upd_idx[dim]) * stride;
        }
        
        out.writeF64(dst_flat, upd_c.readF64(flat_idx));
    }
    
    return out;
}

/// Gather: advanced indexing operation with flexible dimension mappings.
/// JAX equivalent: jax.lax.gather
/// This is a simplified version that handles the most common case.
pub fn gather(operand: Tensor, start_indices: Tensor, slice_sizes: []const usize) Error!Tensor {
    if (operand.rank() != slice_sizes.len) return Error.ShapeMismatch;
    
    // start_indices shape: [batch..., num_indices]
    // Output shape: [batch..., slice_sizes...]
    const num_indices = start_indices.shape.dims[start_indices.rank() - 1];
    if (num_indices != operand.rank()) return Error.ShapeMismatch;
    
    // Compute output shape
    var out_dims: [MAX_RANK]usize = undefined;
    var out_rank: usize = 0;
    
    // Add batch dimensions from start_indices
    for (0..start_indices.rank() - 1) |i| {
        out_dims[out_rank] = start_indices.shape.dims[i];
        out_rank += 1;
    }
    
    // Add slice dimensions
    for (slice_sizes) |size| {
        out_dims[out_rank] = size;
        out_rank += 1;
    }
    
    var out = try Tensor.empty(try Shape.init(out_dims[0..out_rank]), operand.dt);
    out.batch_depth = operand.batch_depth;
    
    const op_c = try operand.toContiguous();
    defer releaseTemp(op_c);
    const si_c = try start_indices.toContiguous();
    defer releaseTemp(si_c);
    
    // Iterate over batch dimensions
    const batch_size = if (start_indices.rank() > 1) blk: {
        var s: usize = 1;
        for (0..start_indices.rank() - 1) |i| s *= start_indices.shape.dims[i];
        break :blk s;
    } else 1;
    
    for (0..batch_size) |batch_idx| {
        // Get start indices for this batch
        var starts: [MAX_RANK]usize = undefined;
        for (0..num_indices) |i| {
            const idx_val = @as(i64, @intFromFloat(si_c.readF64(batch_idx * num_indices + i)));
            const normalized_idx = if (idx_val < 0) idx_val + @as(i64, @intCast(operand.shape.dims[i])) else idx_val;
            const clamped = @max(0, @min(normalized_idx, @as(i64, @intCast(operand.shape.dims[i])) - @as(i64, @intCast(slice_sizes[i]))));
            starts[i] = @intCast(clamped);
        }
        
        // Copy slice
        var slice_idx: [MAX_RANK]usize = @splat(0);
        var slice_numel: usize = 1;
        for (slice_sizes) |s| slice_numel *= s;
        
        for (0..slice_numel) |flat_slice_idx| {
            // Convert to multi-dim
            var temp = flat_slice_idx;
            for (0..slice_sizes.len) |dim| {
                const stride = if (dim + 1 < slice_sizes.len) blk: {
                    var s: usize = 1;
                    for (dim + 1..slice_sizes.len) |d| s *= slice_sizes[d];
                    break :blk s;
                } else 1;
                slice_idx[dim] = temp / stride;
                temp %= stride;
            }
            
            // Compute source index in operand
            var src_flat: usize = 0;
            for (0..operand.rank()) |dim| {
                var stride: usize = 1;
                for (dim + 1..operand.rank()) |d| stride *= operand.shape.dims[d];
                src_flat += (starts[dim] + slice_idx[dim]) * stride;
            }
            
            // Compute destination index in output
            const dst_flat: usize = batch_idx * slice_numel + flat_slice_idx;
            
            out.writeF64(dst_flat, op_c.readF64(src_flat));
        }
    }
    
    return out;
}

pub fn pad(x: Tensor, pads: []const [2]usize, value: f64) Error!Tensor {
    if (pads.len != x.rank()) return Error.ShapeMismatch;
    var dims = x.shape.dims;
    for (0..x.rank()) |i| dims[i] = x.shape.dims[i] + pads[i][0] + pads[i][1];
    var out = try Tensor.empty(Shape.fromParts(dims, x.rank()), x.dt);
    out.batch_depth = x.batch_depth;
    for (0..out.numel()) |i| out.writeF64(i, value);
    var ranges: [MAX_RANK]Range = @splat(Range.all());
    var attrs = Attrs{};
    attrs.nsl = @intCast(x.rank());
    attrs.scalar = value;
    for (0..x.rank()) |i| {
        ranges[i] = .{ .start = @intCast(pads[i][0]), .end = @intCast(pads[i][0] + x.shape.dims[i]) };
        attrs.pads_lo[i] = pads[i][0];
        attrs.pads_hi[i] = pads[i][1];
        attrs.starts[i] = @intCast(pads[i][0]);
        attrs.lens[i] = x.shape.dims[i];
    }
    const view = try sliceView(out, ranges[0..x.rank()]);
    defer releaseTemp(view);
    try copyInto(view, x);
    return finish(.pad, &.{x}, out, attrs);
}

pub fn flip(x: Tensor, axis: i32) Error!Tensor {
    const a = try realAxis(x, axis);
    var out = x;
    // Mirroring an axis leaves the first *logical* element at the far end of
    // that axis. Flipping twice walks back to where it started, and flipping a
    // already-flipped tensor therefore moves the offset backwards — signed.
    out.strides[a] = -x.strides[a];
    out.offset = @intCast(@as(isize, @intCast(x.offset)) +
        @as(isize, @intCast(x.shape.dims[a] - 1)) * x.strides[a]);
    out.id = @import("tensor.zig").Tensor.viewOf(x, out.shape, out.strides, out.offset).id;
    return finish(.flip, &.{x}, out, .{ .axis = axis });
}

pub fn cumsum(x: Tensor, axis: i32, reverse: bool) Error!Tensor {
    const ax = try realAxis(x, axis);
    const xc = try x.toContiguous();
    defer releaseTemp(xc);
    var out = try Tensor.empty(x.shape, x.dt);
    out.batch_depth = x.batch_depth;
    const outer: usize = blk: {
        var c: usize = 1;
        for (0..ax) |i| c *= x.shape.dims[i];
        break :blk c;
    };
    const dim = x.shape.dims[ax];
    const inner: usize = xc.numel() / (outer * dim);
    const step = inner * dim;
    for (0..outer) |o| {
        for (0..inner) |i| {
            var acc: f64 = 0;
            for (0..dim) |d| {
                const t = if (reverse) dim - 1 - d else d;
                const src = o * step + t * inner + i;
                acc += xc.readF64(src);
                out.writeF64(src, acc);
            }
        }
    }
    var attrs = Attrs{ .axis = axis };
    attrs = attrs.withReverse(reverse);
    return finish(.cumsum, &.{x}, out, attrs);
}

pub fn cast(x: Tensor, dt: DType) Error!Tensor {
    if (dt == x.dt) return x.retainScoped();
    var out = try Tensor.empty(x.shape, dt);
    out.batch_depth = x.batch_depth;
    const n = out.numel();
    for (0..n) |i| out.writeF64(i, x.readLogical(n, i));
    var attrs = Attrs{};
    attrs.axis = @intCast(@backingInt(dt));
    return finish(.cast_op, &.{x}, out, attrs);
}

/// Broadcasts x to the shape of `like` (used by the backward pass).
pub fn expandLike(x: Tensor, like: Tensor) Error!Tensor {
    return broadcastTo(x, like.shape);
}

// =============================================================================
// Interpreter used by the JIT
// =============================================================================

/// Applies `prim` to `inputs` with the recorded `attrs`. This is the only
/// entry point the JIT needs.
pub fn applyPrim(prim: Prim, inputs: []const Tensor, attrs: Attrs) Error!Tensor {
    switch (prim) {
        // unary
        .neg => return neg(inputs[0]),
        .abs => return abs(inputs[0]),
        .exp => return exp(inputs[0]),
        .expm1 => return expm1(inputs[0]),
        .log => return log(inputs[0]),
        .log1p => return log1p(inputs[0]),
        .sqrt => return sqrt(inputs[0]),
        .rsqrt => return rsqrt(inputs[0]),
        .square => return square(inputs[0]),
        .recip => return recip(inputs[0]),
        .cbrt => return cbrt(inputs[0]),
        .sin => return sin(inputs[0]),
        .cos => return cos(inputs[0]),
        .tan => return tan(inputs[0]),
        .asin => return asin(inputs[0]),
        .acos => return acos(inputs[0]),
        .atan => return atan(inputs[0]),
        .sinh => return sinh(inputs[0]),
        .cosh => return cosh(inputs[0]),
        .tanh => return tanh(inputs[0]),
        .asinh => return asinh(inputs[0]),
        .acosh => return acosh(inputs[0]),
        .atanh => return atanh(inputs[0]),
        .erf => return erf(inputs[0]),
        .sigmoid => return sigmoid(inputs[0]),
        .softplus => return softplus(inputs[0]),
        .relu => return relu(inputs[0]),
        .floor => return floor(inputs[0]),
        .ceil => return ceil(inputs[0]),
        .round => return round(inputs[0]),
        .trunc => return trunc(inputs[0]),
        .sign => return sign(inputs[0]),
        .isnan => return isnan(inputs[0]),
        .isinf => return isinf(inputs[0]),
        .isfinite => return isfinite(inputs[0]),
        .logical_not => return logicalNot(inputs[0]),
        // binary
        .add => return add(inputs[0], inputs[1]),
        .sub => return sub(inputs[0], inputs[1]),
        .mul => return mul(inputs[0], inputs[1]),
        .div => return div(inputs[0], inputs[1]),
        .pow => return pow(inputs[0], inputs[1]),
        .maximum => return maximum(inputs[0], inputs[1]),
        .minimum => return minimum(inputs[0], inputs[1]),
        .atan2 => return atan2(inputs[0], inputs[1]),
        .hypot => return hypot(inputs[0], inputs[1]),
        .mod_op => return mod(inputs[0], inputs[1]),
        .eq => return eq(inputs[0], inputs[1]),
        .ne => return ne(inputs[0], inputs[1]),
        .lt => return lt(inputs[0], inputs[1]),
        .le => return le(inputs[0], inputs[1]),
        .gt => return gt(inputs[0], inputs[1]),
        .ge => return ge(inputs[0], inputs[1]),
        .logical_and => return logicalAnd(inputs[0], inputs[1]),
        .logical_or => return logicalOr(inputs[0], inputs[1]),
        .logical_xor => return logicalXor(inputs[0], inputs[1]),
        .where => return where(inputs[0], inputs[1], inputs[2]),
        .clamp => {
            const lo = @as(f64, @floatFromInt(attrs.starts[0]));
            const hi = @as(f64, @floatFromInt(attrs.starts[1]));
            return clamp(inputs[0], lo, hi);
        },
        .lerp => return lerp(inputs[0], inputs[1], attrs.scalar),
        // reductions
        .sum => return sum(inputs[0], attrs.axis, attrs.keepDims()),
        .prod_op => return prod(inputs[0], attrs.axis, attrs.keepDims()),
        .max_reduce => return max(inputs[0], attrs.axis, attrs.keepDims()),
        .min_reduce => return min(inputs[0], attrs.axis, attrs.keepDims()),
        .any_reduce => return any(inputs[0], attrs.axis, attrs.keepDims()),
        .all_reduce => return all(inputs[0], attrs.axis, attrs.keepDims()),
        .argmax => return argmax(inputs[0], attrs.axis),
        .argmin => return argmin(inputs[0], attrs.axis),
        // movement
        .reshape => return reshape(inputs[0], Shape.fromParts(attrs.shape, attrs.shape_len)),
        .transpose => {
            var perm: [MAX_RANK]usize = @splat(0);
            for (0..attrs.nsl) |i| perm[i] = attrs.perm[i];
            return transpose(inputs[0], perm[0..attrs.nsl]);
        },
        .broadcast => return broadcastTo(inputs[0], Shape.fromParts(attrs.shape, attrs.shape_len)),
        .expand_dims => return expandDims(inputs[0], attrs.axis),
        .squeeze => return squeeze(inputs[0], attrs.axis),
        .slice => {
            var ranges: [MAX_RANK]Range = @splat(Range.all());
            for (0..attrs.nsl) |i| {
                const start = attrs.starts[i];
                const len = attrs.lens[i];
                ranges[i] = .{
                    .start = start,
                    .end = start + @as(i64, @intCast(len)),
                    .step = attrs.steps[i],
                };
            }
            return sliceRanges(inputs[0], ranges[0..attrs.nsl]);
        },
        .concat_op => return concat(inputs[0], inputs[1], attrs.axis),
        .stack_op => return stack(inputs[0], inputs[1], attrs.axis),
        .take => return take(inputs[0], inputs[1], attrs.axis),
        .scatter_add => return scatterAdd(inputs[0], inputs[1], inputs[2], attrs.axis),
        .pad => {
            var pads: [MAX_RANK][2]usize = @splat(.{ 0, 0 });
            for (0..attrs.nsl) |i| pads[i] = .{ attrs.pads_lo[i], attrs.pads_hi[i] };
            return pad(inputs[0], pads[0..attrs.nsl], attrs.scalar);
        },
        .flip => return flip(inputs[0], attrs.axis),
        .cumsum => return cumsum(inputs[0], attrs.axis, attrs.reverse()),
        .matmul => return matmul(inputs[0], inputs[1]),
        .cast_op => return cast(inputs[0], @fromBackingInt(@intCast(@as(u8, @intCast(attrs.axis))))),
        .copy => return inputs[0].retainScoped(),
        .constant => return inputs[0].retainScoped(),
        .solve, .inv, .det, .cholesky, .triangular_solve => return Error.NotSupported,
        .reverse => return flip(inputs[0], attrs.axis),
        .conv => return Error.NotSupported,
    }
}

// =============================================================================
// Tests
// =============================================================================

test "elementwise and broadcasting" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();

    const a = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 2, 3 }));
    const b = try Tensor.fromArray(f64, &.{ 10, 20, 30 }, Shape.initComptime(&.{3}));
    const c = try add(a, b);
    try std.testing.expect(c.shape.eql(Shape.initComptime(&.{ 2, 3 })));
    try std.testing.expectApproxEqAbs(@as(f64, 11), c.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 36), c.readF64(5), 1e-12);

    const s = try mulScalar(a, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 12), s.readF64(5), 1e-12);

    const m = try maximum(a, try scalarLike(a, 3.5));
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), m.readF64(0), 1e-12);

    const lt_ = try lt(a, try scalarLike(a, 3.5));
    try std.testing.expectEqual(DType.bool, lt_.dtype());
    try std.testing.expect(lt_.readF64(0) == 1);

    const w = try where(lt_, a, try scalarLike(a, 0));
    try std.testing.expectApproxEqAbs(@as(f64, 1), w.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), w.readF64(3), 1e-12);
}

test "reductions" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();

    const a = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 2, 3 }));
    const s = try sum(a, AXIS_NONE, false);
    try std.testing.expectApproxEqAbs(@as(f64, 21), s.item(), 1e-12);
    const s1 = try sum(a, 1, false);
    try std.testing.expectApproxEqAbs(@as(f64, 6), s1.readF64(0), 1e-12);
    const s1k = try sum(a, -1, true);
    try std.testing.expect(s1k.shape.eql(Shape.initComptime(&.{ 2, 1 })));
    const mx = try max(a, 0, false);
    try std.testing.expectApproxEqAbs(@as(f64, 4), mx.readF64(0), 1e-12);
    const am = try argmax(a, 1);
    try std.testing.expectEqual(DType.i64, am.dtype());
    try std.testing.expectApproxEqAbs(@as(f64, 2), am.readF64(0), 1e-12);
    const mu = try mean(a, AXIS_NONE, false);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), mu.item(), 1e-12);
    const cu = try cumsum(a, 1, false);
    try std.testing.expectApproxEqAbs(@as(f64, 6), cu.readF64(2), 1e-12);
}

test "matmul shapes" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();

    const a = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 2, 3 }));
    const b = try Tensor.fromArray(f64, &.{ 7, 8, 9, 10, 11, 12 }, Shape.initComptime(&.{ 3, 2 }));
    const c = try matmul(a, b);
    try std.testing.expect(c.shape.eql(Shape.initComptime(&.{ 2, 2 })));
    try std.testing.expectApproxEqAbs(@as(f64, 58), c.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 139), c.readF64(2), 1e-12);

    // vec @ mat
    const v = try Tensor.fromArray(f64, &.{ 1, 1, 1 }, Shape.initComptime(&.{3}));
    const r = try matmul(v, b);
    try std.testing.expect(r.shape.eql(Shape.initComptime(&.{2})));
    try std.testing.expectApproxEqAbs(@as(f64, 27), r.readF64(0), 1e-12);

    // batched
    const a3 = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 1, 0, 0, 1 }, Shape.initComptime(&.{ 2, 2, 2 }));
    const b3 = try Tensor.fromArray(f64, &.{ 5, 6, 7, 8, 1, 2, 3, 4 }, Shape.initComptime(&.{ 2, 2, 2 }));
    const c3 = try matmul(a3, b3);
    try std.testing.expect(c3.shape.eql(Shape.initComptime(&.{ 2, 2, 2 })));
    try std.testing.expectApproxEqAbs(@as(f64, 19), c3.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 43), c3.readF64(2), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4), c3.readF64(7), 1e-12);
}

test "shape ops" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();

    const a = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 2, 3 }));
    const t = try transpose(a, &.{ 1, 0 });
    try std.testing.expect(t.shape.eql(Shape.initComptime(&.{ 3, 2 })));
    try std.testing.expectApproxEqAbs(@as(f64, 4), t.readLogical(6, 1), 1e-12);
    try std.testing.expect(!t.isContiguous());

    const c = try toContiguousTest(t);
    try std.testing.expect(c.isContiguous());

    const r = try reshape(a, Shape.initComptime(&.{ 3, 2 }));
    try std.testing.expectApproxEqAbs(@as(f64, 5), r.readF64(4), 1e-12);

    const sl = try sliceRanges(a, &.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } });
    try std.testing.expect(sl.shape.eql(Shape.initComptime(&.{ 1, 2 })));
    try std.testing.expectApproxEqAbs(@as(f64, 2), sl.readLogical(2, 0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), sl.readLogical(2, 1), 1e-12);

    const p = try pad(a, &.{ .{ 1, 0 }, .{ 0, 1 } }, 0);
    try std.testing.expect(p.shape.eql(Shape.initComptime(&.{ 3, 4 })));
    try std.testing.expectApproxEqAbs(@as(f64, 0), p.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), p.readF64(4), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), p.readF64(11), 1e-12);

    const f = try flip(a, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 4), f.readLogical(6, 0), 1e-12);

    const cc = try concat(a, a, 0);
    try std.testing.expect(cc.shape.eql(Shape.initComptime(&.{ 4, 3 })));
    try std.testing.expectApproxEqAbs(@as(f64, 1), cc.readF64(6), 1e-12);
}

fn toContiguousTest(t: Tensor) Error!Tensor {
    return t.toContiguous();
}

test "take and scatter" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();

    const table = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 3, 2 }));
    const idx = try Tensor.fromArray(i64, &.{ 2, 0 }, Shape.initComptime(&.{2}));
    const g = try take(table, idx, 0);
    try std.testing.expect(g.shape.eql(Shape.initComptime(&.{ 2, 2 })));
    try std.testing.expectApproxEqAbs(@as(f64, 5), g.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), g.readF64(2), 1e-12);

    const zeros_ = try Tensor.zeros(Shape.initComptime(&.{ 3, 2 }), .f64);
    const up = try Tensor.fromArray(f64, &.{ 10, 20, 30, 40 }, Shape.initComptime(&.{ 2, 2 }));
    const back = try scatterAdd(zeros_, idx, up, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 30), back.readF64(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10), back.readF64(4), 1e-12);
}
