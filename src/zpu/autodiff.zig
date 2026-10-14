//! Reverse mode (and forward mode) automatic differentiation.
//!
//! The forward pass appends `tape.Node`s (see `tape.zig`); this file holds the
//! per-primitive vector-Jacobian rules (`contribution`) and the reverse sweep
//! (`Engine`). Everything is expressed in terms of the ordinary ops, so the
//! backward pass is *itself* differentiable: higher order derivatives like
//! `hvp` come for free.
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");
const prims = @import("prims.zig");
const tape_mod = @import("tape.zig");
const ctxmod = @import("context.zig");
const linalg = @import("linalg.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = tmod.Shape;
pub const Error = tmod.Error;
pub const Node = tape_mod.Node;
pub const Tape = tape_mod.Tape;
pub const Prim = prims.Prim;
pub const Attrs = prims.Attrs;
pub const MAX_RANK = prims.MAX_RANK;

const AZERO = ops.AXIS_NONE;
const ALAST = ops.AXIS_LAST;

fn release(x: Tensor) void {
    ops.releaseTemp(x);
}

/// True when the prim has a meaningful derivative (see `prims.zig`).
pub const differentiable = prims.differentiable;

// =============================================================================
// Shape helpers used by the backward rules
// =============================================================================

/// Reduces `g` (shaped like a broadcast op output) back to `target`.
pub fn reduceToShape(g: Tensor, target: Tensor) Error!Tensor {
    if (g.shape.eql(target.shape)) return g;
    if (g.numel() == target.numel() and g.batch_depth == target.batch_depth) {
        return ops.reshape(g, target.shape);
    }
    return unbroadcast(g, target.shape, target.batch_depth);
}

/// Sums away the axes a broadcast added, and the payload axes of `g` that the
/// operand never had.
pub fn unbroadcast(g0: Tensor, shape: Shape, in_depth: u8) Error!Tensor {
    var g = g0;
    if (g.shape.eql(shape)) return g;
    const d: usize = g.batch_depth;
    const p_cur = g.rank() - d;
    const p_in = shape.rank() - (if (in_depth == g.batch_depth) @as(usize, in_depth) else 0);
    // 1. drop extra leading payload axes (they were added by broadcasting)
    if (p_cur > p_in) {
        var k: usize = p_cur - p_in;
        while (k > 0) : (k -= 1) {
            const s = try ops.sum(g, @intCast(d), false);
            release(g);
            g = s;
        }
    }
    // 2. drop batch axes the operand does not have
    if (in_depth < g.batch_depth) {
        var k: usize = g.batch_depth - in_depth;
        while (k > 0) : (k -= 1) {
            const s = try ops.sum(g, 0, false);
            release(g);
            g = s;
        }
    }
    // 3. collapse axes that were broadcast from 1
    for (0..g.rank()) |i| {
        if (shape.dims[i] == 1 and g.shape.dims[i] != 1) {
            const s = try ops.sum(g, @intCast(i), true);
            release(g);
            g = s;
        }
    }
    if (!g.shape.eql(shape)) {
        const r = try ops.reshape(g, shape);
        release(g);
        g = r;
    }
    return g;
}

/// Re-inserts the axes a reduction dropped, then broadcasts to `shape`.
fn expandReduced(g: Tensor, x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    var gg = g;
    if (!keepdims) {
        // rebuild the reduced shape with 1s on the reduced axes, then broadcast
        const depth = x.batch_depth;
        const prank = x.rank() - depth;
        var red: [MAX_RANK]bool = @splat(false);
        if (axis == ops.AXIS_NONE) {
            for (0..prank) |i| red[i] = true;
        } else {
            const a64: i64 = if (axis >= 0) axis else @as(i64, @intCast(prank)) + axis;
            if (a64 < 0 or a64 >= prank) return Error.InvalidAxis;
            red[@intCast(a64)] = true;
        }
        var dims: [MAX_RANK]usize = @splat(1);
        var n: usize = 0;
        for (0..depth) |i| {
            dims[n] = x.shape.dims[i];
            n += 1;
        }
        for (0..prank) |i| {
            dims[n] = if (red[i]) 1 else x.shape.dims[depth + i];
            n += 1;
        }
        gg = try ops.reshape(g, Shape.fromParts(dims, n));
    }
    return ops.broadcastTo(gg, x.shape);
}

fn zerosLike(x: Tensor) Error!Tensor {
    return Tensor.zeros(x.shape, x.dtype());
}

fn floatLike(x: Tensor) DType {
    return if (x.dtype().isFloat()) x.dtype() else .f32;
}

/// `1` in the element type of `x`.
fn oneLike(x: Tensor) Error!Tensor {
    return Tensor.fromScalarF64(1.0, floatLike(x));
}

fn scalarLike(x: Tensor, v: f64) Error!Tensor {
    return Tensor.fromScalarF64(v, floatLike(x));
}

/// Slicing ranges for the payload axes of `t` (see `Attrs.starts` / `.lens`).
fn payloadRanges(t: Tensor, attrs: Attrs, buf: *[MAX_RANK]tmod.Range) []const tmod.Range {
    const depth = t.batch_depth;
    var n: usize = 0;
    for (depth..t.rank()) |i| {
        buf[n] = .{
            .start = attrs.starts[i],
            .end = attrs.starts[i] + @as(i64, @intCast(attrs.lens[i])),
            .step = attrs.steps[i],
        };
        n += 1;
    }
    return buf[0..n];
}

/// The real index of a payload-relative axis.
fn realAxisOf(t: Tensor, axis: i32) usize {
    if (axis < 0) {
        return @intCast(@as(i64, @intCast(t.rank())) + axis);
    }
    return @as(usize, @intCast(axis)) + t.batch_depth;
}

// =============================================================================
// The backward rules
// =============================================================================

/// Primitives that have no backward rule, and what a gradient through them
/// means. `contribution` answers from this classification, so the compiler
/// refuses to build a primitive that was never classified — a new op cannot
/// quietly drop its gradient.
const Kind = enum { rule, zero, unimplemented };
const NoRule = enum {
    /// Discrete / comparison ops: the derivative really is zero.
    zero,
    /// The rule is not written yet. Differentiating through it is an error, so
    /// it can never pass as a plausible looking zero.
    unimplemented,
};

const no_rule = struct {
    const zero = [_]Prim{
        .floor, .ceil,   .round,     .trunc,     .sign,    .isnan,      .isinf,       .isfinite,
        .argmax, .argmin, .all_reduce, .any_reduce, .constant,
        .eq,     .ne,     .lt,        .le,        .gt,      .ge,
        .logical_and, .logical_or, .logical_xor, .logical_not,
    };
    // `conv` (a 4-D convolution over batches) and the Cholesky factorisation
    // have known VJPs, but they need primitives ZPU does not have yet:
    // `conv` differentiates to another convolution, Cholesky to a triangular
    // mask (Murray 2016). `triangular_solve` is a scaled transpose solve.
    const unimplemented = [_]Prim{ .cholesky, .triangular_solve };

    fn contains(comptime list: []const Prim, comptime p: Prim) bool {
        for (list) |q| if (q == p) return true;
        return false;
    }
};

/// How a gradient flows through `p`: `.rule` (see the switch in
/// `contribution`), `.zero`, or `.unimplemented`.
fn classify(comptime p: Prim) Kind {
    @setEvalBranchQuota(10_000);
    if (no_rule.contains(&no_rule.zero, p)) return .zero;
    if (no_rule.contains(&no_rule.unimplemented, p)) return .unimplemented;
    return .rule;
}

comptime {
    // Every primitive is either differentiable, and then `contribution` has a
    // rule for it, or it is classified above. Anything else is a mistake that
    // would silently drop gradients at run time, so it fails the build instead.
    @setEvalBranchQuota(10_000);
    const info = @typeInfo(Prim).@"enum";
    for (info.field_names, info.field_values) |name, value| {
        const p: Prim = @enumFromInt(value);
        if ((classify(p) != .rule) == prims.differentiable(p))
            @compileError("Prim." ++ name ++ ": `no_rule` and `differentiable()` disagree");
    }
}

/// Returns d(out)/d(input[i]) * g, or null when the input is not
/// differentiable. The result has the shape of `node.inputs[i]`.
pub fn contribution(node: *const Node, i: usize, g: Tensor) Error!?Tensor {
    const a = node.inputs[0];
    const b = if (node.n_inputs > 1) node.inputs[1] else a;
    const y = node.out;
    const attrs = node.attrs;

    const raw: ?Tensor = switch (node.prim) {
        .constant => null,

        // ---- elementwise unary ------------------------------------------------
        .neg => try ops.neg(g),
        .abs => try ops.mul(g, try ops.sign(a)),
        .exp => try ops.mul(g, y),
        .expm1 => try ops.mul(g, try ops.exp(a)),
        .log => try ops.div(g, a),
        .log1p => blk: {
            const one = try oneLike(a);
            const d = try ops.add(a, one);
            break :blk try ops.div(g, d);
        },
        .sqrt => blk: {
            const two = try scalarLike(a, 2);
            const d = try ops.mul(y, two);
            break :blk try ops.div(g, d);
        },
        .rsqrt => blk: {
            const half = try scalarLike(a, -0.5);
            const gn = try ops.mul(g, half);
            break :blk try ops.div(try ops.mul(gn, y), a);
        },
        .square => blk: {
            const two = try scalarLike(a, 2);
            break :blk try ops.mul(g, try ops.mul(two, a));
        },
        .recip => blk: {
            const y2 = try ops.square(y);
            break :blk try ops.neg(try ops.mul(g, y2));
        },
        .cbrt => blk: {
            const three = try scalarLike(a, 3);
            const d = try ops.mul(three, try ops.square(y));
            break :blk try ops.div(g, d);
        },
        .sin => try ops.mul(g, try ops.cos(a)),
        .cos => try ops.neg(try ops.mul(g, try ops.sin(a))),
        .tan => blk: {
            const y2 = try ops.square(y);
            const one = try oneLike(a);
            break :blk try ops.mul(g, try ops.add(one, y2));
        },
        .asin => try ops.div(g, try ops.sqrt(try subOne(try ops.square(a)))),
        .acos => try ops.neg(try ops.div(g, try ops.sqrt(try subOne(try ops.square(a))))),
        .atan => blk: {
            const a2 = try ops.square(a);
            const one = try oneLike(a);
            break :blk try ops.div(g, try ops.add(one, a2));
        },
        .sinh => try ops.mul(g, try ops.cosh(a)),
        .cosh => try ops.mul(g, try ops.sinh(a)),
        .tanh => blk: {
            const one = try oneLike(a);
            const y2 = try ops.square(y);
            break :blk try ops.mul(g, try ops.sub(one, y2));
        },
        .asinh => blk: {
            const a2 = try ops.square(a);
            const one = try oneLike(a);
            break :blk try ops.div(g, try ops.sqrt(try ops.add(one, a2)));
        },
        .acosh => blk: {
            const a2 = try ops.square(a);
            const one = try oneLike(a);
            break :blk try ops.div(g, try ops.sqrt(try ops.sub(a2, one)));
        },
        .atanh => blk: {
            const a2 = try ops.square(a);
            break :blk try ops.div(g, try subOne(a2));
        },
        .erf => blk: {
            const c = try scalarLike(a, 2.0 / @sqrt(std.math.pi));
            const a2 = try ops.square(a);
            const e = try ops.exp(try ops.neg(a2));
            break :blk try ops.mul(g, try ops.mul(c, e));
        },
        .sigmoid => blk: {
            const one = try oneLike(a);
            const s = try ops.sub(one, y);
            break :blk try ops.mul(g, try ops.mul(y, s));
        },
        .softplus => try ops.mul(g, try ops.sigmoid(a)),
        .relu => blk: {
            const zero = try scalarLike(a, 0);
            const pos = try ops.gt(a, zero);
            break :blk try ops.where(pos, g, try zerosLike(g));
        },
        .floor, .ceil, .round, .trunc, .sign, .isnan, .isinf, .isfinite => null,

        // ---- elementwise binary ------------------------------------------------
        .add => g,
        .sub => if (i == 0) g else try ops.neg(g),
        .mul => if (i == 0) try ops.mul(g, b) else try ops.mul(g, a),
        .div => if (i == 0)
            try ops.div(g, b)
        else
            try ops.neg(try ops.div(try ops.mul(g, a), try ops.square(b))),
        .pow => if (i == 0) blk: {
            const one = try oneLike(a);
            const e = try ops.sub(b, one);
            const p = try ops.pow(a, e);
            break :blk try ops.mul(try ops.mul(g, b), p);
        } else blk: {
            const l = try ops.log(a);
            break :blk try ops.mul(g, try ops.mul(y, l));
        },
        .maximum => blk: {
            const alt = try zerosLike(g);
            const ge = try ops.ge(a, b);
            break :blk if (i == 0) try ops.where(ge, g, alt) else try ops.where(ge, alt, g);
        },
        .minimum => blk: {
            const alt = try zerosLike(g);
            const le = try ops.le(a, b);
            break :blk if (i == 0) try ops.where(le, g, alt) else try ops.where(le, alt, g);
        },
        .atan2 => if (i == 0) blk: {
            const den = try ops.add(try ops.square(a), try ops.square(b));
            break :blk try ops.div(try ops.mul(g, b), den);
        } else blk: {
            const den = try ops.add(try ops.square(a), try ops.square(b));
            break :blk try ops.neg(try ops.div(try ops.mul(g, a), den));
        },
        .hypot => if (i == 0) try ops.div(try ops.mul(g, a), y) else try ops.div(try ops.mul(g, b), y),
        .mod_op => if (i == 0) g else blk: {
            // d/dy x mod y = -floor(x/y)
            const q = try ops.floor(try ops.div(a, b));
            break :blk try ops.neg(try ops.mul(g, q));
        },
        .logical_and, .logical_or, .logical_xor, .eq, .ne, .lt, .le, .gt, .ge => null,
        .where => blk: {
            const cond = node.inputs[0];
            if (i == 0) break :blk null;
            if (i == 1) {
                break :blk try ops.where(cond, g, try zerosLike(g));
            }
            break :blk try ops.where(cond, try zerosLike(g), g);
        },

        // ---- reductions --------------------------------------------------------
        .sum => if (i == 0) blk: {
            const gg = try expandReduced(g, a, attrs.axis, attrs.keepDims());
            break :blk gg;
        } else null,
        .max_reduce, .min_reduce => if (i == 0) blk: {
            const gg = try expandReduced(g, a, attrs.axis, attrs.keepDims());
            const yb = try expandReduced(y, a, attrs.axis, attrs.keepDims());
            const eq = try ops.eq(a, yb);
            const mask = try ops.cast(eq, floatLike(a));
            const cnt = try ops.sum(mask, attrs.axis, true);
            const denom = try ops.broadcastTo(cnt, a.shape);
            const share = try ops.div(mask, denom);
            break :blk try ops.mul(gg, share);
        } else null,
        .prod_op => if (i == 0) blk: {
            const one = try oneLike(a);
            const zero = try scalarLike(a, 0);
            const iszero = try ops.eq(a, zero);
            const anyzero = try ops.sum(try ops.cast(iszero, floatLike(a)), attrs.axis, true);
            // general formula: g * y / a, with the zero-safe fallback
            const gg = try expandReduced(g, a, attrs.axis, attrs.keepDims());
            const safe_a = try ops.where(iszero, one, a);
            const safe_y = try ops.where(try ops.gt(anyzero, zero), zero, y);
            const yb = try expandReduced(safe_y, a, attrs.axis, attrs.keepDims());
            break :blk try ops.div(try ops.mul(gg, yb), safe_a);
        } else null,
        .argmax, .argmin, .all_reduce, .any_reduce => null,

        // ---- linear algebra ----------------------------------------------------
        .matmul => blk: {
            const bt = try swapLastTwo(b);
            const at = try swapLastTwo(a);
            if (i == 0) {
                break :blk try ops.matmul(g, bt);
            }
            break :blk try ops.matmul(at, g);
        },
        .solve => blk: {
            if (i == 0) {
                // x = A^-1 b ; dA = -A^-T g b^T
                const at = try swapLastTwo(a);
                const gb = try linalg.solve(at, g);
                const bt = try swapLastTwo(b);
                break :blk try ops.neg(try ops.matmul(gb, bt));
            }
            const at = try swapLastTwo(a);
            break :blk try linalg.solve(at, g);
        },
        .inv => blk: {
            const ai = try linalg.inv(a);
            const ait = try swapLastTwo(ai);
            const r = try ops.matmul(try ops.matmul(ait, g), ait);
            break :blk try ops.neg(r);
        },
        .det => blk: {
            const ai = try linalg.inv(a);
            const ait = try swapLastTwo(ai);
            // g is scalar per batch; scale by det
            var dims: [MAX_RANK]usize = @splat(0);
            for (0..y.rank()) |k| dims[k] = y.shape.dims[k];
            dims[y.rank()] = 1;
            dims[y.rank() + 1] = 1;
            const d = try ops.reshape(y, Shape.fromParts(dims, y.rank() + 2));
            const gs = try ops.reshape(g, d.shape);
            const sc = try ops.mul(gs, d);
            break :blk try ops.mul(sc, ait);
        },
        .conv => return Error.NotSupported,

        // ---- shape / indexing --------------------------------------------------
        .reshape => try ops.reshape(g, a.shape),
        .expand_dims, .squeeze => try ops.reshape(g, a.shape),
        .transpose => blk: {
            var inv: [MAX_RANK]usize = @splat(0);
            for (0..y.rank()) |k| inv[attrs.perm[k]] = k;
            break :blk try ops.transpose(g, inv[0..y.rank()]);
        },
        .broadcast => try reduceToShape(g, a),
        .slice => blk: {
            const out = try zerosLike(a);
            var rbuf: [MAX_RANK]tmod.Range = undefined;
            const dst = try ops.sliceRanges(out, payloadRanges(a, attrs, &rbuf));
            try ops.copyInto(dst, g);
            release(dst);
            break :blk out;
        },
        .cast_op => try ops.cast(g, a.dtype()),
        // `copy` is the identity: hand the incoming gradient straight back.
        .copy => g,
        .concat_op => blk: {
            if (i == 0) {
                break :blk try ops.narrow(g, attrs.axis, 0, a.shape.dims[realAxisOf(a, attrs.axis)]);
            }
            const raxis: usize = realAxisOf(a, attrs.axis);
            const start = a.shape.dims[raxis];
            break :blk try ops.narrow(g, attrs.axis, start, b.shape.dims[raxis]);
        },
        .take => blk: {
            if (i != 0) break :blk null;
            const z = try zerosLike(a);
            break :blk try ops.scatterAdd(z, b, g, attrs.axis);
        },
        .scatter_add => blk: {
            if (i == 0) break :blk g;
            if (i == 1) break :blk null;
            break :blk try ops.take(g, node.inputs[1], attrs.axis);
        },
        .pad => blk: {
            var rbuf: [MAX_RANK]tmod.Range = undefined;
            break :blk try ops.sliceRanges(g, payloadRanges(a, attrs, &rbuf));
        },
        // `reverse` is `flip` along an axis, and both are their own inverse.
        .flip, .reverse => try ops.flip(g, attrs.axis),
        .cumsum => try ops.cumsum(g, attrs.axis, !attrs.reverse()),
        .lerp, .clamp => null,
        .stack_op => blk: {
            // d/dx_j stack = squeeze of the corresponding half
            if (i == 0) break :blk try ops.squeeze(g, attrs.axis);
            break :blk try ops.squeeze(g, attrs.axis);
        },

        // Everything else is classified above: `zero` primitives contribute
        // nothing, `unimplemented` ones refuse to produce a gradient, and a
        // primitive that is in neither place fails to compile.
        inline else => |p| blk: {
            switch (comptime classify(p)) {
                .zero => break :blk null,
                .unimplemented => return Error.NotSupported,
                .rule => @compileError("no backward rule for primitive " ++ @tagName(p)),
            }
        },
    };

    if (raw) |r| {
        if (r.dtype() == .bool) return null;
        const shaped = try reduceToShape(r, node.inputs[i]);
        if (shaped.buf != r.buf) release(r);
        return shaped;
    }
    return null;
}

fn subOne(x: Tensor) Error!Tensor {
    const one = try oneLike(x);
    return ops.sub(one, x);
}

fn swapLastTwo(x: Tensor) Error!Tensor {
    if (x.rank() < 2) {
        return ops.expandDims(x, 0);
    }
    return ops.swapAxes(x, -1, -2);
}

// =============================================================================
// The reverse sweep
// =============================================================================

pub const Engine = struct {
    alloc: std.mem.Allocator,
    grads: std.AutoHashMapUnmanaged(u64, Tensor) = .empty,
    needed: std.AutoHashMapUnmanaged(u64, void) = .empty,

    pub fn init(alloc: std.mem.Allocator) Engine {
        return .{ .alloc = alloc };
    }

    /// Releases every gradient the engine holds *and* the map itself. The
    /// tensors handed out by `get` are retained by the caller, so they survive.
    pub fn deinit(self: *Engine) void {
        // NB: a direct `deinit`, not `releaseTemp` — the engine owns these
        // references, and `releaseTemp` intentionally does nothing while a
        // `Scope` is installed (the common case, since `grad` opens one).
        var it = self.grads.valueIterator();
        while (it.next()) |t| {
            var tt = t.*;
            tt.deinit();
        }
        self.grads.deinit(self.alloc);
        self.needed.deinit(self.alloc);
    }

    pub fn seed(self: *Engine, tensor: Tensor, g: Tensor) !void {
        if (self.grads.getPtr(tensor.id)) |slot| {
            const sum = try ops.add(slot.*, g);
            var old = slot.*;
            old.deinit();
            slot.* = sum.retain();
        } else {
            try self.grads.put(self.alloc, tensor.id, g.retain());
        }
    }

    pub fn gradient(self: *Engine, tensor: Tensor) ?Tensor {
        return self.grads.get(tensor.id);
    }

    /// Reverse sweep. `targets` seeds gradients for the outputs; `wrt` lists the
    /// tensors we care about (they may also be intermediates).
    pub fn run(self: *Engine, tape: *Tape, targets: []const Tensor, seeds: []const Tensor, wrt: []const Tensor) !void {
        for (targets, seeds) |tt, s| try self.seed(tt, s);

        // relevance: only walk nodes that can reach a `wrt` tensor
        for (wrt) |w| try self.needed.put(self.alloc, w.id, {});
        for (targets) |tt| try self.needed.put(self.alloc, tt.id, {});
        var idx: usize = tape.nodes.items.len;
        while (idx > 0) {
            idx -= 1;
            const node = &tape.nodes.items[idx];
            if (self.needed.contains(node.out.id)) {
                var k: usize = 0;
                while (k < node.n_inputs) : (k += 1) {
                    try self.needed.put(self.alloc, node.inputs[k].id, {});
                }
            }
        }

        idx = tape.nodes.items.len;
        while (idx > 0) {
            idx -= 1;
            // Re-read the node on every step: computing a contribution runs ops,
            // and if another recorder is installed those ops are appended to
            // *this* tape, which may reallocate its storage.
            const g = self.grads.get(tape.nodes.items[idx].out.id) orelse continue;
            var k: usize = 0;
            while (k < tape.nodes.items[idx].n_inputs) : (k += 1) {
                const node = &tape.nodes.items[idx];
                const in = node.inputs[k];
                if (!in.requires_grad) continue;
                if (!self.needed.contains(in.id)) continue;
                const c = try contribution(node, k, g);
                if (c) |contrib| try self.seed(in, contrib);
            }
        }
    }

    /// The gradient of `t` (or zeros when it does not depend on the targets).
    pub fn get(self: *Engine, tns: Tensor) !Tensor {
        if (self.grads.get(tns.id)) |g| return g;
        const z = try zerosLike(tns);
        try self.grads.put(self.alloc, tns.id, z.retain());
        return z;
    }
};

/// The identity cotangent: ones shaped like `t`.
pub fn seedOnes(t: Tensor) Error!Tensor {
    return Tensor.ones(t.shape, floatLike(t));
}

test "backward rules compile" {
    const fns = .{ differentiable, reduceToShape, unbroadcast, contribution, seedOnes };
    inline for (fns) |f| _ = &f;
    _ = &Engine.init;
}
