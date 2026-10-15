//! zpu snapshots: what the numerical core produces today.
//!
//! The point of these is not to re-assert what the numerical suite already
//! proves (softmax sums to one, a gradient matches finite differences). It is to
//! pin the *values* — so that a change in the kernels, the summation order, the
//! JIT's constant folding or a dtype promotion shows up as a readable diff
//! instead of an unexplained 3 % slowdown or a subtly different number.
const std = @import("std");
const zpu = @import("zpu");
const snap = @import("../snapshot.zig");

const Snap = snap.Snap;
const Tensor = zpu.Tensor;
const Shape = zpu.Shape;
const alloc = std.testing.allocator;

fn arr(comptime T: type, values: []const T, dims: []const usize) !Tensor {
    return Tensor.fromArray(T, values, try Shape.init(dims));
}

fn vec(values: []const f64) !Tensor {
    return arr(f64, values, &.{values.len});
}

/// A fixed, small "dataset": deterministic values with a spread of magnitudes,
/// signs and a zero — enough that a sign error, an off-by-one or a broken tail
/// loop changes the snapshot.
fn spread(n: usize) !Tensor {
    const buf = try alloc.alloc(f64, n);
    defer alloc.free(buf);
    for (buf, 0..) |*v, i| {
        const f: f64 = @floatFromInt(i);
        v.* = std.math.sin(f * 0.7) * 1.5 + f * 0.01 - 0.25;
    }
    return vec(buf);
}

test "snapshot: tensor rendering and dtypes" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    try s.expectTensor("zpu/tensor/vector-f64", try spread(9), 1e-9);
    try s.expectTensor("zpu/tensor/matrix-f32", try zpu.reshape(try spread(12), try Shape.init(&.{ 3, 4 })), 1e-6);
    try s.expectTensor("zpu/tensor/rank3", try zpu.reshape(try spread(24), try Shape.init(&.{ 2, 3, 4 })), 1e-9);
    try s.expectTensor("zpu/tensor/scalar", try Tensor.fromScalarF64(2.5, .f64), 1e-12);
    try s.expectTensor("zpu/tensor/ints", try arr(i32, &.{ 1, -2, 3, -4, 5, -6 }, &.{ 2, 3 }), 0);
    try s.expectTensor("zpu/tensor/bools", try zpu.reshape(try zpu.cast(try spread(6), .bool), try Shape.init(&.{ 1, 6 })), 0);
    // An empty tensor is a case, not an error: the shape is the whole content.
    try s.expectTensor("zpu/tensor/empty", try Tensor.empty(try Shape.init(&.{ 0 }), .f32), 0);
    try s.finish();
}

test "snapshot: elementwise ops" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const x = try spread(8);
    try s.expectTensor("zpu/ops/exp", try zpu.exp(x), 1e-9);
    try s.expectTensor("zpu/ops/sigmoid", try zpu.sigmoid(x), 1e-9);
    try s.expectTensor("zpu/ops/tanh", try zpu.tanh(x), 1e-9);
    try s.expectTensor("zpu/ops/erf", try zpu.erf(x), 1e-9);
    try s.expectTensor("zpu/ops/sqrt-abs", try zpu.sqrt(try zpu.abs(x)), 1e-9);
    try s.expectTensor("zpu/ops/square-rsqrt", try zpu.rsqrt(try zpu.square(x)), 1e-9);
    try s.expectTensor("zpu/ops/add-broadcast", try zpu.add(x, try vec(&.{1})), 1e-9);
    try s.expectTensor("zpu/ops/mul-scalar", try zpu.mulScalar(x, 2.5), 1e-9);
    try s.expectTensor("zpu/ops/maximum", try zpu.maximum(x, try zpu.neg(x)), 1e-9);
    try s.expectTensor("zpu/ops/where", try zpu.where(
        try zpu.gt(x, try zpu.scalarLike(x, 0)),
        x,
        try zpu.neg(x),
    ), 1e-9);
    try s.expectTensor("zpu/ops/clamp", try zpu.clamp(x, -0.5, 0.5), 1e-9);
    try s.expectTensor("zpu/ops/lerp", try zpu.lerp(x, try zpu.neg(x), 0.25), 1e-9);
    try s.finish();
}

test "snapshot: reductions and shapes" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const x = try zpu.reshape(try spread(12), try Shape.init(&.{ 3, 4 }));
    try s.expectTensor("zpu/reduce/sum-all", try zpu.sumAll(x), 1e-9);
    try s.expectTensor("zpu/reduce/sum-axis0", try zpu.sum(x, 0, false), 1e-9);
    try s.expectTensor("zpu/reduce/sum-axis1-keepdims", try zpu.sum(x, 1, true), 1e-9);
    try s.expectTensor("zpu/reduce/max-axis1", try zpu.max(x, 1, false), 1e-9);
    try s.expectTensor("zpu/reduce/min-axis0", try zpu.min(x, 0, false), 1e-9);
    try s.expectTensor("zpu/reduce/mean-axis1", try zpu.mean(x, 1, false), 1e-9);
    try s.expectTensor("zpu/reduce/variance-axis1", try zpu.variance(x, 1, false), 1e-9);
    try s.expectTensor("zpu/reduce/prod-axis1", try zpu.prod(x, 1, false), 1e-9);
    try s.expectTensor("zpu/reduce/argmax-axis1", try zpu.argmax(x, 1), 0);
    try s.expectTensor("zpu/reduce/cumsum-axis1", try zpu.cumsum(x, 1, false), 1e-9);
    try s.expectTensor("zpu/reduce/cumsum-reverse", try zpu.cumsum(x, 1, true), 1e-9);
    try s.expectTensor("zpu/shape/transpose", try zpu.transpose(x, &.{ 1, 0 }), 1e-9);
    try s.expectTensor("zpu/shape/flip", try zpu.flip(x, 1), 1e-9);
    try s.expectTensor("zpu/shape/pad", try zpu.pad(x, &.{ .{ 1, 0 }, .{ 0, 2 } }, -1), 1e-9);
    try s.expectTensor("zpu/shape/take", try zpu.take(x, try arr(i32, &.{ 2, 0, 2 }, &.{3}), 0), 1e-9);
    try s.expectTensor("zpu/shape/concat", try zpu.concat(x, x, 1), 1e-9);
    try s.expectTensor("zpu/shape/stack", try zpu.stack(x, x, 0), 1e-9);
    try s.finish();
}

test "snapshot: linear algebra" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // A fixed, well-conditioned SPD matrix and a matching right-hand side.
    const a = try arr(f64, &.{
        4.0, 1.0, 0.5,
        1.0, 3.0, 0.25,
        0.5, 0.25, 2.0,
    }, &.{ 3, 3 });
    const b = try vec(&.{ 1, 2, 3 });

    try s.expectTensor("zpu/linalg/matmul-3x3", try zpu.matmul(a, a), 1e-9);
    try s.expectTensor("zpu/linalg/matvec", try zpu.matmul(a, b), 1e-9);
    try s.expectTensor("zpu/linalg/solve", try zpu.linalg.solve(a, b), 1e-9);
    try s.expectTensor("zpu/linalg/inv", try zpu.linalg.inv(a), 1e-9);
    try s.expectTensor("zpu/linalg/det", try zpu.linalg.det(a), 1e-9);
    try s.expectTensor("zpu/linalg/cholesky", try zpu.linalg.cholesky(a, false), 1e-9);
    try s.expectTensor("zpu/linalg/norm", try zpu.linalg.normAll(a), 1e-9);
    try s.expectTensor("zpu/linalg/trace", try zpu.linalg.trace(a), 1e-9);
    try s.expectTensor("zpu/linalg/diag", try zpu.linalg.diag(a), 1e-9);
    try s.expectTensor("zpu/linalg/outer", try zpu.linalg.outer(try vec(&.{ 1, 2 }), try vec(&.{ 3, 4, 5 })), 1e-9);
    try s.expectTensor("zpu/linalg/dot", try zpu.linalg.dot(b, b), 1e-9);
    try s.finish();
}

test "snapshot: softmax, layernorm and a convolution" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const logits = try arr(f64, &.{
        1.0,  2.0,  3.0,
        -1.0, 0.0,  0.5,
        2.5,  -2.5, 0.0,
    }, &.{ 3, 3 });
    try s.expectTensor("zpu/nn/softmax", try zpu.softmax(logits, -1), 1e-9);
    try s.expectTensor("zpu/nn/log-softmax", try zpu.logSoftmax(logits, -1), 1e-9);

    const rows = try arr(f64, &.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }, &.{ 2, 3 });
    const w = try zpu.Tensor.ones(try Shape.init(&.{3}), .f64);
    const bb = try vec(&.{ 0.5, -0.5, 1.0 });
    try s.expectTensor("zpu/nn/layer-norm", try zpu.nn.layerNorm(rows, w, bb, 1e-5, -1), 1e-9);

    // A 1×1×5×5 image through a 1×1×3×3 kernel, values chosen so the result is
    // readable rather than arbitrary.
    const img = try arr(f64, &.{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
    }, &.{ 1, 1, 5, 5 });
    const kernel = try arr(f64, &.{ 1, 0, -1, 2, 0, -2, 1, 0, -1 }, &.{ 1, 1, 3, 3 });
    try s.expectTensor("zpu/nn/conv2d-sobel", try zpu.nn.conv2d(img, kernel, 1, 0), 1e-9);
    try s.finish();
}

test "snapshot: gradients of a composition" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // f(x, w) = sum(softmax(sigmoid(x·w + b)) * target) — a loss with a bit of
    // everything: a matmul, an elementwise non-linearity, a reduction and a
    // second-rank op. Its gradient is what `grad` must reproduce exactly.
    const F = struct {
        fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const x = args[0];
            const w = args[1];
            const b = args[2];
            const target = args[3];
            const h = try zpu.sigmoid(try zpu.matmul(x, w));
            const z = try zpu.add(h, b);
            const p = try zpu.softmax(z, -1);
            const prod = try zpu.mul(p, target);
            return zpu.sumAll(prod);
        }
    };

    const x = try arr(f64, &.{ 0.5, -1.0, 2.0, 0.25, -0.5, 1.5 }, &.{ 2, 3 });
    const w = try arr(f64, &.{ 1.0, -1.0, 0.5, 0.25, 2.0, -0.5 }, &.{ 3, 2 });
    const b = try arr(f64, &.{ 0.1, -0.2 }, &.{2});
    const target = try arr(f64, &.{ 1.0, 0.0, 0.25, 0.75 }, &.{ 2, 2 });
    var args = [_]Tensor{ x, w, b, target };

    const grads = try zpu.grad(alloc, F.call, null, &args, &.{ 0, 1, 2 });
    defer alloc.free(grads);
    for (grads, 0..) |g, i| {
        defer g.deinit();
        const name = try fmtName(alloc, "zpu/grad/arg{d}", .{i});
        defer alloc.free(name);
        try s.expectTensor(name, g, 1e-9);
    }

    // Forward mode must agree with reverse mode on the same function: the
    // directional derivative `J·v` computed through `jvp` has to equal the same
    // contraction of the gradients `grad` returned, which is a property the
    // snapshot cannot express by itself.
    const tangent = try arr(f64, &.{ 0.3, -0.7, 0.2, 0.4, -0.1, 0.6 }, &.{ 2, 3 });
    const zero_w = try Tensor.zeros(w.shape, .f64);
    const zero_b = try Tensor.zeros(b.shape, .f64);
    const zero_target = try Tensor.zeros(target.shape, .f64);
    const dart = [_]Tensor{ tangent, zero_w, zero_b, zero_target };
    const primal = try zpu.jvp(alloc, F.call, null, &args, &dart);
    defer primal.value.deinit();
    defer primal.tangent.deinit();
    try s.expectTensor("zpu/grad/jvp-value", primal.value, 1e-9);
    try s.expectTensor("zpu/grad/jvp-tangent", primal.tangent, 1e-9);

    // ... and check that property here: sum_i <grad_i, tangent_i> == jvp.
    var contracted: f64 = 0;
    const dirs = [_]?Tensor{ tangent, null, null, null };
    for (grads, 0..) |g, i| {
        const d = dirs[i] orelse continue;
        contracted += (try zpu.sumAll(try zpu.mul(g, d))).item();
    }
    const forward = primal.tangent.item();
    if (@abs(contracted - forward) > 1e-9 * @max(1, @abs(forward))) {
        std.debug.print("jvp {d} disagrees with the gradient contraction {d}\n", .{ forward, contracted });
        return error.JvpMismatch;
    }

    // Second order: the Hessian-vector product of a scalar function.
    const Hess = struct {
        fn call(ud: ?*anyopaque, a: []const Tensor) anyerror!Tensor {
            _ = ud;
            const s2 = try zpu.square(a[0]);
            const c = try zpu.cos(a[0]);
            return zpu.sumAll(try zpu.mul(s2, c));
        }
    };
    const pt = try vec(&.{ 0.5, 1.0, -1.5 });
    const dir = try vec(&.{ 1.0, 0.0, -0.5 });
    const hv = try zpu.hvp(alloc, Hess.call, null, &[_]Tensor{pt}, 0, dir);
    defer hv.deinit();
    try s.expectTensor("zpu/grad/hvp", hv, 1e-9);

    try s.finish();
}

test "snapshot: vmap and jit" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const Per = struct {
        fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const a = args[0];
            const b = args[1];
            const p = try zpu.matmul(a, b);
            return zpu.tanh(p);
        }
    };

    // Three matrices, one kernel: `vmap` must give the same answers as a loop.
    const a = try arr(f64, &.{
        1, 2, 3, 4,
        0.5, -1, 2, 0.25,
        2, 0, -1, 1,
    }, &.{ 3, 2, 2 });
    const b = try arr(f64, &.{
        1, 0, 0, 1,
        2, -1, 0.5, 1,
        -1, 1, 1, 0,
    }, &.{ 3, 2, 2 });
    const mapped = try zpu.vmap(alloc, Per.call, null, &[_]Tensor{ a, b });
    defer mapped.deinit();
    try s.expectTensor("zpu/transforms/vmap", mapped, 1e-9);

    // Tracing is deterministic, so the number of steps and the replayed values
    // are both references.
    var jit = zpu.Jit.init(alloc, Per.call, null);
    defer jit.deinit();
    const traced = try jit.call(&[_]Tensor{ a, b });
    defer traced.deinit();
    try s.expectFmt("zpu/transforms/jit-steps", "steps {d}\n", .{jit.numSteps()});
    try s.expectTensor("zpu/transforms/jit-replay", traced, 1e-9);
    try s.finish();
}

test "snapshot: an MLP, its attention, and one Adam step" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // A tiny self-attention block with fixed weights: the pattern this library
    // was built for, pinned end to end.
    const Block = struct {
        fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const x = args[0]; // [T, D]
            const wq = args[1];
            const wk = args[2];
            const wv = args[3];
            const q = try zpu.matmul(x, wq);
            const k = try zpu.matmul(x, wk);
            const v = try zpu.matmul(x, wv);
            // scores = q @ kᵀ / sqrt(D)
            const kt = try zpu.transpose(k, &.{ 1, 0 });
            const scores = try zpu.divScalar(try zpu.matmul(q, kt), 2.0);
            const weights = try zpu.softmax(scores, -1);
            return zpu.matmul(weights, v);
        }
    };

    const x = try arr(f64, &.{
        0.5, -1.0, 2.0,
        1.0, 0.5, -0.5,
        -1.0, 2.0, 0.25,
        0.0, 1.0, 1.0,
    }, &.{ 4, 3 });
    const wq = try arr(f64, &.{
        1, 0, 0.5,
        0, 1, -0.5,
        0.5, 0.5, 1,
    }, &.{ 3, 3 });
    const attn = try Block.call(null, &[_]Tensor{ x, wq, wq, wq });
    try s.expectTensor("zpu/nn/attention", attn, 1e-9);

    // One optimisation step, from fixed parameters and a fixed gradient: the
    // update rule itself is part of the snapshot.
    const params = try vec(&.{ 1.0, -2.0, 0.5 });
    const grads = try vec(&.{ 0.1, -0.25, 0.05 });
    var adam = zpu.optim.Adam.init(alloc, .{ .lr = 0.01 });
    defer adam.deinit();
    var params_arr = [_]Tensor{params};
    const grad_arr = [_]Tensor{grads};
    try adam.step(&params_arr, &grad_arr);
    try s.expectTensor("zpu/optim/adam-step1", params_arr[0], 1e-9);
    for (0..9) |_| try adam.step(&params_arr, &grad_arr);
    try s.expectTensor("zpu/optim/adam-step10", params_arr[0], 1e-9);
    try s.finish();
}

test "snapshot: the random number generator is reproducible" {
    var s = Snap.init(alloc);
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // A generator whose output is part of the contract: a snapshot here is what
    // makes "same seed, same numbers" checkable across releases.
    var rng = zpu.rng.Rng.init(12345);
    const n = try rng.normal(try Shape.init(&.{ 2, 4 }), .f64);
    try s.expectTensor("zpu/rng/normal-seed12345", n, 0);
    var rng2 = zpu.rng.Rng.init(12345);
    const u = try rng2.uniform(try Shape.init(&.{4}), .f64);
    try s.expectTensor("zpu/rng/uniform-seed12345", u, 0);
    try s.finish();
}

fn fmtName(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.allocPrint(a, fmt, args);
}
