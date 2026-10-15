//! Numerical test suite for ZPU.
//!
//! Every expectation here is either an analytic derivative, a value taken from
//! NumPy, or a finite-difference cross-check. Tests run under
//! `std.testing.allocator`, so every one of them also verifies that ZPU does not
//! leak.
const std = @import("std");
const zpu = @import("zpu");

const alloc = std.testing.allocator;
const T = zpu.Tensor;
const S = zpu.Shape;
const Range = zpu.Range;

fn setup() !void {
    try zpu.init(alloc, .{});
}

fn arr(comptime Tp: type, values: []const Tp, dims: []const usize) !T {
    return T.fromArray(Tp, values, try S.init(dims));
}

/// Compares against expected values with an *absolute* tolerance.
fn check(actual: T, expected: []const f64, tol: f64) !void {
    if (actual.numel() != expected.len) return error.LengthMismatch;
    for (expected, 0..) |e, i| {
        const a = actual.readLogical(actual.numel(), i);
        if (std.math.isNan(e) or std.math.isNan(a)) {
            if (!(std.math.isNan(e) and std.math.isNan(a))) return error.NotClose;
            continue;
        }
        if (@abs(a - e) > tol) {
            std.debug.print("element {d}: got {d}, want {d} (tol {d})\n", .{ i, a, e, tol });
            return error.NotClose;
        }
    }
}

/// Drops a reference a transform handed back (transforms return owned tensors,
/// unlike the scope owned tensors the ops inside a scope produce).
fn deinitOwned(t: T) void {
    var tt = t;
    tt.deinit();
}

/// Compares against expected values with a *relative* tolerance (the right
/// measure for large magnitudes, where an absolute epsilon is meaningless).
fn checkRel(actual: T, expected: []const f64, tol: f64) !void {
    if (actual.numel() != expected.len) return error.LengthMismatch;
    for (expected, 0..) |e, i| {
        const a = actual.readLogical(actual.numel(), i);
        if (std.math.isNan(e) or std.math.isNan(a)) {
            if (!(std.math.isNan(e) and std.math.isNan(a))) return error.NotClose;
            continue;
        }
        const denom = @max(@abs(e), 1e-300);
        if (@abs(a - e) / denom > tol) {
            std.debug.print("element {d}: got {d}, want {d} (rel tol {d})\n", .{ i, a, e, tol });
            return error.NotClose;
        }
    }
}

// =============================================================================
// The eager core
// =============================================================================

test "elementwise and broadcasting match numpy" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const a = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    const b = try arr(f64, &.{ 10, 20, 30 }, &.{3});
    const c = try zpu.add(a, b);
    try check(c, &.{ 11, 22, 33, 14, 25, 36 }, 1e-12);

    const d = try zpu.div(c, try zpu.scalarLike(c, 2));
    try check(d, &.{ 5.5, 11, 16.5, 7, 12.5, 18 }, 1e-12);

    const p = try zpu.pow(a, try zpu.scalarLike(a, 2));
    try check(p, &.{ 1, 4, 9, 16, 25, 36 }, 1e-12);

    // transcendental functions (numpy floats, 1e-12)
    const e = try zpu.exp(try zpu.neg(a));
    try check(e, &.{
        0.36787944117144233, 0.1353352832366127,   0.049787068367863944,
        0.01831563888873418, 0.006737946999085467, 0.0024787521766663585,
    }, 1e-15);

    const l = try zpu.log(a);
    try check(l, &.{
        0.0,                0.6931471805599453, 1.0986122886681098,
        1.3862943611198906, 1.6094379124341003, 1.791759469228055,
    }, 1e-12);

    const cond = try zpu.gt(a, try zpu.scalarLike(a, 3));
    try std.testing.expectEqual(zpu.DType.bool, cond.dtype());
    try check(try zpu.cast(cond, .f64), &.{ 0, 0, 0, 1, 1, 1 }, 0);

    const w = try zpu.where(cond, a, try zpu.neg(a));
    try check(w, &.{ -1, -2, -3, 4, 5, 6 }, 1e-12);

    // erf / tanh / sigmoid
    const z = try arr(f64, &.{ -1, 0, 0.5, 1 }, &.{4});
    try check(try zpu.erf(z), &.{ -0.8427007929497149, 0, 0.5204998778130465, 0.8427007929497149 }, 1e-14);
    try check(try zpu.sigmoid(z), &.{ 0.2689414213699951, 0.5, 0.6224593312018546, 0.7310585786300049 }, 1e-12);
    try check(try zpu.tanh(z), &.{ -0.7615941559557649, 0, 0.46211715726000974, 0.7615941559557649 }, 1e-12);
}

test "reductions and softmax" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const a = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    try std.testing.expectApproxEqAbs(@as(f64, 21), (try zpu.sumAll(a)).item(), 1e-12);
    try check(try zpu.sum(a, 0, false), &.{ 5, 7, 9 }, 1e-12);
    try check(try zpu.sum(a, -1, false), &.{ 6, 15 }, 1e-12);
    try std.testing.expect((try zpu.sum(a, 1, true)).shape.eql(zpu.Shape.initComptime(&.{ 2, 1 })));
    try check(try zpu.mean(a, -1, false), &.{ 2, 5 }, 1e-12);
    try check(try zpu.max(a, 0, false), &.{ 4, 5, 6 }, 1e-12);
    try check(try zpu.min(a, 1, false), &.{ 1, 4 }, 1e-12);
    try check(try zpu.cast(try zpu.argmax(a, -1), .f64), &.{ 2, 2 }, 0);
    try check(try zpu.cumsum(a, 1, false), &.{ 1, 3, 6, 4, 9, 15 }, 1e-12);
    try check(try zpu.cumsum(a, 1, true), &.{ 6, 5, 3, 15, 11, 6 }, 1e-12);

    // softmax rows sum to one, and match the numpy value
    const sm = try zpu.softmax(a, -1);
    try check(sm, &.{
        0.09003057317038046, 0.24472847105479767, 0.6652409557748219,
        0.09003057317038046, 0.24472847105479767, 0.6652409557748219,
    }, 1e-12);
    try check(try zpu.sum(sm, -1, false), &.{ 1, 1 }, 1e-12);

    const lsm = try zpu.logSoftmax(a, -1);
    try check(lsm, &.{
        -2.4076059644443806, -1.4076059644443806, -0.40760596444438059,
        -2.4076059644443806, -1.4076059644443806, -0.40760596444438059,
    }, 1e-12);
    const exp_lsm = try zpu.exp(lsm);
    try std.testing.expect(exp_lsm.allClose(sm, 1e-12));
}

test "matmul: matrix, vector and batched" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const a = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    const b = try arr(f64, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    try check(try zpu.matmul(a, b), &.{ 58, 64, 139, 154 }, 1e-12);

    const v = try arr(f64, &.{ 1, 1, 1 }, &.{3});
    try check(try zpu.matmul(v, b), &.{ 27, 30 }, 1e-12);
    try check(try zpu.matmul(a, v), &.{ 6, 15 }, 1e-12);

    const a3 = try arr(f64, &.{ 1, 2, 3, 4, 1, 0, 0, 1 }, &.{ 2, 2, 2 });
    const b3 = try arr(f64, &.{ 5, 6, 7, 8, 1, 2, 3, 4 }, &.{ 2, 2, 2 });
    try check(try zpu.matmul(a3, b3), &.{ 19, 22, 43, 50, 1, 2, 3, 4 }, 1e-12);

    // broadcasting a batch of matrices against one matrix
    const single = try arr(f64, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    const broad = try zpu.matmul(a3, single);
    try std.testing.expect(broad.shape.eql(S.initComptime(&.{ 2, 2, 2 })));
    try check(broad, &.{ 7, 10, 15, 22, 1, 2, 3, 4 }, 1e-12);
}

// =============================================================================
// Automatic differentiation
// =============================================================================

test "grad of exp(square(x))" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const sq = try zpu.square(args[0]);
            defer zpu.releaseTemp(sq);
            const e = try zpu.exp(sq);
            defer zpu.releaseTemp(e);
            return zpu.sumAll(e);
        }
    };

    const x = try arr(f64, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    const g = try zpu.grad(alloc, F.f, null, &.{x}, &.{0});
    defer alloc.free(g);
    // d/dx sum(exp(x^2)) = 2x exp(x^2), analytic and exact to the last bit
    try checkRel(g[0], &.{
        2 * std.math.exp(1.0),
        4 * std.math.exp(4.0),
        6 * std.math.exp(9.0),
        8 * std.math.exp(16.0),
    }, 1e-12);
    var gg = g[0];
    gg.deinit();

    // valueAndGrad also returns the primal
    var r = try zpu.valueAndGrad(alloc, F.f, null, &.{x}, &.{0});
    defer r.deinit();
    const primal = std.math.exp(1.0) + std.math.exp(4.0) + std.math.exp(9.0) + std.math.exp(16.0);
    try std.testing.expectApproxEqRel(primal, r.value.item(), 1e-15);
}

test "grad matches finite differences through many ops" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const x = args[0];
            const h = try zpu.tanh(x);
            defer zpu.releaseTemp(h);
            const e = try zpu.exp(try zpu.mulScalar(x, 0.5));
            defer zpu.releaseTemp(e);
            const l = try zpu.log(try zpu.add(e, try zpu.scalarLike(e, 1)));
            defer zpu.releaseTemp(l);
            const s = try zpu.square(try zpu.sub(h, try zpu.scalarLike(h, 0.2)));
            defer zpu.releaseTemp(s);
            const sum1 = try zpu.add(s, l);
            defer zpu.releaseTemp(sum1);
            const sq = try zpu.sqrt(try zpu.add(try zpu.square(x), try zpu.scalarLike(x, 1)));
            defer zpu.releaseTemp(sq);
            const sum2 = try zpu.add(sum1, sq);
            defer zpu.releaseTemp(sum2);
            return zpu.sumAll(try zpu.mul(sum2, x));
        }
    };

    const xs = [_]f64{ -0.7, 0.3, 1.1 };
    const x = try arr(f64, &xs, &.{3});
    const g = try zpu.grad(alloc, F.f, null, &.{x}, &.{0});
    defer alloc.free(g);
    defer {
        var gg = g[0];
        gg.deinit();
    }

    // central differences
    const eps = 1e-6;
    for (xs, 0..) |v, i| {
        // NB: tensors created while the scope above is active belong to it, so
        // they are deliberately not deinit'ed here.
        const plus = try arr(f64, &.{ xs[0], xs[1], xs[2] }, &.{3});
        const minus = try arr(f64, &.{ xs[0], xs[1], xs[2] }, &.{3});
        plus.writeF64(i, v + eps);
        minus.writeF64(i, v - eps);
        const fp = try F.f(null, &.{plus});
        const fm = try F.f(null, &.{minus});
        const fd = (fp.item() - fm.item()) / (2 * eps);
        const got = g[0].readF64(i);
        try std.testing.expectApproxEqRel(fd, got, 1e-6);
    }
}

test "grad through matmul and softmax cross entropy" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // L = mean(softmaxCE(logits, onehot)) with a single sample; dL/dlogits = sm - y
    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            return zpu.nn.softmaxCrossEntropy(args[0], args[1]);
        }
    };

    const logits = try arr(f64, &.{ 1, 2, 3, 0.5 }, &.{ 1, 4 });
    const labels = try arr(f64, &.{ 0, 0, 1, 0 }, &.{ 1, 4 });
    const g = try zpu.grad(alloc, F.f, null, &.{ logits, labels }, &.{0});
    defer alloc.free(g);
    defer {
        var gg = g[0];
        gg.deinit();
    }
    const sm = try zpu.softmax(logits, -1);
    const expected = try zpu.sub(sm, labels);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected.readF64(i), g[0].readF64(i), 1e-9);
    }

    // gradient of a linear model wrt its weights
    const G = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const pred = try zpu.matmul(args[0], args[1]);
            defer zpu.releaseTemp(pred);
            return zpu.nn.mse(pred, args[2]);
        }
    };
    const x = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    const w = try arr(f64, &.{ 0.1, 0.2 }, &.{ 2, 1 });
    const y = try arr(f64, &.{ 1, 2, 3 }, &.{ 3, 1 });
    const gw = try zpu.grad(alloc, G.f, null, &.{ x, w, y }, &.{1});
    defer alloc.free(gw);
    defer {
        var gg = gw[0];
        gg.deinit();
    }
    // d/dw mean((xw - y)^2) = 2/n * x^T (xw - y) with n = 3
    const pred = try zpu.matmul(x, w);
    const diff = try zpu.sub(pred, y);
    const xt = try zpu.transpose(x, &.{ 1, 0 });
    const xtd = try zpu.matmul(xt, diff);
    const scaled = try zpu.mulScalar(xtd, 2.0 / 3.0);
    try std.testing.expect(gw[0].allClose(scaled, 1e-9));
}

test "grad through gather, pad, slice, cumsum and transpose" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const table = args[0];
            const idx = args[1];
            const rows = try zpu.take(table, idx, 0);
            defer zpu.releaseTemp(rows);
            const padded = try zpu.pad(rows, &.{ .{ 1, 0 }, .{ 0, 0 } }, 0.5);
            defer zpu.releaseTemp(padded);
            const sliced = try zpu.sliceRanges(padded, &.{
                .{ .start = 0, .end = 2 },
                .{ .start = 0, .end = 2 },
            });
            defer zpu.releaseTemp(sliced);
            const t = try zpu.transpose(sliced, &.{ 1, 0 });
            defer zpu.releaseTemp(t);
            const c = try zpu.cumsum(t, 1, false);
            defer zpu.releaseTemp(c);
            return zpu.sumAll(try zpu.square(c));
        }
    };

    const table = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    const idx = try arr(i64, &.{ 2, 0 }, &.{2});
    const g = try zpu.grad(alloc, F.f, null, &.{ table, idx }, &.{0});
    defer alloc.free(g);
    defer {
        var gg = g[0];
        gg.deinit();
    }
    try std.testing.expect(g[0].shape.eql(table.shape));

    // finite difference check on one element
    const eps = 1e-6;
    const base = try F.f(null, &.{ table, idx });
    const perturbed = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, S.initComptime(&.{ 3, 2 }));
    perturbed.writeF64(0, 1.0 + eps);
    const up = try F.f(null, &.{ perturbed, idx });
    const fd = (up.item() - base.item()) / eps;
    try std.testing.expectApproxEqRel(fd, g[0].readF64(0), 1e-4);
}

test "vmap vectorises over a batch" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // per-sample dot product, without unrolling the batch
    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            finalize();
            const prod = try zpu.mul(args[0], args[1]);
            defer zpu.releaseTemp(prod);
            return zpu.sum(prod, zpu.AXIS_NONE, false);
        }
        // counts how often the body runs: vmap must run it once
        var calls: usize = 0;
        fn finalize() void {
            calls += 1;
        }
        fn reset() void {
            calls = 0;
        }
    };

    const xs = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    const ys = try arr(f64, &.{ 10, 20, 30, 40, 50, 60 }, &.{ 3, 2 });
    F.reset();
    const out = try zpu.vmap(alloc, F.f, null, &.{ xs, ys });
    defer deinitOwned(out);
    try std.testing.expectEqual(@as(usize, 1), F.calls);
    try check(out, &.{ 50, 250, 610 }, 1e-12);

    // vmap of a per-sample quadratic form
    const G = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const sq = try zpu.square(args[0]);
            defer zpu.releaseTemp(sq);
            const s = try zpu.sum(sq, zpu.AXIS_NONE, false);
            defer zpu.releaseTemp(s);
            return zpu.mulScalar(s, 2);
        }
    };
    const v = try arr(f64, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    const q = try zpu.vmap(alloc, G.f, null, &.{v});
    defer deinitOwned(q);
    try check(q, &.{ 10, 50 }, 1e-12); // 2 * sum(x^2) per row

    // vmap over a batched matmul
    const H = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            return zpu.matmul(args[0], args[1]);
        }
    };
    const a3 = try arr(f64, &.{ 1, 2, 3, 4, 1, 0, 0, 1 }, &.{ 2, 2, 2 });
    const b3 = try arr(f64, &.{ 5, 6, 7, 8, 1, 2, 3, 4 }, &.{ 2, 2, 2 });
    const mm = try zpu.vmap(alloc, H.f, null, &.{ a3, b3 });
    defer deinitOwned(mm);
    try check(mm, &.{ 19, 22, 43, 50, 1, 2, 3, 4 }, 1e-12);
}

test "jvp and hvp" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // f(x) = sum(x * tanh(x)); df/dx = tanh(x) + x (1 - tanh^2(x))
    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const h = try zpu.tanh(args[0]);
            defer zpu.releaseTemp(h);
            const p = try zpu.mul(args[0], h);
            defer zpu.releaseTemp(p);
            return zpu.sumAll(p);
        }
    };
    const x = try arr(f64, &.{ 0.5, -0.25 }, &.{2});
    const v = try arr(f64, &.{ 1, 1 }, &.{2});
    const r = try zpu.jvp(alloc, F.f, null, &.{x}, &.{v});
    defer {
        var a = r.value;
        a.deinit();
        var b = r.tangent;
        b.deinit();
    }
    const t1 = std.math.tanh(@as(f64, 0.5));
    const t2 = std.math.tanh(@as(f64, -0.25));
    const expected = (t1 + 0.5 * (1 - t1 * t1)) + (t2 + -0.25 * (1 - t2 * t2));
    try std.testing.expectApproxEqAbs(expected, r.tangent.item(), 1e-9);

    // hvp of 0.5 x^T Q x is Q v
    const Q = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const px = args[0];
            const q = args[1];
            const qx = try zpu.matmul(q, px);
            defer zpu.releaseTemp(qx);
            return zpu.mulScalar(try zpu.sumAll(try zpu.mul(px, qx)), 0.5);
        }
    };
    const q = try arr(f64, &.{ 2, 1, 1, 3 }, &.{ 2, 2 });
    const xx = try arr(f64, &.{ 0.3, -0.4 }, &.{2});
    const vv = try arr(f64, &.{ 1, 2 }, &.{2});
    const hv = try zpu.hvp(alloc, Q.f, null, &.{ xx, q }, 0, vv);
    defer {
        var h = hv;
        h.deinit();
    }
    const qv = try zpu.matmul(q, vv);
    for (0..2) |i| try std.testing.expectApproxEqAbs(qv.readF64(i), hv.readF64(i), 1e-9);
}

test "jit traces once and replays the plan" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const F = struct {
        var calls: usize = 0;
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            calls += 1;
            // a constant (folded) and a live computation
            const bias = try T.full(.{}, .f64, 1.5);
            defer zpu.releaseTemp(bias);
            const m = try zpu.matmul(args[0], args[1]);
            defer zpu.releaseTemp(m);
            const s = try zpu.tanh(m);
            defer zpu.releaseTemp(s);
            return zpu.add(s, bias);
        }
    };

    const a = try arr(f64, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    const b = try arr(f64, &.{ 0.5, -0.5, 1, 2 }, &.{ 2, 2 });

    var j = zpu.jit(alloc, F.f, null);
    defer j.deinit();

    const eager = try F.f(null, &.{ a, b }); // scope owned, like every tensor here
    F.calls = 0;

    const out = try j.call(&.{ a, b });
    defer {
        var o = out;
        o.deinit();
    }
    try std.testing.expectEqual(@as(usize, 1), j.trace_count);
    try std.testing.expectEqual(@as(usize, 1), F.calls);
    try std.testing.expect(out.allClose(eager, 1e-12));
    // the constant was folded into a blob
    try std.testing.expect(j.numSteps() < 4);

    // replay with new inputs
    const a2 = try arr(f64, &.{ 2, 0, 0, 2 }, &.{ 2, 2 });
    const out2 = try j.call(&.{ a2, b });
    defer {
        var o = out2;
        o.deinit();
    }
    try std.testing.expectEqual(@as(usize, 1), j.trace_count);
    const m2 = try zpu.matmul(a2, b);
    const e2 = try zpu.add(try zpu.tanh(m2), try T.full(.{}, .f64, 1.5));
    try std.testing.expect(out2.allClose(e2, 1e-12));
}

test "jit of a gradient" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const F = struct {
        fn gradFn(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const inner = struct {
                fn loss(u: ?*anyopaque, a: []const T) anyerror!T {
                    _ = u;
                    const e = try zpu.exp(a[0]);
                    defer zpu.releaseTemp(e);
                    return zpu.sumAll(e);
                }
            };
            const g = try zpu.grad(alloc, inner.loss, null, args, &.{0});
            defer alloc.free(g);
            return g[0];
        }
    };

    const x = try arr(f64, &.{ 0.0, 1.0, 2.0 }, &.{3});
    var j = zpu.jit(alloc, F.gradFn, null);
    defer j.deinit();
    const out = try j.call(&.{x});
    defer {
        var o = out;
        o.deinit();
    }
    try check(out, &.{ 1.0, std.math.e, std.math.e * std.math.e }, 1e-9);
}

// =============================================================================
// Neural network layer, optimisers, rng
// =============================================================================

test "nn: layer norm" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const x = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    const w = try zpu.Tensor.ones(S.initComptime(&.{3}), .f64);
    const b = try zpu.Tensor.zeros(S.initComptime(&.{3}), .f64);
    const eps = 1e-5;
    const y = try zpu.nn.layerNorm(x, w, b, eps, -1);
    // each row now has zero mean, and its variance is the input's scaled by the
    // eps in the denominator: var / (var + eps)
    const mu = try zpu.mean(y, -1, false);
    const var_ = try zpu.variance(y, -1, false);
    const var_x = try zpu.variance(x, -1, false);
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(@as(f64, 0), mu.readF64(i), 1e-12);
        const vx = var_x.readF64(i);
        try std.testing.expectApproxEqRel(vx / (vx + eps), var_.readF64(i), 1e-12);
    }
}

test "nn: dropout in eval mode is the identity" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const x = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    var gen = zpu.rng.Rng.init(42);
    const eval_out = try zpu.nn.dropout(x, 0.5, &gen, false);
    try std.testing.expect(eval_out.allClose(x, 0));
}

test "nn: one-hot" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const idx = try arr(i64, &.{ 0, 2 }, &.{2});
    const oh = try zpu.nn.oneHot(idx, 3, .f64);
    try std.testing.expect(oh.shape.eql(S.initComptime(&.{ 2, 3 })));
    try check(oh, &.{ 1, 0, 0, 0, 0, 1 }, 0);
}

test "nn: max and average pool" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // 2x2 max pooling
    const img = try arr(f64, &.{ 1, 2, 3, 4 }, &.{ 1, 1, 2, 2 });
    try check(try zpu.nn.maxPool2d(img, 2), &.{4}, 1e-12);
    try check(try zpu.nn.avgPool2d(img, 2), &.{2.5}, 1e-12);
}

test "nn: conv2d" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // conv2d against a hand computed result
    const input = try arr(f64, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &.{ 1, 1, 3, 3 });
    const kernel = try arr(f64, &.{ 1, 0, 0, 1 }, &.{ 1, 1, 2, 2 });
    const conv = try zpu.nn.conv2d(input, kernel, 1, 0);
    try std.testing.expect(conv.shape.eql(S.initComptime(&.{ 1, 1, 2, 2 })));
    try check(conv, &.{ 6, 8, 12, 14 }, 1e-12);
}

test "negative strides read back in logical order" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // `flip` is a view with a negative stride, so reading it walks backwards
    // through the buffer. That walk has to be done in signed arithmetic: the
    // accumulated delta is negative even though the element it selects is not.
    const a = try arr(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    const f = try zpu.flip(a, -1);
    try check(f, &.{ 3, 2, 1, 6, 5, 4 }, 0);
    try std.testing.expect(!f.isContiguous());

    // the same walk through a slice of the view, and through toContiguous/clone
    // (both of which copy element by element)
    try check(try zpu.narrow(f, 0, 1, 1), &.{ 6, 5, 4 }, 0);
    try check(try f.toContiguous(), &.{ 3, 2, 1, 6, 5, 4 }, 0);
    try check(try f.clone(), &.{ 3, 2, 1, 6, 5, 4 }, 0);

    // slicing a flipped tensor is where the offset moves backwards
    const g = try zpu.flip(a, 1);
    try check(try zpu.sliceRanges(g, &.{ Range.all(), Range.bounded(0, 2) }), &.{ 3, 2, 6, 5 }, 0);

    // flipping twice is the identity, and a flipped matrix still multiplies:
    // the gemm kernels walk raw memory, so both operands are materialised first
    // and both go through the negative-stride walk
    try check(try zpu.flip(f, -1), &.{ 1, 2, 3, 4, 5, 6 }, 0);
    const m = try zpu.matmul(f, try zpu.transpose(f, &.{ 1, 0 }));
    try check(m, &.{ 14, 32, 32, 77 }, 0);

    // A reverse-strided range is the other way to produce negative strides.
    // `end` is exclusive and a negative `end` counts from the far end (NumPy's
    // rule), so "one before the first element" is spelled `-(dim + 1)`. For
    // simply reversing an axis, `flip` says it better and costs nothing.
    const cols: i64 = @intCast(a.shape.dims[1]);
    const rev = try zpu.sliceRanges(a, &.{ Range.all(), Range.strided(cols - 1, -cols - 1, -1) });
    try check(rev, &.{ 3, 2, 1, 6, 5, 4 }, 0);
}

test "reshape of a strided view owns its materialised copy" {
    // Outside a scope `releaseTemp` frees for real, so a view that forgets to
    // take a reference of its own is a use-after-free — the shape the benchmark
    // runs in (it is what turned into a SIGSEGV in CI).
    try setup();
    const a = try arr(f64, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &.{ 1, 1, 3, 3 });
    defer a.deinit();
    const t = try zpu.transpose(a, &.{ 0, 1, 3, 2 });
    defer t.deinit();
    const r = try zpu.reshape(t, try S.init(&.{ 1, 1, 9 }));
    defer r.deinit();
    try check(r, &.{ 1, 4, 7, 2, 5, 8, 3, 6, 9 }, 1e-12);
}

test "conv2d keeps its temporaries balanced without a scope" {
    // No `Scope` on purpose: `releaseTemp` only no-ops while a tracker is
    // installed, so this is the shape the benchmark runs in (and the shape that
    // catches a temporary released twice).
    try setup();
    const input = try arr(f64, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &.{ 1, 1, 3, 3 });
    defer input.deinit();
    const kernel = try arr(f64, &.{ 1, 0, 0, 1 }, &.{ 1, 1, 2, 2 });
    defer kernel.deinit();
    const conv = try zpu.nn.conv2d(input, kernel, 1, 0);
    defer conv.deinit();
    try check(conv, &.{ 6, 8, 12, 14 }, 1e-12);
}

test "optimisers minimise a quadratic" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const Loss = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            // (w - 3)^2 + (b + 1)^2
            const d1 = try zpu.sub(args[0], try zpu.scalarLike(args[0], 3));
            defer zpu.releaseTemp(d1);
            const d2 = try zpu.add(args[1], try zpu.scalarLike(args[1], 1));
            defer zpu.releaseTemp(d2);
            const s1 = try zpu.square(d1);
            defer zpu.releaseTemp(s1);
            const s2 = try zpu.square(d2);
            defer zpu.releaseTemp(s2);
            return zpu.add(try zpu.sumAll(s1), try zpu.sumAll(s2));
        }
    };

    // scope owned (like every other tensor built inside it)
    const w = try T.full(.{}, .f64, 0);
    const b = try T.full(.{}, .f64, 0);
    var params = [_]T{ w, b };

    var opt = zpu.optim.Adam.init(alloc, .{ .lr = 0.2 });
    defer opt.deinit();

    for (0..400) |_| {
        const g = try zpu.grad(alloc, Loss.f, null, &params, &.{ 0, 1 });
        defer {
            for (g) |*t| {
                var tt = t.*;
                tt.deinit();
            }
            alloc.free(g);
        }
        try opt.step(&params, g);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), params[0].item(), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), params[1].item(), 1e-3);

    // SGD with momentum also converges
    const L2 = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            const d = try zpu.sub(args[0], try zpu.scalarLike(args[0], 3));
            defer zpu.releaseTemp(d);
            return zpu.sumAll(try zpu.square(d));
        }
    };
    const w2 = try T.full(.{}, .f64, 0);
    var params2 = [_]T{w2};
    var sgd = zpu.optim.Sgd.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
    defer sgd.deinit();
    for (0..200) |_| {
        const g = try zpu.grad(alloc, L2.f, null, &.{params2[0]}, &.{0});
        defer {
            for (g) |*t| {
                var tt = t.*;
                tt.deinit();
            }
            alloc.free(g);
        }
        try sgd.step(&params2, g);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), params2[0].item(), 1e-3);
}

test "rng: determinism and distribution" {
    try setup();

    var g1 = zpu.rng.Rng.init(7);
    var g2 = zpu.rng.Rng.init(7);
    const a = try g1.uniform(S.initComptime(&.{ 4, 4 }), .f64);
    defer {
        var t = a;
        t.deinit();
    }
    const b = try g2.uniform(S.initComptime(&.{ 4, 4 }), .f64);
    defer {
        var t = b;
        t.deinit();
    }
    try std.testing.expect(a.eql(b));

    var g3 = zpu.rng.Rng.init(11);
    const n = try g3.normal(S.initComptime(&.{20000}), .f64);
    defer {
        var t = n;
        t.deinit();
    }
    var sum: f64 = 0;
    var sum2: f64 = 0;
    for (0..20000) |i| {
        const v = n.readF64(i);
        sum += v;
        sum2 += v * v;
    }
    const mean = sum / 20000;
    const var_ = sum2 / 20000 - mean * mean;
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), mean, 0.03);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), var_, 0.05);
}

// =============================================================================
// Linear algebra
// =============================================================================

test "linalg with gradients" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const a = try arr(f64, &.{ 3, 1, 1, 2 }, &.{ 2, 2 });
    const b = try arr(f64, &.{ 9, 8 }, &.{2});
    const x = try zpu.linalg.solve(a, b);
    try check(x, &.{ 2, 3 }, 1e-12);

    try std.testing.expectApproxEqAbs(@as(f64, 5), (try zpu.linalg.det(a)).item(), 1e-12);
    const ai = try zpu.linalg.inv(a);
    try check(ai, &.{ 0.4, -0.2, -0.2, 0.6 }, 1e-12);

    // d/dA det(A) = det(A) * A^-T
    const F = struct {
        fn f(ud: ?*anyopaque, args: []const T) anyerror!T {
            _ = ud;
            return zpu.linalg.det(args[0]);
        }
    };
    const g = try zpu.grad(alloc, F.f, null, &.{a}, &.{0});
    defer alloc.free(g);
    defer {
        var gg = g[0];
        gg.deinit();
    }
    // det = 5, A^-T = [[0.4, -0.2], [-0.2, 0.6]]
    try check(g[0], &.{ 2.0, -1.0, -1.0, 3.0 }, 1e-9);

    // cholesky round trip
    const spd = try arr(f64, &.{ 4, 2, 2, 3 }, &.{ 2, 2 });
    const l = try zpu.linalg.cholesky(spd, false);
    const lt = try zpu.transpose(l, &.{ 1, 0 });
    const llt = try zpu.matmul(l, lt);
    try std.testing.expect(llt.allClose(spd, 1e-12));
}

test "dtypes: f32, i64 and bool" {
    try setup();
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const f = try arr(f32, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    try std.testing.expectEqual(zpu.DType.f32, f.dtype());
    const s = try zpu.add(f, f);
    try std.testing.expectEqual(zpu.DType.f32, s.dtype());
    try check(try zpu.cast(s, .f64), &.{ 2, 4, 6, 8 }, 1e-6);

    const i = try arr(i64, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    const ip = try zpu.add(i, i);
    try std.testing.expectEqual(zpu.DType.i64, ip.dtype());
    try check(try zpu.cast(ip, .f64), &.{ 2, 4, 6, 8 }, 0);

    // mixed dtype promotes
    const mixed = try zpu.mul(i, try zpu.scalarLike(i, 2));
    try std.testing.expectEqual(zpu.DType.i64, mixed.dtype());

    const bb = try zpu.lt(i, try zpu.scalarLike(i, 3));
    try std.testing.expectEqual(zpu.DType.bool, bb.dtype());
    const any_ = try zpu.any(bb, zpu.AXIS_NONE, false);
    try std.testing.expect(any_.item() == 1);
    const all_ = try zpu.all(bb, zpu.AXIS_NONE, false);
    try std.testing.expect(all_.item() == 0);
}

test "scopes own what they create" {
    try setup();

    // a tensor kept out of a scope survives it
    var kept: T = undefined;
    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        const t = try T.fromArray(f64, &.{ 1, 2, 3 }, S.initComptime(&.{3}));
        const double = try zpu.mulScalar(t, 4);
        kept = scope.keep(double);
        try std.testing.expect(scope.count() >= 2);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 8), kept.readF64(1), 1e-12);
    kept.deinit();

    // explicit mode: the caller owns everything it creates
    const x = try T.fromArray(f64, &.{ 1, 2, 3 }, S.initComptime(&.{3}));
    defer x.deinit();
    const y = try zpu.mulScalar(x, 2);
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 4), y.readF64(1), 1e-12);
}
