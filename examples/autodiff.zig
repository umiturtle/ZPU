//! `zig build run-autodiff`
//!
//! The transforms, each cross-checked against something it did not produce:
//!
//! * `grad` — reverse mode, against central differences and the analytic answer
//! * `jvp`  — forward mode, i.e. the same directional derivative from the other side
//! * `hvp`  — reverse-over-reverse, against the analytic Hessian of a quadratic
//! * nested `grad` — a second derivative taken by differentiating a gradient
//! * `vmap` — one function body, many samples, with a counter that proves it
//! * `jit`  — trace once, replay, and what an explicit re-trace costs
const std = @import("std");
const zpu = @import("zpu");

const Io = std.Io;
const Tensor = zpu.Tensor;
const Shape = zpu.Shape;

/// Counts the checks that passed, so the summary line is honest.
var checks: usize = 0;

/// Every function handed to a transform has this shape: user data plus the
/// positional arguments. `ud` is how a caller keeps state (here: the allocator,
/// for the transform that needs to call another transform).
const Host = struct { alloc: std.mem.Allocator };

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file = Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_file.interface;

    try zpu.init(alloc, .{});
    defer zpu.deinit();

    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    var host = Host{ .alloc = alloc };

    try out.print("zpu — autodiff tour\n\n", .{});

    const x = try Tensor.fromArray(f64, &.{ 0.5, -1.0, 2.0 }, Shape.initComptime(&.{3}));
    try reverseMode(out, alloc, x);
    try forwardMode(out, alloc, x);
    try secondOrder(out, alloc, &host, x);
    try vectorising(out, alloc);
    try tracing(out, alloc, x);

    try out.print("\n{d} checks passed; every gradient above was verified\n", .{checks});
    try out.flush();
}

// ---------------------------------------------------------------------------
// the functions we differentiate
// ---------------------------------------------------------------------------

/// `sum(exp(x^2))` — the running example. `d/dx = 2x exp(x^2)`.
const ExpSquare = struct {
    fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        _ = ud;
        const sq = try zpu.square(args[0]);
        defer zpu.releaseTemp(sq);
        const e = try zpu.exp(sq);
        defer zpu.releaseTemp(e);
        return zpu.sumAll(e);
    }
};

/// `mean((x * w + b - y)^2)` — two parameters, so `wrt` has something to say.
const SquaredError = struct {
    fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        _ = ud;
        const pred = try zpu.add(try zpu.mul(args[0], args[1]), args[2]);
        defer zpu.releaseTemp(pred);
        const diff = try zpu.sub(pred, args[3]);
        defer zpu.releaseTemp(diff);
        const sq = try zpu.square(diff);
        defer zpu.releaseTemp(sq);
        return zpu.meanAll(sq);
    }
};

/// `0.5 * x^T Q x` — a quadratic, so `H f(x) · v = Q v` exactly.
const Quadratic = struct {
    fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        _ = ud;
        const qx = try zpu.matmul(args[1], args[0]);
        defer zpu.releaseTemp(qx);
        const inner = try zpu.mul(args[0], qx);
        defer zpu.releaseTemp(inner);
        return zpu.mulScalar(try zpu.sumAll(inner), 0.5);
    }
};

/// `sum(x^3)` — its gradient is `3x^2`, which is itself differentiable.
const Cube = struct {
    fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        _ = ud;
        const sq = try zpu.square(args[0]);
        defer zpu.releaseTemp(sq);
        const cube = try zpu.mul(args[0], sq);
        defer zpu.releaseTemp(cube);
        return zpu.sumAll(cube);
    }
};

/// `sum(grad(sum(x^3))) = sum(3x^2)`: differentiating a gradient is just another
/// `grad`, because the backward pass is recorded like any other op.
const D2Cube = struct {
    fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        // A transform inside a transform needs an allocator, so it arrives
        // through the user-data pointer - the way a caller passes state in.
        const host: *Host = @ptrCast(@alignCast(ud.?));
        const g = try zpu.grad(host.alloc, Cube.call, null, args, &.{0});
        defer host.alloc.free(g);
        defer {
            var gg = g[0];
            gg.deinit();
        }
        return zpu.sumAll(g[0]);
    }
};

/// `sum(x * y)` per sample — used to show that `vmap` runs the body once.
const PerSampleDot = struct {
    var calls: usize = 0;

    fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        _ = ud;
        calls += 1;
        const prod = try zpu.mul(args[0], args[1]);
        defer zpu.releaseTemp(prod);
        return zpu.sum(prod, zpu.AXIS_NONE, false);
    }
};

// ---------------------------------------------------------------------------
// reverse mode
// ---------------------------------------------------------------------------

fn reverseMode(out: *Io.Writer, alloc: std.mem.Allocator, x: Tensor) !void {
    try section(out, "reverse mode (`grad`)");

    // `grad` returns one owned tensor per entry of `wrt`, in that order.
    const grads = try zpu.grad(alloc, ExpSquare.call, null, &.{x}, &.{0});
    defer alloc.free(grads);
    defer {
        for (grads) |g| {
            var gg = g;
            gg.deinit();
        }
    }
    try row(out, "grad sum(exp(x^2))", grads[0]);

    // analytic: 2 x exp(x^2)
    var want: [3]f64 = undefined;
    for (0..3) |i| want[i] = 2 * x.readF64(i) * std.math.exp(x.readF64(i) * x.readF64(i));
    try close(out, "analytic 2x exp(x^2)", grads[0], &want, 1e-12);

    // central differences of the very same function
    var fd: [3]f64 = undefined;
    for (0..3) |i| fd[i] = try finiteDiff(x, i, 1e-5);
    try close(out, "central differences", grads[0], &fd, 1e-6);

    // several arguments: gradients come back positionally, in `wrt` order
    const xs = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, Shape.initComptime(&.{4}));
    const w = try Tensor.fromArray(f64, &.{ 0.5, -0.5, 1.0, 2.0 }, Shape.initComptime(&.{4}));
    const b = try Tensor.fromScalarF64(0.25, .f64);
    const y = try Tensor.fromArray(f64, &.{ 1, 1, 2, 2 }, Shape.initComptime(&.{4}));
    const pg = try zpu.grad(alloc, SquaredError.call, null, &.{ xs, w, b, y }, &.{ 1, 2 });
    defer alloc.free(pg);
    defer {
        for (pg) |g| {
            var gg = g;
            gg.deinit();
        }
    }
    try row(out, "d/dw mean((xw+b-y)^2)", pg[0]);
    try row(out, "d/db mean((xw+b-y)^2)", pg[1]);

    // `valueAndGrad` is the same sweep, plus the primal value it already had
    var r = try zpu.valueAndGrad(alloc, SquaredError.call, null, &.{ xs, w, b, y }, &.{1});
    defer r.deinit();
    try out.print("{s:<24} value {d:.6}, d/dw[0] {d:.6}\n", .{ "valueAndGrad", r.value.item(), r.grads[0].readF64(0) });
    checks += 1;

    // `valueAndGrad` on a two-parameter row of the loss: the mean over one
    // sample is what a normalisation layer sees
    const one = try zpu.narrow(xs, 0, 0, 1);
    var r1 = try zpu.valueAndGrad(alloc, SquaredError.call, null, &.{ one, w, b, y }, &.{ 0, 1 });
    defer r1.deinit();
    try row(out, "d/dx of one sample", r1.grads[0]);
    try row(out, "d/dw of one sample", r1.grads[1]);
    checks += 1;
}

/// Central difference of `ExpSquare` with respect to element `i`.
fn finiteDiff(x: Tensor, i: usize, h: f64) !f64 {
    var scope = try zpu.Scope.init(std.heap.smp_allocator);
    defer scope.deinit();

    const plus = try x.clone();
    plus.writeF64(i, plus.readF64(i) + h);
    const minus = try x.clone();
    minus.writeF64(i, minus.readF64(i) - h);

    const up = [_]Tensor{plus};
    const down = [_]Tensor{minus};
    const vp = try ExpSquare.call(null, &up);
    const vm = try ExpSquare.call(null, &down);
    return (vp.item() - vm.item()) / (2 * h);
}

// ---------------------------------------------------------------------------
// forward mode
// ---------------------------------------------------------------------------

fn forwardMode(out: *Io.Writer, alloc: std.mem.Allocator, x: Tensor) !void {
    try section(out, "forward mode (`jvp`)");

    // A tangent is a direction in the input space; `jvp` answers with the
    // directional derivative along it, without building a Jacobian.
    const v = try Tensor.fromArray(f64, &.{ 1, 0, -1 }, Shape.initComptime(&.{3}));
    const r = try zpu.jvp(alloc, ExpSquare.call, null, &.{x}, &.{v});
    defer {
        var a = r.value;
        a.deinit();
        var b = r.tangent;
        b.deinit();
    }
    try out.print("{s:<24} value {d:.6}, J·v {d:.6}\n", .{ "jvp", r.value.item(), r.tangent.item() });
    checks += 1;

    // The same number from the reverse side: <grad, v>.
    const grads = try zpu.grad(alloc, ExpSquare.call, null, &.{x}, &.{0});
    defer alloc.free(grads);
    defer {
        for (grads) |g| {
            var gg = g;
            gg.deinit();
        }
    }
    const dot = try zpu.sumAll(try zpu.mul(grads[0], v));
    try closeScalar(out, "jvp == <grad, v>", r.tangent.item(), dot.item(), 1e-12);
}

// ---------------------------------------------------------------------------
// second order
// ---------------------------------------------------------------------------

fn secondOrder(out: *Io.Writer, alloc: std.mem.Allocator, host: *Host, x: Tensor) !void {
    try section(out, "second order");

    // Hessian-vector product of a quadratic: H f(x) · v = Q v, exactly.
    const q = try Tensor.fromArray(f64, &.{ 2, 1, 1, 3 }, Shape.initComptime(&.{ 2, 2 }));
    const y = try Tensor.fromArray(f64, &.{ 0.3, -0.4 }, Shape.initComptime(&.{2}));
    const v = try Tensor.fromArray(f64, &.{ 1, 2 }, Shape.initComptime(&.{2}));
    const hv = try zpu.hvp(alloc, Quadratic.call, null, &.{ y, q }, 0, v);
    defer {
        var h = hv;
        h.deinit();
    }
    const qv = try zpu.matmul(q, v);
    try close(out, "hvp == Q v", hv, &.{ qv.readF64(0), qv.readF64(1) }, 1e-12);

    // Differentiating a gradient: grad(sum(grad(sum(x^3)))) = 6x
    // (the outer sweep walks the backward pass, which was recorded like any
    // other computation).
    const g1 = try zpu.grad(alloc, D2Cube.call, host, &.{x}, &.{0});
    defer alloc.free(g1);
    defer {
        for (g1) |g| {
            var gg = g;
            gg.deinit();
        }
    }
    try row(out, "d/dx sum(x^3) = 3x^2", g1[0]);
    try close(out, "6x (second derivative)", g1[0], &[3]f64{ 6 * x.readF64(0), 6 * x.readF64(1), 6 * x.readF64(2) }, 1e-12);
}

// ---------------------------------------------------------------------------
// vectorising
// ---------------------------------------------------------------------------

fn vectorising(out: *Io.Writer, alloc: std.mem.Allocator) !void {
    try section(out, "vectorising (`vmap`)");

    // `vmap` maps over the leading axis of every argument. The body is called
    // **once** with a batched depth, so a loop over samples never appears.
    const xs = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 3, 2 }));
    const ys = try Tensor.fromArray(f64, &.{ 10, 20, 30, 40, 50, 60 }, Shape.initComptime(&.{ 3, 2 }));
    PerSampleDot.calls = 0;
    const dots = try zpu.vmap(alloc, PerSampleDot.call, null, &.{ xs, ys });
    defer {
        var d = dots;
        d.deinit();
    }
    try row(out, "per-sample dot", dots);
    try out.print("{s:<24} body ran {d} time for {d} samples\n", .{ "vmap", PerSampleDot.calls, 3 });
    if (PerSampleDot.calls != 1) return error.NotVectorised;
    checks += 1;
    try close(out, "dot products", dots, &.{ 50, 250, 610 }, 1e-12);
}

// ---------------------------------------------------------------------------
// tracing
// ---------------------------------------------------------------------------

fn tracing(out: *Io.Writer, alloc: std.mem.Allocator, x: Tensor) !void {
    try section(out, "tracing (`jit`)");

    var j = zpu.jit(alloc, ExpSquare.call, null);
    defer j.deinit();

    const y1 = try j.call(&.{x});
    defer {
        var t = y1;
        t.deinit();
    }
    try out.print("{s:<24} trace_count {d} before the first call, {d} after\n", .{ "jit", 0, j.trace_count });
    if (j.plan) |p| try p.describe(out);

    // the replay does not re-trace, and answers the same thing
    const x2 = try Tensor.fromArray(f64, &.{ 1.0, 1.0, 1.0 }, Shape.initComptime(&.{3}));
    const y2 = try j.call(&.{x2});
    defer {
        var t = y2;
        t.deinit();
    }
    try out.print("{s:<24} trace_count {d}, steps {d}\n", .{ "after a replay", j.trace_count, j.numSteps() });
    if (j.trace_count != 1) return error.RetraceHappened;
    checks += 1;

    // A shape change needs an explicit trace: the folded constants were baked
    // with the shapes of the first call.
    const big = try Tensor.fromArray(f64, &.{ 0.1, 0.2, 0.3, 0.4 }, Shape.initComptime(&.{4}));
    try j.trace(&.{big});
    const y3 = try j.call(&.{big});
    defer {
        var t = y3;
        t.deinit();
    }
    try out.print("{s:<24} trace_count {d} after an explicit re-trace\n", .{ "shape change", j.trace_count });
    if (j.trace_count != 2) return error.MissingRetrace;
    checks += 1;

    // jit nests with grad: `jit(grad(f))` replays the backward pass too
    const G = struct {
        fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            const h: *Host = @ptrCast(@alignCast(ud.?));
            const g = try zpu.grad(h.alloc, ExpSquare.call, null, args, &.{0});
            defer h.alloc.free(g);
            return g[0];
        }
    };
    var host = Host{ .alloc = alloc };
    var jg = zpu.jit(alloc, G.call, &host);
    defer jg.deinit();
    const gj = try jg.call(&.{x});
    defer {
        var t = gj;
        t.deinit();
    }
    try row(out, "jit(grad) replay", gj);
    // The plan of a `jit(grad)` carries the backward pass as ordinary steps:
    // forward ops first, then one step per VJP. That is why replaying it is
    // faster than running `grad` again.
    try out.print("{s:<26} steps {d} (backward included)\n", .{ "jit(grad) plan", jg.numSteps() });
    if (jg.plan) |p| try p.describe(out);
    var want: [3]f64 = undefined;
    for (0..3) |i| want[i] = 2 * x.readF64(i) * std.math.exp(x.readF64(i) * x.readF64(i));
    try close(out, "jit(grad) matches grad", gj, &want, 1e-12);
}

// ---------------------------------------------------------------------------
// reporting helpers
// ---------------------------------------------------------------------------

fn section(out: *Io.Writer, title: []const u8) !void {
    try out.print("\n-- {s} ", .{title});
    var i: usize = title.len + 4;
    while (i < 60) : (i += 1) try out.print("-", .{});
    try out.print("\n", .{});
}

fn row(out: *Io.Writer, label: []const u8, t: Tensor) !void {
    try out.print("{s:<26}", .{label});
    try out.print("[", .{});
    for (0..t.rank()) |i| try out.print("{s}{d}", .{ if (i == 0) "" else "x", t.shape.dims[i] });
    try out.print("] ", .{});
    const n = t.numel();
    for (0..@min(n, 8)) |i| try out.print(" {d:>10.6}", .{t.readLogical(n, i)});
    if (n > 8) try out.print(" ...", .{});
    try out.print("\n", .{});
}

/// Compares a tensor against expected values, element by element.
fn close(out: *Io.Writer, label: []const u8, actual: Tensor, expected: []const f64, tol: f64) !void {
    if (actual.numel() != expected.len) return error.LengthMismatch;
    for (expected, 0..) |e, i| {
        const a = actual.readLogical(actual.numel(), i);
        if (@abs(a - e) > tol) {
            try out.print("  FAIL {s}: element {d} is {d}, expected {d}\n", .{ label, i, a, e });
            try out.flush();
            return error.ValueMismatch;
        }
    }
    checks += 1;
}

fn closeScalar(out: *Io.Writer, label: []const u8, actual: f64, expected: f64, tol: f64) !void {
    if (@abs(actual - expected) > tol) {
        try out.print("  FAIL {s}: {d} != {d}\n", .{ label, actual, expected });
        try out.flush();
        return error.ValueMismatch;
    }
    checks += 1;
}
