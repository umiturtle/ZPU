//! `zig build run-train-mlp`
//!
//! A complete training run, end to end: a synthetic two-class problem, a small
//! MLP, `valueAndGrad` for the gradients and Adam for the update. It prints the
//! loss curve, the accuracy, the learned decision boundary as ASCII art, and
//! **fails if the model did not learn** — so it doubles as a smoke test of the
//! autodiff and optimiser paths on this machine.
//!
//! The data are two concentric rings: a circle of radius 0.35 (class 0) inside a
//! ring of radius 1.0 (class 1). They are not linearly separable, so a model
//! that only fits a line cannot reach the accuracy this example demands.
const std = @import("std");
const zpu = @import("zpu");

const Io = std.Io;
const Tensor = zpu.Tensor;
const Shape = zpu.Shape;

const samples = 256;
const hidden = 16;
const steps = 400;
const lr = 0.05;
const seed = 0x5EED;

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

    try out.print("zpu — training an MLP on two rings\n\n", .{});

    const x, const y = try rings();
    const w1 = try zpu.nn.init.xavierNormal(Shape.initComptime(&.{ 2, hidden }), .f64, 2, hidden);
    const b1 = try Tensor.zeros(Shape.initComptime(&.{hidden}), .f64);
    const w2 = try zpu.nn.init.xavierNormal(Shape.initComptime(&.{ hidden, hidden }), .f64, hidden, hidden);
    const b2 = try Tensor.zeros(Shape.initComptime(&.{hidden}), .f64);
    const w3 = try zpu.nn.init.xavierNormal(Shape.initComptime(&.{ hidden, 1 }), .f64, hidden, 1);
    const b3 = try Tensor.zeros(Shape.initComptime(&.{1}), .f64);

    var params = [_]Tensor{ w1, b1, w2, b2, w3, b3 };
    const args = [_]Tensor{ x, y, w1, b1, w2, b2, w3, b3 };

    try out.print("{s:<12} {d} samples, 2 features, 2 rings (noise 0.12)\n", .{ "data", samples });
    try out.print("{s:<12} 2 -> {d} -> {d} -> 1 (gelu), {d} parameters\n", .{ "model", hidden, hidden, countParams(&params) });
    try out.print("{s:<12} Adam, lr {d}, {d} full-batch steps\n", .{ "optimiser", lr, steps });

    var adam = zpu.optim.Adam.init(alloc, .{ .lr = lr });
    defer adam.deinit();

    // `valueAndGrad` returns the loss *and* the gradients of the requested
    // arguments in one sweep. Only the parameters are in `wrt`, so the data
    // tensors are not differentiated.
    const wrt = [_]usize{ 2, 3, 4, 5, 6, 7 };
    var first_loss: f64 = 0;
    var last_loss: f64 = 0;

    const t0 = Io.Clock.awake.now(io).nanoseconds;
    var step: usize = 0;
    while (step < steps) : (step += 1) {
        var r = try zpu.valueAndGrad(alloc, Net.loss, null, &args, &wrt);
        defer r.deinit();
        last_loss = r.value.item();
        if (step == 0) first_loss = last_loss;
        if (step % 50 == 0 or step + 1 == steps) {
            try out.print("step {d:>4}   loss {d:.5}\n", .{ step, last_loss });
        }
        try adam.step(&params, r.grads);
    }
    const elapsed_ns = Io.Clock.awake.now(io).nanoseconds - t0;

    const acc = try accuracy(&args);
    const rate = @as(f64, @floatFromInt(steps)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    try out.print("\n{s:<12} {d:.5} -> {d:.5} ({d:.1}x)\n", .{ "loss", first_loss, last_loss, first_loss / last_loss });
    try out.print("{s:<12} {d:.1}% of {d} samples\n", .{ "accuracy", acc * 100, samples });
    try out.print("{s:<12} {d:.1} steps/s ({d:.2} ms per step, gradients included)\n", .{ "throughput", rate, 1000 / rate });

    try boundary(out, &args);
    try out.flush();

    if (last_loss > first_loss * 0.5) return error.ModelDidNotLearn;
    if (acc < 0.95) return error.AccuracyTooLow;
}

// ---------------------------------------------------------------------------
// the problem
// ---------------------------------------------------------------------------

/// The model: `x -> gelu(x·W1 + b1) -> gelu(·W2 + b2) -> W3 + b3` (logits).
const Net = struct {
    fn forward(args: []const Tensor) !Tensor {
        const h1 = try zpu.nn.dense(args[0], args[2], args[3]);
        defer zpu.releaseTemp(h1);
        const a1 = try zpu.nn.gelu(h1);
        defer zpu.releaseTemp(a1);
        const h2 = try zpu.nn.dense(a1, args[4], args[5]);
        defer zpu.releaseTemp(h2);
        const a2 = try zpu.nn.gelu(h2);
        defer zpu.releaseTemp(a2);
        return zpu.nn.dense(a2, args[6], args[7]);
    }

    /// Arguments, in order: `x, y, w1, b1, w2, b2, w3, b3`.
    fn loss(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
        _ = ud;
        const logits = try forward(args);
        defer zpu.releaseTemp(logits);
        // Logits, not probabilities: the fused form is stable for large values.
        return zpu.nn.binaryCrossEntropyWithLogits(logits, args[1]);
    }
};

/// Two concentric rings, generated from a seeded PRNG so a run is reproducible.
fn rings() !struct { Tensor, Tensor } {
    var g = zpu.rng.Rng.init(seed);
    var xs: [samples * 2]f64 = undefined;
    var ys: [samples]f64 = undefined;
    for (0..samples) |i| {
        const inner = i % 2 == 0;
        const radius: f64 = if (inner) 0.35 else 1.0;
        const r = radius + 0.12 * (g.nextF64() * 2 - 1);
        const theta = std.math.tau * g.nextF64();
        xs[2 * i + 0] = r * @cos(theta);
        xs[2 * i + 1] = r * @sin(theta);
        ys[i] = if (inner) 0.0 else 1.0;
    }
    const x = try Tensor.fromArray(f64, &xs, Shape.initComptime(&.{ samples, 2 }));
    // [samples, 1] so it lines up with the logits: broadcasting aligns to the
    // right, so a [samples] target would broadcast into a [samples, samples] grid.
    const y = try Tensor.fromArray(f64, &ys, Shape.initComptime(&.{ samples, 1 }));
    return .{ x, y };
}

fn countParams(params: []const Tensor) usize {
    var n: usize = 0;
    for (params) |p| n += p.numel();
    return n;
}

/// Fraction of samples on the correct side of the decision boundary.
fn accuracy(args: []const Tensor) !f64 {
    const logits = try Net.forward(args);
    defer zpu.releaseTemp(logits);
    var correct: usize = 0;
    for (0..samples) |i| {
        const p = 1.0 / (1.0 + std.math.exp(-logits.readLogical(samples, i)));
        const predicted: f64 = if (p >= 0.5) 1 else 0;
        if (predicted == args[1].readF64(i)) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(samples));
}

/// Prints the learned boundary over `[-1.5, 1.5]^2` with the data on top, so the
/// fit can be *seen* and not only measured.
fn boundary(out: *Io.Writer, args: []const Tensor) !void {
    const cols = 49;
    const rows = 21;
    const extent = 1.5;

    var grid: [rows * cols * 2]f64 = undefined;
    for (0..rows) |r| {
        for (0..cols) |c| {
            const fy = 1.0 - 2.0 * @as(f64, @floatFromInt(r)) / @as(f64, @floatFromInt(rows - 1));
            const fx = 2.0 * @as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(cols - 1)) - 1.0;
            grid[(r * cols + c) * 2 + 0] = fx * extent;
            grid[(r * cols + c) * 2 + 1] = fy * extent;
        }
    }
    const gx = try Tensor.fromArray(f64, &grid, Shape.initComptime(&.{ rows * cols, 2 }));

    // The same model, evaluated on the grid instead of the data. The grid
    // tensor belongs to the scope, exactly like the training tensors.
    var gargs: [8]Tensor = undefined;
    @memcpy(&gargs, args);
    gargs[0] = gx;
    const logits = try Net.forward(&gargs);
    defer zpu.releaseTemp(logits);

    // (Zig 0.17 has no `**` array-repeat operator any more: `@memset` over a
    // fixed-size array is the direct spelling.)
    var cells: [rows * cols]u8 = undefined;
    @memset(&cells, ' ');
    for (0..logits.numel()) |i| {
        cells[i] = if (logits.readLogical(logits.numel(), i) > 0) '#' else '.';
    }
    // the samples themselves: 'o' for class 0, 'x' for class 1
    for (0..samples) |i| {
        const sx = args[0].readLogical(samples * 2, i * 2 + 0);
        const sy = args[0].readLogical(samples * 2, i * 2 + 1);
        const c = @as(isize, @intFromFloat(@round((sx / extent * 0.5 + 0.5) * @as(f64, @floatFromInt(cols - 1)))));
        const r = @as(isize, @intFromFloat(@round((0.5 - sy / extent * 0.5) * @as(f64, @floatFromInt(rows - 1)))));
        if (c < 0 or r < 0 or c >= cols or r >= rows) continue;
        cells[@intCast(r * cols + c)] = if (args[1].readF64(i) > 0.5) 'x' else 'o';
    }

    try out.print("\nlearned boundary: '.' is the class-0 side and '#' the class-1 side,\n", .{});
    try out.print("o = inner ring (class 0), x = outer ring (class 1)\n\n", .{});
    for (0..rows) |r| {
        try out.print("  ", .{});
        try out.writeAll(cells[r * cols .. (r + 1) * cols]);
        try out.print("\n", .{});
    }
}
