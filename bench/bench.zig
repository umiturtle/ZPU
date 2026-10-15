//! ZPU benchmark suite.
//!
//! Run with `zig build bench` (always built in `ReleaseFast`):
//!
//! ```sh
//! zig build bench                          # full run
//! zig build bench -- --filter matmul       # one family of cases
//! zig build bench -- --quick               # fewer iterations
//! zig build bench -- --json bench/out.json # machine readable results
//! zig build bench -- --check               # compare against bench/baseline.txt
//! zig build bench -- --write-baseline      # refresh the baseline
//! ```
//!
//! ## Methodology
//!
//! * Each case prepares its inputs once and then runs one *iteration* (one op,
//!   one gradient, one optimiser step) between two reads of the monotonic clock.
//! * The iteration count is chosen so that every case runs for at least
//!   `--min-time` (default 25 ms) after `--warmup` discarded rounds, so short
//!   cases are not dominated by clock noise.
//! * A case declares how much *work* one iteration performs. The reported rate
//!   is that work divided by the measured time: bytes/s for memory bound
//!   kernels (counted as bytes actually touched), FLOP/s for `matmul`-shaped
//!   kernels, elements/s for pure compute, steps/s for gradients and updates.
//! * Nothing here needs Python, a BLAS, or a profiler: the numbers come out of
//!   the library itself and `bench/baseline.txt` records the last accepted run
//!   so `--check` can fail a regression.
//!
//! The absolute numbers depend on the machine and on the `ReleaseFast` build, so
//! `baseline.txt` is only meaningful for the machine that wrote it — that is
//! what `--write-baseline` is for.
const std = @import("std");
const zgf = @import("zgf");
const zpu = @import("zpu");
const zui = @import("zui");
const bench_options = @import("bench_options");

const Io = std.Io;
const Allocator = std.mem.Allocator;
const Tensor = zpu.Tensor;
const S = zpu.Shape;

/// What the numbers of a case mean.
const Kind = enum {
    /// bytes moved per iteration -> GB/s
    bytes,
    /// floating point operations -> GFLOP/s
    flops,
    /// elements produced -> Melem/s
    items,
    /// one gradient / update / replay -> steps/s
    steps,

    fn unit(k: Kind) []const u8 {
        return switch (k) {
            .bytes => "GB/s",
            .flops => "GFLOP/s",
            .items => "Melem/s",
            .steps => "steps/s",
        };
    }

    /// Scales `work` into the numerator of the reported rate.
    fn scale(k: Kind, work: f64) f64 {
        return switch (k) {
            .bytes, .flops => work / 1e9,
            .items => work / 1e6,
            .steps => work,
        };
    }
};

const Env = struct {
    alloc: Allocator,
    io: Io,

    // memory bound cases
    a32: Tensor,
    b32: Tensor,
    a64: Tensor,
    b64: Tensor,
    // matmul cases
    m32a: Tensor,
    m32b: Tensor,
    m64a: Tensor,
    m64b: Tensor,
    // vmap case: 64 matrices of 32x32
    batch32: Tensor,
    // convolution case
    img: Tensor,
    kernel: Tensor,
    // MLP case
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    adam: zpu.optim.Adam,
    jit_loss: zpu.Jit,

    // UI cases: one `Ui`, one framebuffer, and a body with enough widgets for
    // the numbers to mean something (see `BenchBody`).
    ui: zui.Ui,
    fb: zgf.Image,
    ui_app: UiApp,
    /// The last frame as instances: the buffer a backend is handed, the flat
    /// layout it uploads, and an RGBA target the kernel draws into.
    instance_buf: [max_instances]zui.Instance = undefined,
    instance_count: usize = 0,
    flat: []f32,
    target: []f32,
    /// A second UI, this one a hundred thousand rows long, for the virtual-list
    /// case: the number that matters is that its frame costs the rows you can
    /// see, not the rows it has.
    vlist: zui.Ui,
    /// Alternates each iteration, so the frame really does rebuild: a frame whose
    /// input is identical to the previous one is skipped by design (that is the
    /// `ui frame idle` case).
    ui_wiggle: bool = false,
};

const Case = struct {
    name: []const u8,
    kind: Kind,
    work: f64,
    run: *const fn (*Env) anyerror!void,
};

const mlp_batch = 64;
const mlp_in = 64;
const mlp_hidden = 64;
const mlp_out = 10;

/// `sumAll(mean(gelu(x @ w1 + b1) @ w2 + b2)^2)` — a small two layer MLP.
fn mlpLoss(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    const h = try zpu.nn.dense(args[0], args[1], args[2]);
    defer zpu.releaseTemp(h);
    const act = try zpu.nn.gelu(h);
    defer zpu.releaseTemp(act);
    const y = try zpu.nn.dense(act, args[3], args[4]);
    defer zpu.releaseTemp(y);
    const sq = try zpu.square(y);
    defer zpu.releaseTemp(sq);
    const per_row = try zpu.mean(sq, -1, false);
    defer zpu.releaseTemp(per_row);
    return zpu.sumAll(per_row);
}

/// Runs one gradient of `mlpLoss` and releases everything it produced.
fn oneGradient(env: *Env) !void {
    var scope = try zpu.Scope.init(env.alloc);
    defer scope.deinit();
    const args = [_]Tensor{ env.x, env.w1, env.b1, env.w2, env.b2 };
    const grads = try zpu.grad(env.alloc, mlpLoss, null, &args, &.{ 1, 2, 3, 4 });
    defer env.alloc.free(grads);
    for (grads) |g| {
        var gg = g;
        gg.deinit();
    }
}

// ---------------------------------------------------------------- the cases

fn caseAddF32(env: *Env) !void {
    var out = try zpu.add(env.a32, env.b32);
    out.deinit();
}

fn caseMulF64(env: *Env) !void {
    var out = try zpu.mul(env.a64, env.b64);
    out.deinit();
}

fn caseExpF32(env: *Env) !void {
    var out = try zpu.exp(env.a32);
    out.deinit();
}

fn caseSumF32(env: *Env) !void {
    var out = try zpu.sumAll(env.a32);
    out.deinit();
}

fn caseMatmulF32(env: *Env) !void {
    var out = try zpu.matmul(env.m32a, env.m32b);
    out.deinit();
}

fn caseMatmulF64(env: *Env) !void {
    var out = try zpu.matmul(env.m64a, env.m64b);
    out.deinit();
}

fn caseBatchMatmul(env: *Env) !void {
    var out = try zpu.matmul(env.batch32, env.batch32);
    out.deinit();
}

fn caseConv2d(env: *Env) !void {
    var out = try zpu.nn.conv2d(env.img, env.kernel, 1, 1);
    out.deinit();
}

fn caseLayerNorm(env: *Env) !void {
    var out = try zpu.nn.layerNorm(env.x, env.ln_w, env.ln_b, 1e-5, -1);
    out.deinit();
}

fn caseGradMlp(env: *Env) !void {
    try oneGradient(env);
}

fn caseAdamStep(env: *Env) !void {
    var scope = try zpu.Scope.init(env.alloc);
    defer scope.deinit();
    var params = [_]Tensor{ env.w1, env.b1, env.w2, env.b2 };
    const args = [_]Tensor{ env.x, env.w1, env.b1, env.w2, env.b2 };
    const grads = try zpu.grad(env.alloc, mlpLoss, null, &args, &.{ 1, 2, 3, 4 });
    defer env.alloc.free(grads);
    defer for (grads) |g| {
        var gg = g;
        gg.deinit();
    };
    try env.adam.step(&params, grads);
}

fn caseJitGradMlp(env: *Env) !void {
    const args = [_]Tensor{ env.x, env.w1, env.b1, env.w2, env.b2 };
    var out = try env.jit_loss.call(&args);
    out.deinit();
}

// --------------------------------------------------------------------- zui
//
// Three things are worth measuring in a UI library, and the three are different
// by orders of magnitude: laying a frame out (the body runs), noticing that a
// frame *cannot* have changed (the body does not run), and turning the display
// list into pixels. A GPU backend replaces the third; it does not replace the
// first two.

const ui_width: usize = 320;
const ui_height: usize = 200;

/// Room for the instance buffer: the bench body paints well under this, and a
/// fixed array keeps the case free of allocation (the frame itself never
/// allocates either).
const max_instances: usize = 512;

const UiApp = struct {
    gain: f32 = 0.4,
    toggled: bool = false,
    radio: usize = 0,
    clicks: usize = 0,
    buf: [32]u8 = undefined,
    field: zui.TextField = undefined,

    fn init(self: *UiApp) void {
        self.field = zui.TextField.init(&self.buf);
    }
};

/// A body with every widget family in it: buttons, toggles, radios, a slider, a
/// text field, a progress bar, a scrolling list and a collapsing section.
const BenchBody = struct {
    app: *UiApp,

    pub fn run(self: @This(), u: *zui.Ui) zui.Ui.Error!void {
        try u.container("root", .{ .w = .full, .h = .full, .pad = .all(12), .gap = 8, .bg = u.theme.bg }, BenchRoot{ .app = self.app });
    }

    const BenchRoot = struct {
        app: *UiApp,
        pub fn run(self: @This(), u: *zui.Ui) zui.Ui.Error!void {
            try u.label("benchmark", .{ .color = u.theme.text_dim });
            try u.container("controls", .{ .dir = .x, .gap = 6, .w = .full }, Controls{ .app = self.app });
            _ = try u.slider("gain", &self.app.gain, .{ .min = 0, .max = 1, .w = .{ .px = 150 } });
            _ = try u.textEdit("name", &self.app.field, .{ .placeholder = "type" });
            try u.progress(self.app.gain, .{ .w = 240 });
            try u.scrollable("list", .{ .style = .{ .w = .full, .h = .{ .px = 60 }, .gap = 2, .bg = u.theme.surface.withAlpha(0.6), .border = 1, .border_color = u.theme.border, .radius = 5 } }, Rows{});
            _ = try u.collapsing("about", "About", .{}, About{});
        }
    };

    const Controls = struct {
        app: *UiApp,
        pub fn run(self: @This(), u: *zui.Ui) zui.Ui.Error!void {
            if (try u.button("Click me", .{ .w = .{ .px = 100 } })) self.app.clicks += 1;
            _ = try u.checkbox("on", &self.app.toggled, .{});
            _ = try u.radio("a", &self.app.radio, 0, .{});
            _ = try u.radio("b", &self.app.radio, 1, .{});
        }
    };

    const Rows = struct {
        const names = [_][]const u8{ "row 0", "row 1", "row 2", "row 3", "row 4", "row 5", "row 6", "row 7" };
        pub fn run(_: @This(), u: *zui.Ui) zui.Ui.Error!void {
            for (names, 0..) |name, i| {
                try u.container(name, .{ .w = .full, .h = .{ .px = 18 }, .pad = .xy(6, 1), .bg = u.theme.widget.withAlpha(if (i % 2 == 0) 0.75 else 0.4), .radius = 3 }, RowLabel{ .text = name });
            }
        }
        const RowLabel = struct {
            text: []const u8,
            pub fn run(self: @This(), u: *zui.Ui) zui.Ui.Error!void {
                try u.label(self.text, .{});
            }
        };
    };

    const About = struct {
        pub fn run(_: @This(), u: *zui.Ui) zui.Ui.Error!void {
            try u.label("A frame is a function of the input and the retained state.", .{ .color = u.theme.text_dim });
        }
    };
};

/// A hundred thousand rows in a viewport two rows tall. The frame that builds
/// this is the frame that builds any other: the rows off screen are never called.
const VListBody = struct {
    pub fn run(_: @This(), u: *zui.Ui) zui.Ui.Error!void {
        try u.virtualList("big", 100_000, .{
            .style = .{ .w = .full, .h = .{ .px = ui_height }, .gap = 2 },
            .row_height = 20,
        }, VListBody{}, row);
    }

    fn row(_: @This(), u: *zui.Ui, i: usize, box: zui.Rect) zui.Ui.Error!void {
        try u.paintRect(box, u.theme.widget);
        var buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&buf, "row {d}", .{i}) catch unreachable;
        try u.label(label, .{});
    }
};

/// One frame of the virtual list, with the pointer moving so the frame rebuilds.
fn caseVirtualList(env: *Env) !void {
    env.ui_wiggle = !env.ui_wiggle;
    const x: f32 = if (env.ui_wiggle) 58 else 59;
    _ = try env.vlist.run(.{ .pointer = .{ .x = x, .y = 47 } }, VListBody{});
}

/// One frame with the pointer moving over a widget: the body runs, the picture
/// changes.
fn caseUiFrame(env: *Env) !void {
    env.ui_wiggle = !env.ui_wiggle;
    const x: f32 = if (env.ui_wiggle) 58 else 59;
    _ = try env.ui.run(.{ .pointer = .{ .x = x, .y = 47 } }, BenchBody{ .app = &env.ui_app });
}

/// One frame in which nothing happened: the display list is reused, the body is
/// not called.
fn caseUiIdleFrame(env: *Env) !void {
    const input = zui.Input{ .pointer = .{ .x = 9999, .y = 9999 } };
    _ = try env.ui.run(input, BenchBody{ .app = &env.ui_app });
}

/// The display list of the last frame, rasterised into a framebuffer.
fn caseUiRaster(env: *Env) !void {
    zui.render.draw(&env.ui.list, &env.fb, .{ .clear = env.ui.theme.bg });
}

/// The same, at 2x device pixels — what a HiDPI host pays.
fn caseUiRaster2x(env: *Env) !void {
    zui.render.draw(&env.ui.list, &env.fb, .{ .clear = env.ui.theme.bg, .scale = 2 });
}

/// The last frame walked into an instance buffer: one record per shape, text
/// expanded per glyph. This is pure list traversal — no pixels — so it is the
/// cheapest part of a GPU frame and worth its own number.
fn caseUiInstances(env: *Env) !void {
    env.instance_count = zui.instances.collect(&env.ui.list, .{}, &env.instance_buf).len;
}

/// The same records flattened into the layout a device buffer wants.
fn caseUiPack(env: *Env) !void {
    _ = zui.pipeline.pack(env.instance_buf[0..env.instance_count], env.flat);
}

/// The frame drawn *from the buffer*, with the same coverage maths the
/// software renderer uses — what a backend that runs invocations in order does.
fn caseUiRasterViaInstances(env: *Env) !void {
    zui.pipeline.draw(&env.ui.list, &env.fb, .{ .clear = env.ui.theme.bg }, &env.instance_buf);
}

/// A whole frame through the `zbackend` kernel into an RGBA target: the device
/// path, on whatever backend is present (the CPU, on the machines CI runs on).
fn caseUiKernelDispatch(env: *Env) !void {
    env.instance_count = zui.instances.collect(&env.ui.list, .{}, &env.instance_buf).len;
    // The same "clear, then draw" the software case does: the kernel blends over
    // whatever the target already holds, so a target reused across frames has to
    // start from the background.
    const bg = env.ui.theme.bg;
    for (0..ui_width * ui_height) |i| {
        env.target[i * 4] = bg.r;
        env.target[i * 4 + 1] = bg.g;
        env.target[i * 4 + 2] = bg.b;
        env.target[i * 4 + 3] = 1;
    }
    try zui.pipeline.dispatch(.cpu, env.instance_buf[0..env.instance_count], env.target, ui_width, ui_height, .{}, env.alloc);
}

const cases = [_]Case{
    .{ .name = "add f32 [1M]", .kind = .bytes, .work = 3 * 1_000_000 * 4, .run = caseAddF32 },
    .{ .name = "mul f64 [1M]", .kind = .bytes, .work = 3 * 1_000_000 * 8, .run = caseMulF64 },
    .{ .name = "exp f32 [1M]", .kind = .bytes, .work = 2 * 1_000_000 * 4, .run = caseExpF32 },
    .{ .name = "sum f32 [1M]", .kind = .bytes, .work = 1_000_000 * 4, .run = caseSumF32 },
    .{ .name = "matmul f32 512x512x512", .kind = .flops, .work = 2.0 * 512 * 512 * 512, .run = caseMatmulF32 },
    .{ .name = "matmul f64 256x256x256", .kind = .flops, .work = 2.0 * 256 * 256 * 256, .run = caseMatmulF64 },
    .{ .name = "batch matmul f32 64x32x32", .kind = .flops, .work = 64.0 * 2 * 32 * 32 * 32, .run = caseBatchMatmul },
    .{ .name = "conv2d f32 1x3x32x32 k3", .kind = .flops, .work = 2.0 * 1 * 8 * 32 * 32 * 3 * 3 * 3, .run = caseConv2d },
    .{ .name = "layerNorm f32 [64,64]", .kind = .bytes, .work = 2 * 64 * 64 * 4, .run = caseLayerNorm },
    .{ .name = "grad MLP f32 (batch 64)", .kind = .steps, .work = 1, .run = caseGradMlp },
    .{ .name = "adam step MLP f32", .kind = .steps, .work = 1, .run = caseAdamStep },
    .{ .name = "jit(grad MLP) f32", .kind = .steps, .work = 1, .run = caseJitGradMlp },
    .{ .name = "ui frame build 320x200", .kind = .steps, .work = 1, .run = caseUiFrame },
    .{ .name = "ui frame idle (reused)", .kind = .steps, .work = 1, .run = caseUiIdleFrame },
    .{ .name = "ui raster 320x200", .kind = .items, .work = ui_width * ui_height, .run = caseUiRaster },
    .{ .name = "ui raster 320x200 @2x", .kind = .items, .work = ui_width * ui_height * 4, .run = caseUiRaster2x },
    .{ .name = "ui frame -> instances", .kind = .steps, .work = 1, .run = caseUiInstances },
    .{ .name = "ui instance pack", .kind = .steps, .work = 1, .run = caseUiPack },
    .{ .name = "ui raster via instances", .kind = .items, .work = ui_width * ui_height, .run = caseUiRasterViaInstances },
    .{ .name = "ui kernel dispatch", .kind = .items, .work = ui_width * ui_height, .run = caseUiKernelDispatch },
    .{ .name = "ui virtual list 100k rows", .kind = .steps, .work = 1, .run = caseVirtualList },
};

// ------------------------------------------------------------- the harness

const Options = struct {
    filter: ?[]const u8 = null,
    json_path: ?[]const u8 = null,
    check: bool = false,
    write_baseline: bool = false,
    quick: bool = false,
    min_time_ns: i128 = 25 * std.time.ns_per_ms,
    warmup: usize = 2,
    /// tolerated relative regression before `--check` fails
    tolerance: f64 = 0.10,
};

const Result = struct {
    name: []const u8,
    unit: []const u8,
    rate: f64,
    ns_per_iter: f64,
    iters: usize,
};

fn nowNs(io: Io) i128 {
    return @as(i128, @intCast(Io.Clock.awake.now(io).nanoseconds));
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const alloc = init.gpa;

    var stdout_buffer: [0x1000]u8 = undefined;
    var stdout_file = Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_file.interface;

    var opts = Options{};
    var filter_from_args: ?[]const u8 = null;
    const args = try init.minimal.args.toSlice(arena);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--filter")) {
            i += 1;
            if (i == args.len) return usage(out);
            filter_from_args = args[i];
        } else if (std.mem.eql(u8, arg, "--json")) {
            i += 1;
            if (i == args.len) return usage(out);
            opts.json_path = args[i];
        } else if (std.mem.eql(u8, arg, "--min-time")) {
            i += 1;
            if (i == args.len) return usage(out);
            const ms = try std.fmt.parseFloat(f64, args[i]);
            opts.min_time_ns = @intFromFloat(ms * @as(f64, std.time.ns_per_ms));
        } else if (std.mem.eql(u8, arg, "--tolerance")) {
            i += 1;
            if (i == args.len) return usage(out);
            opts.tolerance = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--quick")) {
            opts.quick = true;
        } else if (std.mem.eql(u8, arg, "--check")) {
            opts.check = true;
        } else if (std.mem.eql(u8, arg, "--write-baseline")) {
            opts.write_baseline = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return usage(out);
        } else {
            try out.print("unknown option: {s}\n", .{arg});
            return usage(out);
        }
    }
    // the build script passes the same options as -D flags
    if (filter_from_args == null) {
        if (bench_options.bench_filter) |f| filter_from_args = f;
    }
    if (opts.json_path == null) {
        if (bench_options.bench_json) |p| opts.json_path = p;
    }
    opts.filter = filter_from_args;
    if (opts.quick) opts.min_time_ns = 5 * std.time.ns_per_ms;

    try zpu.init(alloc, .{ .threads = 1 });
    defer zpu.deinit();

    const env = try alloc.create(Env);
    defer alloc.destroy(env);
    env.* = try initEnv(alloc, io);
    defer deinitEnv(env);

    try out.print(
        \\zpu benchmarks — {s} build, iteration >= {d} ms
        \\
        \\
    , .{ @tagName(@import("builtin").mode), @divTrunc(opts.min_time_ns, std.time.ns_per_ms) });

    var results: std.ArrayListUnmanaged(Result) = .empty;
    defer results.deinit(alloc);

    try out.print("{s:<28} {s:>12} {s:>12} {s:>10}\n", .{ "case", "rate", "unit", "ns/iter" });
    try out.print("{s:-<28} {s:->12} {s:->12} {s:->10}\n", .{ "", "", "", "" });
    // Flush the header and every case: a run that crashes (or is killed) should
    // still say which case it got to.
    try out.flush();

    for (cases) |case| {
        if (opts.filter) |f| {
            if (std.mem.indexOf(u8, case.name, f) == null) continue;
        }
        const timer = try Timer.start(env, case, opts);
        const rate = case.kind.scale(case.work) / (timer.ns_per_iter / std.time.ns_per_s);
        try results.append(alloc, .{
            .name = case.name,
            .unit = case.kind.unit(),
            .rate = rate,
            .ns_per_iter = timer.ns_per_iter,
            .iters = timer.iters,
        });
        try out.print("{s:<28} {d:>12.3} {s:>12} {d:>10.1}\n", .{ case.name, rate, case.kind.unit(), timer.ns_per_iter });
        try out.flush();
    }
    try out.flush();

    var failed = false;
    if (opts.check) failed = try checkBaseline(io, out, results.items, opts.tolerance);
    if (opts.write_baseline) try writeBaseline(io, out, results.items);
    if (opts.json_path) |path| try writeJson(io, out, path, results.items);

    if (failed) {
        try out.print("\nbaseline check FAILED\n", .{});
        try out.flush();
        std.process.exit(1);
    }
}

fn usage(out: *Io.Writer) Io.Writer.Error!void {
    try out.print(
        \\zpu-bench [options]
        \\
        \\  --filter <text>      only run cases whose name contains <text>
        \\  --json <path>        also write the results as JSON
        \\  --check              compare against bench/baseline.txt and exit 1 on a regression
        \\  --write-baseline     record the current numbers as bench/baseline.txt
        \\  --tolerance <frac>   allowed regression for --check (default 0.10)
        \\  --min-time <ms>      minimum time per case (default 25)
        \\  --quick              same as --min-time 5
        \\  --help               this text
        \\
    , .{});
}

const Timer = struct {
    ns_per_iter: f64,
    iters: usize,

    /// Calibrates the iteration count, then measures a fresh round of them.
    fn start(env: *Env, case: Case, opts: Options) !Timer {
        for (0..opts.warmup) |_| try case.run(env);

        var iters: usize = 1;
        var ns_per_iter: f64 = 0;
        // grow the batch until one round lasts at least `min_time_ns`
        while (true) {
            const t0 = nowNs(env.io);
            for (0..iters) |_| try case.run(env);
            const elapsed: f64 = @floatFromInt(nowNs(env.io) - t0);
            ns_per_iter = elapsed / @as(f64, @floatFromInt(iters));
            if (elapsed >= @as(f64, @floatFromInt(opts.min_time_ns)) or iters >= 1_000_000) break;
            const wanted: f64 = @as(f64, @floatFromInt(opts.min_time_ns)) / @max(ns_per_iter, 1);
            iters = @intFromFloat(@max(wanted, @as(f64, @floatFromInt(iters)) * 2));
            iters = @min(iters, 1_000_000);
        }
        // one more, clean round at the final iteration count
        const t0 = nowNs(env.io);
        for (0..iters) |_| try case.run(env);
        const elapsed: f64 = @floatFromInt(nowNs(env.io) - t0);
        return .{ .ns_per_iter = elapsed / @as(f64, @floatFromInt(iters)), .iters = iters };
    }
};

fn initEnv(alloc: Allocator, io: Io) !Env {
    const vec_shape = try S.init(&.{1_000_000});
    const mat_shape = try S.init(&.{ 512, 512 });
    const mat64_shape = try S.init(&.{ 256, 256 });
    const batch_shape = try S.init(&.{ 64, 32, 32 });
    const img_shape = try S.init(&.{ 1, 3, 32, 32 });
    const kern_shape = try S.init(&.{ 8, 3, 3, 3 });
    const x_shape = try S.init(&.{ mlp_batch, mlp_in });
    const w1_shape = try S.init(&.{ mlp_in, mlp_hidden });
    const b1_shape = try S.init(&.{mlp_hidden});
    const w2_shape = try S.init(&.{ mlp_hidden, mlp_out });
    const b2_shape = try S.init(&.{mlp_out});
    const ln_shape = try S.init(&.{mlp_in});

    var rng = zpu.rng.Rng.init(0x5eed);
    var env = Env{
        .alloc = alloc,
        .io = io,
        .a32 = try rng.uniform(vec_shape, .f32),
        .b32 = try rng.uniform(vec_shape, .f32),
        .a64 = try rng.uniform(vec_shape, .f64),
        .b64 = try rng.uniform(vec_shape, .f64),
        .m32a = try rng.normal(mat_shape, .f32),
        .m32b = try rng.normal(mat_shape, .f32),
        .m64a = try rng.normal(mat64_shape, .f64),
        .m64b = try rng.normal(mat64_shape, .f64),
        .batch32 = try rng.normal(batch_shape, .f32),
        .img = try rng.normal(img_shape, .f32),
        .kernel = try rng.normal(kern_shape, .f32),
        .x = try rng.normal(x_shape, .f32),
        .w1 = try rng.normal(w1_shape, .f32),
        .b1 = try zpu.Tensor.zeros(b1_shape, .f32),
        .w2 = try rng.normal(w2_shape, .f32),
        .b2 = try zpu.Tensor.zeros(b2_shape, .f32),
        .ln_w = try zpu.Tensor.ones(ln_shape, .f32),
        .ln_b = try zpu.Tensor.zeros(ln_shape, .f32),
        .adam = zpu.optim.Adam.init(alloc, .{ .lr = 1e-3 }),
        .jit_loss = zpu.jit(alloc, mlpLoss, null),
        .ui = zui.Ui.init(alloc, .{ .width = @floatFromInt(ui_width), .height = @floatFromInt(ui_height) }),
        .fb = try zgf.Image.init(alloc, ui_width, ui_height, 3),
        .ui_app = .{},
        .flat = try alloc.alloc(f32, max_instances * zui.pipeline.stride),
        .target = try alloc.alloc(f32, ui_width * ui_height * 4),
        .vlist = zui.Ui.init(alloc, .{ .width = @floatFromInt(ui_width), .height = @floatFromInt(ui_height) }),
    };
    env.ui_app.init();
    // One frame before the first case: the display list the raster cases draw is
    // the one a real host would have produced, and the idle case needs a
    // previous frame to reuse.
    _ = try env.ui.run(.{ .pointer = .{ .x = 58, .y = 47 } }, BenchBody{ .app = &env.ui_app });
    // force a trace up front so the first timed iteration is a replay
    const jit_args = [_]Tensor{ env.x, env.w1, env.b1, env.w2, env.b2 };
    var first = try env.jit_loss.call(&jit_args);
    first.deinit();
    return env;
}

fn deinitEnv(env: *Env) void {
    const owned = [_]*Tensor{
        &env.a32,  &env.b32,  &env.a64,     &env.b64, &env.m32a,   &env.m32b,
        &env.m64a, &env.m64b, &env.batch32, &env.img, &env.kernel, &env.x,
        &env.w1,   &env.b1,   &env.w2,      &env.b2,  &env.ln_w,   &env.ln_b,
    };
    for (owned) |t| t.deinit();
    env.adam.deinit();
    env.jit_loss.deinit();
    env.ui.deinit();
    env.vlist.deinit();
    env.fb.deinit();
    env.alloc.free(env.flat);
    env.alloc.free(env.target);
}

// ---------------------------------------------------------------- reporting

fn readFileOrNull(io: Io, path: []const u8, buf: []u8) ?[]const u8 {
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    const n = file.readPositionalAll(io, buf, 0) catch return null;
    return buf[0..n];
}

fn checkBaseline(io: Io, out: *Io.Writer, results: []const Result, tolerance: f64) !bool {
    // baseline lives next to this source file: <repo>/bench/baseline.txt
    var file_buf: [64 * 1024]u8 = undefined;
    const text = readFileOrNull(io, "bench/baseline.txt", &file_buf) orelse {
        try out.print("\nno bench/baseline.txt yet — run with --write-baseline first\n", .{});
        return false;
    };

    var failed = false;
    try out.print("\n{s:<28} {s:>12} {s:>12} {s:>8}\n", .{ "case", "baseline", "now", "ratio" });
    try out.print("{s:-<28} {s:->12} {s:->12} {s:->8}\n", .{ "", "", "", "" });
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var checked: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var fields = std.mem.tokenizeAny(u8, line, " \t");
        const name = fields.next() orelse continue;
        const value_text = fields.next() orelse continue;
        const base = std.fmt.parseFloat(f64, value_text) catch continue;
        for (results) |r| {
            if (!std.mem.eql(u8, r.name, name)) continue;
            const ratio = if (base == 0) 1.0 else r.rate / base;
            const status = if (ratio < 1.0 - tolerance) "  FAIL" else "";
            if (status.len != 0) failed = true;
            try out.print("{s:<28} {d:>12.3} {d:>12.3} {d:>8.3}{s}\n", .{ r.name, base, r.rate, ratio, status });
            checked += 1;
            break;
        }
    }
    try out.print("{d} cases compared, tolerance {d:.0}%\n", .{ checked, tolerance * 100 });
    return failed;
}

fn writeBaseline(io: Io, out: *Io.Writer, results: []const Result) !void {
    var buffer: [64 * 1024]u8 = undefined;
    var w = Io.Writer.fixed(&buffer);
    try w.print("# zpu benchmark baseline — written by `zig build bench -- --write-baseline`\n", .{});
    try w.print("# machine specific: re-record after changing CPU, compiler or kernel.\n", .{});
    try w.print("# format: <case name> <rate>\n", .{});
    for (results) |r| try w.print("{s} {d:.6}\n", .{ r.name, r.rate });

    const file = try Io.Dir.cwd().createFile(io, "bench/baseline.txt", .{});
    defer file.close(io);
    try file.writeStreamingAll(io, w.buffered());
    try out.print("\nwrote bench/baseline.txt ({d} cases)\n", .{results.len});
}

fn writeJson(io: Io, out: *Io.Writer, path: []const u8, results: []const Result) !void {
    var buffer: [64 * 1024]u8 = undefined;
    var w = Io.Writer.fixed(&buffer);
    // builtin fields, so the file is self describing
    try w.print("{{\n  \"cases\": [\n", .{});
    for (results, 0..) |r, i| {
        try w.print(
            "    {{ \"name\": \"{s}\", \"rate\": {d:.6}, \"unit\": \"{s}\", \"ns_per_iter\": {d:.3}, \"iters\": {d} }}{s}\n",
            .{ r.name, r.rate, r.unit, r.ns_per_iter, r.iters, if (i + 1 == results.len) "" else "," },
        );
    }
    try w.print("  ]\n}}\n", .{});

    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, w.buffered());
    try out.print("wrote {s}\n", .{path});
}
