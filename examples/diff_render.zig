//! `zig build run-diff-render`
//!
//! Differentiable rendering: the image is a *function* of the scene parameters
//! (centre, radius, three colour channels) and the loss is a function of the
//! pixels, so `zpu.grad` walks from the pixels back to the parameters. Nothing
//! in the renderer is a rasteriser in the classical sense — a signed distance,
//! a sigmoid and an `over` blend are all it takes, and every one of them is a
//! `zpu` op, which is what makes the whole picture differentiable.
//!
//! The target image is *drawn* by zgf (`verticalGradient` + `drawCircle`), so
//! the two implementations of "a circle on a gradient" have to agree: the fit
//! has to find the drawn centre, radius and colour again — to within a third of
//! a pixel, once the half-pixel difference between "pixel index" and "pixel
//! centre" is accounted for.
//!
//! The fit anneals the edge softness (16 px down to 0.8 px). That is the one
//! non-obvious trick in differentiable rendering: with a sharp edge, moving a
//! shape that sits 30 px away from where it belongs changes *no* pixel at all,
//! so the gradient w.r.t. its centre is zero and gradient descent cannot even
//! start. A wide edge lets every pixel see the target from far away; the sharp
//! stages then finish the job.
//!
//! What the example verifies, beyond the fit: the autodiff gradient against a
//! central finite difference of the same loss (agreement to ~1e-9), the sign of
//! the gradient w.r.t. centre and radius against where the truth is, that one
//! optimizer step along the gradient really lowers the loss, and that the final
//! picture matches the target's silhouette area. It writes
//! `zig-out/diff-render.ppm`: target | start | fit | 8x error.
const std = @import("std");
const zpu = @import("zpu");
const zgf = @import("zgf");

const Io = std.Io;
const Color = zgf.Color;
const Image = zgf.Image;
const Shape = zpu.Shape;
const Tensor = zpu.Tensor;

const W = 128;
const H = 96;
/// Softness of the disc's edge, in pixels, and how many Adam steps each stage
/// gets. This is the whole trick: a hard edge has a zero gradient almost
/// everywhere, a ramp does not — and a *wide* ramp lets a pixel see the target
/// from far away, which is what lets the centre move at all.
const blur_stages = [_]f64{ 16, 8, 4, 2, 0.8 };
const steps_per_stage = [_]usize{ 50, 50, 50, 60, 120 };
const lr = 0.5;
/// A deliberately small rate, for the single-step check below.
const probe_lr = 0.02;

const sky_top = Color{ .r = 0.05, .g = 0.07, .b = 0.18 };
const sky_bottom = Color{ .r = 0.85, .g = 0.45, .b = 0.25 };

/// What zgf draws, and what the fit has to find again.
const truth: Pose = .{ .cx = 84.0, .cy = 38.0, .r = 22.0, .colour = .{ 0.95, 0.93, 0.86 } };
/// Where the fit starts: off centre, too small, grey. The scene is deliberately
/// wrong so that the image has to *move* to fit.
const start: Pose = .{ .cx = 40.0, .cy = 60.0, .r = 14.0, .colour = .{ 0.5, 0.5, 0.5 } };

/// A disc in pixel units and a colour in `0..1`.
///
/// The coordinates are in the *pixel-centre* frame the SDF works in: pixel
/// `(x, y)` sits at `(x + 0.5, y + 0.5)`. zgf's `drawCircle` fills by pixel
/// index, so the disc it draws for `truth` is centred at `(cx + 0.5, cy + 0.5)`
/// in this frame — and the fit has to land there, not at `cx`. That half-pixel
/// is why the checks below compare against `truth.cx + 0.5`: it is a statement
/// about how the two rasterisers line up.
const Pose = struct { cx: f64, cy: f64, r: f64, colour: [3]f64 };

/// The six parameters, in the order the loss takes them.
const names = [_][]const u8{ "cx", "cy", "radius", "red", "green", "blue" };

/// Parameter `i` of a `Pose`, flattened into the array `loss` receives.
fn poseParam(p: Pose, i: usize) f64 {
    return if (i < 3) switch (i) {
        0 => p.cx,
        1 => p.cy,
        else => p.r,
    } else p.colour[i - 3];
}

var checks: usize = 0;

/// Everything the loss needs that is not a parameter, plus the target pixels.
const Scene = struct {
    target: Tensor, // [H, W, 3]
    blur: f64, // current stage's edge softness
    xs: Tensor, // [1, W] pixel centres
    ys: Tensor, // [H, 1] pixel centres
    bg: [3]Tensor, // [H, 1] background per channel

    fn coverage(self: *const Scene, cx: Tensor, cy: Tensor, radius: Tensor) !Tensor {
        const dx = try zpu.sub(self.xs, cx);
        defer zpu.releaseTemp(dx);
        const dy = try zpu.sub(self.ys, cy);
        defer zpu.releaseTemp(dy);
        const dx2 = try zpu.square(dx);
        defer zpu.releaseTemp(dx2);
        const dy2 = try zpu.square(dy);
        defer zpu.releaseTemp(dy2);
        const d2 = try zpu.add(dx2, dy2); // [H, W] by broadcasting
        defer zpu.releaseTemp(d2);
        // `sqrt` has an infinite derivative at 0, so the squared distance gets
        // a floor: the pixel at the very centre of the disc must not produce a
        // NaN gradient.
        const floored = try zpu.addScalar(d2, 1e-4);
        defer zpu.releaseTemp(floored);
        const dist = try zpu.sqrt(floored);
        defer zpu.releaseTemp(dist);
        const sdf = try zpu.sub(dist, radius);
        defer zpu.releaseTemp(sdf);
        const scaled = try zpu.divScalar(sdf, -self.blur);
        defer zpu.releaseTemp(scaled);
        return zpu.sigmoid(scaled); // 1 inside the disc, 0 outside
    }

    /// The picture: coverage, then one `over` blend per channel.
    fn render(self: *const Scene, args: []const Tensor) !Tensor {
        const cov = try self.coverage(args[0], args[1], args[2]);
        defer zpu.releaseTemp(cov);
        const chans = [3]Tensor{
            try mixChannel(self.bg[0], args[3], cov),
            try mixChannel(self.bg[1], args[4], cov),
            try mixChannel(self.bg[2], args[5], cov),
        };
        defer {
            for (chans) |c| zpu.releaseTemp(c);
        }
        // [H, W] per channel -> [H, W, 3] for `Image.fromTensor`.
        const e0 = try zpu.expandDims(chans[0], -1);
        defer zpu.releaseTemp(e0);
        const e1 = try zpu.expandDims(chans[1], -1);
        defer zpu.releaseTemp(e1);
        const e2 = try zpu.expandDims(chans[2], -1);
        defer zpu.releaseTemp(e2);
        const rg = try zpu.concat(e0, e1, -1);
        defer zpu.releaseTemp(rg);
        return zpu.concat(rg, e2, -1); // [H, W, 3]
    }
};

/// `colour * cov + background * (1 - cov)`: the compositor's `over`, written
/// out of `zpu` ops so that it has a derivative like everything else.
fn mixChannel(background: Tensor, colour: Tensor, cov: Tensor) !Tensor {
    const shifted = try zpu.subScalar(cov, 1.0);
    defer zpu.releaseTemp(shifted);
    const one_minus = try zpu.neg(shifted);
    defer zpu.releaseTemp(one_minus);
    const under = try zpu.mul(background, one_minus);
    defer zpu.releaseTemp(under);
    const over = try zpu.mul(cov, colour);
    defer zpu.releaseTemp(over);
    return zpu.add(under, over);
}

fn loss(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    const scene: *Scene = @ptrCast(@alignCast(ud.?));
    const rgb = try scene.render(args);
    defer zpu.releaseTemp(rgb);
    const diff = try zpu.sub(rgb, scene.target);
    defer zpu.releaseTemp(diff);
    const sq = try zpu.square(diff);
    defer zpu.releaseTemp(sq);
    return zpu.sumAll(sq);
}

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

    // ------------------------------------------------------------- the target
    var target_img = try Image.init(alloc, W, H, 3);
    defer target_img.deinit();
    target_img.verticalGradient(sky_top, sky_bottom);
    target_img.drawCircle(@intFromFloat(truth.cx), @intFromFloat(truth.cy), @intFromFloat(truth.r), .{
        .r = @floatCast(truth.colour[0]),
        .g = @floatCast(truth.colour[1]),
        .b = @floatCast(truth.colour[2]),
    });
    try save(io, target_img, "zig-out/diff-render-target.ppm");

    // Everything created while the scope is alive lives until `scope.deinit()`,
    // so the long lived pieces are deliberately not freed by hand: `xs`, `ys`,
    // the background and the parameters are all still needed at the end.
    const target = try imageToTensor(target_img);
    const xs = try zpu.reshape(try Tensor.linspace(0.5, W - 0.5, W, .f64), Shape.initComptime(&.{ 1, W }));
    const ys = try zpu.reshape(try Tensor.linspace(0.5, H - 0.5, H, .f64), Shape.initComptime(&.{ H, 1 }));
    const row_t = try zpu.reshape(try Tensor.linspace(0, 1, H, .f64), Shape.initComptime(&.{ H, 1 }));

    // The renderer's background is the same gradient zgf drew, from the same two
    // colours: the fit only has to explain the disc.
    var bg: [3]Tensor = undefined;
    for (0..3) |c| {
        const delta = channel(sky_bottom, c) - channel(sky_top, c);
        const ramp = try zpu.mulScalar(row_t, delta); // [H, 1]
        defer zpu.releaseTemp(ramp);
        bg[c] = try zpu.addScalar(ramp, channel(sky_top, c));
    }

    var scene = Scene{ .target = target, .xs = xs, .ys = ys, .bg = bg, .blur = blur_stages[0] };

    var params: [6]Tensor = undefined;
    for (0..6) |k| params[k] = try Tensor.fromScalarF64(poseParam(start, k), .f64);

    const wrt = [_]usize{ 0, 1, 2, 3, 4, 5 };
    var first = try zpu.valueAndGrad(alloc, loss, &scene, &params, &wrt);
    defer first.deinit();
    const first_loss = first.value.item();

    try out.print("\ndifferentiable renderer: {d}x{d}, {d} parameters, soft edge {d} px\n", .{ W, H, 6, blur_stages[0] });
    try out.print("loss at the start {d:.4} (per pixel {d:.6})\n\n", .{ first_loss, first_loss / (W * H * 3) });

    // ---------------------------------------------- is this really a gradient?
    // The definition of a derivative, checked: central differences of the very
    // same loss against the sweep's answer. Done at the *coarsest* blur, where
    // the picture is smooth on the scale of the finite-difference step.
    try out.print("gradient at the start (blur {d} px)\n", .{scene.blur});
    try out.print("{s:<8} {s:>14} {s:>14} {s:>9}\n", .{ "param", "autodiff", "finite diff", "rel err" });
    const fd_of = [_]usize{ 0, 2, 3 };
    for (fd_of) |i| {
        const analytic = first.grads[i].item();
        const numeric = try finiteDiff(&scene, &params, i, 1e-4);
        const rel = @abs(analytic - numeric) / @max(1e-12, @abs(analytic));
        try out.print("{s:<8} {d:>14.8} {d:>14.8} {e:>9.2}\n", .{ names[i], analytic, numeric, rel });
        try lt(out, "the sweep agrees with finite differences", rel, 1e-6);
    }

    // Which way does the gradient point? Downhill, and the truth is downhill
    // from the start.
    const cx_grad = first.grads[0].item();
    const cy_grad = first.grads[1].item();
    const r_grad = first.grads[2].item();
    try lt(out, "the centre moves towards the target (x)", cx_grad * (truth.cx - start.cx), 0);
    try lt(out, "the centre moves towards the target (y)", cy_grad * (truth.cy - start.cy), 0);
    try lt(out, "the radius grows towards the target", r_grad * (truth.r - start.r), 0);

    // The update rule the fit itself uses, at a small learning rate: a gradient
    // is only worth having if stepping along it goes downhill. (A *large* first
    // Adam step can overshoot a colour channel, which is exactly why the fit
    // anneals the blur and starts with a modest rate.)
    var probe_opt = zpu.optim.Adam.init(alloc, .{ .lr = probe_lr });
    defer probe_opt.deinit();
    var probe: [6]Tensor = undefined;
    for (params, 0..) |p, k| probe[k] = try p.clone();
    try probe_opt.step(&probe, first.grads);
    const after = try loss(&scene, &probe);
    try out.print("\none Adam step at lr {d}: loss {d:.4} -> {d:.4}\n", .{ probe_lr, first_loss, after.item() });
    try lt(out, "an optimizer step along the gradient lowers the loss", after.item(), first_loss);

    // -------------------------------------------------------------- the fit
    // Coarse to fine: a wide soft edge lets a pixel see the target from far
    // away, a narrow one gives the sharp picture. Annealing the blur is what
    // makes the *position* of a shape visible to gradient descent at all —
    // with a 1 px edge and a disc 30 px away, `d loss / d centre` is zero for
    // any practical purpose.
    var adam = zpu.optim.Adam.init(alloc, .{ .lr = lr });
    defer adam.deinit();

    const t0 = Io.Clock.awake.now(io).nanoseconds;
    var total_steps: usize = 0;
    var last_loss = first_loss;
    try out.print("\n{s:>5} {s:>6} {s:>12} {s:>12}\n", .{ "step", "blur", "loss", "per pixel" });
    for (blur_stages, steps_per_stage) |stage_blur, stage_steps| {
        scene.blur = stage_blur;
        var step: usize = 0;
        while (step < stage_steps) : (step += 1) {
            var r = try zpu.valueAndGrad(alloc, loss, &scene, &params, &wrt);
            defer r.deinit();
            last_loss = r.value.item();
            if (step % 40 == 0 or step + 1 == stage_steps) {
                try out.print("{d:>5} {d:>6.2} {d:>12.4} {d:>12.6}\n", .{ total_steps, stage_blur, last_loss, last_loss / (W * H * 3) });
            }
            try adam.step(&params, r.grads);
            total_steps += 1;
        }
    }
    const elapsed_ns = Io.Clock.awake.now(io).nanoseconds - t0;
    const rate = @as(f64, @floatFromInt(total_steps)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    // --------------------------------------------------------------- the look
    const fitted = try scene.render(&params);
    var fitted_img = try Image.fromTensor(alloc, fitted);
    defer fitted_img.deinit();
    try save(io, fitted_img, "zig-out/diff-render-fit.ppm");

    var start_params: [6]Tensor = undefined;
    for (0..6) |k| start_params[k] = try Tensor.fromScalarF64(poseParam(start, k), .f64);
    const initial = try scene.render(&start_params);

    const residual = try zpu.sub(fitted, target);
    const absres = try zpu.abs(residual);
    const amplified = try zpu.clamp(try zpu.mulScalar(absres, 8.0), 0, 1);

    // target | start | fit | 8x error, side by side, through `concat`.
    const panel_a = try zpu.concat(target, initial, 1);
    const panel_b = try zpu.concat(panel_a, fitted, 1);
    const montage = try zpu.concat(panel_b, amplified, 1);
    var montage_img = try Image.fromTensor(alloc, montage);
    defer montage_img.deinit();
    try save(io, montage_img, "zig-out/diff-render.ppm");

    // ------------------------------------------------------------- the report
    const mean_abs = try zpu.meanAll(absres);

    try out.print("\n{s:<8} {s:>10} {s:>10} {s:>10}\n", .{ "param", "start", "fitted", "drawn" });
    for (0..6) |k| {
        try out.print("{s:<8} {d:>10.4} {d:>10.4} {d:>10.4}\n", .{ names[k], poseParam(start, k), params[k].item(), poseParam(truth, k) });
    }
    try out.print("\n{s:<12} {d:.4} -> {d:.4} ({d:.0}x), {d} steps\n", .{ "loss", first_loss, last_loss, first_loss / last_loss, total_steps });
    try out.print("{s:<12} mean |error| {d:.6} ({d:.3}% of the range)\n", .{ "residual", mean_abs.item(), mean_abs.item() * 100 });
    try out.print("{s:<12} {d:.0} steps/s (gradients included)\n", .{ "throughput", rate });
    try preview(out, montage_img);

    // The checks that make this an example rather than a screenshot.
    // The floor of this fit is the drawn hard edge against the renderer's soft
    // ramp, so the ratio is what is checked, not the absolute loss.
    try check(out, "the loss falls by more than 30x", first_loss / last_loss, 30);
    // zgf fills pixel *indices*; the SDF lives in pixel *centres*.
    try close(out, "the fitted centre x (pixel centres)", params[0].item(), truth.cx + 0.5, 0.3);
    try close(out, "the fitted centre y (pixel centres)", params[1].item(), truth.cy + 0.5, 0.3);
    try close(out, "the fitted radius", params[2].item(), truth.r, 0.5);
    for (0..3) |c| try close(out, "the fitted colour", params[3 + c].item(), truth.colour[c], 0.05);
    try lt(out, "the mean pixel error is under 1%", mean_abs.item(), 0.01);
    try check(out, "the fit explains the target's area", coverage_area(fitted_img), coverage_area(target_img) * 0.92);

    try out.print("\nwrote zig-out/diff-render.ppm (target | start | fit | 8x error)\n", .{});
    try out.print("{d} checks passed\n", .{checks});
    try out.flush();
}

/// The loss with one parameter nudged, evaluated at `p ± h`: the definition of
/// `d loss / d p`, which is exactly what the sweep claims to give.
fn finiteDiff(scene: *Scene, params: []const Tensor, i: usize, h: f64) !f64 {
    const plus = try Tensor.fromScalarF64(params[i].item() + h, .f64);
    const minus = try Tensor.fromScalarF64(params[i].item() - h, .f64);
    // The other five parameters are shared with `params`, which keeps them
    // alive; the scope owns everything the loss builds.
    var probe_plus: [6]Tensor = undefined;
    var probe_minus: [6]Tensor = undefined;
    for (params, 0..) |p, j| {
        probe_plus[j] = if (j == i) plus else p;
        probe_minus[j] = if (j == i) minus else p;
    }
    const a = try loss(scene, &probe_plus);
    const b = try loss(scene, &probe_minus);
    return (a.item() - b.item()) / (2 * h);
}

fn save(io: Io, img: Image, path: []const u8) !void {
    try Io.Dir.cwd().createDirPath(io, "zig-out");
    try img.savePpm(io, path);
}

/// Copies an 8-bit image into a `[H, W, 3] f64` tensor — the reverse of
/// `Image.fromTensor`, so both directions of the zgf/zpu handover are used.
fn imageToTensor(img: Image) !Tensor {
    const t = try Tensor.zeros(Shape.initComptime(&.{ H, W, 3 }), .f64);
    for (0..H) |y| {
        for (0..W) |x| {
            const p = img.getPixel(x, y);
            const base = (y * W + x) * 3;
            t.writeF64(base + 0, @as(f64, p.r));
            t.writeF64(base + 1, @as(f64, p.g));
            t.writeF64(base + 2, @as(f64, p.b));
        }
    }
    return t;
}

fn channel(c: Color, i: usize) f64 {
    return switch (i) {
        0 => @as(f64, c.r),
        1 => @as(f64, c.g),
        else => @as(f64, c.b),
    };
}

/// How many pixels the disc covers, read off the *image* rather than the
/// parameters: the target's hard edge and the fit's soft edge have to agree on
/// this even though their profiles differ.
fn coverage_area(img: Image) f64 {
    var n: usize = 0;
    // The disc's green channel is ~0.93 where the sky is at most 0.45, so a
    // plain threshold finds the silhouette in either image.
    for (0..H) |y| {
        for (0..W) |x| {
            if (img.getPixel(x, y).g > 0.6) n += 1;
        }
    }
    return @floatFromInt(n);
}

fn preview(out: *Io.Writer, img: Image) !void {
    const cols = 72;
    const rows = 18;
    const ramp = " .:-=+*#%@";
    try out.print("\n", .{});
    for (0..rows) |r| {
        try out.print("  ", .{});
        for (0..cols) |c| {
            const x = c * img.width() / cols;
            const y = r * img.height() / rows;
            const p = img.getPixel(x, y);
            const v = 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;
            const idx: usize = @intFromFloat(@round(std.math.clamp(v, 0, 1) * @as(f32, ramp.len - 1)));
            try out.print("{c}", .{ramp[idx]});
        }
        try out.print("\n", .{});
    }
}

/// `value` has to clear `floor`.
fn check(out: *Io.Writer, label: []const u8, value: f64, floor: f64) !void {
    if (!(value > floor)) {
        try out.print("  FAIL {s}: {d} is not above {d}\n", .{ label, value, floor });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn lt(out: *Io.Writer, label: []const u8, value: f64, ceiling: f64) !void {
    if (!(value < ceiling)) {
        try out.print("  FAIL {s}: {d} is not below {d}\n", .{ label, value, ceiling });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn close(out: *Io.Writer, label: []const u8, got: f64, want: f64, tolerance: f64) !void {
    if (!(@abs(got - want) <= tolerance)) {
        try out.print("  FAIL {s}: {d} is not within {d} of {d}\n", .{ label, got, tolerance, want });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}
