//! `zig build run-vector-art`
//!
//! A 2-D poster drawn entirely with ZGF: primitives for the geometry, the
//! numeric core for the two effects that need arithmetic over the whole image
//! (a bloom pass built from `blur` and `add`, and the sky gradient), and a small
//! barycentric triangle filler for the mountains. Writes
//! `zig-out/vector-art.ppm` and prints a terminal preview.
const std = @import("std");
const zpu = @import("zpu");
const zgf = @import("zgf");

const Io = std.Io;
const Image = zgf.Image;
const Color = zgf.Color;

const width = 176;
const height = 110;
const horizon = 74;

var checks: usize = 0;

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

    var img = try Image.init(alloc, width, height, 3);
    defer img.deinit();

    try sky(&img);
    try stars(&img);
    const bloom = try sun(&img);
    try mountains(&img);
    try water(&img);

    // The bloom is the interesting part: `blur` is a tensor operation, so the
    // glow is `blur(sun) * 0.6` added back onto the sky, then clamped to 0..1.
    // The result comes back as an image through `Image.fromTensor`.
    const glow = try zpu.mulScalar(bloom, 0.6);
    const lit = try zpu.add(img.pixels, glow);
    const clamped = try zpu.clamp(lit, 0, 1);
    var final = try Image.fromTensor(alloc, clamped);
    defer final.deinit();

    try out.print("zgf — vector art\n\n", .{});
    try preview(out, final);
    try report(out, final);

    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/vector-art.ppm";
    try final.savePpm(io, path);
    try out.print("\nwrote {s} ({d}x{d})\n", .{ path, width, height });
    try out.print("{d} checks passed\n", .{checks});
    try out.flush();
}

fn sky(img: *Image) !void {
    // A vertical gradient from night blue to dawn orange, plus a warmer band
    // just above the horizon.
    img.verticalGradient(.{ .r = 0.03, .g = 0.05, .b = 0.16 }, .{ .r = 0.85, .g = 0.42, .b = 0.22 });
    for (0..horizon) |y| {
        const t = @as(f64, @floatFromInt(y)) / @as(f64, @floatFromInt(horizon));
        const warm = @max(0.0, (t - 0.55) / 0.45);
        for (0..width) |x| {
            const p = img.getPixel(x, y);
            img.setPixel(@intCast(x), @intCast(y), .{
                .r = p.r + @as(f32, @floatCast(warm * 0.45)),
                .g = p.g + @as(f32, @floatCast(warm * 0.18)),
                .b = p.b - @as(f32, @floatCast(warm * 0.05)),
            });
        }
    }
}

fn stars(img: *Image) !void {
    var g = zpu.rng.Rng.init(0x57A25);
    for (0..140) |_| {
        const x: isize = @intFromFloat(g.nextF64() * @as(f64, @floatFromInt(width)));
        const y: isize = @intFromFloat(g.nextF64() * @as(f64, @floatFromInt(horizon)));
        const bright = 0.35 + 0.65 * g.nextF64();
        // Fade the stars out near the bright horizon so the sky reads as a sky.
        const fade = 1.0 - @as(f64, @floatFromInt(y)) / @as(f64, @floatFromInt(horizon));
        const v: f32 = @floatCast(bright * fade * fade);
        img.setPixel(x, y, .{ .r = v, .g = v, .b = v * 1.1 });
    }
}

/// Draws the sun into a *separate* image and returns it as a tensor: that layer
/// is what gets blurred for the bloom.
fn sun(img: *Image) !zpu.Tensor {
    const cx: isize = width / 2 + 34;
    const cy: isize = horizon - 16;
    const radius: isize = 9;

    var glow = try Image.init(img.alloc, width, height, 3);
    defer glow.deinit();
    glow.fill(.{ .r = 0, .g = 0, .b = 0 });

    // disc, then a couple of rings that fade outwards
    glow.drawCircle(cx, cy, radius, .{ .r = 1.0, .g = 0.96, .b = 0.78 });
    glow.drawCircle(cx, cy, radius - 2, .{ .r = 1.0, .g = 1.0, .b = 0.9 });
    var r: isize = radius + 1;
    while (r <= radius + 4) : (r += 1) {
        const t = 1.0 - @as(f64, @floatFromInt(r - radius)) / 5.0;
        var x = cx - r;
        while (x <= cx + r) : (x += 1) {
            const dy = @sqrt(@as(f64, @floatFromInt(r * r)) - @as(f64, @floatFromInt((x - cx) * (x - cx))));
            const c = Color{ .r = @floatCast(t * 0.8), .g = @floatCast(t * 0.7), .b = @floatCast(t * 0.4) };
            glow.setPixel(x, cy - @as(isize, @intFromFloat(dy)), c);
            glow.setPixel(x, cy + @as(isize, @intFromFloat(dy)), c);
        }
    }
    // the sky behind the sun is drawn by the caller; the sun itself goes in
    img.drawCircle(cx, cy, radius, .{ .r = 1.0, .g = 0.96, .b = 0.78 });

    // the bloom layer is the glow, blurred: separable Gaussian, plain `zpu` ops
    const blurred = try glow.blur(6);
    return blurred;
}

fn mountains(img: *Image) !void {
    // Three ridges, back to front, each a filled polygon.
    const ridges = [_]struct { height: f64, color: Color }{
        .{ .height = 30, .color = .{ .r = 0.16, .g = 0.13, .b = 0.26 } },
        .{ .height = 22, .color = .{ .r = 0.12, .g = 0.10, .b = 0.20 } },
        .{ .height = 14, .color = .{ .r = 0.07, .g = 0.06, .b = 0.13 } },
    };
    const base: f64 = @floatFromInt(horizon);
    for (ridges, 0..) |ridge, i| {
        var g = zpu.rng.Rng.init(0xB000 + i * 977);
        var x: f64 = -12;
        while (x < @as(f64, @floatFromInt(width))) {
            const w = 26 + 34 * g.nextF64();
            const peak = base - ridge.height * (0.45 + 0.55 * g.nextF64());
            const left = x;
            const right = x + w;
            const mid = x + w / 2;
            // a triangle per column pair: (left, base) (mid, peak) (right, base)
            fillTriangle(img, .{ left, base }, .{ mid, peak }, .{ right, base }, ridge.color);
            x = mid;
        }
    }
    // a snow cap on the tallest peak for contrast
    img.drawCircle(width / 3 + 6, horizon - 26, 3, .{ .r = 0.82, .g = 0.86, .b = 0.95 });
}

fn water(img: *Image) !void {
    // The lower band is the sky mirrored, dimmed and rippled.
    for (horizon..height) |y| {
        const depth = @as(f64, @floatFromInt(y - horizon)) / @as(f64, @floatFromInt(height - horizon));
        const src_y = horizon - @as(usize, @intFromFloat(depth * 26.0));
        for (0..width) |x| {
            const ripple = @sin(@as(f64, @floatFromInt(x)) * 0.35 + depth * 9.0) * depth * 3.0;
            const sx = @as(isize, @intFromFloat(@as(f64, @floatFromInt(x)) + ripple));
            const p = img.getPixel(@intCast(std.math.clamp(sx, 0, @as(isize, width - 1))), @min(src_y, horizon - 1));
            const dim: f32 = @floatCast(0.55 + 0.25 * (1.0 - depth));
            img.setPixel(@intCast(x), @intCast(y), .{ .r = p.r * dim, .g = p.g * dim, .b = p.b * dim });
        }
    }
}

/// Fills a triangle with a barycentric test over its bounding box: the same
/// maths a GPU rasteriser does, minus the parallelism.
fn fillTriangle(img: *Image, a: [2]f64, b: [2]f64, c: [2]f64, color: Color) void {
    const y0: isize = @intFromFloat(@floor(@min(a[1], @min(b[1], c[1]))));
    const y1: isize = @intFromFloat(@ceil(@max(a[1], @max(b[1], c[1]))));
    const x0: isize = @intFromFloat(@floor(@min(a[0], @min(b[0], c[0]))));
    const x1: isize = @intFromFloat(@ceil(@max(a[0], @max(b[0], c[0]))));
    const area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0) return;

    var y = y0;
    while (y <= y1) : (y += 1) {
        var x = x0;
        while (x <= x1) : (x += 1) {
            const p = .{ @as(f64, @floatFromInt(x)) + 0.5, @as(f64, @floatFromInt(y)) + 0.5 };
            const w0 = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / area;
            const w1 = ((c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])) / area;
            const w2 = ((a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])) / area;
            if (w0 < 0 or w1 < 0 or w2 < 0) continue;
            img.setPixel(x, y, color);
        }
    }
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

fn preview(out: *Io.Writer, img: Image) !void {
    const cols = 64;
    const rows = 20;
    const ramp = " .:-=+*#%@";
    try out.print("preview\n", .{});
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

fn report(out: *Io.Writer, img: Image) !void {
    // Three properties of the composition, measured on the finished image:
    // the sun outshines everything, the sky brightens towards the horizon, and
    // the water is not a flat fill (its ripples show up as variance).
    const sun_luma = luma(img.getPixel(width / 2 + 34, horizon - 16));
    const sky_top = luma(img.getPixel(width / 4, 3));
    const sky_horizon = luma(img.getPixel(width / 4, horizon - 3));

    var dark: usize = 0;
    for (0..horizon) |y| {
        for (0..width) |x| {
            if (luma(img.getPixel(x, y)) < 0.10) dark += 1;
        }
    }

    var sum: f64 = 0;
    var sum2: f64 = 0;
    var n: usize = 0;
    for (horizon..height) |y| {
        for (0..width) |x| {
            const v = luma(img.getPixel(x, y));
            sum += v;
            sum2 += v * v;
            n += 1;
        }
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const variance = sum2 / @as(f64, @floatFromInt(n)) - mean * mean;

    try out.print("\nsun {d:.3}, sky top {d:.3}, sky at the horizon {d:.3}\n", .{ sun_luma, sky_top, sky_horizon });
    try out.print("dark pixels above the horizon (mountains): {d}, water variance {d:.5}\n", .{ dark, variance });
    try check(out, "the sun is the brightest thing in the frame", sun_luma, sky_horizon * 2.0);
    try check(out, "the sky brightens towards the horizon", sky_horizon, sky_top * 2.0);
    try check(out, "the mountains cover a real area", @floatFromInt(dark), 800);
    try check(out, "the water has ripples", variance, 0.0005);
}

fn luma(c: Color) f64 {
    return @as(f64, @floatCast(0.299 * c.r + 0.587 * c.g + 0.114 * c.b));
}
