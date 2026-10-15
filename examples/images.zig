//! `zig build run-images`
//!
//! ZGF's image layer: an image is a `zpu.Tensor` of shape `[H, W, C]`, so
//! drawing writes into it and filters are ordinary tensor expressions.
//!
//! The example paints a small scene, runs every filter, works out a greyscale
//! conversion *with `matmul`* (the numeric core doing graphics work), round
//! trips the image through the 8-bit PPM codec and through a file on disk, and
//! prints an ASCII preview. Each step is checked, so a wrong filter or a broken
//! codec fails the run.
const std = @import("std");
const zpu = @import("zpu");
const zgf = @import("zgf");

const Io = std.Io;
const Image = zgf.Image;
const Color = zgf.Color;
const Tensor = zpu.Tensor;

const width = 96;
const height = 96;

/// Counts the checks that passed.
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

    try out.print("zgf — images, filters and codecs\n\n", .{});

    var img = try paint(alloc);
    defer img.deinit();
    try out.print("{s:<14} {d}x{d}, {d} channels, {d} floats\n", .{ "scene", width, height, img.channels(), img.numel() });
    try preview(out, img);

    try filters(out, img);
    try greyscale(out, img);
    try codec(out, alloc, io, img);

    try out.print("\n{d} checks passed\n", .{checks});
    try out.flush();
}

/// Draws the scene the rest of the example analyses.
fn paint(alloc: std.mem.Allocator) !Image {
    var img = try Image.init(alloc, width, height, 3);
    errdefer img.deinit();

    // background, top to bottom
    img.verticalGradient(.{ .r = 0.05, .g = 0.07, .b = 0.12 }, .{ .r = 0.35, .g = 0.30, .b = 0.45 });
    // a filled circle and a rectangle, then a thick diagonal
    img.drawCircle(width / 3, height / 3, height / 5, .{ .r = 0.95, .g = 0.85, .b = 0.25 });
    img.drawRect(width / 2, height / 2, width - 12, height - 10, .{ .r = 0.10, .g = 0.55, .b = 0.45 });
    img.drawLineThick(4, height - 6, width - 5, 6, 3, .{ .r = 0.95, .g = 0.40, .b = 0.35 });
    return img;
}

// ---------------------------------------------------------------------------
// filters
// ---------------------------------------------------------------------------

fn filters(out: *Io.Writer, img: Image) !void {
    try section(out, "filters (all of them are plain tensor ops)");

    const blurred = try img.blur(3);
    const sharpened = try img.sharpen(2, 1.5);
    const edges = try img.sobel();
    const darker = try img.brightness(0.8);
    const punchy = try img.contrast(1.4);

    try out.print("{s:<12} {s:>9} {s:>9} {s:>9}\n", .{ "variant", "mean", "stddev", "max" });
    try stats(out, "original", img.pixels);
    try stats(out, "blur r=3", blurred);
    try stats(out, "sharpen", sharpened);
    try stats(out, "sobel", edges);
    try stats(out, "0.8x", darker);
    try stats(out, "contrast", punchy);

    // A blur is low-pass: it removes exactly the high frequencies the standard
    // deviation measures, so it has to go down. Sharpening is the opposite.
    try lt(out, "blur lowers stddev", stddev(blurred), stddev(img.pixels));
    try gt(out, "sharpen raises stddev", stddev(sharpened), stddev(img.pixels));
    try lt(out, "0.8x lowers mean", mean(darker), mean(img.pixels));

    // Sobel is an edge detector: the inside of the rectangle is flat (so ~0),
    // while the circle's rim is a step and lights up.
    const inside = try zpu.narrow(try zpu.narrow(edges, 0, 60, 3), 1, 60, 3);
    const rim = try zpu.narrow(try zpu.narrow(edges, 0, height / 3, 1), 1, width / 3 - height / 5, 1);
    try out.print("{s:<12} sobel max inside the rectangle {d:.4}, on the rim {d:.4}\n", .{ "edges", maxOf(inside), maxOf(rim) });
    try lt(out, "sobel ~0 inside the rectangle", maxOf(inside), 0.05);
    try gt(out, "sobel large on the rim", maxOf(rim), 0.5);
}

// ---------------------------------------------------------------------------
// greyscale as a matrix product
// ---------------------------------------------------------------------------

fn greyscale(out: *Io.Writer, img: Image) !void {
    try section(out, "greyscale with `matmul`");

    // [H*W, 3] @ [3, 1] -> [H*W, 1]: the luminance weights are a matrix, and the
    // image is the other operand. This is the whole point of keeping pixels in
    // the same tensor type the numeric core uses: the library's own matmul,
    // broadcasting and dtype rules apply to images.
    const flat = try zpu.reshape(img.pixels, zpu.Shape.initComptime(&.{ height * width, 3 }));
    const weights = try Tensor.fromArray(f32, &.{ 0.299, 0.587, 0.114 }, zpu.Shape.initComptime(&.{ 3, 1 }));
    const luma = try zpu.matmul(flat, weights);
    const luma_img = try zpu.reshape(luma, zpu.Shape.initComptime(&.{ height, width, 1 }));

    // pure red is exactly the first weight
    const red = try Tensor.fromArray(f32, &.{ 1, 0, 0 }, zpu.Shape.initComptime(&.{ 1, 3 }));
    const red_luma = try zpu.matmul(red, weights);
    try near(out, "luma(red) == 0.299", red_luma.readF64(0), 0.299, 1e-6);

    // A one-channel image is built from the same bytes path as a colour one, and
    // the mono codec writes P5 rather than P6.
    const grey_bytes = try std.heap.smp_allocator.alloc(u8, height * width);
    defer std.heap.smp_allocator.free(grey_bytes);
    for (0..height * width) |i| {
        const v = luma_img.readF64(i);
        grey_bytes[i] = @intFromFloat(@round(@min(@max(v, 0), 1) * 255));
    }
    var grey = try Image.fromPixels(std.heap.smp_allocator, width, height, 1, grey_bytes);
    defer grey.deinit();
    try near(out, "grey channels", @floatFromInt(grey.channels()), 1, 0);
    try near(out, "grey pixels", @floatFromInt(grey.numel()), @floatFromInt(height * width), 0);
    try gt(out, "luma is not all black", maxOf(luma_img), 0.05);
}

// ---------------------------------------------------------------------------
// codec and file round trip
// ---------------------------------------------------------------------------

fn codec(out: *Io.Writer, alloc: std.mem.Allocator, io: std.Io, img: Image) !void {
    try section(out, "8-bit codec and disk round trip");

    const ppm = try img.toPpm(alloc);
    defer alloc.free(ppm);
    try out.print("{s:<14} {d} bytes\n", .{ "PPM", ppm.len });
    try expectPrefix(out, "P6 header", ppm, "P6\n96 96\n255\n");
    // header ("P6", size, 255) plus exactly three bytes per pixel
    const body_start = std.mem.indexOf(u8, ppm, "255\n").? + 4;
    try near(out, "PPM size", @floatFromInt(ppm.len), @floatFromInt(body_start + width * height * 3), 0);

    // toPixels/fromPixels round trip: the only loss allowed is 8-bit rounding
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    var back = try Image.fromPixels(alloc, width, height, 3, pixels);
    defer back.deinit();
    var worst: f64 = 0;
    for (0..img.numel()) |i| {
        const d = @abs(img.pixels.readF64(i) - back.pixels.readF64(i));
        if (d > worst) worst = d;
    }
    try lt(out, "quantisation error", worst, 1.0 / 255.0 + 1e-6);

    // Through the filesystem: write it, read it back, compare the bytes.
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/images-demo.ppm";
    img.savePpm(io, path) catch |err| {
        try out.print("{s:<14} could not write {s}: {t}\n", .{ "file", path, err });
        return err;
    };
    var loaded = try zgf.loadPpm(alloc, io, path);
    defer loaded.deinit();
    try near(out, "loaded size", @floatFromInt(loaded.width() * loaded.height() * loaded.channels()), @floatFromInt(width * height * 3), 0);
    const loaded_ppm = try loaded.toPpm(alloc);
    defer alloc.free(loaded_ppm);
    try expectPrefix(out, "same bytes on disk", loaded_ppm, ppm[0..ppm.len]);
    try out.print("{s:<14} wrote and re-read {s}\n", .{ "file", path });
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

/// A small ASCII impression of the image, so the scene is visible in a terminal.
fn preview(out: *Io.Writer, img: Image) !void {
    const cols = 48;
    const rows = 16;
    const ramp = " .:-=+*#%@";
    try out.print("\n{s}\n", .{"preview (luminance)"});
    for (0..rows) |r| {
        try out.print("  ", .{});
        for (0..cols) |c| {
            const y = r * height / rows;
            const x = c * width / cols;
            const v = 0.299 * img.pixels.readF64((y * width + x) * 3 + 0) +
                0.587 * img.pixels.readF64((y * width + x) * 3 + 1) +
                0.114 * img.pixels.readF64((y * width + x) * 3 + 2);
            const idx: usize = @intFromFloat(@round(@min(@max(v, 0), 1) * @as(f64, ramp.len - 1)));
            try out.print("{c}", .{ramp[idx]});
        }
        try out.print("\n", .{});
    }
}

fn stats(out: *Io.Writer, label: []const u8, t: Tensor) !void {
    try out.print("{s:<12} {d:>9.4} {d:>9.4} {d:>9.4}\n", .{ label, mean(t), stddev(t), maxOf(t) });
}

fn mean(t: Tensor) f64 {
    var s: f64 = 0;
    for (0..t.numel()) |i| s += t.readLogical(t.numel(), i);
    return s / @as(f64, @floatFromInt(t.numel()));
}

fn stddev(t: Tensor) f64 {
    const m = mean(t);
    var s: f64 = 0;
    for (0..t.numel()) |i| {
        const d = t.readLogical(t.numel(), i) - m;
        s += d * d;
    }
    return @sqrt(s / @as(f64, @floatFromInt(t.numel())));
}

fn maxOf(t: Tensor) f64 {
    var m = -std.math.inf(f64);
    for (0..t.numel()) |i| m = @max(m, t.readLogical(t.numel(), i));
    return m;
}

fn lt(out: *Io.Writer, label: []const u8, a: f64, b: f64) !void {
    if (!(a < b)) {
        try out.print("  FAIL {s}: {d} !< {d}\n", .{ label, a, b });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn gt(out: *Io.Writer, label: []const u8, a: f64, b: f64) !void {
    if (!(a > b)) {
        try out.print("  FAIL {s}: {d} !> {d}\n", .{ label, a, b });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn near(out: *Io.Writer, label: []const u8, a: f64, b: f64, tol: f64) !void {
    if (@abs(a - b) > tol) {
        try out.print("  FAIL {s}: {d} vs {d} (tol {d})\n", .{ label, a, b, tol });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn expectPrefix(out: *Io.Writer, label: []const u8, got: []const u8, want: []const u8) !void {
    if (got.len < want.len or !std.mem.eql(u8, got[0..want.len], want)) {
        try out.print("  FAIL {s}: got {d} bytes starting {s}\n", .{ label, got.len, got[0..@min(got.len, 8)] });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}
