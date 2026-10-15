//! zgf snapshots: what the graphics layer draws.
//!
//! An image cannot be compared byte for byte and stay useful — a one-ulp change
//! in `sin` would rewrite every pixel. So `snapshot.zig` writes an image as
//! *size and channel count* (exact), *luminance statistics* (tolerant) and an
//! *ASCII rendition* (exact). The rendition is the part that earns its keep: a
//! circle that moves one pixel, a gradient that reverses, a blur that gets
//! sharper — all of those are visible in a diff of the picture, which a hash of
//! the bytes would never show.
const std = @import("std");
const zgf = @import("zgf");
const zpu = @import("zpu");
const snap = @import("../snapshot.zig");

const Snap = snap.Snap;
const Color = zgf.Color;

/// `zgf.Color` has named fields; a three-number shorthand keeps the tests about
/// pictures rather than struct literals.
fn rgb(r: f32, g: f32, b: f32) Color {
    return .{ .r = r, .g = g, .b = b };
}
const alloc = std.testing.allocator;

/// A small scene every drawing snapshot shares: a filled background, a
/// rectangle, a circle and a line, so a change in *any* of them shows up in one
/// place as well as in the focused test.
fn scene(w: usize, h: usize) !zgf.Image {
    var img = try zgf.Image.init(alloc, w, h, 3);
    img.fill(rgb(0.08, 0.09, 0.12));
    img.drawRect(4, 4, 24, 20, rgb(0.9, 0.3, 0.2));
    img.drawCircle(44, 18, 8, rgb(0.2, 0.7, 0.4));
    img.drawLine(4, 28, 60, 28, rgb(0.6, 0.6, 0.9));
    return img;
}

test "snapshot: drawing primitives" {
    var s = Snap.init(alloc);
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    // Every tensor a filter returns is owned by this scope, so the leak checker
    // in `std.testing.allocator` covers the graphics layer too.
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    var img = try scene(64, 32);
    defer img.deinit();
    try s.expectImage("zgf/draw/scene", &img, 64, 16);

    // Each primitive on its own, so a mismatch points at one of them.
    var rect = try zgf.Image.init(alloc, 32, 32, 3);
    defer rect.deinit();
    rect.fill(rgb(0, 0, 0));
    rect.drawRect(3, 5, 28, 26, rgb(1, 1, 1));
    try s.expectImage("zgf/draw/rect", &rect, 32, 16);

    var circle = try zgf.Image.init(alloc, 32, 32, 3);
    defer circle.deinit();
    circle.fill(rgb(0, 0, 0));
    circle.drawCircle(16, 16, 12, rgb(1, 0.5, 0));
    try s.expectImage("zgf/draw/circle", &circle, 32, 16);

    var line = try zgf.Image.init(alloc, 32, 32, 3);
    defer line.deinit();
    line.fill(rgb(0, 0, 0));
    line.drawLine(2, 30, 30, 2, rgb(1, 1, 1));
    line.drawLineThick(2, 2, 30, 2, 3, rgb(0.2, 0.8, 1));
    try s.expectImage("zgf/draw/line", &line, 32, 16);

    var grad = try zgf.Image.init(alloc, 32, 32, 3);
    defer grad.deinit();
    grad.verticalGradient(rgb(1, 0, 0), rgb(0, 0, 1));
    try s.expectImage("zgf/draw/gradient", &grad, 32, 16);
    try s.finish();
}

test "snapshot: filters" {
    var s = Snap.init(alloc);
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    // Every tensor a filter returns is owned by this scope, so the leak checker
    // in `std.testing.allocator` covers the graphics layer too.
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    var img = try scene(64, 32);
    defer img.deinit();

    // Filters return tensors, so the snapshot is written through an image view
    // of the result: same [H, W, C] data, same text.
    const blurred = try img.blur(2);
    try s.expectTensor("zgf/filter/blur", blurred, 1e-6);
    const edged = try img.sobel();
    try s.expectTensor("zgf/filter/sobel", edged, 1e-6);
    const sharp = try img.sharpen(1, 1.5);
    try s.expectTensor("zgf/filter/sharpen", sharp, 1e-6);
    const bright = try img.brightness(1.4);
    try s.expectTensor("zgf/filter/brightness", bright, 1e-6);
    const contrast = try img.contrast(1.8);
    try s.expectTensor("zgf/filter/contrast", contrast, 1e-6);
    try s.finish();
}

test "snapshot: transfer and codecs" {
    var s = Snap.init(alloc);
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    // Every tensor a filter returns is owned by this scope, so the leak checker
    // in `std.testing.allocator` covers the graphics layer too.
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    var img = try scene(8, 8);
    defer img.deinit();

    // u8 sRGB conversion: the round trip through 8-bit must lose nothing beyond
    // the quantisation, and the quantised values are the snapshot.
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    try s.expectFmt("zgf/transfer/pixels-u8-first-row", "{any}\n", .{pixels[0 .. 8 * 3]});

    var back = try zgf.Image.fromPixels(alloc, 8, 8, 3, pixels);
    defer back.deinit();
    try s.expectTensor("zgf/transfer/roundtrip", back.pixels, 1e-6);

    // The P6 encoder is text-readable enough to snapshot: a header and no
    // compression, so a change in the writer is a visible change.
    const ppm = try img.toPpm(alloc);
    defer alloc.free(ppm);
    try s.expectFmt("zgf/codec/ppm-header", "header {s}\nbytes {d}\n", .{
        ppm[0..std.mem.indexOfScalar(u8, ppm, '\n').?], ppm.len,
    });
    var decoded = try zgf.decodePpm(alloc, ppm);
    defer decoded.deinit();
    try s.expectTensor("zgf/codec/ppm-roundtrip", decoded.pixels, 1e-6);
    try s.finish();
}

test "snapshot: a convolution kernel is generated deterministically" {
    var s = Snap.init(alloc);
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    // Every tensor a filter returns is owned by this scope, so the leak checker
    // in `std.testing.allocator` covers the graphics layer too.
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // The Gaussian kernel is generated, not tabulated: its coefficients are a
    // reference like any other computed value.
    for ([_]usize{ 1, 2, 4 }) |r| {
        const k = try zgf.gaussianKernel(alloc, r);
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "zgf/kernel/gaussian-{d}", .{r});
        try s.expectTensor(name, k, 1e-9);
    }
    try s.finish();
}
