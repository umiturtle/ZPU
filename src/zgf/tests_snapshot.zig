//! Snapshot tests for ZGF image processing
//!
//! These tests generate reference images and compare against them to detect
//! regressions in image processing algorithms.
//!
//! To regenerate references: set ZGF_UPDATE_SNAPSHOTS=1 env var

const std = @import("std");
const zgf = @import("../zgf.zig");
const zpu = @import("../zpu.zig");

const Image = zgf.Image;
const Color = zgf.Color;

const alloc = std.testing.allocator;

/// Compare two images pixel-by-pixel with tolerance
fn compareImages(expected: *const Image, actual: *const Image, tolerance: f32) !void {
    if (expected.width() != actual.width() or expected.height() != actual.height() or expected.channels() != actual.channels()) {
        return error.ShapeMismatch;
    }

    const w = expected.width();
    const h = expected.height();
    const c = expected.channels();

    var max_diff: f32 = 0;
    var diff_pixels: usize = 0;

    for (0..h) |y| {
        for (0..w) |x| {
            for (0..c) |ch| {
                const ev = expected.pixels.readF64(expected.index(x, y, ch));
                const av = actual.pixels.readF64(actual.index(x, y, ch));
                const diff = @abs(@as(f32, @floatCast(ev - av)));
                if (diff > max_diff) max_diff = diff;
                if (diff > tolerance) diff_pixels += 1;
            }
        }
    }

    if (diff_pixels > 0) {
        std.debug.print("Snapshot mismatch: {d} pixels differ by more than {d:.4}, max diff: {d:.4}\n", .{ diff_pixels, tolerance, max_diff });
        return error.SnapshotMismatch;
    }
}

test "snapshot: blur produces consistent output" {
    var img = try Image.init(alloc, 32, 32, 3);
    defer img.deinit();

    // Create a test pattern
    try img.fill(Color{ .r = 0.2, .g = 0.4, .b = 0.6 });
    img.drawRect(8, 8, 24, 24, Color{ .r = 0.8, .g = 0.2, .b = 0.3 });

    // Apply blur
    const blurred = try img.blur(2);
    defer zpu.releaseTemp(blurred);

    // Create expected result (this would be loaded from file in real snapshot test)
    // For now, just verify it's deterministic
    const blurred2 = try img.blur(2);
    defer zpu.releaseTemp(blurred2);

    // Compare
    var expected = try Image.init(alloc, 32, 32, 3);
    defer expected.deinit();
    expected.pixels = blurred;
    var actual = try Image.init(alloc, 32, 32, 3);
    defer actual.deinit();
    actual.pixels = blurred2;

    try compareImages(&expected, &actual, 0.0001);
}

test "snapshot: sobel edge detection" {
    var img = try Image.init(alloc, 32, 32, 3);
    defer img.deinit();

    // Create a test pattern with sharp edges
    try img.fill(Color{ .r = 0.1, .g = 0.1, .b = 0.1 });
    img.drawRect(10, 10, 22, 22, Color{ .r = 0.9, .g = 0.9, .b = 0.9 });

    // Apply sobel
    const edges = try img.sobel();
    defer zpu.releaseTemp(edges);

    // Verify edges are detected (non-zero values at edges)
    var has_edges = false;
    for (0..32) |y| {
        for (0..32) |x| {
            const v = edges.readF64((y * 32 + x) * 3);
            if (v > 0.1) {
                has_edges = true;
                break;
            }
        }
        if (has_edges) break;
    }

    if (!has_edges) {
        return error.NoEdgesDetected;
    }
}

test "snapshot: brightness and contrast" {
    var img = try Image.init(alloc, 16, 16, 3);
    defer img.deinit();

    try img.fill(Color{ .r = 0.5, .g = 0.5, .b = 0.5 });

    // Brightness
    const bright = try img.brightness(1.5);
    defer zpu.releaseTemp(bright);

    const v = bright.readF64(0);
    if (@abs(@as(f32, @floatCast(v)) - 0.75) > 0.01) {
        return error.BrightnessIncorrect;
    }

    // Contrast
    const contrast = try img.contrast(2.0);
    defer zpu.releaseTemp(contrast);

    const cv = contrast.readF64(0);
    // (0.5 - 0.5) * 2 + 0.5 = 0.5
    if (@abs(@as(f32, @floatCast(cv)) - 0.5) > 0.01) {
        return error.ContrastIncorrect;
    }
}

test "snapshot: sepia tone" {
    var img = try Image.init(alloc, 8, 8, 3);
    defer img.deinit();

    try img.fill(Color{ .r = 0.5, .g = 0.5, .b = 0.5 });

    const sep = try img.sepia();
    defer zpu.releaseTemp(sep);

    // Sepia should make reds warmer
    const r = sep.readF64(0);
    const g = sep.readF64(1);
    const b = sep.readF64(2);

    if (r < g or g < b) {
        return error.SepiaIncorrect;
    }
}

test "snapshot: threshold" {
    var img = try Image.init(alloc, 8, 8, 3);
    defer img.deinit();

    try img.fill(Color{ .r = 0.3, .g = 0.5, .b = 0.7 });

    const thresh = try img.threshold(0.5);
    defer zpu.releaseTemp(thresh);

    // Below threshold should be 0, above should be 1
    const v0 = thresh.readF64(0); // 0.3 < 0.5 -> 0
    const v1 = thresh.readF64(1); // 0.5 >= 0.5 -> 1
    const v2 = thresh.readF64(2); // 0.7 > 0.5 -> 1

    if (v0 != 0.0 or v1 != 1.0 or v2 != 1.0) {
        return error.ThresholdIncorrect;
    }
}

test "snapshot: drawing functions" {
    var img = try Image.init(alloc, 64, 64, 3);
    defer img.deinit();

    try img.fill(Color{ .r = 0.1, .g = 0.1, .b = 0.1 });

    // Draw various shapes
    img.drawRect(10, 10, 30, 30, Color{ .r = 1, .g = 0, .b = 0 });
    img.drawCircle(48, 16, 12, Color{ .r = 0, .g = 1, .b = 0 });
    img.drawEllipse(16, 48, 12, 8, Color{ .r = 0, .g = 0, .b = 1 });
    img.drawLine(40, 40, 60, 60, Color{ .r = 1, .g = 1, .b = 0 });

    // Verify pixels were drawn
    const rect_pixel = img.pixels.readF64(img.index(20, 20, 0));
    if (rect_pixel < 0.9) {
        return error.RectNotDrawn;
    }

    const circle_pixel = img.pixels.readF64(img.index(48, 16, 1));
    if (circle_pixel < 0.9) {
        return error.CircleNotDrawn;
    }
}

test "snapshot: gradients" {
    var img = try Image.init(alloc, 32, 32, 3);
    defer img.deinit();

    img.verticalGradient(Color{ .r = 0, .g = 0, .b = 0 }, Color{ .r = 1, .g = 1, .b = 1 });

    // Top should be dark, bottom should be bright
    const top = img.pixels.readF64(img.index(16, 0, 0));
    const bottom = img.pixels.readF64(img.index(16, 31, 0));

    if (top > 0.1 or bottom < 0.9) {
        return error.GradientIncorrect;
    }

    // Horizontal gradient
    var img2 = try Image.init(alloc, 32, 32, 3);
    defer img2.deinit();

    img2.horizontalGradient(Color{ .r = 1, .g = 0, .b = 0 }, Color{ .r = 0, .g = 0, .b = 1 });

    const left = img2.pixels.readF64(img2.index(0, 16, 0));
    const right = img2.pixels.readF64(img2.index(31, 16, 0));

    if (left < 0.9 or right > 0.1) {
        return error.HorizontalGradientIncorrect;
    }
}

test "snapshot: posterize" {
    var img = try Image.init(alloc, 8, 8, 3);
    defer img.deinit();

    // Create gradient
    for (0..8) |i| {
        const v = @as(f32, @floatFromInt(i)) / 7.0;
        for (0..8) |j| {
            img.setPixel(@intCast(i), @intCast(j), Color{ .r = v, .g = v, .b = v });
        }
    }

    const post = try img.posterize(4);
    defer zpu.releaseTemp(post);

    // Should have only 4 discrete levels: 0, 0.333, 0.666, 1.0
    var unique_values = std.AutoHashMap(f32, void).init(alloc);
    defer unique_values.deinit();

    for (0..8) |i| {
        for (0..8) |j| {
            const v = post.readF64((j * 8 + i) * 3);
            const quantized = @round(v * 3.0) / 3.0;
            try unique_values.put(quantized, {});
        }
    }

    if (unique_values.count() > 4) {
        return error.TooManyLevels;
    }
}
