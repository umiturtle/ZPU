//! Snapshot tests for ZUI rendering
//!
//! These tests render UI elements and compare against reference images to detect
//! regressions in the rendering pipeline.

const std = @import("std");
const zui = @import("../zui.zig");
const zgf = @import("../zgf.zig");
const zpu = @import("../zpu.zig");

const Ui = zui.Ui;
const Input = zui.Input;
const Color = zui.Color;

const alloc = std.testing.allocator;

/// Render a simple UI and verify the output
fn renderUi(ui: *Ui, input: Input) !zgf.Image {
    const width = 320;
    const height = 240;

    // Run the UI frame
    const result = try ui.run(input, struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.container("root", .{
                .w = .full,
                .h = .full,
                .pad = .all(10),
                .gap = 8,
                .bg = u.theme.bg,
            }, struct {
                pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                    try u.label("Snapshot Test", .{ .color = u.theme.text });
                    _ = try u.button("Click me", .{});
                    _ = try u.slider("Value", &@as(f32, 0.5), .{ .min = 0, .max = 1 });
                }
            }{});
        }
    }{});

    // Render to image
    var img = try zgf.Image.init(alloc, width, height, 4);
    errdefer img.deinit();

    const display_list = result.display_list;
    try zui.render.renderToImage(&img, display_list, .{});

    return img;
}

test "snapshot: basic UI rendering" {
    var ui = try Ui.init(alloc, .{});
    defer ui.deinit();

    const input = Input{};

    var img = try renderUi(&ui, input);
    defer img.deinit();

    // Verify the image was rendered (not all black)
    var has_content = false;
    for (0..img.height()) |y| {
        for (0..img.width()) |x| {
            const r = img.pixels.readF64(img.index(x, y, 0));
            const g = img.pixels.readF64(img.index(x, y, 1));
            const b = img.pixels.readF64(img.index(x, y, 2));
            if (r > 0.05 or g > 0.05 or b > 0.05) {
                has_content = true;
                break;
            }
        }
        if (has_content) break;
    }

    if (!has_content) {
        return error.NoContentRendered;
    }
}

test "snapshot: button rendering" {
    var ui = try Ui.init(alloc, .{});
    defer ui.deinit();

    const width = 200;
    const height = 100;

    // Render a button
    const input = Input{};
    const result = try ui.run(input, struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.container("root", .{
                .w = .full,
                .h = .full,
                .bg = u.theme.bg,
            }, struct {
                pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                    _ = try u.button("Test Button", .{ .w = .{ .px = 120 } });
                }
            }{});
        }
    }{});

    // Render to image
    var img = try zgf.Image.init(alloc, width, height, 4);
    defer img.deinit();

    try zui.render.renderToImage(&img, result.display_list, .{});

    // Verify button was rendered (should have some non-background pixels)
    var button_pixels: usize = 0;
    for (0..height) |y| {
        for (0..width) |x| {
            const r = img.pixels.readF64(img.index(x, y, 0));
            if (r > 0.2) {
                button_pixels += 1;
            }
        }
    }

    if (button_pixels < 100) {
        return error.ButtonNotRendered;
    }
}

test "snapshot: text rendering" {
    var ui = try Ui.init(alloc, .{});
    defer ui.deinit();

    const width = 200;
    const height = 100;

    const input = Input{};
    const result = try ui.run(input, struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.container("root", .{
                .w = .full,
                .h = .full,
                .bg = Color{ .r = 0, .g = 0, .b = 0, .a = 1 },
            }, struct {
                pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                    try u.label("Hello, World!", .{ .color = Color{ .r = 1, .g = 1, .b = 1, .a = 1 } });
                }
            }{});
        }
    }{});

    // Render to image
    var img = try zgf.Image.init(alloc, width, height, 4);
    defer img.deinit();

    try zui.render.renderToImage(&img, result.display_list, .{});

    // Verify text was rendered (should have white pixels on black background)
    var text_pixels: usize = 0;
    for (0..height) |y| {
        for (0..width) |x| {
            const r = img.pixels.readF64(img.index(x, y, 0));
            const g = img.pixels.readF64(img.index(x, y, 1));
            const b = img.pixels.readF64(img.index(x, y, 2));
            if (r > 0.5 and g > 0.5 and b > 0.5) {
                text_pixels += 1;
            }
        }
    }

    if (text_pixels < 10) {
        return error.TextNotRendered;
    }
}

test "snapshot: layout consistency" {
    var ui = try Ui.init(alloc, .{});
    defer ui.deinit();

    const width = 300;
    const height = 200;

    // Render twice with same input - should produce identical output
    const input = Input{};

    var img1 = try renderUi(&ui, input);
    defer img1.deinit();

    var img2 = try renderUi(&ui, input);
    defer img2.deinit();

    // Compare pixel by pixel
    var diff_pixels: usize = 0;
    for (0..height) |y| {
        for (0..width) |x| {
            for (0..4) |c| {
                const v1 = img1.pixels.readF64(img1.index(x, y, c));
                const v2 = img2.pixels.readF64(img2.index(x, y, c));
                if (@abs(v1 - v2) > 0.001) {
                    diff_pixels += 1;
                }
            }
        }
    }

    if (diff_pixels > 0) {
        std.debug.print("Layout not deterministic: {d} pixels differ\n", .{diff_pixels});
        return error.LayoutNotDeterministic;
    }
}

test "snapshot: hover state" {
    var ui = try Ui.init(alloc, .{});
    defer ui.deinit();

    const width = 200;
    const height = 100;

    // Render button without hover
    const input1 = Input{ .pointer = .{ .x = 1000, .y = 1000 } }; // Far away
    const result1 = try ui.run(input1, struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.container("root", .{ .w = .full, .h = .full, .bg = u.theme.bg }, struct {
                pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                    _ = try u.button("Hover Test", .{});
                }
            }{});
        }
    }{});

    var img1 = try zgf.Image.init(alloc, width, height, 4);
    defer img1.deinit();
    try zui.render.renderToImage(&img1, result1.display_list, .{});

    // Render button with hover
    const input2 = Input{ .pointer = .{ .x = 50, .y = 50 } }; // Over button
    const result2 = try ui.run(input2, struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.container("root", .{ .w = .full, .h = .full, .bg = u.theme.bg }, struct {
                pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                    _ = try u.button("Hover Test", .{});
                }
            }{});
        }
    }{});

    var img2 = try zgf.Image.init(alloc, width, height, 4);
    defer img2.deinit();
    try zui.render.renderToImage(&img2, result2.display_list, .{});

    // Images should be different (hover changes appearance)
    var diff_pixels: usize = 0;
    for (0..height) |y| {
        for (0..width) |x| {
            const r1 = img1.pixels.readF64(img1.index(x, y, 0));
            const r2 = img2.pixels.readF64(img2.index(x, y, 0));
            if (@abs(r1 - r2) > 0.01) {
                diff_pixels += 1;
            }
        }
    }

    if (diff_pixels == 0) {
        return error.HoverNotDetected;
    }
}
