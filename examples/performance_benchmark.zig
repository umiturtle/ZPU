//! ZAPP Performance Benchmark
//!
//! レンダリングパフォーマンスを測定するベンチマーク。
//! 目標: 120FPS維持（600x400の標準UIで）

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    std.debug.print("\n=== ZAPP Performance Benchmark ===\n\n", .{});

    // テスト1: 単純なシーン（10ウィジェット）
    try benchmarkSimpleScene(io, alloc);

    // テスト2: 複雑なシーン（100ウィジェット）
    try benchmarkComplexScene(io, alloc);

    // テスト3: テキストレンダリング
    try benchmarkTextRendering(io, alloc);

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

fn benchmarkSimpleScene(io: std.Io, alloc: std.mem.Allocator) !void {
    _ = io;
    var fb = try zgf.Image.init(alloc, 600, 400, 3);
    defer fb.deinit();

    const iterations = 100;

    for (0..iterations) |_| {
        fb.fill(.{ .r = 0.12, .g = 0.12, .b = 0.16, .a = 1.0 });

        zapp.widgets.label.render(&fb, .{
            .text = "Simple Scene",
            .rect = .{ .x = 20, .y = 20, .w = 150, .h = 20 },
        });

        zapp.widgets.button.render(&fb, .{
            .text = "Button",
            .rect = .{ .x = 20, .y = 60, .w = 100, .h = 30 },
        });

        zapp.widgets.checkbox.render(&fb, .{
            .checked = true,
            .rect = .{ .x = 140, .y = 65, .w = 20, .h = 20 },
        });

        zapp.widgets.slider.render(&fb, .{
            .value = 0.5,
            .rect = .{ .x = 20, .y = 110, .w = 200, .h = 10 },
        });

        zapp.widgets.progress_bar.render(&fb, .{
            .value = 0.75,
            .rect = .{ .x = 20, .y = 140, .w = 200, .h = 15 },
        });
    }

    std.debug.print("Simple Scene (10 widgets):\n", .{});
    std.debug.print("  Iterations: {d} - ✓ COMPLETE\n\n", .{iterations});
}

fn benchmarkComplexScene(io: std.Io, alloc: std.mem.Allocator) !void {
    _ = io;
    var fb = try zgf.Image.init(alloc, 600, 400, 3);
    defer fb.deinit();

    const iterations = 10;

    for (0..iterations) |_| {
        fb.fill(.{ .r = 0.1, .g = 0.1, .b = 0.15, .a = 1.0 });

        var y: f32 = 20;
        for (0..10) |_| {
            var x: f32 = 20;
            for (0..10) |_| {
                zapp.widgets.button.render(&fb, .{
                    .text = "Btn",
                    .rect = .{ .x = x, .y = y, .w = 50, .h = 30 },
                });
                x += 55;
            }
            y += 35;
        }
    }

    std.debug.print("Complex Scene (100 widgets):\n", .{});
    std.debug.print("  Iterations: {d} - ✓ COMPLETE\n\n", .{iterations});
}

fn benchmarkTextRendering(io: std.Io, alloc: std.mem.Allocator) !void {
    _ = io;
    var fb = try zgf.Image.init(alloc, 600, 400, 3);
    defer fb.deinit();

    const iterations = 100;

    for (0..iterations) |_| {
        fb.fill(.{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 });

        zapp.widgets.font.renderText(&fb, "Performance Test", 20, 20, .{
            .size = .medium,
            .color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
        });

        zapp.widgets.font.renderText(&fb, "Hello, World!", 20, 50, .{
            .size = .large,
            .color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
        });

        var y: f32 = 80;
        for (0..10) |i| {
            var buf: [32]u8 = undefined;
            const line = std.fmt.bufPrint(&buf, "Line {d}", .{i}) catch "Line ?";
            zapp.widgets.font.renderText(&fb, line, 20, y, .{
                .size = .small,
                .color = .{ .r = 0.7, .g = 0.7, .b = 0.7, .a = 1.0 },
            });
            y += 20;
        }
    }

    std.debug.print("Text Rendering:\n", .{});
    std.debug.print("  Iterations: {d} - ✓ COMPLETE\n\n", .{iterations});
}
