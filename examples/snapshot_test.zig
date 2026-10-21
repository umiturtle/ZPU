//! ZAPP Snapshot Test
//!
//! スナップショットテストを生成して、視覚的な回帰を検出する。
//! フォント、ウィジェット、レイアウトのレンダリング結果を画像として保存。

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");

const Dir = std.Io.Dir;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    std.debug.print("\n=== ZAPP Snapshot Test ===\n\n", .{});

    Dir.cwd().createDirPath(io, "snapshots/zapp/snapshot_test") catch {};

    // テスト1: フォントサイズ比較
    try testFontSizes(io, alloc);

    // テスト2: 全ウィジェット表示
    try testAllWidgets(io, alloc);

    // テスト3: レイアウトテスト
    try testLayout(io, alloc);

    std.debug.print("\n=== Snapshot Test Complete ===\n", .{});
}

fn testFontSizes(io: std.Io, alloc: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(alloc, 400, 200, 3);
    defer fb.deinit();

    // 背景をクリア
    fb.fill(.{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 });

    // 異なるフォントサイズでテキストを描画
    const sizes = [_]zapp.widgets.font.FontSize{ .small, .medium, .large, .xlarge };
    const labels = [_][]const u8{ "Small (8px)", "Medium (14px)", "Large (18px)", "XLarge (24px)" };
    const y_positions = [_]f32{ 20, 60, 110, 160 };

    for (sizes, labels, y_positions) |size, label, y| {
        zapp.widgets.font.renderText(&fb, label, 20, y, .{
            .size = size,
            .color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
        });
    }

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = "snapshots/zapp/snapshot_test/font_sizes.ppm";
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} (400x200)\n", .{path});
}

fn testAllWidgets(io: std.Io, alloc: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(alloc, 400, 300, 3);
    defer fb.deinit();

    // 背景
    fb.fill(.{ .r = 0.12, .g = 0.12, .b = 0.16, .a = 1.0 });

    // Label
    zapp.widgets.label.render(&fb, .{
        .text = "Label Widget",
        .rect = .{ .x = 20, .y = 20, .w = 100, .h = 20 },
        .color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
    });

    // Button (normal)
    zapp.widgets.button.render(&fb, .{
        .text = "Button",
        .rect = .{ .x = 20, .y = 60, .w = 100, .h = 30 },
        .state = .normal,
    });

    // Button (hover)
    zapp.widgets.button.render(&fb, .{
        .text = "Hover",
        .rect = .{ .x = 140, .y = 60, .w = 100, .h = 30 },
        .state = .hover,
    });

    // Checkbox (checked)
    zapp.widgets.checkbox.render(&fb, .{
        .checked = true,
        .rect = .{ .x = 20, .y = 110, .w = 20, .h = 20 },
    });

    // Checkbox (unchecked)
    zapp.widgets.checkbox.render(&fb, .{
        .checked = false,
        .rect = .{ .x = 60, .y = 110, .w = 20, .h = 20 },
    });

    // Slider
    zapp.widgets.slider.render(&fb, .{
        .value = 0.6,
        .rect = .{ .x = 20, .y = 150, .w = 200, .h = 10 },
    });

    // ProgressBar
    zapp.widgets.progress_bar.render(&fb, .{
        .value = 0.75,
        .rect = .{ .x = 20, .y = 180, .w = 200, .h = 15 },
    });

    // TextInput
    zapp.widgets.text_input.render(&fb, .{
        .text = "Hello",
        .rect = .{ .x = 20, .y = 210, .w = 150, .h = 30 },
    });

    // Separator
    zapp.widgets.separator.render(&fb, .{
        .rect = .{ .x = 20, .y = 260, .w = 360, .h = 10 },
    });

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = "snapshots/zapp/snapshot_test/all_widgets.ppm";
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} (400x300)\n", .{path});
}

fn testLayout(io: std.Io, alloc: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(alloc, 300, 300, 3);
    defer fb.deinit();

    // 背景
    fb.fill(.{ .r = 0.1, .g = 0.1, .b = 0.15, .a = 1.0 });

    // 矩形の描画（レイアウトテスト）
    zapp.widgets.drawing.fillRect(&fb, .{ .x = 20, .y = 20, .w = 100, .h = 50 }, .{
        .r = 0.3,
        .g = 0.6,
        .b = 0.9,
        .a = 1.0,
    });

    zapp.widgets.drawing.fillRect(&fb, .{ .x = 140, .y = 20, .w = 100, .h = 50 }, .{
        .r = 0.9,
        .g = 0.3,
        .b = 0.3,
        .a = 1.0,
    });

    zapp.widgets.drawing.fillRect(&fb, .{ .x = 20, .y = 90, .w = 220, .h = 30 }, .{
        .r = 0.3,
        .g = 0.9,
        .b = 0.3,
        .a = 1.0,
    });

    // 枠線テスト
    zapp.widgets.drawing.drawRectBorder(&fb, .{ .x = 20, .y = 140, .w = 260, .h = 80 }, .{
        .r = 0.8,
        .g = 0.8,
        .b = 0.8,
        .a = 1.0,
    }, 2.0);

    // セパレーター
    zapp.widgets.separator.render(&fb, .{
        .rect = .{ .x = 20, .y = 240, .w = 260, .h = 10 },
        .orientation = .horizontal,
    });

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = "snapshots/zapp/snapshot_test/layout.ppm";
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} (300x300)\n", .{path});
}
