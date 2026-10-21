//! Adaptive Rendering Demo
//!
//! 適応的レンダリングとジェスチャー入力の統合デモ
//! ズーム・パン操作に応じて品質が動的に変化します

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");
const rich = zapp.rich_render;
const adaptive = zapp.adaptive_render;
const gesture = zapp.gesture;
const Dir = std.Io.Dir;
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("\n=== Adaptive Rendering Demo ===\n\n", .{});

    Dir.cwd().createDirPath(io, "snapshots/zapp/adaptive") catch {};

    // シナリオ1: 通常の表示
    try demoNormalView(io, alloc);

    // シナリオ2: 2倍ズーム
    try demoZoomedView(io, alloc, 2.0);

    // シナリオ3: 4倍ズーム（最大品質）
    try demoZoomedView(io, alloc, 4.0);

    // シナリオ4: 0.5倍ズーム（最低品質）
    try demoZoomedView(io, alloc, 0.5);

    // シナリオ5: 小さな画面
    try demoSmallScreen(io, alloc);

    // シナリオ6: 大きな画面
    try demoLargeScreen(io, alloc);

    std.debug.print("\n=== Demo Complete ===\n", .{});
    std.debug.print("Generated 6 snapshots showing adaptive quality\n", .{});
}

fn demoNormalView(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Scenario 1: Normal View (1x zoom, 800x600)\n", .{});

    const viewport = adaptive.Viewport{
        .width = 800,
        .height = 600,
        .scale_factor = 1.0,
        .zoom = 1.0,
    };

    try renderScene(io, alloc, viewport, "normal");
}

fn demoZoomedView(io: Io, alloc: std.mem.Allocator, zoom: f32) !void {
    var buf: [32]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "zoom_{d}x", .{zoom});

    std.debug.print("Scenario: Zoomed View ({d}x zoom)\n", .{zoom});

    const viewport = adaptive.Viewport{
        .width = 800,
        .height = 600,
        .scale_factor = 1.0,
        .zoom = zoom,
    };

    try renderScene(io, alloc, viewport, name);
}

fn demoSmallScreen(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Scenario 5: Small Screen (400x300)\n", .{});

    const viewport = adaptive.Viewport{
        .width = 400,
        .height = 300,
        .scale_factor = 1.0,
        .zoom = 1.0,
    };

    try renderScene(io, alloc, viewport, "small_screen");
}

fn demoLargeScreen(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Scenario 6: Large Screen (2560x1440)\n", .{});

    const viewport = adaptive.Viewport{
        .width = 2560,
        .height = 1440,
        .scale_factor = 1.0,
        .zoom = 1.0,
    };

    try renderScene(io, alloc, viewport, "large_screen");
}

fn renderScene(io: Io, alloc: std.mem.Allocator, viewport: adaptive.Viewport, name: []const u8) !void {
    // 適応的コンテキストを作成
    const config = adaptive.AdaptiveConfig{
        .base_quality = .medium,
    };

    var ctx = adaptive.AdaptiveContext.init(config, viewport);

    // 品質を計算
    const quality = ctx.current_quality;
    const render_config = ctx.toRenderConfig();

    std.debug.print("  → Calculated quality: {s}\n", .{@tagName(quality)});

    // 画像を作成
    const img_width = @min(@as(u32, @intFromFloat(viewport.width)), 1920);
    const img_height = @min(@as(u32, @intFromFloat(viewport.height)), 1080);

    var fb = try zgf.Image.init(alloc, img_width, img_height, 4);
    defer fb.deinit();

    // 背景
    fb.fill(.{ .r = 0.1, .g = 0.1, .b = 0.15, .a = 1.0 });

    // 品質表示
    const font = zapp.vector_text.SimpleFont.withScale(1.5);
    var quality_buf: [64]u8 = undefined;
    const quality_text = try std.fmt.bufPrint(&quality_buf, "Quality: {s}", .{@tagName(quality)});
    font.drawText(&fb, quality_text, 20, 20, .{ .r = 0.9, .g = 0.9, .b = 0.9 });

    var zoom_buf: [64]u8 = undefined;
    const zoom_text = try std.fmt.bufPrint(&zoom_buf, "Zoom: {d:.1}x", .{viewport.zoom});
    font.drawText(&fb, zoom_text, 20, 40, .{ .r = 0.7, .g = 0.7, .b = 0.7 });

    var size_buf: [64]u8 = undefined;
    const size_text = try std.fmt.bufPrint(&size_buf, "Size: {}x{}", .{ img_width, img_height });
    font.drawText(&fb, size_text, 20, 60, .{ .r = 0.7, .g = 0.7, .b = 0.7 });

    // UIコンポーネントを描画
    try drawUIComponents(&fb, render_config, quality);

    // 保存
    var path_buf: [256]u8 = undefined;
    const ppm_path = try std.fmt.bufPrint(&path_buf, "snapshots/zapp/adaptive/{s}.ppm", .{name});

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const f = try Dir.createFile(Dir.cwd(), io, ppm_path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s}\n\n", .{ppm_path});
}

fn drawUIComponents(fb: *zgf.Image, config: rich.RenderConfig, quality: rich.RenderQuality) !void {
    const font = zapp.vector_text.SimpleFont.withScale(1.2);

    // カード1: シャドウ付き（品質依存）
    const card1_x: f32 = 50;
    const card1_y: f32 = 120;
    rich.drawShadowRect(
        fb,
        card1_x,
        card1_y,
        200,
        120,
        12,
        .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 },
        .{ .offset_y = 4, .blur = 8 },
        config,
    );
    font.drawText(fb, "Card with Shadow", card1_x + 15, card1_y + 15, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    font.drawText(fb, "(Quality dependent)", card1_x + 15, card1_y + 35, .{ .r = 0.6, .g = 0.6, .b = 0.6 });

    // カード2: グラデーション（品質依存）
    const card2_x: f32 = 280;
    const card2_y: f32 = 120;
    rich.drawGradientRect(
        fb,
        card2_x,
        card2_y,
        200,
        120,
        .{
            .from = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
            .to = .{ .r = 0.2, .g = 0.4, .b = 0.7, .a = 1.0 },
            .direction = .vertical,
        },
        config,
    );
    font.drawText(fb, "Gradient Fill", card2_x + 15, card2_y + 15, .{ .r = 1.0, .g = 1.0, .b = 1.0 });
    font.drawText(fb, "(Steps: " ++ ")", card2_x + 15, card2_y + 35, .{ .r = 0.9, .g = 0.9, .b = 0.9 });

    // ボタン群
    const btn_y: f32 = 280;

    // Primaryボタン
    rich.drawShadowRect(
        fb,
        50,
        btn_y,
        120,
        45,
        8,
        .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
        .{ .offset_y = 2, .blur = 4 },
        config,
    );
    font.drawText(fb, "Primary", 80, btn_y + 15, .{ .r = 1.0, .g = 1.0, .b = 1.0 });

    // Secondaryボタン
    rich.drawRoundedRect(
        fb,
        200,
        btn_y,
        120,
        45,
        8,
        .{ .r = 0.3, .g = 0.3, .b = 0.35, .a = 1.0 },
        config,
    );
    font.drawText(fb, "Secondary", 225, btn_y + 15, .{ .r = 0.9, .g = 0.9, .b = 0.9 });

    // Progress bar
    const progress_y: f32 = 360;
    rich.drawRoundedRect(fb, 50, progress_y, 400, 20, 10, .{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 }, config);
    rich.drawGradientRect(
        fb,
        50,
        progress_y,
        300,
        20,
        .{
            .from = .{ .r = 0.3, .g = 0.8, .b = 0.4, .a = 1.0 },
            .to = .{ .r = 0.2, .g = 0.7, .b = 0.3, .a = 1.0 },
            .direction = .horizontal,
        },
        config,
    );
    font.drawText(fb, "Progress: 75%", 50, progress_y + 30, .{ .r = 0.9, .g = 0.9, .b = 0.9 });

    // 品質レベルの視覚的表示
    const indicator_y: f32 = 440;
    font.drawText(fb, "Quality Indicators:", 50, indicator_y, .{ .r = 0.9, .g = 0.9, .b = 0.9 });

    // セグメント数を表示（角の丸みに影響）
    const segments: u32 = switch (quality) {
        .low => 0,
        .medium => 4,
        .high => 8,
        .ultra => 16,
    };

    var seg_buf: [32]u8 = undefined;
    const seg_text = try std.fmt.bufPrint(&seg_buf, "Corner segments: {d}", .{segments});
    font.drawText(fb, seg_text, 70, indicator_y + 25, .{ .r = 0.7, .g = 0.7, .b = 0.7 });

    // AA表示
    const aa_text = if (config.antialiasing) "AA: Enabled" else "AA: Disabled";
    font.drawText(fb, aa_text, 70, indicator_y + 45, .{ .r = 0.7, .g = 0.7, .b = 0.7 });

    // シャドウサンプル数
    const shadow_samples: u32 = switch (quality) {
        .low => 0,
        .medium => 2,
        .high => 4,
        .ultra => 8,
    };

    var shadow_buf: [32]u8 = undefined;
    const shadow_text = try std.fmt.bufPrint(&shadow_buf, "Shadow samples: {d}", .{shadow_samples});
    font.drawText(fb, shadow_text, 70, indicator_y + 65, .{ .r = 0.7, .g = 0.7, .b = 0.7 });
}
