//! ZAPP Rich UI Demo
//!
//! リッチレンダリング機能を使ったデモンストレーション
//! コンパイル時の品質設定で見た目が変化します

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");
const rich = zapp.rich_render;
const Dir = std.Io.Dir;
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    
    // 4つの品質レベルでレンダリング
    const qualities = [_]rich.RenderQuality{ .low, .medium, .high, .ultra };
    const names = [_][]const u8{ "low", "medium", "high", "ultra" };
    
    for (qualities, names) |quality, name| {
        try renderWithQuality(io, alloc, quality, name);
    }
    
    std.debug.print("\n=== Rich UI Demo Complete ===\n", .{});
    std.debug.print("Generated 4 snapshots with different quality levels\n", .{});
}

fn renderWithQuality(io: Io, alloc: std.mem.Allocator, quality: rich.RenderQuality, name: []const u8) !void {
    var fb = try zgf.Image.init(alloc, 800, 600, 4);
    defer fb.deinit();
    
    const config = rich.RenderConfig{
        .global_quality = quality,
        .shadows = true,
        .gradients = true,
        .antialiasing = quality == .high or quality == .ultra,
    };
    
    // 背景
    fb.fill(.{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 });
    
    // タイトル
    const font = zapp.vector_text.SimpleFont.withScale(2.0);
    font.drawText(&fb, "Rich UI - ", 20, 20, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    font.drawText(&fb, name, 200, 20, .{ .r = 0.3, .g = 0.8, .b = 1.0 });
    
    // カード1: シャドウ付き
    const card1_x: f32 = 30;
    const card1_y: f32 = 80;
    rich.drawShadowRect(
        &fb,
        card1_x,
        card1_y,
        230,
        150,
        12,
        .{ .r = 0.25, .g = 0.25, .b = 0.3, .a = 1.0 },
        .{ .offset_y = 4, .blur = 8 },
        config,
    );
    font.drawText(&fb, "Card with Shadow", card1_x + 20, card1_y + 20, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // カード2: グラデーション
    const card2_x: f32 = 285;
    const card2_y: f32 = 80;
    rich.drawGradientRect(
        &fb,
        card2_x,
        card2_y,
        230,
        150,
        .{
            .from = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
            .to = .{ .r = 0.2, .g = 0.4, .b = 0.7, .a = 1.0 },
            .direction = .vertical,
        },
        config,
    );
    font.drawText(&fb, "Gradient Card", card2_x + 20, card2_y + 20, .{ .r = 1.0, .g = 1.0, .b = 1.0 });
    
    // カード3: ボーダー付き
    const card3_x: f32 = 540;
    const card3_y: f32 = 80;
    rich.drawBorderedRect(
        &fb,
        card3_x,
        card3_y,
        230,
        150,
        12,
        .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 },
        .{ .width = 2, .color = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 } },
        config,
    );
    font.drawText(&fb, "Bordered Card", card3_x + 20, card3_y + 20, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // ボタン群
    const btn_y: f32 = 270;
    
    // Primaryボタン
    rich.drawShadowRect(
        &fb,
        30,
        btn_y,
        140,
        50,
        8,
        .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
        .{ .offset_y = 2, .blur = 4 },
        config,
    );
    font.drawText(&fb, "Primary", 60, btn_y + 15, .{ .r = 1.0, .g = 1.0, .b = 1.0 });
    
    // Secondaryボタン
    rich.drawRoundedRect(
        &fb,
        190,
        btn_y,
        140,
        50,
        8,
        .{ .r = 0.4, .g = 0.4, .b = 0.45, .a = 1.0 },
        config,
    );
    font.drawText(&fb, "Secondary", 215, btn_y + 15, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // Dangerボタン
    rich.drawGradientRect(
        &fb,
        350,
        btn_y,
        140,
        50,
        .{
            .from = .{ .r = 0.9, .g = 0.3, .b = 0.3, .a = 1.0 },
            .to = .{ .r = 0.7, .g = 0.2, .b = 0.2, .a = 1.0 },
            .direction = .vertical,
        },
        config,
    );
    font.drawText(&fb, "Danger", 385, btn_y + 15, .{ .r = 1.0, .g = 1.0, .b = 1.0 });
    
    // Outlinedボタン
    rich.drawBorderedRect(
        &fb,
        510,
        btn_y,
        140,
        50,
        8,
        .{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 },
        .{ .width = 2, .color = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 } },
        config,
    );
    font.drawText(&fb, "Outlined", 540, btn_y + 15, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // プログレスバー
    const progress_y: f32 = 360;
    rich.drawRoundedRect(&fb, 30, progress_y, 740, 20, 10, .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 }, config);
    rich.drawGradientRect(
        &fb,
        30,
        progress_y,
        555,
        20,
        .{
            .from = .{ .r = 0.3, .g = 0.8, .b = 0.4, .a = 1.0 },
            .to = .{ .r = 0.2, .g = 0.7, .b = 0.3, .a = 1.0 },
            .direction = .horizontal,
        },
        config,
    );
    font.drawText(&fb, "Progress: 75%", 30, progress_y + 30, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // スライダー
    const slider_y: f32 = 430;
    rich.drawRoundedRect(&fb, 30, slider_y + 8, 740, 8, 4, .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 }, config);
    rich.drawGradientRect(
        &fb,
        30,
        slider_y + 8,
        518,
        8,
        .{
            .from = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
            .to = .{ .r = 0.2, .g = 0.5, .b = 0.8, .a = 1.0 },
            .direction = .horizontal,
        },
        config,
    );
    // スライダーハンドル
    zapp.vector.fillCircleOptimized(&fb, 548, slider_y + 12, 12, .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 });
    font.drawText(&fb, "Volume: 70%", 30, slider_y - 15, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // チェックボックス
    const check_y: f32 = 500;
    rich.drawRoundedRect(&fb, 30, check_y, 24, 24, 4, .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 }, config);
    zapp.vector.drawLineFast(&fb, 35, check_y + 12, 42, check_y + 18, .{ .r = 1.0, .g = 1.0, .b = 1.0 });
    zapp.vector.drawLineFast(&fb, 42, check_y + 18, 50, check_y + 8, .{ .r = 1.0, .g = 1.0, .b = 1.0 });
    font.drawText(&fb, "Enable feature", 65, check_y + 3, .{ .r = 0.9, .g = 0.9, .b = 0.9 });
    
    // 品質設定の表示
    const quality_text = switch (quality) {
        .low => "Quality: LOW (fastest, minimal)",
        .medium => "Quality: MEDIUM (balanced)",
        .high => "Quality: HIGH (smooth edges)",
        .ultra => "Quality: ULTRA (maximum detail)",
    };
    font.drawText(&fb, quality_text, 30, 560, .{ .r = 0.6, .g = 0.6, .b = 0.6 });
    
    // 保存
    var path_buf: [256]u8 = undefined;
    const ppm_path = try std.fmt.bufPrint(&path_buf, "snapshots/zapp/rich_ui_{s}.ppm", .{name});
    
    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);
    
    const f = try Dir.createFile(Dir.cwd(), io, ppm_path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);
    
    std.debug.print("✓ {s} ({}x{})\n", .{ ppm_path, fb.width(), fb.height() });
}
