//! ZAPP Widget Drawing Utilities
//!
//! 共通の描画プリミティブを提供するモジュール。

const std = @import("std");
const zgf = @import("zgf");
const VirtualRect = @import("../coordinates.zig").VirtualRect;
const font = @import("font.zig");

/// 矩形を塗りつぶす
pub fn fillRect(fb: *zgf.Image, rect: VirtualRect, color: zgf.Color) void {
    const fw: f32 = @floatFromInt(fb.width());
    const fh: f32 = @floatFromInt(fb.height());
    
    // 座標をクランプ
    const x0: f32 = @max(rect.x, 0);
    const y0: f32 = @max(rect.y, 0);
    const x1: f32 = @min(rect.x + rect.w, fw);
    const y1: f32 = @min(rect.y + rect.h, fh);
    
    if (x0 >= x1 or y0 >= y1) return;
    
    const px0: u32 = @intFromFloat(@round(x0));
    const py0: u32 = @intFromFloat(@round(y0));
    const px1: u32 = @intFromFloat(@round(x1));
    const py1: u32 = @intFromFloat(@round(y1));
    
    var y: u32 = py0;
    while (y < py1) : (y += 1) {
        var x: u32 = px0;
        while (x < px1) : (x += 1) {
            fb.setPixel(@intCast(x), @intCast(y), color);
        }
    }
}

/// 矩形の枠線を描画
pub fn drawRectBorder(fb: *zgf.Image, rect: VirtualRect, color: zgf.Color, thickness: f32) void {
    // 上辺
    fillRect(fb, .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = thickness }, color);
    // 下辺
    fillRect(fb, .{ .x = rect.x, .y = rect.y + rect.h - thickness, .w = rect.w, .h = thickness }, color);
    // 左辺
    fillRect(fb, .{ .x = rect.x, .y = rect.y, .w = thickness, .h = rect.h }, color);
    // 右辺
    fillRect(fb, .{ .x = rect.x + rect.w - thickness, .y = rect.y, .w = thickness, .h = rect.h }, color);
}

/// 水平線（セパレーター）を描画
pub fn drawHorizontalLine(fb: *zgf.Image, x: f32, y: f32, length: f32, color: zgf.Color) void {
    fillRect(fb, .{ .x = x, .y = y, .w = length, .h = 1.0 }, color);
}

/// 垂直線を描画
pub fn drawVerticalLine(fb: *zgf.Image, x: f32, y: f32, height: f32, color: zgf.Color) void {
    fillRect(fb, .{ .x = x, .y = y, .w = 1.0, .h = height }, color);
}

/// テキストを描画
pub fn drawText(fb: *zgf.Image, text: []const u8, x: f32, y: f32, color: zgf.Color) void {
    font.renderText(fb, text, x, y, .{
        .size = .medium,
        .color = color,
    });
}

/// テキストを描画（フォント設定付き）
pub fn drawTextWithConfig(fb: *zgf.Image, text: []const u8, x: f32, y: f32, config: font.FontConfig) void {
    font.renderText(fb, text, x, y, config);
}

/// スライダーのつまみを描画
pub fn drawSliderKnob(fb: *zgf.Image, rect: VirtualRect, color: zgf.Color) void {
    // 角丸矩形（簡易）
    fillRect(fb, rect, color);
}

/// チェックマークを描画
pub fn drawCheckmark(fb: *zgf.Image, rect: VirtualRect, color: zgf.Color) void {
    const line_width: f32 = 2.0;
    const x = rect.x;
    const y = rect.y;
    const w = rect.w;
    const h = rect.h;
    
    // V字の左側
    fillRect(fb, .{
        .x = x + w * 0.2,
        .y = y + h * 0.5,
        .w = line_width,
        .h = h * 0.3,
    }, color);
    
    // V字の右側（長い）
    fillRect(fb, .{
        .x = x + w * 0.5,
        .y = y + h * 0.4,
        .w = line_width,
        .h = h * 0.5,
    }, color);
}

// テスト: fillRectが正しく動作するか
test "fillRect - basic fill" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 10, 10, 3);
    defer img.deinit();
    
    fillRect(&img, .{ .x = 2, .y = 2, .w = 4, .h = 4 }, .{ .r = 1.0, .g = 0.0, .b = 0.0 });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    // (3,3)が赤になっているはず
    const idx = (3 * 10 + 3) * 3;
    try std.testing.expectEqual(@as(u8, 255), pixels[idx]); // R
    try std.testing.expectEqual(@as(u8, 0), pixels[idx + 1]); // G
    try std.testing.expectEqual(@as(u8, 0), pixels[idx + 2]); // B
}

test "fillRect - out of bounds clamping" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 10, 10, 3);
    defer img.deinit();
    
    // 範囲外の矩形
    fillRect(&img, .{ .x = -5, .y = -5, .w = 20, .h = 20 }, .{ .r = 0.5, .g = 0.5, .b = 0.5 });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    // 全ピクセルが0.5（128）になっているはず
    for (pixels) |p| {
        try std.testing.expectEqual(@as(u8, 128), p);
    }
}
