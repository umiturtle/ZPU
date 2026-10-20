//! ZAPP UI v2 - Enhanced Vector Rendering
//! 
//! 高品質なベクターレンダリングによるUIコンポーネント

const std = @import("std");
const zgf = @import("zgf");
const vector = @import("vector.zig");
const vector_text = @import("vector_text.zig");

// ============================================================================
// Enhanced Vector Drawing Primitives
// ============================================================================

/// アンチエイリアス付きライン描画 (Wu's algorithm簡易版)
pub fn drawLineAA(fb: *zgf.Image, x0: f32, y0: f32, x1: f32, y1: f32, color: zgf.Color) void {
    // 簡易アンチエイリアス: 複数サンプルで平均化
    const samples = 3;
    const offsets = [_]f32{ -0.25, 0, 0.25 };
    
    var i: usize = 0;
    while (i < samples) : (i += 1) {
        const offset = offsets[i];
        vector.drawLineFast(fb, x0 + offset, y0 + offset, x1 + offset, y1 + offset, color);
    }
}

/// グラデーション付き矩形描画
pub fn drawGradientRect(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color_top: zgf.Color,
    color_bottom: zgf.Color,
) void {
    const y_start: i32 = @intFromFloat(y);
    const y_end: i32 = @intFromFloat(y + h);
    const x_start: i32 = @intFromFloat(x);
    const x_end: i32 = @intFromFloat(x + w);
    
    var yi = y_start;
    while (yi < y_end) : (yi += 1) {
        const t: f32 = @as(f32, @floatFromInt(yi - y_start)) / @as(f32, @floatFromInt(y_end - y_start));
        const color = lerpColor(color_top, color_bottom, t);
        
        var xi = x_start;
        while (xi < x_end) : (xi += 1) {
            fb.setPixel(@intCast(xi), @intCast(yi), color);
        }
    }
}

/// 色補間
fn lerpColor(a: zgf.Color, b: zgf.Color, t: f32) zgf.Color {
    return .{
        .r = a.r + (b.r - a.r) * t,
        .g = a.g + (b.g - a.g) * t,
        .b = a.b + (b.b - a.b) * t,
        .a = a.a + (b.a - a.a) * t,
    };
}

/// シャドウ付き矩形描画
pub fn drawShadowRect(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: zgf.Color,
    shadow_color: zgf.Color,
    shadow_offset: f32,
    shadow_blur: f32,
) void {
    // シャドウを描画 (簡易ブラー)
    const blur_samples: i32 = @intFromFloat(shadow_blur);
    var i: i32 = 0;
    while (i <= blur_samples) : (i += 1) {
        const alpha_scale: f32 = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(blur_samples + 1));
        const shadow_c = zgf.Color{
            .r = shadow_color.r,
            .g = shadow_color.g,
            .b = shadow_color.b,
            .a = shadow_color.a * alpha_scale * 0.3,
        };
        
        vector.fillRectOptimized(
            fb,
            @intFromFloat(x + shadow_offset + @as(f32, @floatFromInt(i))),
            @intFromFloat(y + shadow_offset + @as(f32, @floatFromInt(i))),
            @intFromFloat(x + w + shadow_offset + @as(f32, @floatFromInt(i))),
            @intFromFloat(y + h + shadow_offset + @as(f32, @floatFromInt(i))),
            shadow_c,
        );
    }
    
    // メイン矩形を描画
    vector.fillRectOptimized(
        fb,
        @intFromFloat(x),
        @intFromFloat(y),
        @intFromFloat(x + w),
        @intFromFloat(y + h),
        color,
    );
}

/// 高品質な角丸矩形 (スーパーサンプリング)
pub fn drawRoundedRectHQ(fb: *zgf.Image, rect_x: f32, rect_y: f32, rect_w: f32, rect_h: f32, radius: f32, color: zgf.Color) void {
    if (radius <= 0) {
        vector.fillRectOptimized(
            fb,
            @intFromFloat(rect_x),
            @intFromFloat(rect_y),
            @intFromFloat(rect_x + rect_w),
            @intFromFloat(rect_y + rect_h),
            color,
        );
        return;
    }
    
    // 中央部分
    vector.fillRectOptimized(fb, @intFromFloat(rect_x + radius), @intFromFloat(rect_y), @intFromFloat(rect_x + rect_w - radius), @intFromFloat(rect_y + rect_h), color);
    vector.fillRectOptimized(fb, @intFromFloat(rect_x), @intFromFloat(rect_y + radius), @intFromFloat(rect_x + rect_w), @intFromFloat(rect_y + rect_h - radius), color);
    
    // 四隅の円弧
    vector.fillCircleOptimized(fb, rect_x + radius, rect_y + radius, radius, color);
    vector.fillCircleOptimized(fb, rect_x + rect_w - radius, rect_y + radius, radius, color);
    vector.fillCircleOptimized(fb, rect_x + radius, rect_y + rect_h - radius, radius, color);
    vector.fillCircleOptimized(fb, rect_x + rect_w - radius, rect_y + rect_h - radius, radius, color);
}

/// アイコン描画 (ベクター)
pub fn drawIcon(fb: *zgf.Image, icon: Icon, x: f32, y: f32, size: f32, color: zgf.Color) void {
    const s = size / 24.0; // 24x24グリッド基準
    
    switch (icon) {
        .check => {
            vector.drawLineFast(fb, x + 4 * s, y + 12 * s, x + 10 * s, y + 18 * s, color);
            vector.drawLineFast(fb, x + 10 * s, y + 18 * s, x + 20 * s, y + 6 * s, color);
        },
        .cross => {
            vector.drawLineFast(fb, x + 6 * s, y + 6 * s, x + 18 * s, y + 18 * s, color);
            vector.drawLineFast(fb, x + 18 * s, y + 6 * s, x + 6 * s, y + 18 * s, color);
        },
        .arrow_right => {
            vector.drawLineFast(fb, x + 4 * s, y + 12 * s, x + 20 * s, y + 12 * s, color);
            vector.drawLineFast(fb, x + 14 * s, y + 6 * s, x + 20 * s, y + 12 * s, color);
            vector.drawLineFast(fb, x + 14 * s, y + 18 * s, x + 20 * s, y + 12 * s, color);
        },
        .menu => {
            vector.drawLineFast(fb, x + 4 * s, y + 6 * s, x + 20 * s, y + 6 * s, color);
            vector.drawLineFast(fb, x + 4 * s, y + 12 * s, x + 20 * s, y + 12 * s, color);
            vector.drawLineFast(fb, x + 4 * s, y + 18 * s, x + 20 * s, y + 18 * s, color);
        },
        .search => {
            vector.drawCircleOptimized(fb, x + 10 * s, y + 10 * s, 6 * s, color);
            vector.drawLineFast(fb, x + 14 * s, y + 14 * s, x + 20 * s, y + 20 * s, color);
        },
        .star => {
            var i: usize = 0;
            while (i < 5) : (i += 1) {
                const angle1: f32 = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / 5.0 - std.math.pi / 2.0;
                const angle2: f32 = @as(f32, @floatFromInt(i + 1)) * 2.0 * std.math.pi / 5.0 - std.math.pi / 2.0;
                const cx = x + 12 * s;
                const cy = y + 12 * s;
                const r = 10 * s;
                vector.drawLineFast(
                    fb,
                    cx + r * @cos(angle1),
                    cy + r * @sin(angle1),
                    cx + r * @cos(angle2),
                    cy + r * @sin(angle2),
                    color,
                );
            }
        },
    }
}

pub const Icon = enum {
    check,
    cross,
    arrow_right,
    menu,
    search,
    star,
};

// ============================================================================
// Tests
// ============================================================================

test "vector_ui - lerpColor" {
    const a: zgf.Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 };
    const b: zgf.Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 };
    const mid = lerpColor(a, b, 0.5);
    
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.g, 1e-6);
}

test "vector_ui - drawRoundedRectHQ" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    drawRoundedRectHQ(&fb, 10, 10, 80, 80, 8, .{ .r = 1, .g = 0, .b = 0 });
    
    const pixels = try fb.toPixels(allocator);
    defer allocator.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}

test "vector_ui - drawIcon" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    drawIcon(&fb, .check, 10, 10, 24, .{ .r = 0, .g = 1, .b = 0 });
    drawIcon(&fb, .cross, 40, 10, 24, .{ .r = 1, .g = 0, .b = 0 });
}
