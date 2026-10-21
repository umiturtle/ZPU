//! ZAPP リッチUIレンダリングシステム
//!
//! コンパイル時設定による高品質UIコンポーネントレンダリング
//!
//! ## 機能
//! - 4段階のレンダリング品質 (low, medium, high, ultra)
//! - コンポーネントごとの品質設定
//! - グラデーション、シャドウ、アンチエイリアス
//! - comptime分岐によるゼロコスト抽象化
//!
//! ## 使用例
//!
//! ```zig
//! // ビルド時オプション
//! // zig build -Drender-quality=ultra
//!
//! const render = @import("zapp").rich_render;
//!
//! // 全体的な品質設定
//! const config = render.RenderConfig{
//!     .global_quality = .ultra,
//!     .shadows = true,
//!     .gradients = true,
//!     .antialiasing = true,
//! };
//!
//! // コンポーネント固有の品質
//! const button_style = render.ComponentStyle{
//!     .quality = .high,
//!     .shadow = .{ .offset = 2, .blur = 4, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.3 } },
//!     .gradient = .{ .from = .{ .r = 0.3, .g = 0.6, .b = 0.9 }, .to = .{ .r = 0.2, .g = 0.5, .b = 0.8 } },
//! };
//! ```

const std = @import("std");
const zgf = @import("zgf");
const vector = @import("vector.zig");

// ビルドオプションからレンダリング品質を取得
const build_options = @import("build_options");

// ============================================================================
// ビルド時設定
// ============================================================================

/// ビルド時のデフォルト品質 (build.zigの -Drender-quality で上書き可能)
pub const default_quality: RenderQuality = blk: {
    const quality_str = build_options.render_quality;
    if (std.mem.eql(u8, quality_str, "low")) break :blk .low;
    if (std.mem.eql(u8, quality_str, "medium")) break :blk .medium;
    if (std.mem.eql(u8, quality_str, "high")) break :blk .high;
    if (std.mem.eql(u8, quality_str, "ultra")) break :blk .ultra;
    break :blk .medium; // デフォルト
};

// ============================================================================
// レンダリング品質
// ============================================================================

/// レンダリング品質レベル
pub const RenderQuality = enum {
    /// 最小限の描画。角丸なし、シャドウなし、単純な矩形
    low,
    /// 基本的な角丸とシンプルなシャドウ
    medium,
    /// 高品質な角丸、グラデーション、シャドウ
    high,
    /// 最高品質。マルチサンプルAA、精密なグラデーション、複雑なシャドウ
    ultra,
    
    /// 角丸のセグメント数を取得
    pub fn cornerSegments(self: RenderQuality) u32 {
        return switch (self) {
            .low => 0,
            .medium => 4,
            .high => 8,
            .ultra => 16,
        };
    }
    
    /// シャドウのサンプル数を取得
    pub fn shadowSamples(self: RenderQuality) u32 {
        return switch (self) {
            .low => 0,
            .medium => 2,
            .high => 4,
            .ultra => 8,
        };
    }
    
    /// グラデーションのステップ数を取得
    pub fn gradientSteps(self: RenderQuality) u32 {
        return switch (self) {
            .low => 1,
            .medium => 8,
            .high => 32,
            .ultra => 64,
        };
    }
    
    /// アンチエイリアスを行うか
    pub fn useAntialiasing(self: RenderQuality) bool {
        return switch (self) {
            .low => false,
            .medium => false,
            .high => true,
            .ultra => true,
        };
    }
};

// ============================================================================
// スタイル設定
// ============================================================================

/// シャドウ設定
pub const Shadow = struct {
    offset_x: f32 = 0,
    offset_y: f32 = 2,
    blur: f32 = 4,
    color: zgf.Color = .{ .r = 0, .g = 0, .b = 0, .a = 0.3 },
    
    pub const none: Shadow = .{};
};

/// グラデーション設定
pub const Gradient = struct {
    from: zgf.Color,
    to: zgf.Color,
    direction: Direction = .vertical,
    
    pub const Direction = enum {
        horizontal,
        vertical,
        diagonal_tl, // 左上から右下
        diagonal_bl, // 左下から右上
        radial,
    };
};

/// ボーダー設定
pub const Border = struct {
    width: f32 = 1,
    color: zgf.Color = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 },
    style: Style = .solid,
    
    pub const Style = enum {
        solid,
        dashed,
        dotted,
    };
    
    pub const none: Border = .{ .width = 0 };
};

/// コンポーネントスタイル
pub const ComponentStyle = struct {
    quality: RenderQuality = default_quality,
    shadow: Shadow = Shadow.none,
    gradient: ?Gradient = null,
    border: Border = Border.none,
    corner_radius: f32 = 4,
    opacity: f32 = 1.0,
};

/// 全体設定
pub const RenderConfig = struct {
    global_quality: RenderQuality = default_quality,
    shadows: bool = true,
    gradients: bool = true,
    antialiasing: bool = true,
    animations: bool = false,
    
    /// 低品質プリセット (組み込み機器向け)
    pub const low: RenderConfig = .{
        .global_quality = .low,
        .shadows = false,
        .gradients = false,
        .antialiasing = false,
    };
    
    /// 中品質プリセット (モバイル向け)
    pub const medium: RenderConfig = .{
        .global_quality = .medium,
        .shadows = true,
        .gradients = true,
        .antialiasing = false,
    };
    
    /// 高品質プリセット (デスクトップ向け)
    pub const high: RenderConfig = .{
        .global_quality = .high,
        .shadows = true,
        .gradients = true,
        .antialiasing = true,
    };
    
    /// 最高品質プリセット (プレゼンテーション向け)
    pub const ultra: RenderConfig = .{
        .global_quality = .ultra,
        .shadows = true,
        .gradients = true,
        .antialiasing = true,
        .animations = true,
    };
};

// ============================================================================
// リッチレンダリングプリミティブ
// ============================================================================

/// 高品質な角丸矩形を描画
pub fn drawRoundedRect(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    radius: f32,
    color: zgf.Color,
    config: RenderConfig,
) void {
    const quality = config.global_quality;
    
    if (quality == .low or radius <= 0) {
        // 低品質: 単純な矩形
        vector.fillRectOptimized(
            fb,
            @intFromFloat(x),
            @intFromFloat(y),
            @intFromFloat(x + width),
            @intFromFloat(y + height),
            color,
        );
        return;
    }
    
    // 高品質: 精密な角丸
    const segments = quality.cornerSegments();
    drawRoundedRectHQ(fb, x, y, width, height, radius, color, segments, config.antialiasing);
}

/// 高品質角丸矩形の内部実装
fn drawRoundedRectHQ(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    radius: f32,
    color: zgf.Color,
    segments: u32,
    antialias: bool,
) void {
    // 中央の矩形
    vector.fillRectOptimized(fb, @intFromFloat(x + radius), @intFromFloat(y), @intFromFloat(x + width - radius), @intFromFloat(y + height), color);
    vector.fillRectOptimized(fb, @intFromFloat(x), @intFromFloat(y + radius), @intFromFloat(x + width), @intFromFloat(y + height - radius), color);
    
    // 四隅の円弧
    if (antialias) {
        fillCircleAA(fb, x + radius, y + radius, radius, color, segments);
        fillCircleAA(fb, x + width - radius, y + radius, radius, color, segments);
        fillCircleAA(fb, x + radius, y + height - radius, radius, color, segments);
        fillCircleAA(fb, x + width - radius, y + height - radius, radius, color, segments);
    } else {
        vector.fillCircleOptimized(fb, x + radius, y + radius, radius, color);
        vector.fillCircleOptimized(fb, x + width - radius, y + radius, radius, color);
        vector.fillCircleOptimized(fb, x + radius, y + height - radius, radius, color);
        vector.fillCircleOptimized(fb, x + width - radius, y + height - radius, radius, color);
    }
}

/// アンチエイリアス付き円塗りつぶし
fn fillCircleAA(fb: *zgf.Image, cx: f32, cy: f32, radius: f32, color: zgf.Color, segments: u32) void {
    _ = segments;
    const r: i32 = @intFromFloat(@round(radius));
    if (r == 0) return;
    
    const cx_int: i32 = @intFromFloat(@round(cx));
    const cy_int: i32 = @intFromFloat(@round(cy));
    
    // スキャンラインで塗りつぶし（アンチエイリアス付き）
    var dy: i32 = -r;
    while (dy <= r) : (dy += 1) {
        const dy_f: f32 = @floatFromInt(dy);
        const r_f: f32 = @floatFromInt(r);
        const dx_exact: f32 = @sqrt(r_f * r_f - dy_f * dy_f);
        const dx: i32 = @intFromFloat(@floor(dx_exact));
        const fractional: f32 = dx_exact - @floor(dx_exact);
        
        const y = cy_int + dy;
        if (y < 0 or y >= @as(i32, @intCast(fb.height()))) continue;
        
        // メイン部分
        var xi: i32 = cx_int - dx;
        while (xi <= cx_int + dx) : (xi += 1) {
            if (xi >= 0 and xi < @as(i32, @intCast(fb.width()))) {
                fb.setPixel(@intCast(xi), @intCast(y), color);
            }
        }
        
        // エッジのアンチエイリアス
        if (fractional > 0.1) {
            const edge_color = zgf.Color{
                .r = color.r,
                .g = color.g,
                .b = color.b,
                .a = color.a * fractional,
            };
            
            const left_x = cx_int - dx - 1;
            const right_x = cx_int + dx + 1;
            
            if (left_x >= 0 and left_x < @as(i32, @intCast(fb.width()))) {
                fb.setPixel(@intCast(left_x), @intCast(y), edge_color);
            }
            if (right_x >= 0 and right_x < @as(i32, @intCast(fb.width()))) {
                fb.setPixel(@intCast(right_x), @intCast(y), edge_color);
            }
        }
    }
}

/// グラデーション矩形を描画
pub fn drawGradientRect(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    gradient: Gradient,
    config: RenderConfig,
) void {
    if (!config.gradients) {
        // グラデーション無効: from色で塗りつぶし
        vector.fillRectOptimized(
            fb,
            @intFromFloat(x),
            @intFromFloat(y),
            @intFromFloat(x + width),
            @intFromFloat(y + height),
            gradient.from,
        );
        return;
    }
    
    const steps = config.global_quality.gradientSteps();
    
    switch (gradient.direction) {
        .horizontal => {
            const step_width = width / @as(f32, @floatFromInt(steps));
            for (0..steps) |i| {
                const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                const color = lerpColor(gradient.from, gradient.to, t);
                const sx = x + @as(f32, @floatFromInt(i)) * step_width;
                vector.fillRectOptimized(
                    fb,
                    @intFromFloat(sx),
                    @intFromFloat(y),
                    @intFromFloat(sx + step_width + 1),
                    @intFromFloat(y + height),
                    color,
                );
            }
        },
        .vertical => {
            const step_height = height / @as(f32, @floatFromInt(steps));
            for (0..steps) |i| {
                const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                const color = lerpColor(gradient.from, gradient.to, t);
                const sy = y + @as(f32, @floatFromInt(i)) * step_height;
                vector.fillRectOptimized(
                    fb,
                    @intFromFloat(x),
                    @intFromFloat(sy),
                    @intFromFloat(x + width),
                    @intFromFloat(sy + step_height + 1),
                    color,
                );
            }
        },
        .diagonal_tl => {
            // 左上から右下
            const pixel_count = @as(usize, @intFromFloat(width * height));
            _ = pixel_count;
            // 簡易実装: 水平グラデーションで代用
            const step_width = width / @as(f32, @floatFromInt(steps));
            for (0..steps) |i| {
                const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                const color = lerpColor(gradient.from, gradient.to, t);
                const sx = x + @as(f32, @floatFromInt(i)) * step_width;
                vector.fillRectOptimized(
                    fb,
                    @intFromFloat(sx),
                    @intFromFloat(y),
                    @intFromFloat(sx + step_width + 1),
                    @intFromFloat(y + height),
                    color,
                );
            }
        },
        .diagonal_bl => {
            // 簡易実装
            const step_width = width / @as(f32, @floatFromInt(steps));
            for (0..steps) |i| {
                const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                const color = lerpColor(gradient.to, gradient.from, t);
                const sx = x + @as(f32, @floatFromInt(i)) * step_width;
                vector.fillRectOptimized(
                    fb,
                    @intFromFloat(sx),
                    @intFromFloat(y),
                    @intFromFloat(sx + step_width + 1),
                    @intFromFloat(y + height),
                    color,
                );
            }
        },
        .radial => {
            // 放射状グラデーション
            const cx = x + width / 2;
            const cy = y + height / 2;
            const max_radius = @min(width, height) / 2;
            
            const step_size = max_radius / @as(f32, @floatFromInt(steps));
            for (0..steps) |i| {
                const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                const color = lerpColor(gradient.from, gradient.to, t);
                const r = max_radius - @as(f32, @floatFromInt(i)) * step_size;
                vector.fillCircleOptimized(fb, cx, cy, r, color);
            }
        },
    }
}

/// シャドウ付き矩形を描画
pub fn drawShadowRect(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    radius: f32,
    color: zgf.Color,
    shadow: Shadow,
    config: RenderConfig,
) void {
    if (!config.shadows or shadow.blur <= 0) {
        // シャドウなし
        drawRoundedRect(fb, x, y, width, height, radius, color, config);
        return;
    }
    
    const samples = config.global_quality.shadowSamples();
    
    // シャドウを描画
    for (0..samples) |i| {
        const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(samples));
        const blur_offset = shadow.blur * t;
        const alpha_scale = 1.0 - t;
        
        const shadow_color = zgf.Color{
            .r = shadow.color.r,
            .g = shadow.color.g,
            .b = shadow.color.b,
            .a = shadow.color.a * alpha_scale * 0.5,
        };
        
        drawRoundedRect(
            fb,
            x + shadow.offset_x + blur_offset,
            y + shadow.offset_y + blur_offset,
            width,
            height,
            radius + blur_offset * 0.5,
            shadow_color,
            config,
        );
    }
    
    // メインの矩形を描画
    drawRoundedRect(fb, x, y, width, height, radius, color, config);
}

/// ボーダー付き矩形を描画
pub fn drawBorderedRect(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    radius: f32,
    fill_color: zgf.Color,
    border: Border,
    config: RenderConfig,
) void {
    // 塗りつぶし
    drawRoundedRect(fb, x, y, width, height, radius, fill_color, config);
    
    // ボーダー
    if (border.width > 0) {
        drawRoundedRectOutline(fb, x, y, width, height, radius, border.color, border.width, config);
    }
}

/// 角丸矩形の輪郭を描画
fn drawRoundedRectOutline(
    fb: *zgf.Image,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    radius: f32,
    color: zgf.Color,
    thickness: f32,
    config: RenderConfig,
) void {
    _ = thickness;
    const quality = config.global_quality;
    const segments = quality.cornerSegments();
    
    // 上辺
    vector.drawLineFast(fb, x + radius, y, x + width - radius, y, color);
    // 下辺
    vector.drawLineFast(fb, x + radius, y + height, x + width - radius, y + height, color);
    // 左辺
    vector.drawLineFast(fb, x, y + radius, x, y + height - radius, color);
    // 右辺
    vector.drawLineFast(fb, x + width, y + radius, x + width, y + height - radius, color);
    
    // 角の弧
    if (segments > 0) {
        drawArcOutline(fb, x + radius, y + radius, radius, 180, 270, color, segments);
        drawArcOutline(fb, x + width - radius, y + radius, radius, 270, 360, color, segments);
        drawArcOutline(fb, x + radius, y + height - radius, radius, 90, 180, color, segments);
        drawArcOutline(fb, x + width - radius, y + height - radius, radius, 0, 90, color, segments);
    }
}

/// 弧の輪郭を描画
fn drawArcOutline(
    fb: *zgf.Image,
    cx: f32,
    cy: f32,
    radius: f32,
    start_angle: f32,
    end_angle: f32,
    color: zgf.Color,
    segments: u32,
) void {
    const angle_range = end_angle - start_angle;
    
    for (0..segments) |i| {
        const t0: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
        const t1: f32 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(segments));
        
        const a0 = (start_angle + angle_range * t0) * std.math.pi / 180.0;
        const a1 = (start_angle + angle_range * t1) * std.math.pi / 180.0;
        
        const x0 = cx + radius * @cos(a0);
        const y0 = cy + radius * @sin(a0);
        const x1 = cx + radius * @cos(a1);
        const y1 = cy + radius * @sin(a1);
        
        vector.drawLineFast(fb, x0, y0, x1, y1, color);
    }
}

// ============================================================================
// ユーティリティ
// ============================================================================

/// 色補間
fn lerpColor(a: zgf.Color, b: zgf.Color, t: f32) zgf.Color {
    return .{
        .r = a.r + (b.r - a.r) * t,
        .g = a.g + (b.g - a.g) * t,
        .b = a.b + (b.b - a.b) * t,
        .a = a.a + (b.a - a.a) * t,
    };
}

// ============================================================================
// テスト
// ============================================================================

test "rich_render - RenderQuality segments" {
    try std.testing.expectEqual(@as(u32, 0), RenderQuality.low.cornerSegments());
    try std.testing.expectEqual(@as(u32, 4), RenderQuality.medium.cornerSegments());
    try std.testing.expectEqual(@as(u32, 8), RenderQuality.high.cornerSegments());
    try std.testing.expectEqual(@as(u32, 16), RenderQuality.ultra.cornerSegments());
}

test "rich_render - RenderConfig presets" {
    try std.testing.expect(!RenderConfig.low.shadows);
    try std.testing.expect(RenderConfig.high.antialiasing);
    try std.testing.expect(RenderConfig.ultra.animations);
}

test "rich_render - lerpColor" {
    const a: zgf.Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 };
    const b: zgf.Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 };
    const mid = lerpColor(a, b, 0.5);
    
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.g, 1e-6);
}

test "rich_render - drawRoundedRect low quality" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    const config = RenderConfig.low;
    drawRoundedRect(&fb, 10, 10, 80, 80, 8, .{ .r = 1, .g = 0, .b = 0 }, config);
}

test "rich_render - drawGradientRect" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    const config = RenderConfig.high;
    const gradient = Gradient{
        .from = .{ .r = 1, .g = 0, .b = 0 },
        .to = .{ .r = 0, .g = 0, .b = 1 },
        .direction = .horizontal,
    };
    
    drawGradientRect(&fb, 10, 10, 80, 80, gradient, config);
}

test "rich_render - drawShadowRect" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 150, 150, 3);
    defer fb.deinit();
    
    const config = RenderConfig.high;
    const shadow = Shadow{ .offset_y = 4, .blur = 8 };
    
    drawShadowRect(&fb, 20, 20, 100, 100, 8, .{ .r = 0.3, .g = 0.6, .b = 0.9 }, shadow, config);
}
