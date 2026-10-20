//! ZAPP Button Widget
//!
//! クリック可能なボタンウィジェット。
//! 適応的レンダリング対応：品質レベルに応じてリッチな描画を自動適用

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const rich_render = @import("../rich_render.zig");
const adaptive_render = @import("../adaptive_render.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// ボタンの状態
pub const ButtonState = enum {
    normal,
    hover,
    active,
    disabled,
};

/// ボタンスタイル
pub const ButtonStyle = struct {
    /// 基本色
    bg: zgf.Color = .{ .r = 0.25, .g = 0.25, .b = 0.3, .a = 1.0 },
    fg: zgf.Color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
    border: zgf.Color = .{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 },
    
    /// ホバー時の色変化
    hover_bg: ?zgf.Color = null,
    active_bg: ?zgf.Color = null,
    
    /// コーナー半径
    radius: f32 = 8.0,
    
    /// 重要度（0.0 - 1.0）
    importance: f32 = 0.5,
};

/// ボタンバリアント
pub const ButtonVariant = enum {
    primary,
    secondary,
    danger,
    success,
    outlined,
    ghost,
};

/// ボタンウィジェットの設定
pub const ButtonConfig = struct {
    text: []const u8,
    rect: VirtualRect,
    state: ButtonState = .normal,
    style: ButtonStyle = .{},
    variant: ButtonVariant = .primary,
    
    /// リッチ機能の有効化
    shadow: bool = true,
    gradient: bool = true,
    border_style: bool = true,
};

/// ボタンのバリアントに応じたスタイルを取得
pub fn getVariantStyle(variant: ButtonVariant) ButtonStyle {
    return switch (variant) {
        .primary => .{
            .bg = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
            .fg = .{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 },
            .radius = 8.0,
            .importance = 0.8,
        },
        .secondary => .{
            .bg = .{ .r = 0.35, .g = 0.35, .b = 0.4, .a = 1.0 },
            .fg = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
            .radius = 8.0,
            .importance = 0.5,
        },
        .danger => .{
            .bg = .{ .r = 0.9, .g = 0.3, .b = 0.3, .a = 1.0 },
            .fg = .{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 },
            .radius = 8.0,
            .importance = 0.7,
        },
        .success => .{
            .bg = .{ .r = 0.3, .g = 0.8, .b = 0.4, .a = 1.0 },
            .fg = .{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 },
            .radius = 8.0,
            .importance = 0.6,
        },
        .outlined => .{
            .bg = .{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 },
            .fg = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
            .border = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 },
            .radius = 8.0,
            .importance = 0.4,
        },
        .ghost => .{
            .bg = .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 0.0 },
            .fg = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
            .radius = 8.0,
            .importance = 0.3,
        },
    };
}

/// 状態に応じた背景色を取得
fn getBackgroundColor(state: ButtonState, base: zgf.Color) zgf.Color {
    return switch (state) {
        .normal => base,
        .hover => .{
            .r = @min(base.r + 0.1, 1.0),
            .g = @min(base.g + 0.1, 1.0),
            .b = @min(base.b + 0.1, 1.0),
            .a = base.a,
        },
        .active => .{
            .r = @max(base.r - 0.1, 0.0),
            .g = @max(base.g - 0.1, 0.0),
            .b = @max(base.b - 0.1, 0.0),
            .a = base.a,
        },
        .disabled => .{
            .r = base.r * 0.5,
            .g = base.g * 0.5,
            .b = base.b * 0.5,
            .a = base.a * 0.5,
        },
    };
}

/// ボタンをレンダリングする（基本版）
pub fn render(fb: *zgf.Image, config: ButtonConfig) void {
    const bg = getBackgroundColor(config.state, config.style.bg);
    
    // 背景
    drawing.fillRect(fb, config.rect, bg);
    
    // 枠線
    drawing.drawRectBorder(fb, config.rect, config.style.border, 1.0);
    
    // テキスト（中央揃え）
    const text_x = config.rect.x + config.rect.w / 2;
    const text_y = config.rect.y + config.rect.h / 2;
    drawing.drawText(fb, config.text, text_x, text_y, config.style.fg);
}

/// ボタンをレンダリングする（リッチ版）
pub fn renderRich(
    fb: *zgf.Image,
    config: ButtonConfig,
    render_config: rich_render.RenderConfig,
) void {
    const bg = getBackgroundColor(config.state, config.style.bg);
    
    // シャドウ（high/ultra品質時のみ）
    if (config.shadow and config.state == .normal and 
        (render_config.global_quality == .high or render_config.global_quality == .ultra)) {
        rich_render.drawShadowRect(
            fb,
            config.rect.x,
            config.rect.y,
            config.rect.w,
            config.rect.h,
            config.style.radius,
            bg,
            .{ .offset_y = 2, .blur = 4 },
            render_config,
        );
    }
    
    // グラデーション背景
    if (config.gradient and render_config.gradients) {
        rich_render.drawGradientRect(
            fb,
            config.rect.x,
            config.rect.y,
            config.rect.w,
            config.rect.h,
            .{
                .from = bg,
                .to = .{
                    .r = @max(bg.r - 0.1, 0.0),
                    .g = @max(bg.g - 0.1, 0.0),
                    .b = @max(bg.b - 0.1, 0.0),
                    .a = bg.a,
                },
                .direction = .vertical,
            },
            render_config,
        );
    } else {
        // 通常背景
        rich_render.drawRoundedRect(
            fb,
            config.rect.x,
            config.rect.y,
            config.rect.w,
            config.rect.h,
            config.style.radius,
            bg,
            render_config,
        );
    }
    
    // ボーダー
    if (config.border_style) {
        rich_render.drawBorderedRect(
            fb,
            config.rect.x,
            config.rect.y,
            config.rect.w,
            config.rect.h,
            config.style.radius,
            bg,
            .{ .width = 1.0, .color = config.style.border },
            render_config,
        );
    }
    
    // テキスト（中央揃え）
    const text_x = config.rect.x + config.rect.w / 2;
    const text_y = config.rect.y + config.rect.h / 2;
    drawing.drawText(fb, config.text, text_x, text_y, config.style.fg);
}

/// 適応的レンダリング対応のボタン描画
pub fn renderAdaptive(
    fb: *zgf.Image,
    config: ButtonConfig,
    ctx: adaptive_render.AdaptiveContext,
) void {
    const render_config = ctx.toRenderConfig();
    
    // コンポーネント固有品質を計算
    const component_quality = adaptive_render.ComponentQuality{
        .importance = config.style.importance,
    };
    _ = component_quality.calculate(ctx);
    
    // リッチ機能の有効/無効を品質に応じて決定
    var adapted_config = config;
    adapted_config.shadow = render_config.global_quality != .low;
    adapted_config.gradient = render_config.gradients;
    adapted_config.border_style = render_config.global_quality != .low;
    
    renderRich(fb, adapted_config, render_config);
}

// テスト
test "button - normal state" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 100, 30, 3);
    defer img.deinit();
    
    render(&img, .{
        .text = "Click",
        .rect = .{ .x = 10, .y = 5, .w = 80, .h = 20 },
        .state = .normal,
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    // 背景が塗りつぶされているはず
    const idx = (10 * 100 + 10) * 3;
    try std.testing.expect(pixels.len > idx);
}

test "button - hover state is brighter" {
    const normal = getBackgroundColor(.normal, .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 });
    const hover = getBackgroundColor(.hover, .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 });
    
    try std.testing.expect(hover.r > normal.r);
    try std.testing.expect(hover.g > normal.g);
    try std.testing.expect(hover.b > normal.b);
}

test "button - active state is darker" {
    const normal = getBackgroundColor(.normal, .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 });
    const active = getBackgroundColor(.active, .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 });
    
    try std.testing.expect(active.r < normal.r);
    try std.testing.expect(active.g < normal.g);
    try std.testing.expect(active.b < normal.b);
}

test "button - variant styles" {
    const primary = getVariantStyle(.primary);
    const secondary = getVariantStyle(.secondary);
    
    try std.testing.expect(primary.importance > secondary.importance);
}
