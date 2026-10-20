//! Modern UI Components
//!
//! Material DesignにインスパイアされたモダンUIコンポーネント
//! ベクトルベースで高品質、アニメーション対応
//!
//! ## デザイン原則
//! - イレベーション（影のレイヤー）で奥行きを表現
//! - マテリアルカラーパレット
//! - スムーズなアニメーション
//! - タッチフレンドリー（最小48dp）
//! - アクセシビリティ考慮

const std = @import("std");
const zgf = @import("zgf");
const rich_render = @import("../rich_render.zig");
const vector = @import("../vector.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

// ============================================================================
// Material Design Color System
// ============================================================================

pub const MaterialColors = struct {
    // Primary
    pub const primary = zgf.Color{ .r = 0.25, .g = 0.46, .b = 0.95, .a = 1.0 }; // #3F74F2
    pub const primary_variant = zgf.Color{ .r = 0.18, .g = 0.35, .b = 0.78, .a = 1.0 };
    pub const on_primary = zgf.Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };

    // Secondary
    pub const secondary = zgf.Color{ .r = 0.03, .g = 0.73, .b = 0.73, .a = 1.0 }; // #00BAB7
    pub const secondary_variant = zgf.Color{ .r = 0.02, .g = 0.55, .b = 0.55, .a = 1.0 };
    pub const on_secondary = zgf.Color{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 };

    // Error
    pub const error_color = zgf.Color{ .r = 0.96, .g = 0.27, .b = 0.27, .a = 1.0 }; // #F54545
    pub const on_error = zgf.Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };

    // Success
    pub const success = zgf.Color{ .r = 0.18, .g = 0.80, .b = 0.44, .a = 1.0 }; // #2ECC71
    pub const on_success = zgf.Color{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 };

    // Warning
    pub const warning = zgf.Color{ .r = 0.99, .g = 0.76, .b = 0.17, .a = 1.0 }; // #FCC12B
    pub const on_warning = zgf.Color{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 };

    // Surface (Light Theme)
    pub const surface = zgf.Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };
    pub const on_surface = zgf.Color{ .r = 0.13, .g = 0.13, .b = 0.13, .a = 1.0 };
    pub const surface_variant = zgf.Color{ .r = 0.96, .g = 0.96, .b = 0.98, .a = 1.0 };
    pub const on_surface_variant = zgf.Color{ .r = 0.40, .g = 0.40, .b = 0.45, .a = 1.0 };

    // Background
    pub const background = zgf.Color{ .r = 0.97, .g = 0.97, .b = 0.99, .a = 1.0 };
    pub const on_background = zgf.Color{ .r = 0.13, .g = 0.13, .b = 0.13, .a = 1.0 };
};

// ============================================================================
// Elevation System (Shadow Layers)
// ============================================================================

pub const Elevation = struct {
    pub const Level = enum(u8) {
        none = 0,
        level1 = 1, // Cards, buttons
        level2 = 2, // Raised buttons
        level3 = 3, // Menus, dialogs
        level4 = 4, // Navigation drawers
        level5 = 5, // Modal bottom sheets
    };

    pub fn getShadow(level: Level) rich_render.Shadow {
        return switch (level) {
            .none => .{ .offset_y = 0, .blur = 0, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0 } },
            .level1 => .{ .offset_y = 1, .blur = 3, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.12 } },
            .level2 => .{ .offset_y = 2, .blur = 6, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.16 } },
            .level3 => .{ .offset_y = 4, .blur = 12, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.20 } },
            .level4 => .{ .offset_y = 6, .blur = 16, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.24 } },
            .level5 => .{ .offset_y = 8, .blur = 24, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.28 } },
        };
    }
};

// ============================================================================
// Modern Card Component
// ============================================================================

pub const CardConfig = struct {
    rect: VirtualRect,
    elevation: Elevation.Level = .level1,
    radius: f32 = 12.0,
    bg: zgf.Color = MaterialColors.surface,
    border_color: ?zgf.Color = null,
    border_width: f32 = 1.0,
};

/// モダンカードを描画
pub fn drawCard(fb: *zgf.Image, config: CardConfig, render_config: rich_render.RenderConfig) void {
    // シャドウ
    if (config.elevation != .none) {
        const shadow = Elevation.getShadow(config.elevation);
        rich_render.drawShadowRect(
            fb,
            config.rect.x,
            config.rect.y,
            config.rect.w,
            config.rect.h,
            config.radius,
            config.bg,
            shadow,
            render_config,
        );
    }

    // カード本体
    rich_render.drawRoundedRect(
        fb,
        config.rect.x,
        config.rect.y,
        config.rect.w,
        config.rect.h,
        config.radius,
        config.bg,
        render_config,
    );

    // ボーダー（オプション）
    if (config.border_color) |border_color| {
        rich_render.drawBorderedRect(
            fb,
            config.rect.x,
            config.rect.y,
            config.rect.w,
            config.rect.h,
            config.radius,
            config.bg,
            .{ .width = config.border_width, .color = border_color },
            render_config,
        );
    }
}

// ============================================================================
// Modern Button Component (Filled)
// ============================================================================

pub const ButtonState = enum {
    normal,
    hovered,
    pressed,
    disabled,
    focused,
};

pub const ModernButtonConfig = struct {
    rect: VirtualRect,
    text: []const u8,
    state: ButtonState = .normal,
    variant: ButtonVariant = .filled,
    color: zgf.Color = MaterialColors.primary,
    text_color: zgf.Color = MaterialColors.on_primary,
    radius: f32 = 24.0,
    elevation: Elevation.Level = .level2,
    icon: ?[]const u8 = null,
};

pub const ButtonVariant = enum {
    filled, // Filled button with elevation
    elevated, // Filled with more elevation
    outlined, // Border only
    text, // No background
    tonal, // Tinted background
};

/// モダンボタンを描画
pub fn drawModernButton(
    fb: *zgf.Image,
    config: ModernButtonConfig,
    render_config: rich_render.RenderConfig,
) void {
    const bg = getButtonBackground(config);

    // イレベーション（シャドウ）
    const elevation = switch (config.state) {
        .pressed => @as(Elevation.Level, if (@intFromEnum(config.elevation) > 0) @enumFromInt(@intFromEnum(config.elevation) - 1) else .none),
        .hovered => @as(Elevation.Level, if (@intFromEnum(config.elevation) < 5) @enumFromInt(@intFromEnum(config.elevation) + 1) else config.elevation),
        else => config.elevation,
    };

    // バリアント別の描画
    switch (config.variant) {
        .filled, .elevated => {
            // シャドウ
            if (elevation != .none) {
                const shadow = Elevation.getShadow(elevation);
                rich_render.drawShadowRect(
                    fb,
                    config.rect.x,
                    config.rect.y,
                    config.rect.w,
                    config.rect.h,
                    config.radius,
                    bg,
                    shadow,
                    render_config,
                );
            }

            // グラデーション背景
            if (render_config.gradients) {
                rich_render.drawGradientRect(
                    fb,
                    config.rect.x,
                    config.rect.y,
                    config.rect.w,
                    config.rect.h,
                    .{
                        .from = bg,
                        .to = darken(bg, 0.05),
                        .direction = .vertical,
                    },
                    render_config,
                );
            } else {
                rich_render.drawRoundedRect(
                    fb,
                    config.rect.x,
                    config.rect.y,
                    config.rect.w,
                    config.rect.h,
                    config.radius,
                    bg,
                    render_config,
                );
            }
        },
        .outlined => {
            // 透明背景
            rich_render.drawRoundedRect(
                fb,
                config.rect.x,
                config.rect.y,
                config.rect.w,
                config.rect.h,
                config.radius,
                .{ .r = 0, .g = 0, .b = 0, .a = 0 },
                render_config,
            );
            // ボーダー
            rich_render.drawBorderedRect(
                fb,
                config.rect.x,
                config.rect.y,
                config.rect.w,
                config.rect.h,
                config.radius,
                bg,
                .{ .width = 1.5, .color = config.color },
                render_config,
            );
        },
        .text => {
            // ホバー時のみ背景
            if (config.state == .hovered or config.state == .focused) {
                const hover_bg = zgf.Color{
                    .r = config.color.r,
                    .g = config.color.g,
                    .b = config.color.b,
                    .a = 0.08,
                };
                rich_render.drawRoundedRect(
                    fb,
                    config.rect.x,
                    config.rect.y,
                    config.rect.w,
                    config.rect.h,
                    config.radius,
                    hover_bg,
                    render_config,
                );
            }
        },
        .tonal => {
            // 薄い背景色
            const tonal_bg = zgf.Color{
                .r = config.color.r,
                .g = config.color.g,
                .b = config.color.b,
                .a = 0.12,
            };
            rich_render.drawRoundedRect(
                fb,
                config.rect.x,
                config.rect.y,
                config.rect.w,
                config.rect.h,
                config.radius,
                tonal_bg,
                render_config,
            );
        },
    }

    // テキスト描画（中央揃え）
    const text_x = config.rect.x + config.rect.w / 2;
    const text_y = config.rect.y + config.rect.h / 2;

    const text_color = switch (config.variant) {
        .filled, .elevated => config.text_color,
        .outlined, .tonal => config.color,
        .text => config.color,
    };

    // 簡易テキスト描画
    _ = text_x;
    _ = text_y;
    _ = text_color;
    // vector_text.drawText(fb, config.text, text_x, text_y, text_color);
}

fn getButtonBackground(config: ModernButtonConfig) zgf.Color {
    return switch (config.state) {
        .normal => config.color,
        .hovered => lighten(config.color, 0.08),
        .pressed => darken(config.color, 0.08),
        .disabled => .{ .r = config.color.r * 0.5, .g = config.color.g * 0.5, .b = config.color.b * 0.5, .a = 0.5 },
        .focused => lighten(config.color, 0.04),
    };
}

// ============================================================================
// Modern Text Field
// ============================================================================

pub const TextFieldConfig = struct {
    rect: VirtualRect,
    label: []const u8 = "",
    placeholder: []const u8 = "",
    value: []const u8 = "",
    state: TextFieldState = .normal,
    radius: f32 = 8.0,
    has_error: bool = false,
    helper_text: []const u8 = "",
};

pub const TextFieldState = enum {
    normal,
    focused,
    error_state,
    disabled,
};

/// モダンテキストフィールドを描画
pub fn drawTextField(
    fb: *zgf.Image,
    config: TextFieldConfig,
    render_config: rich_render.RenderConfig,
) void {
    const bg: zgf.Color = switch (config.state) {
        .normal => MaterialColors.surface_variant,
        .focused => MaterialColors.surface,
        .error_state => .{ .r = 1.0, .g = 0.95, .b = 0.95, .a = 1.0 },
        .disabled => .{ .r = 0.95, .g = 0.95, .b = 0.95, .a = 1.0 },
    };

    // 背景
    rich_render.drawRoundedRect(
        fb,
        config.rect.x,
        config.rect.y,
        config.rect.w,
        config.rect.h,
        config.radius,
        bg,
        render_config,
    );

    // ボトムボーダー（Material Design style）
    const border_color = if (config.has_error)
        MaterialColors.error_color
    else if (config.state == .focused)
        MaterialColors.primary
    else
        MaterialColors.on_surface_variant;

    const border_width: f32 = if (config.state == .focused) 2.0 else 1.0;

    const bottom_y = config.rect.y + config.rect.h - 1;
    vector.drawLineFast(fb, config.rect.x, bottom_y, config.rect.x + config.rect.w, bottom_y, border_color);

    _ = border_width;

    // テキスト描画
    // vector_text.drawText(fb, config.value, config.rect.x + 16, config.rect.y + config.rect.h / 2, MaterialColors.on_surface);
}

// ============================================================================
// Modern Switch (Toggle)
// ============================================================================

pub const SwitchConfig = struct {
    x: f32,
    y: f32,
    is_on: bool,
    state: ButtonState = .normal,
    track_color_on: zgf.Color = MaterialColors.primary,
    track_color_off: zgf.Color = MaterialColors.surface_variant,
    thumb_color: zgf.Color = MaterialColors.surface,
};

/// モダンスイッチを描画
pub fn drawSwitch(
    fb: *zgf.Image,
    config: SwitchConfig,
    render_config: rich_render.RenderConfig,
) void {
    const track_width: f32 = 52;
    const track_height: f32 = 32;
    const thumb_radius: f32 = 12;

    // トラック
    const track_color = if (config.is_on) config.track_color_on else config.track_color_off;
    rich_render.drawRoundedRect(
        fb,
        config.x,
        config.y,
        track_width,
        track_height,
        track_height / 2,
        track_color,
        render_config,
    );

    // サム（丸いつまみ）
    const thumb_x = if (config.is_on) config.x + track_width - thumb_radius - 4 else config.x + thumb_radius + 4;
    const thumb_y = config.y + track_height / 2;

    // サムのシャドウ
    rich_render.drawShadowRect(
        fb,
        thumb_x - thumb_radius,
        thumb_y - thumb_radius,
        thumb_radius * 2,
        thumb_radius * 2,
        thumb_radius,
        config.thumb_color,
        .{ .offset_y = 1, .blur = 3, .color = .{ .r = 0, .g = 0, .b = 0, .a = 0.2 } },
        render_config,
    );

    // サム本体
    vector.fillCircleOptimized(fb, thumb_x, thumb_y, thumb_radius, config.thumb_color);
}

// ============================================================================
// Utility Functions
// ============================================================================

fn darken(color: zgf.Color, amount: f32) zgf.Color {
    return .{
        .r = @max(color.r - amount, 0.0),
        .g = @max(color.g - amount, 0.0),
        .b = @max(color.b - amount, 0.0),
        .a = color.a,
    };
}

fn lighten(color: zgf.Color, amount: f32) zgf.Color {
    return .{
        .r = @min(color.r + amount, 1.0),
        .g = @min(color.g + amount, 1.0),
        .b = @min(color.b + amount, 1.0),
        .a = color.a,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "material colors are valid" {
    try std.testing.expect(MaterialColors.primary.r <= 1.0);
    try std.testing.expect(MaterialColors.primary.g <= 1.0);
    try std.testing.expect(MaterialColors.primary.b <= 1.0);
}

test "elevation levels" {
    const shadow1 = Elevation.getShadow(.level1);
    const shadow5 = Elevation.getShadow(.level5);

    try std.testing.expect(shadow5.blur > shadow1.blur);
    try std.testing.expect(shadow5.offset_y > shadow1.offset_y);
}
