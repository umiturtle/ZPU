//! Theme system for ZUI
//!
//! Defines visual appearance: colors, sizes, spacing, animation.
//! Themes are dependency-free and can be tested independently.
//!
//! # Design
//! - All colors are `Color` (from types.zig)
//! - Sizes are in logical pixels
//! - Two built-in themes: `dark` and `light`
//! - Custom themes can override any field

const std = @import("std");
const types = @import("types.zig");

const Color = types.Color;

/// Visual theme: colors, sizes, spacing, animation
pub const Theme = struct {
    // Text colors
    text: Color = Color.hex(0xe6e9ef, 1),
    text_dim: Color = Color.hex(0x9aa3b2, 1),
    
    // Background colors
    bg: Color = Color.hex(0x171a21, 1),
    surface: Color = Color.hex(0x232838, 1),
    
    // Border and track
    border: Color = Color.hex(0x3a4257, 1),
    track: Color = Color.hex(0x11141b, 1),
    
    // Accent (interactive elements)
    accent: Color = Color.hex(0x5b8cff, 1),
    accent_text: Color = Color.hex(0xffffff, 1),
    
    // Widget colors
    widget: Color = Color.hex(0x2c3242, 1),
    widget_hover: Color = Color.hex(0x39415a, 1),
    
    // Special elements
    radio_dot: Color = Color.hex(0x9fb4ff, 1),

    // Sizes (logical pixels)
    font_size: f32 = 14,
    item_h: f32 = 26,
    radius: f32 = 5,
    gap: f32 = 6,
    pad: f32 = 8,
    
    /// Animation speed (1/seconds)
    anim_rate: f32 = 18,

    /// Dark theme (default)
    pub const dark = Theme{};
    
    /// Light theme
    pub const light = Theme{
        .text = Color.hex(0x1b1e26, 1),
        .text_dim = Color.hex(0x5b6273, 1),
        .bg = Color.hex(0xf3f4f7, 1),
        .surface = Color.hex(0xffffff, 1),
        .border = Color.hex(0xc9cfdd, 1),
        .track = Color.hex(0xdfe3ec, 1),
        .accent = Color.hex(0x2f6bff, 1),
        .widget = Color.hex(0xe3e7f0, 1),
        .widget_hover = Color.hex(0xd2d9e8, 1),
        .radio_dot = Color.hex(0x2f6bff, 1),
    };
    
    /// Create a custom theme by overriding fields
    pub fn with(self: Theme, overrides: anytype) Theme {
        var result = self;
        inline for (std.meta.fields(@TypeOf(overrides))) |field| {
            @field(result, field.name) = @field(overrides, field.name);
        }
        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "theme - dark theme defaults" {
    const theme = Theme.dark;
    
    try std.testing.expect(theme.font_size > 0);
    try std.testing.expect(theme.item_h > 0);
    try std.testing.expect(theme.radius >= 0);
    try std.testing.expect(theme.gap >= 0);
    try std.testing.expect(theme.pad >= 0);
    try std.testing.expect(theme.anim_rate > 0);
}

test "theme - light theme is different from dark" {
    const dark = Theme.dark;
    const light = Theme.light;
    
    // Light theme should have lighter background
    try std.testing.expect(light.bg.r > dark.bg.r);
    try std.testing.expect(light.bg.g > dark.bg.g);
    try std.testing.expect(light.bg.b > dark.bg.b);
}

test "theme - with overrides" {
    const theme = Theme.dark.with(.{
        .font_size = 16,
        .accent = Color.hex(0xff0000, 1),
    });
    
    try std.testing.expectEqual(@as(f32, 16), theme.font_size);
    try std.testing.expectEqual(@as(f32, 1), theme.accent.r);
    try std.testing.expectEqual(@as(f32, 0), theme.accent.g);
    try std.testing.expectEqual(@as(f32, 0), theme.accent.b);
    
    // Other fields should remain unchanged
    try std.testing.expectEqual(Theme.dark.item_h, theme.item_h);
}

test "theme - color hex conversion" {
    const color = Color.hex(0xff8040, 1);
    
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), color.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), color.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), color.b, 0.01);
}
