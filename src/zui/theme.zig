//! Theme definition for ZUI - colors, sizes, and visual parameters.

const types = @import("types.zig");

const Color = types.Color;

pub const Theme = struct {
    text: Color = Color.hex(0xe6e9ef, 1),
    text_dim: Color = Color.hex(0x9aa3b2, 1),
    bg: Color = Color.hex(0x171a21, 1),
    surface: Color = Color.hex(0x232838, 1),
    border: Color = Color.hex(0x3a4257, 1),
    track: Color = Color.hex(0x11141b, 1),
    accent: Color = Color.hex(0x5b8cff, 1),
    accent_text: Color = Color.hex(0xffffff, 1),
    widget: Color = Color.hex(0x2c3242, 1),
    widget_hover: Color = Color.hex(0x39415a, 1),
    radio_dot: Color = Color.hex(0x9fb4ff, 1),

    font_size: f32 = 14,
    item_h: f32 = 26,
    radius: f32 = 5,
    gap: f32 = 6,
    pad: f32 = 8,
    /// How fast hover/press glows settle, in 1/seconds.
    anim_rate: f32 = 18,

    pub const dark = Theme{};
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
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "theme dark and light exist" {
    const dark = Theme.dark;
    const light = Theme.light;
    try std.testing.expect(dark.font_size == 14);
    try std.testing.expect(light.font_size == 14);
    try std.testing.expect(dark.bg.r != light.bg.r);
}

test "theme has default values" {
    const theme = Theme{};
    try std.testing.expect(theme.gap == 6);
    try std.testing.expect(theme.pad == 8);
    try std.testing.expect(theme.radius == 5);
    try std.testing.expect(theme.anim_rate == 18);
}
