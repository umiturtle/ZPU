//! Response struct - widget interaction state.

const types = @import("types.zig");

const Rect = types.Rect;

/// Response from a widget interaction: what happened this frame.
/// A widget receives this once per frame, so it never has
/// to be asked twice.
pub const Response = struct {
    hovered: bool = false,
    /// The pointer is down and this widget owns the press.
    active: bool = false,
    /// Press *and* release inside, or a keyboard activation: the edge that fires
    /// an action.
    clicked: bool = false,
    focused: bool = false,
    disabled: bool = false,
    /// Animated `0..1` hover glow and press amount.
    hover_t: f32 = 0,
    press_t: f32 = 0,
    rect: Rect = .{},
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "response default values" {
    const resp = Response{};
    try std.testing.expect(!resp.hovered);
    try std.testing.expect(!resp.active);
    try std.testing.expect(!resp.clicked);
    try std.testing.expect(!resp.focused);
    try std.testing.expect(!resp.disabled);
    try std.testing.expect(resp.hover_t == 0);
    try std.testing.expect(resp.press_t == 0);
}

test "response can be created with custom values" {
    const resp = Response{
        .hovered = true,
        .active = true,
        .clicked = false,
        .focused = true,
        .disabled = false,
        .hover_t = 0.5,
        .press_t = 0.7,
    };
    try std.testing.expect(resp.hovered);
    try std.testing.expect(resp.active);
    try std.testing.expect(!resp.clicked);
    try std.testing.expect(resp.focused);
    try std.testing.expect(resp.hover_t == 0.5);
    try std.testing.expect(resp.press_t == 0.7);
}
