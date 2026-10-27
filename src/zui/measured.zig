//! Measured - the result of measuring a container.

const types = @import("types.zig");
const Vec2 = types.Vec2;

/// The result of measuring a container's children.
pub const Measured = struct {
    content: Vec2,
    fixed_main: f32 = 0,
    fill_content: f32 = 0,
    weight: f32 = 0,
    gaps: f32 = 0,
    max_cross: f32 = 0,
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "measured struct" {
    const measured = Measured{
        .content = Vec2{ .x = 100, .y = 200 },
        .fixed_main = 50,
        .fill_content = 30,
        .weight = 1.0,
        .gaps = 10,
        .max_cross = 150,
    };
    try std.testing.expect(measured.content.x == 100);
    try std.testing.expect(measured.content.y == 200);
    try std.testing.expect(measured.fixed_main == 50);
    try std.testing.expect(measured.fill_content == 30);
    try std.testing.expect(measured.weight == 1.0);
    try std.testing.expect(measured.gaps == 10);
    try std.testing.expect(measured.max_cross == 150);
}
