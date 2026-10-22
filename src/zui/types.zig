//! Hub type for ZUI - re-exports from level 0 modules
//!
//! This file serves as a central hub that re-exports types from the
//! dependency-free level 0 modules (vec2, rect, color, id) while adding
//! ZUI-specific types like Alignment and Size.
//!
//! Dependency graph:
//!   types.zig → vec2.zig, rect.zig, color.zig, id.zig (re-exports)
//!   types.zig → (defines: Alignment, Size)
//!
//! All level 0 modules are dependency-free and can be tested independently.

const std = @import("std");

// Re-export from level 0 modules
pub const vec2 = @import("vec2.zig");
pub const rect = @import("rect.zig");
pub const color = @import("color.zig");
pub const id = @import("id.zig");

// Re-export main types for convenience
pub const Vec2 = vec2.Vec2;
pub const Rect = rect.Rect;
pub const Color = color.Color;
pub const Id = id.Id;

// Re-export supporting types from level 0 modules
pub const Axis = rect.Axis;
pub const Edges = rect.Edges;

/// Cross-axis placement inside a row or a column.
pub const Alignment = enum { start, center, end, stretch };

/// What a widget or a container asks for along one axis.
///
/// `auto` means "as small as my content allows"; `fill` means "a share of what
/// is left over" (a share of exactly the remainder when the total share is 1).
pub const Size = union(enum) {
    auto,
    px: f32,
    fill: f32,

    pub const full: Size = .{ .fill = 1.0 };

    pub fn resolve(self: Size, content: f32, avail: f32) f32 {
        return switch (self) {
            .auto => content,
            .px => |v| v,
            .fill => |w| @max(0, avail) * w,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "hub re-exports are compatible" {
    // Verify that re-exported types work as expected
    const v = Vec2.init(1, 2);
    try std.testing.expectEqual(@as(f32, 1), v.x);
    try std.testing.expectEqual(@as(f32, 2), v.y);

    const r = Rect.init(0, 0, 10, 20);
    try std.testing.expectEqual(@as(f32, 10), r.w);

    const c = Color.rgb(1, 0, 0);
    try std.testing.expectEqual(@as(f32, 1), c.r);

    const i = Id.hash(0, "test");
    try std.testing.expect(!i.isNone());
}

test "hub-specific types work" {
    // Test Axis
    const v = Vec2.init(3, 4);
    try std.testing.expectEqual(@as(f32, 3), v.index(.x));
    try std.testing.expectEqual(@as(f32, 4), v.index(.y));

    const flipped = v.along(.y);
    try std.testing.expectEqual(@as(f32, 4), flipped.x);
    try std.testing.expectEqual(@as(f32, 3), flipped.y);

    // Test Edges
    const e = Edges.all(5);
    try std.testing.expectEqual(@as(f32, 10), e.sum(.x));
    try std.testing.expectEqual(@as(f32, 10), e.sum(.y));

    // Test Size
    const s1 = Size{ .px = 100 };
    try std.testing.expectEqual(@as(f32, 100), s1.resolve(50, 200));

    const s2 = Size.full;
    try std.testing.expectEqual(@as(f32, 200), s2.resolve(50, 200));

    const s3 = Size{ .auto = {} };
    try std.testing.expectEqual(@as(f32, 50), s3.resolve(50, 200));
}
