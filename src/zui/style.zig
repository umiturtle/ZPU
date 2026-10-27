//! Style definition - what a widget or container asks for.

const types = @import("types.zig");

const Size = types.Size;
const Edges = types.Edges;
const Axis = types.Axis;
const Alignment = types.Alignment;
const Color = types.Color;

/// What a widget or a container asks for. `w`/`h` are along *screen* axes and are
/// mapped onto the current container's main/cross axis by the layout.
pub const Style = struct {
    w: Size = .auto,
    h: Size = .auto,
    pad: Edges = .{},
    /// Distance to the next sibling (containers: between children).
    gap: f32 = 0,
    dir: Axis = .y,
    /// Cross-axis placement of this container's children.
    cross_align: Alignment = .start,
    /// Main-axis placement of the whole group when there is slack.
    justify: Alignment = .start,
    clip: bool = false,
    bg: Color = Color.transparent,
    border: f32 = 0,
    border_color: Color = Color.hex(0xffffff, 0),
    radius: f32 = 0,
    min_w: f32 = 0,
    min_h: f32 = 0,

    pub fn sizeAlong(self: Style, axis: Axis) Size {
        return if (axis == .x) self.w else self.h;
    }

    /// The minimum on `axis`; a box is never laid out smaller than this.
    pub fn minAlong(self: Style, axis: Axis) f32 {
        return if (axis == .x) self.min_w else self.min_h;
    }
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "style default values" {
    const style = Style{};
    try std.testing.expect(style.w == .auto);
    try std.testing.expect(style.h == .auto);
    try std.testing.expect(style.gap == 0);
    try std.testing.expect(style.dir == .y);
    try std.testing.expect(style.cross_align == .start);
    try std.testing.expect(style.justify == .start);
    try std.testing.expect(!style.clip);
    try std.testing.expect(style.border == 0);
    try std.testing.expect(style.radius == 0);
}

test "style sizeAlong returns correct size" {
    var style = Style{ .w = .{ .fill = 1.0 }, .h = .{ .px = 100 } };
    try std.testing.expectEqual(Size{ .fill = 1.0 }, style.sizeAlong(.x));
    try std.testing.expectEqual(Size{ .px = 100 }, style.sizeAlong(.y));
}

test "style minAlong returns correct minimum" {
    var style = Style{ .min_w = 50, .min_h = 75 };
    try std.testing.expect(style.minAlong(.x) == 50);
    try std.testing.expect(style.minAlong(.y) == 75);
}
