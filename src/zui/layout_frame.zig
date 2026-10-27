//! LayoutFrame - one container's layout state

const std = @import("std");
const types = @import("types.zig");

const Axis = types.Axis;
const Alignment = types.Alignment;
const Rect = types.Rect;

const Mode = enum { measure, draw };

/// One container's layout state; the top of the stack is where the next widget
/// goes.
pub const LayoutFrame = struct {
    mode: Mode,
    dir: Axis = .y,
    /// Content box (draw) or the offered space (measure).
    content: Rect = .{},
    gap: f32 = 0,
    cross_align: Alignment = .start,
    /// Main-axis cursor and cross extent, in absolute coordinates.
    cursor: f32 = 0,
    cross: f32 = 0,
    /// Accumulated extents of the non-`fill` children (draw: what is fixed;
    /// measure: the same, plus gaps).
    fixed_main: f32 = 0,
    /// What the `fill` children asked for by content, and their total weight.
    fill_content: f32 = 0,
    weight: f32 = 0,
    gaps: f32 = 0,
    max_cross: f32 = 0,
    children: usize = 0,
    /// Space the `fill` children share (draw mode).
    fill_extent: f32 = 0,

    pub fn contentMain(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.w else self.content.h;
    }

    pub fn contentCross(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.h else self.content.w;
    }

    pub fn cursorStart(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.x else self.content.y;
    }

    /// What is left of the content box's main axis.
    pub fn remaining(self: *const LayoutFrame) f32 {
        return @max(0, self.contentMain() - (self.cursor - self.cursorStart()));
    }
};

// ============================================================================
// Tests
// ============================================================================

test "layout frame contentMain returns height for vertical layout" {
    var frame: LayoutFrame = .{
        .mode = .draw,
        .dir = .y,
        .content = .{ .x = 0, .y = 0, .w = 100, .h = 200 },
    };
    try std.testing.expectEqual(@as(f32, 200), frame.contentMain());
}

test "layout frame contentMain returns width for horizontal layout" {
    var frame: LayoutFrame = .{
        .mode = .draw,
        .dir = .x,
        .content = .{ .x = 0, .y = 0, .w = 100, .h = 200 },
    };
    try std.testing.expectEqual(@as(f32, 100), frame.contentMain());
}

test "layout frame remaining calculates correctly" {
    var frame: LayoutFrame = .{
        .mode = .draw,
        .dir = .y,
        .content = .{ .x = 0, .y = 0, .w = 100, .h = 200 },
        .cursor = 50,
    };
    try std.testing.expectEqual(@as(f32, 150), frame.remaining());
}

test "layout frame remaining does not go negative" {
    var frame: LayoutFrame = .{
        .mode = .draw,
        .dir = .y,
        .content = .{ .x = 0, .y = 0, .w = 100, .h = 200 },
        .cursor = 250,
    };
    try std.testing.expectEqual(@as(f32, 0), frame.remaining());
}
