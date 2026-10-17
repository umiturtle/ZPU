//! Style system for ZUI
//!
//! Defines layout and sizing for widgets and containers.
//! Styles are dependency-free and can be tested independently.
//!
//! # Design
//! - `Size` can be `auto`, `fill`, or a fixed value
//! - `Edges` defines padding/margin on 4 sides
//! - `Style` combines size, padding, direction, alignment
//! - All layout calculations use these primitives

const std = @import("std");
const types = @import("types.zig");

const Axis = types.Axis;
const Edges = types.Edges;
const Size = types.Size;

/// What a widget or container asks for
pub const Style = struct {
    /// Width along screen X axis
    w: Size = .auto,
    /// Height along screen Y axis
    h: Size = .auto,
    /// Padding on 4 sides
    pad: Edges = .{},
    /// Distance to next sibling (or between children in containers)
    gap: f32 = 0,
    /// Layout direction (row or column)
    dir: Axis = .y,
    /// Cross-axis placement of children
    align: types.Alignment = .start,
    /// Main-axis placement of children
    justify: types.Alignment = .start,
    /// Whether this widget should be able to grow
    grow: bool = false,

    /// Create a style for a row (horizontal layout)
    pub fn row() Style {
        return .{ .dir = .x };
    }

    /// Create a style for a column (vertical layout)
    pub fn col() Style {
        return .{ .dir = .y };
    }

    /// Set width to a fixed value
    pub fn width(self: Style, w: f32) Style {
        var copy = self;
        copy.w = .{ .exact = w };
        return copy;
    }

    /// Set height to a fixed value
    pub fn height(self: Style, h: f32) Style {
        var copy = self;
        copy.h = .{ .exact = h };
        return copy;
    }

    /// Set both width and height
    pub fn size(self: Style, w: f32, h: f32) Style {
        return self.width(w).height(h);
    }

    /// Fill available space
    pub fn fill(self: Style) Style {
        var copy = self;
        copy.w = .fill;
        copy.h = .fill;
        return copy;
    }

    /// Set padding on all sides
    pub fn padding(self: Style, all: f32) Style {
        var copy = self;
        copy.pad = .{ .top = all, .right = all, .bottom = all, .left = all };
        return copy;
    }

    /// Set horizontal padding
    pub fn padX(self: Style, left: f32, right: f32) Style {
        var copy = self;
        copy.pad.left = left;
        copy.pad.right = right;
        return copy;
    }

    /// Set vertical padding
    pub fn padY(self: Style, top: f32, bottom: f32) Style {
        var copy = self;
        copy.pad.top = top;
        copy.pad.bottom = bottom;
        return copy;
    }

    /// Set gap between children
    pub fn withGap(self: Style, gap: f32) Style {
        var copy = self;
        copy.gap = gap;
        return copy;
    }

    /// Set alignment
    pub fn withAlign(self: Style, align: types.Alignment) Style {
        var copy = self;
        copy.align = align;
        return copy;
    }

    /// Set justify
    pub fn withJustify(self: Style, justify: types.Alignment) Style {
        var copy = self;
        copy.justify = justify;
        return copy;
    }

    /// Get total padding along an axis
    pub fn padAlong(self: Style, axis: Axis) f32 {
        return switch (axis) {
            .x => self.pad.left + self.pad.right,
            .y => self.pad.top + self.pad.bottom,
        };
    }

    /// Get minimum size along an axis
    pub fn minAlong(self: Style, axis: Axis) f32 {
        const s = if (axis == .x) self.w else self.h;
        return switch (s) {
            .exact => |v| v + self.padAlong(axis),
            .auto, .fill => self.padAlong(axis),
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "style - default values" {
    const style = Style{};
    
    try std.testing.expectEqual(types.Size.auto, style.w);
    try std.testing.expectEqual(types.Size.auto, style.h);
    try std.testing.expectEqual(@as(f32, 0), style.gap);
    try std.testing.expectEqual(Axis.y, style.dir);
    try std.testing.expect(!style.grow);
}

test "style - row creates horizontal layout" {
    const style = Style.row();
    
    try std.testing.expectEqual(Axis.x, style.dir);
}

test "style - col creates vertical layout" {
    const style = Style.col();
    
    try std.testing.expectEqual(Axis.y, style.dir);
}

test "style - width sets exact width" {
    const style = Style{}.width(100);
    
    try std.testing.expectEqual(@as(f32, 100), style.w.exact);
}

test "style - height sets exact height" {
    const style = Style{}.height(50);
    
    try std.testing.expectEqual(@as(f32, 50), style.h.exact);
}

test "style - size sets both dimensions" {
    const style = Style{}.size(100, 50);
    
    try std.testing.expectEqual(@as(f32, 100), style.w.exact);
    try std.testing.expectEqual(@as(f32, 50), style.h.exact);
}

test "style - fill sets both to fill" {
    const style = Style{}.fill();
    
    try std.testing.expectEqual(types.Size.fill, style.w);
    try std.testing.expectEqual(types.Size.fill, style.h);
}

test "style - padding sets all sides" {
    const style = Style{}.padding(10);
    
    try std.testing.expectEqual(@as(f32, 10), style.pad.top);
    try std.testing.expectEqual(@as(f32, 10), style.pad.right);
    try std.testing.expectEqual(@as(f32, 10), style.pad.bottom);
    try std.testing.expectEqual(@as(f32, 10), style.pad.left);
}

test "style - padX sets horizontal padding" {
    const style = Style{}.padX(10, 20);
    
    try std.testing.expectEqual(@as(f32, 10), style.pad.left);
    try std.testing.expectEqual(@as(f32, 20), style.pad.right);
    try std.testing.expectEqual(@as(f32, 0), style.pad.top);
    try std.testing.expectEqual(@as(f32, 0), style.pad.bottom);
}

test "style - padY sets vertical padding" {
    const style = Style{}.padY(10, 20);
    
    try std.testing.expectEqual(@as(f32, 10), style.pad.top);
    try std.testing.expectEqual(@as(f32, 20), style.pad.bottom);
    try std.testing.expectEqual(@as(f32, 0), style.pad.left);
    try std.testing.expectEqual(@as(f32, 0), style.pad.right);
}

test "style - withGap sets gap" {
    const style = Style{}.withGap(8);
    
    try std.testing.expectEqual(@as(f32, 8), style.gap);
}

test "style - padAlong calculates correctly" {
    const style = Style{}.padding(10);
    
    try std.testing.expectEqual(@as(f32, 20), style.padAlong(.x));
    try std.testing.expectEqual(@as(f32, 20), style.padAlong(.y));
}

test "style - padAlong with asymmetric padding" {
    const style = Style{}.padX(10, 20);
    
    try std.testing.expectEqual(@as(f32, 30), style.padAlong(.x));
    try std.testing.expectEqual(@as(f32, 0), style.padAlong(.y));
}

test "style - minAlong with exact size" {
    const style = Style{}.width(100).padding(10);
    
    try std.testing.expectEqual(@as(f32, 120), style.minAlong(.x));
}

test "style - minAlong with auto size" {
    const style = Style{}.padding(10);
    
    try std.testing.expectEqual(@as(f32, 20), style.minAlong(.x));
}

test "style - chaining methods" {
    const style = Style.row()
        .width(100)
        .height(50)
        .padding(10)
        .withGap(8)
        .withAlign(.center);
    
    try std.testing.expectEqual(Axis.x, style.dir);
    try std.testing.expectEqual(@as(f32, 100), style.w.exact);
    try std.testing.expectEqual(@as(f32, 50), style.h.exact);
    try std.testing.expectEqual(@as(f32, 10), style.pad.top);
    try std.testing.expectEqual(@as(f32, 8), style.gap);
    try std.testing.expectEqual(types.Alignment.center, style.align);
}
