//! Layout helper functions - layout stack operations and queries.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const LayoutFrame = ui_mod.LayoutFrame;
const Vec2 = types.Vec2;

/// Returns a pointer to the top layout frame.
pub fn top(self: *Ui) *LayoutFrame {
    return &self.layouts[self.depth - 1];
}

/// Pushes a layout frame onto the layout stack.
pub fn pushLayout(self: *Ui, frame: LayoutFrame) void {
    std.debug.assert(self.depth < ui_mod.max_depth);
    self.layouts[self.depth] = frame;
    self.depth += 1;
}

/// Pops the top layout frame from the layout stack.
pub fn popLayout(self: *Ui) void {
    std.debug.assert(self.depth > 0);
    self.depth -= 1;
}

/// Returns the space a child may use for wrapping: the container's cross extent.
pub fn available(self: *Ui) Vec2 {
    const lf = top(self);
    return .{ .x = lf.contentCross(), .y = lf.contentMain() };
}

/// Returns the content width of the current layout.
pub fn contentWidth(self: *Ui) f32 {
    return top(self).contentCross();
}

// ============================================================================
// Tests
// ============================================================================

test "layout functions exist" {
    const topFunc = @TypeOf(top);
    const pushFunc = @TypeOf(pushLayout);
    const popFunc = @TypeOf(popLayout);
    const availFunc = @TypeOf(available);
    const widthFunc = @TypeOf(contentWidth);

    const top_info = @typeInfo(topFunc);
    const push_info = @typeInfo(pushFunc);
    const pop_info = @typeInfo(popFunc);
    const avail_info = @typeInfo(availFunc);
    const width_info = @typeInfo(widthFunc);

    try std.testing.expect(top_info == .@"fn");
    try std.testing.expect(push_info == .@"fn");
    try std.testing.expect(pop_info == .@"fn");
    try std.testing.expect(avail_info == .@"fn");
    try std.testing.expect(width_info == .@"fn");
}
