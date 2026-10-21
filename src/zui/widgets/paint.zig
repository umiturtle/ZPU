//! Paint functions - draw primitives to the display list.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Rect = types.Rect;
const Vec2 = types.Vec2;
const Color = types.Color;

/// Draws a filled rectangle.
pub fn paintRect(self: *Ui, r: Rect, color: Color) Ui.Error!void {
    if (self.measuring or color.a <= 0 or r.isEmpty()) return;
    try self.list.push(self.alloc, .{ .rect = .{ .r = r, .color = color } });
}

/// Draws a rounded rectangle with optional border.
pub fn paintRounded(self: *Ui, r: Rect, radius: f32, color: Color, border: f32, border_color: Color) Ui.Error!void {
    if (self.measuring or r.isEmpty()) return;
    if (color.a <= 0 and (border <= 0 or border_color.a <= 0)) return;
    try self.list.push(self.alloc, .{
        .rounded = .{ .r = r, .radius = radius, .color = color, .border = border, .border_color = border_color },
    });
}

/// Draws a line between two points.
pub fn paintLine(self: *Ui, a: Vec2, b: Vec2, width: f32, color: Color) Ui.Error!void {
    if (self.measuring or color.a <= 0) return;
    try self.list.push(self.alloc, .{ .line = .{ .a = a, .b = b, .width = width, .color = color } });
}

/// Draws a filled triangle.
pub fn paintTriangle(self: *Ui, a: Vec2, b: Vec2, c: Vec2, color: Color) Ui.Error!void {
    if (self.measuring or color.a <= 0) return;
    try self.list.push(self.alloc, .{ .triangle = .{ .a = a, .b = b, .c = c, .color = color } });
}

/// Draws one line of text with baseline at `y`. The bytes are copied into the frame
/// arena, so the caller's string may die immediately.
pub fn paintText(self: *Ui, x: f32, baseline: f32, size: f32, color: Color, content: []const u8) Ui.Error!void {
    if (self.measuring or color.a <= 0 or content.len == 0) return;
    const owned = try self.arena.allocator().dupe(u8, content);
    try self.list.push(self.alloc, .{
        .text = .{ .pos = .{ .x = x, .y = baseline }, .size = size, .color = color, .text = owned },
    });
}

/// Pushes a clip rectangle onto the clip stack.
pub fn pushClip(self: *Ui, r: Rect) Ui.Error!void {
    self.clip = self.clip.intersect(r);
    try self.list.push(self.alloc, .{ .push_clip = self.clip });
}

/// Pops the clip rectangle from the clip stack.
pub fn popClip(self: *Ui, previous: Rect) Ui.Error!void {
    self.clip = previous;
    try self.list.push(self.alloc, .{ .pop_clip = {} });
}

// ============================================================================
// Tests
// ============================================================================

test "paint functions exist" {
    // Just verify the functions exist and have correct signature
    const RectFunc = @TypeOf(paintRect);
    const RoundedFunc = @TypeOf(paintRounded);
    const LineFunc = @TypeOf(paintLine);
    const TriangleFunc = @TypeOf(paintTriangle);
    const TextFunc = @TypeOf(paintText);
    
    const rect_info = @typeInfo(RectFunc);
    const rounded_info = @typeInfo(RoundedFunc);
    const line_info = @typeInfo(LineFunc);
    const triangle_info = @typeInfo(TriangleFunc);
    const text_info = @typeInfo(TextFunc);
    
    try std.testing.expect(rect_info == .@"fn");
    try std.testing.expect(rounded_info == .@"fn");
    try std.testing.expect(line_info == .@"fn");
    try std.testing.expect(triangle_info == .@"fn");
    try std.testing.expect(text_info == .@"fn");
}
