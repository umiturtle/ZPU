//! Helper functions - drawFrame, handleFocusKeys, paintTooltip, pushRootLayout, wrapText.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");
const font = @import("../font.zig");

const Ui = ui_mod.Ui;
const Style = ui_mod.Style;
const Rect = types.Rect;
const Vec2 = types.Vec2;
const Id = types.Id;

/// Creates a draw-mode layout frame from measured dimensions.
pub fn drawFrame(style: Style, content_rect: Rect, measured: Ui.Measured) ui_mod.LayoutFrame {
    var frame = ui_mod.LayoutFrame{
        .mode = .draw,
        .dir = style.dir,
        .content = content_rect,
        .gap = style.gap,
        .cross_align = style.cross_align,
        .cursor = if (style.dir == .x) content_rect.x else content_rect.y,
        .cross = if (style.dir == .x) content_rect.h else content_rect.w,
        .fixed_main = measured.fixed_main,
        .fill_content = measured.fill_content,
        .weight = measured.weight,
        .gaps = measured.gaps,
        .max_cross = measured.max_cross,
    };
    frame.fill_extent = @max(0, frame.contentMain() - measured.fixed_main);
    return frame;
}

/// Handles Tab/Shift+Tab focus navigation.
pub fn handleFocusKeys(self: *Ui) void {
    const list = self.focusables_prev.items;
    if (!self.input.keyPressed(.tab) or list.len == 0) return;
    const delta: isize = if (self.input.shift) -1 else 1;
    var idx: isize = 0;
    for (list, 0..) |id, i| {
        if (id.eql(self.ctx.focus)) {
            idx = @as(isize, @intCast(i)) + delta;
            break;
        }
    }
    const n: isize = @intCast(list.len);
    self.ctx.focus = list[@intCast(@mod(idx, n))];
}

/// Draws the tooltip panel.
pub fn paintTooltip(self: *Ui, tip: ui_mod.Tip) Ui.Error!void {
    const size = self.theme.font_size;
    const m = font.measure(tip.text, size);
    const pad = 5;
    const w = m.width + pad * 2;
    const h = m.height + pad * 2;
    var r = Rect{ .x = tip.anchor.x + 6, .y = tip.anchor.y + tip.anchor.h + 5, .w = w, .h = h };
    if (r.y + r.h > self.viewport.y) r.y = @max(0, tip.anchor.y - r.h - 5);
    r.x = std.math.clamp(r.x, 0, @max(0, self.viewport.x - r.w));
    try self.paintRounded(r, self.theme.radius, self.theme.surface, 1, self.theme.border);
    try self.paintText(r.x + pad, r.y + pad + m.ascent, size, self.theme.text, tip.text);
}

/// Pushes the root layout frame at the start of a frame.
pub fn pushRootLayout(self: *Ui) void {
    self.layouts[0] = .{
        .mode = .draw,
        .dir = .y,
        .content = .{ .x = 0, .y = 0, .w = self.viewport.x, .h = self.viewport.y },
        .gap = self.theme.gap,
        .cursor = 0,
        .cross = self.viewport.x,
    };
    self.depth = 1;
}

/// Wraps text content into lines.
pub fn wrapText(self: *Ui, content: []const u8, size: f32, max_width: f32) []font.Line {
    const a = self.arena.allocator();
    if (content.len == 0) {
        const one = a.alloc(font.Line, 1) catch return &.{};
        one[0] = .{ .start = 0, .len = 0 };
        return one;
    }
    const buf = a.alloc(font.Line, @min(self.max_lines, content.len + 1)) catch return &.{};
    return font.wrap(content, size, max_width, buf);
}

// ============================================================================
// Tests
// ============================================================================

test "helper functions exist" {
    // Just verify the functions exist and have correct signature
    const DrawFunc = @TypeOf(drawFrame);
    const FocusFunc = @TypeOf(handleFocusKeys);
    const TooltipFunc = @TypeOf(paintTooltip);
    const RootFunc = @TypeOf(pushRootLayout);
    const WrapFunc = @TypeOf(wrapText);
    
    const draw_info = @typeInfo(DrawFunc);
    const focus_info = @typeInfo(FocusFunc);
    const tooltip_info = @typeInfo(TooltipFunc);
    const root_info = @typeInfo(RootFunc);
    const wrap_info = @typeInfo(WrapFunc);
    
    try std.testing.expect(draw_info == .@"fn");
    try std.testing.expect(focus_info == .@"fn");
    try std.testing.expect(tooltip_info == .@"fn");
    try std.testing.expect(root_info == .@"fn");
    try std.testing.expect(wrap_info == .@"fn");
}
