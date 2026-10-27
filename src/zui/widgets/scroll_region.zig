//! Scroll region helper - handles scroll offset and scrollbar rendering.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Edges = types.Edges;
const Id = types.Id;
const Rect = types.Rect;
const Vec2 = types.Vec2;

/// Manages scroll offset and scrollbar rendering for a scrollable region.
pub fn scrollRegion(self: *Ui, key: Id, r: Rect, view: Rect, content_h: f32, bar: f32, pad: Edges) Ui.Error!f32 {
    const max_scroll = @max(0, content_h - view.h);

    var scroll = self.ctx.scrollOf(key).y;
    if (r.contains(self.input.pointer) and !self.input.scroll.eql(Vec2.zero)) {
        // Positive `y` is "scroll down", which moves the content up: the
        // offset grows.
        scroll += self.input.scroll.y;
        self.markDirty();
    }
    scroll = std.math.clamp(scroll, 0, max_scroll);

    const fit = content_h <= view.h + 0.5;
    if (!fit) {
        const thumb_h = @max(24, view.h * (view.h / content_h));
        const track = Rect.init(view.x + view.w, r.y + pad.t, bar, r.h - pad.sum(.y));
        const thumb = Rect.init(track.x, track.y + (track.h - thumb_h) * (if (max_scroll > 0) scroll / max_scroll else 0), bar, thumb_h);
        const resp = self.respond(self.idOf("bar"), track, false, true);
        if (resp.active and !self.input.pointer_released) {
            const rel = (self.input.pointer.y - track.y - thumb_h * 0.5) / @max(1, track.h - thumb_h);
            scroll = std.math.clamp(rel, 0, 1) * max_scroll;
            self.markDirty();
        }
        const rail = Rect.init(track.x + 2, track.y, @max(2, track.w - 4), track.h);
        try self.paintRounded(rail, rail.w * 0.5, self.theme.track.withAlpha(0.6), 0, Color.transparent);
        try self.paintRounded(Rect.init(thumb.x + 2, thumb.y, @max(2, thumb.w - 4), thumb.h), rail.w * 0.5, self.theme.border, 0, Color.transparent);
    }
    self.ctx.setScroll(key, .{ .x = 0, .y = scroll });
    return scroll;
}

// ============================================================================
// Tests
// ============================================================================

test "scroll region exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(scrollRegion);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
