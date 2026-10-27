//! Response handler - manages widget interaction state (hover, focus, click).

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Id = types.Id;
const Rect = types.Rect;
const Response = ui_mod.Response;

const hover_salt: u64 = 0x9e3779b97f4a7c15;
const press_salt: u64 = 0x517cc1b727220a95;

/// Calculates the interaction state for a widget (hover, focus, click).
pub fn respond(self: *Ui, id: Id, r: Rect, focusable: bool, enabled: bool) Response {
    var resp = Response{ .rect = r };
    if (self.measuring) return resp;
    resp.disabled = !enabled;

    const visible = r.overlaps(self.clip);
    const hovered = enabled and visible and r.contains(self.input.pointer);
    if (hovered) self.ctx.hot = id;
    resp.hovered = hovered;

    if (focusable and enabled) {
        self.focusables.append(self.alloc, id) catch {};
        if (hovered and self.input.pointer_pressed) self.requestFocus(id);
        if (self.ctx.focus.eql(id)) {
            if (self.input.keyPressed(.enter) or self.input.keyPressed(.space)) resp.clicked = true;
            if (self.input.keyPressed(.escape)) self.ctx.focus = Id.none;
        }
    }
    resp.focused = self.ctx.focus.eql(id);

    if (hovered and self.input.pointer_pressed and self.ctx.active.isNone()) self.ctx.active = id;
    if (self.ctx.active.eql(id)) {
        resp.active = true;
        if (self.input.pointer_released) {
            if (hovered) resp.clicked = true;
            self.ctx.active = Id.none;
        }
    }
    if (!enabled and self.ctx.active.eql(id)) self.ctx.active = Id.none;

    resp.hover_t = self.ctx.animate(.{ .h = id.h ^ hover_salt }, if (hovered) 1 else 0, self.theme.anim_rate, self.input.dt);
    resp.press_t = self.ctx.animate(
        .{ .h = id.h ^ press_salt },
        if (resp.active and hovered) 1 else 0,
        self.theme.anim_rate * 1.6,
        self.input.dt,
    );
    return resp;
}

// ============================================================================
// Tests
// ============================================================================

test "respond exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(respond);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
