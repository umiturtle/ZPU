//! Text edit widget - a single-line text input field.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");
const types = @import("../types.zig");
const text_mod = @import("../text.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Rect = types.Rect;
const Size = types.Size;
const TextField = text_mod.TextField;

pub const TextEditOptions = struct {
    placeholder: []const u8 = "",
    w: Size = .full,
    enabled: bool = true,
};

/// The one genuinely retained widget: a caret is a cursor, and a cursor is
/// not a function of the string. Everything a *caller* sees — the value and
/// the change flag — stays immediate.
pub fn textEdit(self: *Ui, id: []const u8, field: *TextField, opts: TextEditOptions) Ui.Error!bool {
    const key = self.idOf(id);
    const size = self.theme.font_size;
    const metrics = font.measure("", size);
    const r = self.occupy(
        .{ .w = opts.w, .h = Size{ .px = @max(self.theme.item_h, metrics.height + 4) }, .min_w = 80 },
        .{ .x = @max(80, metrics.width), .y = metrics.height },
    );
    const resp = self.respond(key, r, true, opts.enabled);
    if (self.measuring) return false;

    const edit = self.ctx.edit(key);
    var changed = false;
    if (resp.hovered and self.input.pointer_pressed and !self.ctx.focus.eql(key)) {
        self.requestFocus(key);
        edit.moveTo(field.offsetAtX(size, self.input.pointer.x - (r.x + self.theme.pad) + edit.scroll));
    } else if (resp.focused and self.input.pointer_pressed) {
        edit.moveTo(field.offsetAtX(size, self.input.pointer.x - (r.x + self.theme.pad) + edit.scroll));
    }

    if (resp.focused) {
        for (self.input.text) |cp| {
            if (cp < 32 or cp > 126) continue;
            if (edit.hasSelection()) {
                const sel = edit.selection();
                edit.moveTo(field.deleteRange(sel[0], sel[1]));
            }
            edit.moveTo(field.insert(edit.caret, @intCast(cp)));
            changed = true;
        }
        if (self.input.keyPressed(.backspace)) {
            if (edit.hasSelection()) {
                const sel = edit.selection();
                edit.moveTo(field.deleteRange(sel[0], sel[1]));
            } else if (edit.caret > 0) {
                edit.moveTo(field.deleteRange(TextField.stepLeft(edit.caret), edit.caret));
            }
            changed = true;
        }
        if (self.input.keyPressed(.delete)) {
            if (edit.hasSelection()) {
                const sel = edit.selection();
                edit.moveTo(field.deleteRange(sel[0], sel[1]));
            } else {
                _ = field.deleteRange(edit.caret, field.stepRight(edit.caret));
            }
            changed = true;
        }
        if (self.input.keyPressed(.left)) edit.moveTo(TextField.stepLeft(edit.caret));
        if (self.input.keyPressed(.right)) edit.moveTo(field.stepRight(edit.caret));
        if (self.input.keyPressed(.home)) edit.moveTo(0);
        if (self.input.keyPressed(.end)) edit.moveTo(field.len);
    }
    if (changed) self.markDirty();

    // Keep the caret inside the box by scrolling the text itself.
    const inner_w = @max(1, r.w - self.theme.pad * 2);
    const caret_x = field.xOfOffset(size, edit.caret);
    if (caret_x - edit.scroll > inner_w) edit.scroll = caret_x - inner_w;
    if (caret_x - edit.scroll < 0) edit.scroll = caret_x;
    edit.scroll = @max(0, edit.scroll);

    const fill = Color.lerp(self.theme.track, self.theme.surface, resp.hover_t * 0.35);
    try self.paintRounded(r, self.theme.radius * 0.6, fill, 1, if (resp.focused) self.theme.accent else self.theme.border);

    const text_x = r.x + self.theme.pad - edit.scroll;
    const baseline = r.y + (r.h - metrics.height) * 0.5 + metrics.ascent;
    const previous_clip = self.clip;
    try self.pushClip(r.shrink(1));
    if (field.len == 0 and !resp.focused) {
        try self.paintText(r.x + self.theme.pad, baseline, size, self.theme.text_dim.scale(0.8), opts.placeholder);
    } else {
        if (edit.hasSelection()) {
            const sel = edit.selection();
            const x0 = text_x + field.xOfOffset(size, sel[0]);
            const x1 = text_x + field.xOfOffset(size, sel[1]);
            try self.paintRounded(Rect.init(x0, r.y + 3, @max(1, x1 - x0), r.h - 6), 2, self.theme.accent.withAlpha(0.35), 0, Color.transparent);
        }
        try self.paintText(text_x, baseline, size, self.theme.text, field.slice());
    }
    if (resp.focused) {
        // A blinking caret, solid right after an edit so typing never looks
        // laggy. The blink is a function of the frame's `time`, which the
        // host supplies — the library has no clock of its own.
        if (@mod(self.input.time, 1.0) < 0.5 or edit.caret_age < 0.35) {
            try self.paintRect(Rect.init(text_x + caret_x, r.y + 4, 1.5, r.h - 8), self.theme.text);
        }
    }
    try self.popClip(previous_clip);
    edit.caret_age += self.input.dt;
    edit.focused = true;
    return changed and opts.enabled;
}

// ============================================================================
// Tests
// ============================================================================

test "text edit options defaults" {
    const opts = TextEditOptions{};
    try std.testing.expectEqual(@as([]const u8, ""), opts.placeholder);
    try std.testing.expect(opts.enabled);
}
