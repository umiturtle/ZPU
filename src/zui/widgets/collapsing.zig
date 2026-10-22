//! Collapsing header widget - a header that shows or hides a section.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Size = types.Size;
const Style = ui_mod.Style;

pub const CollapsingOptions = struct {
    /// State on the first frame; after that the retained flag wins.
    open: bool = false,
};

/// A header that shows or hides a section. The open flag is retained — it is
/// exactly the kind of state a body cannot recompute.
pub fn collapsing(self: *Ui, id: []const u8, title: []const u8, opts: CollapsingOptions, body: anytype) Ui.Error!bool {
    const key = self.idOf(id);
    const size = self.theme.font_size;
    const m = font.measure(title, size);
    const header_h = @max(self.theme.item_h, m.height + 4);
    const header = self.occupy(
        .{ .w = .full, .h = Size{ .px = header_h }, .cross_align = .start },
        .{ .x = m.width + self.theme.pad * 3, .y = m.height },
    );
    const resp = self.respond(key, header, true, true);
    if (self.measuring) return self.ctx.flag(key, opts.open);

    var open = self.ctx.flag(key, opts.open);
    if (resp.clicked) {
        open = !open;
        self.ctx.setFlag(key, open);
        self.markDirty();
    }
    try self.paintRounded(header, self.theme.radius, Color.lerp(self.theme.surface, self.theme.widget_hover, resp.hover_t), 1, self.theme.border);
    const ax = header.x + self.theme.pad + 3;
    const ay = header.y + header_h * 0.5;
    if (open) {
        try self.paintTriangle(.{ .x = ax - 4, .y = ay - 2 }, .{ .x = ax + 4, .y = ay - 2 }, .{ .x = ax, .y = ay + 3 }, self.theme.text_dim);
    } else {
        try self.paintTriangle(.{ .x = ax - 2, .y = ay - 4 }, .{ .x = ax - 2, .y = ay + 4 }, .{ .x = ax + 3, .y = ay }, self.theme.text_dim);
    }
    const baseline = header.y + (header_h - m.height) * 0.5 + m.ascent;
    try self.paintText(ax + self.theme.pad, baseline, size, self.theme.text, title);
    if (!open) return open;

    self.pushId(id);
    defer self.popId();
    const inner_style = Style{
        .dir = .y,
        .gap = self.theme.gap,
        .pad = .{ .l = self.theme.pad, .r = self.theme.pad, .b = self.theme.pad, .t = 2 },
    };
    const avail = self.available();
    const measured = try self.measureContainer(inner_style, avail, body);
    const box = self.occupy(.{ .w = .full, .h = .auto, .pad = inner_style.pad }, measured.content);
    if (self.measuring) return open;
    var frame = @import("../ui.zig").Ui.drawFrame(inner_style, box.inset(inner_style.pad), measured);
    frame.cursor = box.y + inner_style.pad.t;
    self.pushLayout(frame);
    defer self.popLayout();
    try body.run(self);
    return open;
}

// ============================================================================
// Tests
// ============================================================================

test "collapsing options defaults" {
    const opts = CollapsingOptions{};
    try std.testing.expect(!opts.open);
}
