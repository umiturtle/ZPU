//! Virtual list widget - a list that only builds the rows you can see.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Rect = types.Rect;
const Style = ui_mod.Style;

pub const VirtualOptions = struct {
    style: Style = .{},
    /// Every row is exactly this tall. A fixed height is what makes the
    /// visible range a division instead of a measuring pass over rows that
    /// will never be drawn — and it is the honest contract: a list that
    /// wants variable heights has to know them before it builds the rows.
    row_height: f32 = 22,
    /// Layout inside one row (padding, gap, direction).
    row_style: Style = .{},
    bar: f32 = 10,
};

/// A list that only builds the rows you can see.
///
/// `rows` may be a million: the frame costs one row per visible line plus a
/// scrollbar, because the children that are off screen are never called.
/// The row function receives its index **and its box**, placed at
/// `view.y - scroll + i * (row_height + gap)` — the geometry a virtual list
/// has to get exactly right, and the thing its test checks.
///
/// A row is *not* measured, only drawn: its height is the one you gave,
/// which is why a million rows cost nothing and why a row that wants a
/// background can just paint its own box.
pub fn virtualList(
    self: *Ui,
    id: []const u8,
    rows: usize,
    opts: VirtualOptions,
    ctx: anytype,
    comptime rowFn: fn (@TypeOf(ctx), *Ui, usize, Rect) Ui.Error!void,
) Ui.Error!void {
    self.pushId(id);
    defer self.popId();
    const key = self.idOf("scroll");

    const avail = self.available();
    const gap = opts.style.gap;
    const row_h = @max(0, opts.row_height);
    const pitch = row_h + gap;
    const total: f32 = if (rows == 0) 0 else @as(f32, @floatFromInt(rows)) * pitch - gap;

    const r = self.occupy(opts.style, .{ .x = @min(avail.x, opts.style.min_w), .y = total });
    if (self.measuring) return;
    if (r.isEmpty()) return;

    try self.paintRounded(r, opts.style.radius, opts.style.bg, opts.style.border, opts.style.border_color);

    const view = r.inset(.{ .l = opts.style.pad.l, .t = opts.style.pad.t, .r = opts.style.pad.r + opts.bar, .b = opts.style.pad.b });
    const scroll = try self.scrollRegion(key, r, view, total, opts.bar, opts.style.pad);
    if (view.h <= 0 or pitch <= 0) return;

    // The visible range: two divisions, not a walk. Row `i` is visible when
    // `i * pitch < scroll + view.h`, so the exclusive end is that quotient
    // rounded up; `first` is the same quotient rounded down. No slack row,
    // because a row that cannot be seen is a row that should not be built.
    const first_f = @floor(scroll / pitch);
    const first: usize = @min(rows, @as(usize, @intFromFloat(@max(0, first_f))));
    const last: usize = @min(rows, @as(usize, @intFromFloat(@max(0, @ceil((scroll + view.h) / pitch)))));
    if (last <= first) return;

    const row_style = opts.row_style;
    const previous_clip = self.clip;
    try self.pushClip(view);
    defer self.popClip(previous_clip) catch {};

    var i = first;
    while (i < last) : (i += 1) {
        const row_y = view.y - scroll + @as(f32, @floatFromInt(i)) * pitch;
        const box = Rect{
            .x = view.x + row_style.pad.l,
            .y = row_y + row_style.pad.t,
            .w = @max(0, view.w - row_style.pad.sum(.x)),
            .h = @max(0, row_h - row_style.pad.sum(.y)),
        };
        // A layout frame per row, sized by the caller's `row_height`: no
        // measure pass, no intermediate container, and the next row's
        // position comes from its index rather than from what this one used.
        self.pushLayout(.{
            .mode = .draw,
            .dir = row_style.dir,
            .content = box,
            .gap = row_style.gap,
            .cross_align = row_style.cross_align,
            .cursor = if (row_style.dir == .x) box.x else box.y,
            .cross = if (row_style.dir == .x) box.h else box.w,
        });
        defer self.popLayout();
        self.pushIdIndex(i);
        defer self.popId();
        try rowFn(ctx, self, i, box);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "virtual options defaults" {
    const opts = VirtualOptions{};
    try std.testing.expectEqual(@as(f32, 22), opts.row_height);
    try std.testing.expectEqual(@as(f32, 10), opts.bar);
}
