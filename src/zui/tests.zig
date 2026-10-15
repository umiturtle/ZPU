//! Integration tests for the widget layer: layout geometry, the interaction
//! state machine, the idle rule, focus order, scrolling and text editing.
//!
//! These run headless — no window, no GPU and no clock — because everything the
//! library does is a function of `(Input, retained state)`. A test drives frames
//! by handing it inputs and reads the result out of the display list: exactly
//! the structure a renderer or a GPU backend consumes.
const std = @import("std");
const zui = @import("../zui.zig");

const DisplayList = zui.DisplayList;
const Input = zui.Input;
const Key = zui.Key;
const Rect = zui.Rect;
const Color = zui.Color;
const TextField = zui.TextField;
const Ui = zui.Ui;
/// The only error a UI body can produce: layout and painting allocate.
const Error = zui.Ui.Error;
const testing = std.testing;

/// The pointer at `(x, y)`, pressed on this frame, then held, then released:
/// a whole click in three values.
fn press(x: f32, y: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y }, .pointer_down = true, .pointer_pressed = true, .time = 0.1 };
}
fn hold(x: f32, y: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y }, .pointer_down = true, .time = 0.2 };
}
fn release(x: f32, y: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y }, .pointer_released = true, .time = 0.3 };
}
/// A frame with no input at all: the pointer parked outside every widget, so
/// nothing is hovered and the frame can be skipped.
fn idle() Input {
    return .{ .pointer = .{ .x = 10000, .y = 10000 }, .time = 1 };
}
fn keyed(code: zui.KeyCode) Input {
    return .{ .keys = &.{.{ .code = code }}, .time = 2 };
}
fn typed(cp: u21) Input {
    return .{ .text = &.{cp}, .time = 2 };
}

/// Runs idle frames until the library can skip one, or gives up after `limit`.
/// This is what a host's event loop does while a hover glow settles; a test that
/// wants to see the idle rule kick in has to let the animation finish first.
fn settle(ui: *Ui, body: anytype, limit: u32) !u32 {
    var frames: u32 = 0;
    while (frames < limit and !ui.can_skip) : (frames += 1) _ = try ui.run(idle(), body);
    return frames;
}

/// Every rectangle in the list, in order: the geometry a test asserts on.
fn rects(alloc: std.mem.Allocator, list: *const DisplayList) ![]Rect {
    var out = std.ArrayList(Rect).empty;
    errdefer out.deinit(alloc);
    for (list.commands.items) |c| {
        switch (c) {
            .rect => |r| try out.append(alloc, r.r),
            .rounded => |r| try out.append(alloc, r.r),
            else => {},
        }
    }
    return out.toOwnedSlice(alloc);
}

/// The text runs in the list, in order.
fn texts(alloc: std.mem.Allocator, list: *const DisplayList) ![][]const u8 {
    var out = std.ArrayList([]const u8).empty;
    errdefer out.deinit(alloc);
    for (list.commands.items) |c| {
        switch (c) {
            .text => |t| try out.append(alloc, t.text),
            else => {},
        }
    }
    return out.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------- test bodies
//
// A struct with a `run` method *is* a closure in Zig: the instance holds the
// captures and costs nothing to build. These are the bodies the tests drive.

const TwoButtons = struct {
    clicked: *usize = undefined,
    pub fn run(self: @This(), u: *Ui) Error!void {
        if (try u.button("one", .{ .w = .{ .px = 40 } })) self.clicked.* += 1;
        if (try u.button("two", .{ .w = .{ .px = 40 } })) self.clicked.* += 1;
    }
};

const PaddedColumn = struct {
    pub fn run(self: @This(), u: *Ui) Error!void {
        _ = self;
        try u.container("panel", .{
            .w = .{ .px = 100 },
            .h = .{ .px = 90 },
            .pad = .all(10),
            .gap = 4,
            .bg = Color.hex(0x202020, 1),
        }, Inner{});
    }
    const Inner = struct {
        pub fn run(_: @This(), u: *Ui) Error!void {
            _ = try u.button("one", .{ .w = .{ .px = 80 } });
            _ = try u.button("two", .{ .w = .{ .px = 80 } });
        }
    };
};

const FillRow = struct {
    pub fn run(self: @This(), u: *Ui) Error!void {
        _ = self;
        try u.container("row", .{
            .dir = .x,
            .w = .{ .px = 200 },
            .h = .{ .px = 30 },
            .gap = 0,
            .bg = Color.hex(0x202020, 1),
        }, Inner{});
    }
    const Inner = struct {
        pub fn run(_: @This(), u: *Ui) Error!void {
            _ = try u.button("fixed", .{ .w = .{ .px = 80 } });
            try u.container("flex", .{ .w = .full, .h = .full, .bg = Color.hex(0x102030, 1) }, Empty{});
        }
    };
    const Empty = struct {
        pub fn run(_: @This(), u: *Ui) Error!void {
            _ = u;
        }
    };
};

const Disabled = struct {
    clicked: *usize = undefined,
    pub fn run(self: @This(), u: *Ui) Error!void {
        if (try u.button("no", .{ .w = .{ .px = 60 }, .enabled = false })) self.clicked.* += 1;
    }
};

const OneSlider = struct {
    value: *f32 = undefined,
    changed: *usize = undefined,
    pub fn run(self: @This(), u: *Ui) Error!void {
        if (try u.slider("gain", self.value, .{ .min = 0, .max = 10, .w = .{ .px = 100 }, .show_value = false })) {
            self.changed.* += 1;
        }
    }
};

const TallColumn = struct {
    pub fn run(self: @This(), u: *Ui) Error!void {
        _ = self;
        try u.scrollable("list", .{ .style = .{
            .w = .{ .px = 100 },
            .h = .{ .px = 50 },
            .clip = true,
            .bg = Color.hex(0x202020, 1),
        } }, Inner{});
    }
    const Inner = struct {
        pub fn run(_: @This(), u: *Ui) Error!void {
            for (0..5) |i| {
                const label = switch (i) {
                    0 => "row0",
                    1 => "row1",
                    2 => "row2",
                    3 => "row3",
                    else => "row4",
                };
                try u.container(label, .{ .w = .full, .h = .{ .px = 20 }, .bg = Color.hex(0x223344, 1) }, Empty{});
            }
        }
    };
    const Empty = struct {
        pub fn run(_: @This(), u: *Ui) Error!void {
            _ = u;
        }
    };
};

const Collapsible = struct {
    pub fn run(self: @This(), u: *Ui) Error!void {
        _ = self;
        _ = try u.collapsing("section", "Header", .{}, Inner{});
    }
    const Inner = struct {
        pub fn run(_: @This(), u: *Ui) Error!void {
            try u.label("hidden text", .{});
        }
    };
};

const Field = struct {
    field: *TextField = undefined,
    changed: *usize = undefined,
    pub fn run(self: @This(), u: *Ui) Error!void {
        if (try u.textEdit("name", self.field, .{ .w = .{ .px = 120 }, .placeholder = "type" })) self.changed.* += 1;
    }
};

// ------------------------------------------------------------------- the tests

test "a padded column places its children with gaps" {
    var ui = Ui.init(testing.allocator, .{ .width = 120, .height = 100 });
    defer ui.deinit();

    _ = try ui.run(idle(), PaddedColumn{});
    const rs = try rects(testing.allocator, &ui.list);
    defer testing.allocator.free(rs);

    // panel, then two buttons inside it
    try testing.expectEqual(@as(usize, 3), rs.len);
    try testing.expectEqual(Rect.init(0, 0, 100, 90), rs[0]);
    try testing.expectEqual(Rect.init(10, 10, 80, 26), rs[1]);
    try testing.expectEqual(Rect.init(10, 40, 80, 26), rs[2]);
}

test "a fill child takes what the fixed children leave" {
    var ui = Ui.init(testing.allocator, .{ .width = 220, .height = 40 });
    defer ui.deinit();

    _ = try ui.run(idle(), FillRow{});
    const rs = try rects(testing.allocator, &ui.list);
    defer testing.allocator.free(rs);

    // row, fixed button (50 px), flex container (the rest)
    try testing.expectEqual(@as(usize, 3), rs.len);
    try testing.expectEqual(@as(f32, 80), rs[1].w);
    try testing.expectEqual(@as(f32, 120), rs[2].w);
    try testing.expectEqual(@as(f32, 80), rs[2].x);
    // ...and the row's cross extent is filled too.
    try testing.expectEqual(@as(f32, 30), rs[2].h);
}

test "a click fires once, and only for a press that started inside" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();
    var clicks: usize = 0;

    // Press and release inside the first button.
    _ = try ui.run(press(20, 13), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 0), clicks);
    _ = try ui.run(release(20, 13), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 1), clicks);

    // Release outside the button: no second click.
    _ = try ui.run(press(20, 13), TwoButtons{ .clicked = &clicks });
    _ = try ui.run(release(150, 90), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 1), clicks);

    // Press outside and release inside: still no click — the press owns the
    // gesture, and it never belonged to this widget.
    _ = try ui.run(press(150, 90), TwoButtons{ .clicked = &clicks });
    _ = try ui.run(release(20, 13), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 1), clicks);

    // The second button sits below the first one and works the same way.
    _ = try ui.run(press(20, 45), TwoButtons{ .clicked = &clicks });
    _ = try ui.run(release(20, 45), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 2), clicks);
}

test "a disabled widget takes no input" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();
    var clicks: usize = 0;

    _ = try ui.run(press(20, 13), Disabled{ .clicked = &clicks });
    _ = try ui.run(release(20, 13), Disabled{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 0), clicks);
    // ...and it does not report itself as hot either.
    try testing.expect(ui.ctx.hot.h == 0);
}

test "the frame is reused when nothing can have changed" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();
    var clicks: usize = 0;

    const first = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    try testing.expect(first.rebuilt);
    try testing.expect(!first.skipped);
    const commands = first.commands;
    const revision = first.revision;

    // A frame with no input at all: same revision, no rebuild, same commands.
    const second = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    try testing.expect(second.skipped);
    try testing.expect(!second.rebuilt);
    try testing.expectEqual(revision, second.revision);
    try testing.expectEqual(commands, second.commands);
    try testing.expect(ui.list.reused);

    // Input brings it back: the body runs and the revision moves.
    _ = try ui.run(press(20, 13), TwoButtons{ .clicked = &clicks });
    try testing.expect(ui.list.revision > revision);

    // ...and so does an explicit "my data changed".
    _ = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    const before = ui.list.revision;
    ui.markDirty();
    _ = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(before + 1, ui.list.revision);

    // An animation keeps the frame alive even with no input: hover a button and
    // the glow has to have time to move.
    _ = try ui.run(hold(20, 13), TwoButtons{ .clicked = &clicks });
    const hovering = ui.list.revision;
    const still = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    try testing.expect(still.rebuilt);
    try testing.expect(ui.list.revision > hovering);
}

test "hover and press are retained states, not per-frame ones" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();
    var clicks: usize = 0;

    _ = try ui.run(hold(20, 13), TwoButtons{ .clicked = &clicks });
    const hot = ui.ctx.hot;
    try testing.expect(!hot.isNone());

    // Move away: the pointer is elsewhere, so nothing is hot any more.
    _ = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    try testing.expect(ui.ctx.hot.isNone());

    // Press inside and hold: the widget owns the press, and keeps owning it even
    // when the pointer leaves the widget, until the button is released.
    _ = try ui.run(press(20, 13), TwoButtons{ .clicked = &clicks });
    const active = ui.ctx.active;
    try testing.expect(!active.isNone());
    _ = try ui.run(hold(150, 90), TwoButtons{ .clicked = &clicks });
    try testing.expect(ui.ctx.active.eql(active));
}

test "tab moves focus in build order and the keyboard activates" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();
    var clicks: usize = 0;

    // Frame 1 builds the focus ring.
    _ = try ui.run(idle(), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 2), ui.ctx.focusables.items.len);
    try testing.expect(ui.ctx.focus.isNone());

    // Tab selects the first, Tab again the second, wrapping around.
    _ = try ui.run(keyed(.tab), TwoButtons{ .clicked = &clicks });
    const first = ui.ctx.focus;
    try testing.expect(!first.isNone());
    _ = try ui.run(keyed(.tab), TwoButtons{ .clicked = &clicks });
    const second = ui.ctx.focus;
    try testing.expect(!second.eql(first));
    _ = try ui.run(keyed(.tab), TwoButtons{ .clicked = &clicks });
    try testing.expect(ui.ctx.focus.eql(first));

    // `enter` on the focused widget activates it, exactly like a click.
    _ = try ui.run(keyed(.enter), TwoButtons{ .clicked = &clicks });
    try testing.expectEqual(@as(usize, 1), clicks);
}

test "a slider maps the pointer onto its range" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 60 });
    defer ui.deinit();
    var value: f32 = 0;
    var changes: usize = 0;
    const body = OneSlider{ .value = &value, .changed = &changes };

    _ = try ui.run(press(50, 13), body);
    const low = value;
    _ = try ui.run(hold(120, 13), body);
    const high = value;
    try testing.expect(low > 0);
    try testing.expect(high > low);

    // Dragging past either end clamps instead of wrapping.
    _ = try ui.run(hold(-100, 13), body);
    try testing.expectEqual(@as(f32, 0), value);
    _ = try ui.run(hold(1000, 13), body);
    try testing.expectEqual(@as(f32, 10), value);

    // A release ends the drag: moving afterwards must not change anything.
    _ = try ui.run(release(1000, 13), body);
    _ = try ui.run(hold(50, 13), body);
    try testing.expectEqual(@as(f32, 10), value);
    try testing.expect(changes > 0);
}

test "a scroll view moves its children and clamps at both ends" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();

    _ = try ui.run(idle(), TallColumn{});
    const before = try rects(testing.allocator, &ui.list);
    defer testing.allocator.free(before);
    // The region paints itself, then its rail and thumb, then the rows: 5 rows
    // of 20 px in a 50 px view, the first row at the top.
    try testing.expectEqual(Rect.init(0, 0, 100, 50), before[0]);
    try testing.expectEqual(@as(f32, 0), before[3].y);
    try testing.expectEqual(@as(f32, 20), before[3].h);

    // A wheel event scrolls down by exactly what it says (clamped to the end).
    var wheel = Input{ .scroll = .{ .x = 0, .y = 30 }, .time = 3 };
    _ = try ui.run(wheel, TallColumn{});
    var after = try rects(testing.allocator, &ui.list);
    try testing.expectEqual(@as(f32, -30), after[3].y);
    testing.allocator.free(after);

    // Scrolling past the end clamps at `content - view` (100 - 50 = 50).
    wheel = Input{ .scroll = .{ .x = 0, .y = 1000 }, .time = 4 };
    _ = try ui.run(wheel, TallColumn{});
    after = try rects(testing.allocator, &ui.list);
    try testing.expectEqual(@as(f32, -50), after[3].y);
    testing.allocator.free(after);

    // ...and scrolling back up stops at zero.
    wheel = Input{ .scroll = .{ .x = 0, .y = -1000 }, .time = 5 };
    _ = try ui.run(wheel, TallColumn{});
    after = try rects(testing.allocator, &ui.list);
    try testing.expectEqual(@as(f32, 0), after[3].y);
    testing.allocator.free(after);
}

test "a collapsing header hides and shows its body, and remembers" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 100 });
    defer ui.deinit();

    _ = try ui.run(idle(), Collapsible{});
    var t = try texts(testing.allocator, &ui.list);
    defer testing.allocator.free(t);
    try testing.expectEqual(@as(usize, 1), t.len);
    try testing.expectEqualStrings("Header", t[0]);

    // Click the header: the body appears.
    _ = try ui.run(press(40, 13), Collapsible{});
    _ = try ui.run(release(40, 13), Collapsible{});
    testing.allocator.free(t);
    t = try texts(testing.allocator, &ui.list);
    try testing.expectEqual(@as(usize, 2), t.len);
    try testing.expectEqualStrings("hidden text", t[1]);

    // The open flag survives frames with no input at all. The frames right after
    // the click still run — the release was input, and then the hover glow has
    // to settle — but once it has, the frame is skipped and the picture reused
    // exactly as it stands.
    const frames = try settle(&ui, Collapsible{}, 240);
    try testing.expect(frames > 0);
    try testing.expect(frames < 240);
    const result = try ui.run(idle(), Collapsible{});
    try testing.expect(result.skipped);
    try testing.expectEqual(@as(usize, 2), t.len);

    // Click again to close.
    _ = try ui.run(press(40, 13), Collapsible{});
    _ = try ui.run(release(40, 13), Collapsible{});
    testing.allocator.free(t);
    t = try texts(testing.allocator, &ui.list);
    try testing.expectEqual(@as(usize, 1), t.len);
}

test "typing into a field inserts, deletes and keeps the caret" {
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 60 });
    defer ui.deinit();
    var buf: [32]u8 = undefined;
    var field = TextField.init(&buf);
    var changes: usize = 0;
    const body = Field{ .field = &field, .changed = &changes };

    _ = try ui.run(idle(), body);
    // Nothing to show yet: the placeholder is drawn instead of the value.
    var t = try texts(testing.allocator, &ui.list);
    try testing.expectEqual(@as(usize, 1), t.len);
    try testing.expectEqualStrings("type", t[0]);

    // Focus it by clicking, then type.
    _ = try ui.run(press(10, 13), body);
    _ = try ui.run(release(10, 13), body);
    _ = try ui.run(typed('h'), body);
    _ = try ui.run(typed('i'), body);
    try testing.expectEqualStrings("hi", field.slice());
    try testing.expectEqual(@as(usize, 2), changes);

    // Backspace removes the last character, and the caret follows.
    _ = try ui.run(keyed(.backspace), body);
    try testing.expectEqualStrings("h", field.slice());
    try testing.expectEqual(@as(usize, 1), ui.ctx.edit(ui.idOf("name")).caret);

    // Home/end move the caret without touching the text; typing at the caret
    // inserts in the middle.
    _ = try ui.run(keyed(.home), body);
    try testing.expectEqual(@as(usize, 0), ui.ctx.edit(ui.idOf("name")).caret);
    _ = try ui.run(typed('X'), body);
    try testing.expectEqualStrings("Xh", field.slice());

    testing.allocator.free(t);
    t = try texts(testing.allocator, &ui.list);
    defer testing.allocator.free(t);
    try testing.expectEqualStrings("Xh", t[0]);
}

test "the display list is balanced and the tree pops itself" {
    var ui = Ui.init(testing.allocator, .{ .width = 120, .height = 100 });
    defer ui.deinit();

    _ = try ui.run(idle(), PaddedColumn{});
    try testing.expect(ui.list.isBalanced());
    try testing.expectEqual(@as(usize, 0), ui.depth);
    try testing.expectEqual(@as(usize, 0), ui.id_top);

    // An empty frame still reports the frame count, so a host can measure its
    // own loop.
    try testing.expectEqual(@as(u64, 1), ui.stats().frames);
    _ = try ui.run(idle(), PaddedColumn{});
    try testing.expectEqual(@as(u64, 2), ui.stats().frames);
    try testing.expectEqual(@as(u64, 1), ui.stats().skipped);
}

test "a label wraps inside its container and the lines are all drawn" {
    const Long = struct {
        pub fn run(self: @This(), u: *Ui) Error!void {
            _ = self;
            try u.container("box", .{ .w = .{ .px = 80 }, .h = .{ .px = 60 } }, Inner{});
        }
        const Inner = struct {
            pub fn run(_: @This(), u: *Ui) Error!void {
                try u.label("the quick brown fox jumps", .{});
            }
        };
    };
    var ui = Ui.init(testing.allocator, .{ .width = 100, .height = 100 });
    defer ui.deinit();

    _ = try ui.run(idle(), Long{});
    const t = try texts(testing.allocator, &ui.list);
    defer testing.allocator.free(t);
    try testing.expect(t.len >= 3);
    for (t) |line| {
        const w = zui.font.measure(line, ui.theme.font_size).width;
        try testing.expect(w <= 80 + 1e-3);
    }
}

/// The baseline of every text run in the list, in order: how a test sees where
/// rows landed.
fn textBaselines(alloc: std.mem.Allocator, list: *const DisplayList) ![]f32 {
    var out = std.ArrayList(f32).empty;
    errdefer out.deinit(alloc);
    for (list.commands.items) |c| {
        switch (c) {
            .text => |t| try out.append(alloc, t.pos.y),
            else => {},
        }
    }
    return out.toOwnedSlice(alloc);
}

const BigList = struct {
    rows: usize,
    built: *usize,

    pub fn run(self: @This(), u: *Ui) Error!void {
        try u.virtualList("big", self.rows, .{
            .style = .{ .w = .{ .px = 100 }, .h = .{ .px = 48 }, .pad = .all(2) },
            .row_height = 20,
        }, self, row);
    }

    fn row(self: @This(), u: *Ui, i: usize, box: Rect) Error!void {
        self.built.* += 1;
        var buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&buf, "row {d}", .{i}) catch unreachable;
        // The row paints its own background: it owns its box, and nothing
        // measured it.
        try u.paintRect(box, Color.hex(0x222222, 1));
        try u.label(label, .{});
    }
};

test "a virtual list builds only the rows you can see" {
    var built: usize = 0;
    var ui = Ui.init(testing.allocator, .{ .width = 120, .height = 60 });
    defer ui.deinit();

    _ = try ui.run(idle(), BigList{ .rows = 1_000_000, .built = &built });
    // A million rows, and the frame built a handful of them: the rows off screen
    // are never called, which is the entire reason the widget exists.
    try testing.expect(built > 0);
    try testing.expect(built < 6);
    try testing.expect(ui.list.isBalanced());
    // The region paints itself, its rail and its thumb, and then the rows.
    const all = try rects(testing.allocator, &ui.list);
    defer testing.allocator.free(all);
    // The bar is there (a million rows do not fit into 48 px), and the first
    // row starts inside the region's padding. The region's own box is a
    // `rounded` command, so the first plain rectangle is the rail.
    try testing.expectEqual(Rect.init(90, 2, 6, 44), all[0]);
    try testing.expectEqual(Rect.init(2, 2, 86, 20), all[2]);
    try testing.expectEqual(@as(usize, 5), all.len);
}

test "a virtual list places rows by their index, scrolled or not" {
    var built: usize = 0;
    var ui = Ui.init(testing.allocator, .{ .width = 120, .height = 60 });
    defer ui.deinit();

    _ = try ui.run(idle(), BigList{ .rows = 1000, .built = &built });
    const top = try textBaselines(testing.allocator, &ui.list);
    defer testing.allocator.free(top);
    try testing.expect(top.len >= 3);
    // One pitch apart, because the row height is what the list was told.
    try testing.expectApproxEqAbs(@as(f32, 20), top[1] - top[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 20), top[2] - top[1], 1e-4);

    // A 5 px wheel event moves every row by exactly 5 px — no reflow, no
    // rounding: the offset is subtracted from the index-derived position.
    const wheel = Input{ .pointer = .{ .x = 50, .y = 20 }, .scroll = .{ .x = 0, .y = 5 }, .time = 5 };
    built = 0;
    _ = try ui.run(wheel, BigList{ .rows = 1000, .built = &built });
    const scrolled = try textBaselines(testing.allocator, &ui.list);
    defer testing.allocator.free(scrolled);
    try testing.expectApproxEqAbs(top[0] - 5, scrolled[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 20), scrolled[1] - scrolled[0], 1e-4);
    // Still only the visible rows.
    try testing.expect(built < 6);
}

test "a tooltip is drawn last, and only while its widget is hot" {
    const Body = struct {
        anchor: Rect,

        pub fn run(self: @This(), u: *Ui) Error!void {
            _ = try u.button("hover me", .{});
            // A button's id is the hash of its content, and it is the id that
            // says which widget's hotness shows the tip.
            try u.tooltip(u.idOf("hover me"), self.anchor, "tip");
        }
    };
    var ui = Ui.init(testing.allocator, .{ .width = 200, .height = 120 });
    defer ui.deinit();

    // Hovered: the tip is queued, and it is the last thing in the list — that is
    // what "on top" means for a flat command list.
    const body = Body{ .anchor = Rect.init(10, 10, 40, 20) };
    _ = try ui.run(.{ .pointer = .{ .x = 20, .y = 13 }, .time = 1 }, body);
    const t = try texts(testing.allocator, &ui.list);
    defer testing.allocator.free(t);
    try testing.expect(t.len >= 2);
    try testing.expectEqualStrings("tip", t[t.len - 1]);

    // Not hovered: no tip anywhere, not even a stale one from last frame.
    _ = try ui.run(idle(), body);
    const away = try texts(testing.allocator, &ui.list);
    defer testing.allocator.free(away);
    for (away) |line| try testing.expect(!std.mem.eql(u8, line, "tip"));

    // The anchor is inside the widget, so a tip below it never covers the thing
    // it describes; near the bottom edge it flips above instead of leaving the
    // window.
    const low = Body{ .anchor = Rect.init(10, 100, 40, 20) };
    _ = try ui.run(.{ .pointer = .{ .x = 20, .y = 13 }, .time = 2 }, low);
    const flipped = try textBaselines(testing.allocator, &ui.list);
    defer testing.allocator.free(flipped);
    try testing.expect(flipped.len >= 2);
    try testing.expect(flipped[flipped.len - 1] < 100);
}
