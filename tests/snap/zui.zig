//! zui snapshots: the frames a script produces.
//!
//! A UI is the hardest thing in this repository to test by assertion alone,
//! because "the picture is right" is not a property — it is a reference. These
//! snapshots are that reference, at three levels of the same frame:
//!
//! * the **display list** — the commands and their coordinates, exactly;
//! * the **display tree** — the same commands nested by clip, so the structure
//!   of the layout is visible rather than inferred from a flat stream;
//! * the **instances** — the display list after the walk that turns it into
//!   drawable quads, which is where the renderers actually consume it;
//! * the **pixels** — the software renderer's output, as ASCII luminance.
//!
//! All four come from one frame built from scripted input, so a widget that
//! moves by one pixel shows up as a diff in all four and the reviewer can pick
//! the level that explains it.
const std = @import("std");
const zgf = @import("zgf");
const zpu = @import("zpu");
const zui = @import("zui");
const snap = @import("../snapshot.zig");

const Snap = snap.Snap;
const Input = zui.Input;
const Rect = zui.Rect;
const Ui = zui.Ui;
const alloc = std.testing.allocator;

const width: f32 = 240;
const height: f32 = 160;

/// The application state, which in an immediate API *is* the UI: there is no
/// shadow tree to keep in step with the picture.
const App = struct {
    clicks: usize = 0,
    hit_id: zui.Id = .none,
    toggled: bool = false,
    radio: usize = 0,
    gain: f32 = 0.25,
    buf: [32]u8 = undefined,
    field: zui.TextField = undefined,

    fn init(self: *App) void {
        self.field = zui.TextField.init(&self.buf);
    }
};

/// The frame body: a struct with `run(self, ui)` — Zig's closure.
const Body = struct {
    app: *App,

    pub fn run(self: @This(), u: *Ui) Ui.Error!void {
        try u.container("root", .{
            .w = .full,
            .h = .full,
            .pad = .all(10),
            .gap = 6,
            .bg = u.theme.bg,
        }, Root{ .app = self.app });
    }

    const Root = struct {
        app: *App,

        pub fn run(self: @This(), u: *Ui) Ui.Error!void {
            const id = u.idOf("hit");
            self.app.hit_id = id;
            if (try u.buttonId(id, "Click me", .{ .w = .{ .px = 90 } })) self.app.clicks += 1;
            try u.container("toggles", .{ .dir = .x, .gap = 10, .w = .full, .h = .{ .px = 16 } }, Toggles{ .app = self.app });
            try u.container("values", .{ .dir = .x, .gap = 10, .w = .full, .h = .{ .px = 14 } }, Values{ .app = self.app });
            _ = try u.textEdit("name", &self.app.field, .{ .placeholder = "type a word" });
            try u.virtualList("rows", row_count, .{
                .style = .{
                    .w = .full,
                    .h = .{ .px = 34 },
                    .gap = 1,
                    .bg = u.theme.surface.withAlpha(0.6),
                    .border = 1,
                    .border_color = u.theme.border,
                    .radius = 4,
                },
                .row_height = 10,
                .row_style = .{ .pad = .xy(4, 0) },
            }, Rows{}, Rows.row);
        }

        const row_count = 40;
    };

    /// The toggles sit in a row so the frame fits a small viewport: the layout
    /// itself is part of what the snapshot pins.
    const Toggles = struct {
        app: *App,

        pub fn run(self: @This(), u: *Ui) Ui.Error!void {
            _ = try u.checkbox("enabled", &self.app.toggled, .{});
            _ = try u.radio("low", &self.app.radio, 0, .{});
            _ = try u.radio("high", &self.app.radio, 1, .{});
        }
    };

    const Values = struct {
        app: *App,

        pub fn run(self: @This(), u: *Ui) Ui.Error!void {
            _ = try u.slider("gain", &self.app.gain, .{ .min = 0, .max = 1, .w = .{ .px = 120 } });
            try u.progress(self.app.gain, .{ .w = 90 });
        }
    };

    const Rows = struct {
        fn row(_: @This(), u: *Ui, i: usize, box: Rect) Ui.Error!void {
            const bg: f32 = if (i % 2 == 0) 0.75 else 0.4;
            try u.paintRounded(box, 2, u.theme.widget.withAlpha(bg), 0, zui.Color.transparent);
            var buf: [16]u8 = undefined;
            try u.label(std.fmt.bufPrint(&buf, "row {d}", .{i}) catch unreachable, .{});
        }
    };
};

// ------------------------------------------------------------------ the script

fn parked() Input {
    return .{ .pointer = .{ .x = 10000, .y = 10000 } };
}
fn moved(x: f32, y: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y } };
}
fn down(x: f32, y: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y }, .pointer_down = true, .pointer_pressed = true };
}
fn up(x: f32, y: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y }, .pointer_released = true };
}
fn typed(cp: u21) Input {
    return .{ .pointer = .{ .x = 10000, .y = 10000 }, .text = &.{cp} };
}

/// Finds where a painted string sits: widgets do not exist outside the frame, so
/// a host — or a test — reads the widget's anchor back out of the display list.
fn textAnchor(ui: *const Ui, needle: []const u8) ?zui.Vec2 {
    for (ui.list.commands.items) |c| {
        switch (c) {
            .text => |t| {
                if (!std.mem.eql(u8, t.text, needle)) continue;
                const m = zui.font.measure(t.text, t.size);
                return .{ .x = t.pos.x + m.width * 0.5, .y = t.pos.y - m.ascent + m.height * 0.5 };
            },
            else => {},
        }
    }
    return null;
}

/// The slider's track is the only rounded box of its height, which is how the
/// script finds it without the widget existing outside the frame.
fn findRounded(ui: *const Ui, h: f32) ?Rect {
    for (ui.list.commands.items) |c| {
        switch (c) {
            .rounded => |r| if (r.r.h == h) return r.r,
            else => {},
        }
    }
    return null;
}

test "snapshot: one frame, four ways" {
    var s = Snap.init(alloc);
    var app = App{};
    app.init();
    var ui = Ui.init(alloc, .{ .width = width, .height = height });
    defer ui.deinit();

    _ = try ui.run(parked(), Body{ .app = &app });

    try s.expectFmt("zui/frame/summary", "commands {d} balanced {} bounds {d:.2},{d:.2} {d:.2}x{d:.2}\n", .{
        ui.list.len(),
        ui.list.isBalanced(),
        ui.list.bounds.x,
        ui.list.bounds.y,
        ui.list.bounds.w,
        ui.list.bounds.h,
    });
    try s.expectDisplayList("zui/frame/display-list", &ui.list);
    try s.expectFmt("zui/frame/tree", "{f}", .{snap.Tree{ .list = &ui.list, .name = "root" }});

    var scratch: [4096]zui.Instance = undefined;
    const insts = zui.instances.collect(&ui.list, .{}, &scratch);
    try s.expectInstances("zui/frame/instances", insts, 400);

    var img = try zgf.Image.init(alloc, @as(usize, @intFromFloat(width)), @as(usize, @intFromFloat(height)), 3);
    defer img.deinit();
    zui.render.draw(&ui.list, &img, .{ .clear = ui.theme.bg });
    try s.expectImage("zui/frame/pixels", &img, 80, 32);
    try s.finish();
}

test "snapshot: a scripted interaction" {
    var s = Snap.init(alloc);
    var app = App{};
    app.init();
    var ui = Ui.init(alloc, .{ .width = width, .height = height });
    defer ui.deinit();
    const body = Body{ .app = &app };

    // The interaction a host would script: hover the button, press it, toggle the
    // checkbox, drag the slider, type a word — and then let a frame with no
    // events at all run, which must reuse the picture it already has.
    _ = try ui.run(parked(), body);
    const button = textAnchor(&ui, "Click me") orelse return error.MissingButton;
    _ = try ui.run(moved(button.x, button.y), body);
    _ = try ui.run(down(button.x, button.y), body);
    _ = try ui.run(up(button.x, button.y), body);

    const box = textAnchor(&ui, "enabled") orelse return error.MissingCheckbox;
    _ = try ui.run(down(box.x - 6, box.y), body);
    _ = try ui.run(up(box.x - 6, box.y), body);

    const track = findRounded(&ui, 6) orelse return error.MissingSlider;
    const track_y = track.y + track.h * 0.5;
    _ = try ui.run(down(track.x + track.w * 0.3, track_y), body);
    _ = try ui.run(moved(track.x + track.w * 0.8, track_y), body);
    _ = try ui.run(up(track.x + track.w * 0.8, track_y), body);

    const field = textAnchor(&ui, "type a word") orelse return error.MissingField;
    _ = try ui.run(down(field.x, field.y), body);
    _ = try ui.run(up(field.x, field.y), body);
    _ = try ui.run(typed('z'), body);
    _ = try ui.run(typed('i'), body);
    _ = try ui.run(typed('g'), body);

    const idle = try ui.run(parked(), body);
    try s.expectFmt(
        "zui/interaction/state",
        "clicks {d} toggled {} radio {d} gain {d:.4} text \"{s}\" skipped {} revision {d}\n",
        .{ app.clicks, app.toggled, app.radio, app.gain, app.field.slice(), idle.skipped, ui.list.revision },
    );
    try s.expectDisplayList("zui/interaction/display-list", &ui.list);
    try s.expectFmt("zui/interaction/tree", "{f}", .{snap.Tree{ .list = &ui.list, .name = "interaction" }});

    var img = try zgf.Image.init(alloc, @as(usize, @intFromFloat(width)), @as(usize, @intFromFloat(height)), 3);
    defer img.deinit();
    zui.render.draw(&ui.list, &img, .{ .clear = ui.theme.bg });
    try s.expectImage("zui/interaction/pixels", &img, 80, 32);

    // The same list at 2× device pixels: the geometry doubles, the *logical*
    // coordinates in the display list do not.
    var big = try zgf.Image.init(alloc, @as(usize, @intFromFloat(width)) * 2, @as(usize, @intFromFloat(height)) * 2, 3);
    defer big.deinit();
    zui.render.draw(&ui.list, &big, .{ .scale = 2, .clear = ui.theme.bg });
    try s.expectImage("zui/interaction/pixels-2x", &big, 80, 32);
    try s.finish();
}

test "snapshot: the glyph atlas is a generated artifact" {
    var s = Snap.init(alloc);
    try zpu.init(alloc, .{});
    defer zpu.deinit();

    // The atlas is built from the embedded font, so its contents are a property
    // of the font data and the rasteriser. Pinning it means a change in either
    // is a reviewable diff instead of a mysterious pixel shift.
    const pixels = zui.pipeline.atlasPixels();
    const w = zui.pipeline.atlas_size[0];
    const h = zui.pipeline.atlas_size[1];
    try s.expectFmt("zui/atlas/size", "atlas {d}x{d} pixels {d}\n", .{ w, h, pixels.len });

    var img = try zgf.Image.init(alloc, w, h, 1);
    defer img.deinit();
    for (0..h) |y| {
        for (0..w) |x| {
            const v: f32 = @as(f32, @floatFromInt(pixels[y * w + x])) / 255.0;
            img.setPixel(@intCast(x), @intCast(y), .{ .r = v, .g = v, .b = v });
        }
    }
    try s.expectImage("zui/atlas/pixels", &img, 96, 24);
    try s.finish();
}

test "snapshot: layout rules resolve to numbers" {
    var s = Snap.init(alloc);
    var ui = Ui.init(alloc, .{ .width = width, .height = height });
    defer ui.deinit();

    // A layout-only frame: the boxes a set of containers and alignment rules
    // produce, with the text stripped to what matters (a label has no text in
    // this frame, so the list is pure geometry).
    const Layouts = struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.container("a", .{ .w = .full, .h = .full, .pad = .all(8), .gap = 4, .bg = u.theme.bg }, Inner{});
        }
        const Inner = struct {
            pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                try u.container("row", .{ .dir = .x, .gap = 4, .h = .{ .px = 20 }, .bg = u.theme.surface }, Row{});
                try u.container("col", .{ .dir = .y, .gap = 2, .w = .full, .bg = u.theme.surface }, Col{});
            }
        };
        const Row = struct {
            pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                var i: usize = 0;
                while (i < 3) : (i += 1) {
                    try u.container("cell", .{ .w = .{ .px = 20 }, .h = .{ .px = 20 }, .bg = u.theme.widget }, Empty{});
                }
            }
        };
        const Col = struct {
            pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                try u.container("wide", .{ .w = .full, .h = .{ .px = 12 }, .bg = u.theme.widget }, Empty{});
                try u.container("half", .{ .w = .{ .fill = 0.5 }, .h = .{ .px = 12 }, .bg = u.theme.widget }, Empty{});
            }
        };
        /// An empty container: the box it reserves is the thing under test.
        const Empty = struct {
            pub fn run(_: @This(), u: *Ui) Ui.Error!void {
                _ = u;
            }
        };
    };

    _ = try ui.run(parked(), Layouts{});
    try s.expectDisplayList("zui/layout/boxes", &ui.list);
    try s.expectFmt("zui/layout/tree", "{f}", .{snap.Tree{ .list = &ui.list, .name = "layout" }});
    try s.finish();
}
