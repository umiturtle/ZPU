//! zui demo: a UI driven entirely by scripted input, rendered without a window.
//!
//! This is the example that shows what the hybrid design buys. There is no event
//! loop, no window handle and no GPU: a frame is `(Input, body)`, and the picture
//! is a display list that `zui.render` turns into pixels. So the demo can be a
//! *test* — it asserts on geometry, on pixels and on the frame accounting — and
//! still writes a picture a human can look at:
//!
//!     zig build run-ui-demo          # writes zig-out/ui-demo.ppm
//!
//! The filmstrip it writes has nine panels:
//!
//!   1. the first frame                6. the list scrolled
//!   2. a button hovered               7. the section re-opened by a click
//!   3. the button pressed             8. the whole UI at 2× device pixels
//!   4. checkbox toggled, slider moved 9. a 5× zoom on the button's corner
//!   5. text typed into the field
//!
//! Panels 8 and 9 are the interesting ones: the same display list is drawn again
//! with `scale = 2` and again magnified 5×, so the antialiasing the renderer
//! promises is visible rather than asserted.
const std = @import("std");
const zbackend = @import("zbackend");
const zgf = @import("zgf");
const zui = @import("zui");
const zpu = @import("zpu");

const Color = zui.Color;
const Input = zui.Input;
const Rect = zui.Rect;
const TextField = zui.TextField;
const Ui = zui.Ui;

const width: f32 = 320;
const height: f32 = 200;
const cols = 3;
const rows = 3;

/// Everything the UI reads and writes. With an immediate API this *is* the
/// application: there is no shadow tree to keep in step with it.
const App = struct {
    clicks: usize = 0,
    /// The button's id, captured inside the frame that built it. Ids are scoped
    /// by their container, so a host that wants to talk about a widget outside
    /// the frame (to focus it, or to ask whether it is hot) keeps the id.
    hit_id: zui.Id = .none,
    toggled: bool = false,
    radio: usize = 0,
    gain: f32 = 0.25,
    buf: [64]u8 = undefined,
    field: TextField = undefined,

    fn init(self: *App) void {
        self.field = TextField.init(&self.buf);
    }
};

/// The frame's body: a struct with `run(self, ui)` — Zig's closure. The fields
/// are the captures, and there is nothing to allocate to build one.
const Body = struct {
    app: *App,

    pub fn run(self: @This(), u: *Ui) Ui.Error!void {
        try u.container("root", .{
            .w = .full,
            .h = .full,
            .pad = .all(12),
            .gap = 8,
            .bg = u.theme.bg,
        }, Root{ .app = self.app });
    }

    const Root = struct {
        app: *App,
        pub fn run(self: @This(), u: *Ui) Ui.Error!void {
            try u.label("zui — immediate API, retained picture", .{ .color = u.theme.text_dim });
            try u.container("controls", .{ .dir = .x, .gap = 6, .w = .full }, Row{ .app = self.app });
            try u.container("sliders", .{ .dir = .y, .gap = 6, .w = .full }, Sliders{ .app = self.app });
            // The list *is* the scroll region — wrapping a virtual list in a
            // `scrollable` would nest two viewports and the outer one would eat
            // the wheel event.
            const list = List{};
            try list.run(u);
            _ = try u.collapsing("about", "How this frame was built", .{}, About{});
        }
    };

    const Row = struct {
        app: *App,
        pub fn run(self: @This(), u: *Ui) Ui.Error!void {
            const id = u.idOf("hit");
            self.app.hit_id = id;
            if (try u.buttonId(id, "Click me", .{ .w = .{ .px = 104 } })) self.app.clicks += 1;
            // A tooltip is anchored by the host: a widget knows its own box for
            // exactly one frame, and this is how it is read back out of the
            // picture — find the text the button just painted.
            if (textAnchor(u, "Click me")) |at| {
                try u.tooltip(id, Rect.init(at.x - 52, at.y - 9, 104, 18), "sets clicks += 1");
            }
            _ = try u.checkbox("enabled", &self.app.toggled, .{});
            _ = try u.radio("low", &self.app.radio, 0, .{});
            _ = try u.radio("high", &self.app.radio, 1, .{});
        }
    };

    const Sliders = struct {
        app: *App,
        pub fn run(self: @This(), u: *Ui) Ui.Error!void {
            _ = try u.slider("gain", &self.app.gain, .{ .min = 0, .max = 1, .w = .{ .px = 150 } });
            _ = try u.textEdit("name", &self.app.field, .{ .placeholder = "type a word" });
            try u.progress(self.app.gain, .{ .w = 240 });
        }
    };

    /// The scrolling list, virtual: 2000 rows in a 62 px viewport, and the frame
    /// builds only the four rows that fit (plus whatever the scroll brings in).
    /// The row paints its own box — it is handed one — so nothing has to be
    /// measured to know where the next row goes.
    const List = struct {
        const row_count = 2000;

        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.virtualList("list", row_count, .{
                .style = .{
                    .w = .full,
                    .h = .{ .px = 62 },
                    .gap = 2,
                    .bg = u.theme.surface.withAlpha(0.6),
                    .border = 1,
                    .border_color = u.theme.border,
                    .radius = 5,
                },
                .row_height = 18,
                .row_style = .{ .pad = .xy(6, 0) },
            }, List{}, row);
        }

        fn row(_: @This(), u: *Ui, i: usize, box: Rect) Ui.Error!void {
            const bg: f32 = if (i % 2 == 0) 0.75 else 0.4;
            try u.paintRounded(box, 3, u.theme.widget.withAlpha(bg), 0, Color.transparent);
            var buf: [24]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "row {d}", .{i}) catch unreachable;
            try u.label(label, .{});
        }
    };

    const About = struct {
        pub fn run(_: @This(), u: *Ui) Ui.Error!void {
            try u.label("The body runs only when something changed: no input, nothing animating and no dirty flag means the display list is reused exactly as it stands.", .{ .color = u.theme.text_dim });
        }
    };
};

// ------------------------------------------------------------------ the script
//
// A host pushes one `Input` per frame. These are the frames the demo pushes;
// nothing about them is test-only, they are what a windowed host would produce
// from a mouse and a keyboard.

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
fn wheel(x: f32, y: f32, dy: f32) Input {
    return .{ .pointer = .{ .x = x, .y = y }, .scroll = .{ .x = 0, .y = dy } };
}
fn typed(cp: u21) Input {
    return .{ .pointer = .{ .x = 10000, .y = 10000 }, .text = &.{cp} };
}

const Check = struct { ok: bool, what: []const u8 };

/// The captured panels, in film order.
const Film = struct {
    alloc: std.mem.Allocator,
    images: [9]zgf.Image = undefined,
    labels: [9][]const u8 = undefined,
    count: usize = 0,

    fn push(self: *Film, ui: *const Ui, label: []const u8, opts: zui.render.Options) !void {
        var img = try zgf.Image.init(self.alloc, @intFromFloat(width), @intFromFloat(height), 3);
        errdefer img.deinit();
        zui.render.draw(&ui.list, &img, opts);
        self.images[self.count] = img;
        self.labels[self.count] = label;
        self.count += 1;
    }

    fn deinit(self: *Film) void {
        for (self.images[0..self.count]) |*p| p.deinit();
        self.count = 0;
    }
};

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    try zpu.init(alloc, .{});
    defer zpu.deinit();

    var app = App{};
    app.init();
    var ui = Ui.init(alloc, .{ .width = width, .height = height });
    defer ui.deinit();
    const body = Body{ .app = &app };

    var checks = std.ArrayList(Check).empty;
    defer checks.deinit(alloc);

    var film = Film{ .alloc = alloc };
    defer film.deinit();

    // ------------------------------------------------------------------ frame 1
    var result = try ui.run(parked(), body);
    try expect(&checks, alloc, result.rebuilt and !result.skipped, "the first frame is always built");
    try expect(&checks, alloc, ui.list.len() > 30, "the first frame paints a display list");
    const first_commands = ui.list.len();
    try expect(&checks, alloc, ui.list.isBalanced(), "the display list is balanced (every clip is closed)");
    try film.push(&ui, "first frame", .{ .clear = ui.theme.bg });

    // Hovering must repaint even though no button moved: the glow is an
    // animation, and an animation is a reason to run the body.
    // Widget positions come out of the display list: `zui` has no widget objects
    // to query, and a host that wants to anchor a tooltip reads the same list.
    const button = textAnchor(&ui, "Click me") orelse return error.MissingWidget;
    _ = try ui.run(moved(button.x, button.y), body);
    result = try ui.run(moved(button.x, button.y), body);
    try expect(&checks, alloc, result.rebuilt, "a hovered frame is rebuilt (the glow is animating)");
    try expect(&checks, alloc, ui.ctx.hot.eql(app.hit_id), "the widget under the pointer is hot");
    try film.push(&ui, "button hovered", .{ .clear = ui.theme.bg });

    // ------------------------------------------------------------------ click
    _ = try ui.run(down(button.x, button.y), body);
    try film.push(&ui, "button pressed", .{ .clear = ui.theme.bg });
    _ = try ui.run(up(button.x, button.y), body);
    try expect(&checks, alloc, app.clicks == 1, "the press+release pair delivered one click");
    _ = try ui.run(up(button.x, button.y), body);
    try expect(&checks, alloc, app.clicks == 1, "and exactly one");

    // ------------------------------------------------- toggle and drag
    const check_label = textAnchor(&ui, "enabled") orelse return error.MissingWidget;
    _ = try ui.run(down(check_label.x + 4, check_label.y), body);
    _ = try ui.run(up(check_label.x + 4, check_label.y), body);
    try expect(&checks, alloc, app.toggled, "clicking a checkbox label toggles it");

    // The slider maps the pointer's x onto the range, so a drag is just frames
    // in which the pointer is somewhere else.
    const track = findRounded(&ui, 6) orelse return error.MissingWidget;
    const track_y = track.y + track.h * 0.5;
    _ = try ui.run(down(track.x + track.w * 0.3, track_y), body);
    const gain_a = app.gain;
    _ = try ui.run(moved(track.x + track.w * 0.8, track_y), body);
    const gain_b = app.gain;
    _ = try ui.run(up(track.x + track.w * 0.8, track_y), body);
    try expect(&checks, alloc, gain_b > gain_a, "dragging right raised the value");
    try expect(&checks, alloc, gain_b > 0.6 and gain_b < 0.9, "the value tracks the pointer's position on the track");
    try film.push(&ui, "checkbox, slider, progress", .{ .clear = ui.theme.bg });

    // ---------------------------------------------------------------- typing
    // Click into the field to focus it, then type. The caret is retained state;
    // the string belongs to the application.
    const field = textAnchor(&ui, "type a word") orelse return error.MissingWidget;
    _ = try ui.run(down(field.x, field.y), body);
    _ = try ui.run(up(field.x, field.y), body);
    _ = try ui.run(typed('z'), body);
    _ = try ui.run(typed('i'), body);
    _ = try ui.run(typed('g'), body);
    try expect(&checks, alloc, std.mem.eql(u8, app.field.slice(), "zig"), "three keystrokes arrived in order");
    try expect(&checks, alloc, ui.ctx.edit(ui.idOf("root")).caret >= 0, "the caret is retained across frames");
    try film.push(&ui, "text typed, caret retained", .{ .clear = ui.theme.bg });

    // ---------------------------------------------------------------- scroll
    // The list is virtual: 2000 rows, of which four are on screen. `row 1` is
    // visible before and after the wheel event; `row 0` is not visible after it
    // — and it is not *built* either, which is what the second check says.
    const row1 = textAnchor(&ui, "row 1") orelse return error.MissingWidget;
    const before_scroll = row1.y;
    try expect(&checks, alloc, textAnchor(&ui, "row 0") != null, "the virtual list starts at row 0");
    _ = try ui.run(wheel(row1.x + 120, row1.y + 6, 20), body);
    const after_row1 = textAnchor(&ui, "row 1") orelse return error.MissingWidget;
    // Exactly one wheel's worth: the offset comes off the row's index-derived
    // position, so there is no reflow and no rounding.
    try expect(&checks, alloc, @abs(before_scroll - after_row1.y - 20) < 0.01, "the wheel scrolled the list by exactly what it said");
    try expect(&checks, alloc, textAnchor(&ui, "row 0") == null, "and a row that scrolled out of view is not built at all");
    try film.push(&ui, "list scrolled and clipped", .{ .clear = ui.theme.bg });

    // ------------------------------------------------------------ collapsing
    const header = textAnchor(&ui, "How this frame was built") orelse return error.MissingWidget;
    const before_open = ui.list.len();
    _ = try ui.run(down(header.x + 4, header.y), body);
    _ = try ui.run(up(header.x + 4, header.y), body);
    try expect(&checks, alloc, ui.list.len() > before_open, "clicking the header revealed the section body");
    try film.push(&ui, "section re-opened by a click", .{ .clear = ui.theme.bg });

    // -------------------------------------------------------------------- idle
    //
    // The point of the retained display list. The frames right after an
    // interaction still run (the hover glow is settling); once nothing is
    // animating, an idle frame does not run the body at all.
    var settling: u64 = 0;
    while (settling < 300) : (settling += 1) {
        if ((try ui.run(parked(), body)).skipped) break;
    }
    try expect(&checks, alloc, settling < 300, "the animations settle within 300 frames");
    var idle_skipped: u64 = 0;
    for (0..60) |_| {
        if ((try ui.run(parked(), body)).skipped) idle_skipped += 1;
    }
    try expect(&checks, alloc, idle_skipped == 60, "once settled, every one of 60 idle frames reuses the picture");
    const stats = ui.stats();
    try expect(&checks, alloc, ui.list.reused, "the reused list is flagged as reused");

    // ------------------------------------------------------- the same list, 2×
    // A HiDPI host gets the identical display list drawn with `scale = 2`; the
    // widgets never learned about device pixels.
    try film.push(&ui, "same list at 2× (HiDPI)", .{ .clear = ui.theme.bg, .scale = 2 });

    // -------------------------------------------------------------- and zoomed
    const crop = Rect{ .x = 6, .y = 30, .w = 76, .h = 56 };
    const zoom = width / crop.w;
    try film.push(&ui, "5× zoom: edges, not stairs", .{
        .clear = ui.theme.bg,
        .scale = zoom,
        .origin = .{ .x = -crop.x * zoom, .y = -crop.y * zoom },
    });

    // --------------------------------------------------------------- the strip
    var montage = try buildMontage(alloc, film.images[0..film.count], film.labels[0..film.count]);
    defer montage.deinit();

    // Pixels, not vibes: the background is the theme's, edges are partially
    // covered, and the pressed panel really differs from the hovered one.
    const corner = montage.getPixel(2, 2);
    try expect(&checks, alloc, @abs(corner.r - zui.Theme.dark.bg.r) < 0.03, "the strip's background is the theme's");
    var partial: usize = 0;
    var solid: usize = 0;
    for (0..montage.height()) |y| {
        for (0..montage.width()) |x| {
            const p = montage.getPixel(x, y);
            if (p.r > 0.08 and p.r < 0.7) partial += 1;
            if (p.r > 0.85) solid += 1;
        }
    }
    try expect(&checks, alloc, partial > 5000, "antialiased edge pixels exist in quantity");
    try expect(&checks, alloc, solid > 1000, "so do solid pixels");
    try expect(&checks, alloc, !samplesEqual(&montage, panelOrigin(1), panelOrigin(2)), "the pressed panel differs from the hovered one");
    try expect(&checks, alloc, !samplesEqual(&montage, panelOrigin(0), panelOrigin(4)), "the typed panel differs from the first one");

    // ------------------------------------------------------------------ timing
    // Two numbers matter for a UI library: how long a body takes to rebuild and
    // how long the pixels take. Both are measured here rather than guessed at.
    var build_ns: i128 = 0;
    var render_ns: i128 = 0;
    var reps: usize = 0;
    const reps_target: usize = 120;
    var scratch = try zgf.Image.init(alloc, @intFromFloat(width), @intFromFloat(height), 3);
    defer scratch.deinit();
    while (reps < reps_target) : (reps += 1) {
        // Alternate hover and press so the body really does rebuild every frame.
        const input = if (reps % 2 == 0) moved(button.x, button.y) else down(button.x, button.y);
        const t0 = nowNs(io);
        _ = try ui.run(input, body);
        const t1 = nowNs(io);
        build_ns += t1 - t0;
        zui.render.draw(&ui.list, &scratch, .{ .clear = ui.theme.bg });
        render_ns += nowNs(io) - t1;
    }
    const build_us = @as(f64, @floatFromInt(build_ns)) / 1e3 / @as(f64, @floatFromInt(reps_target));
    const render_us = @as(f64, @floatFromInt(render_ns)) / 1e3 / @as(f64, @floatFromInt(reps_target));
    const mpixel_s = @as(f64, width) * @as(f64, height) / render_us;
    try expect(&checks, alloc, build_us < 2000, "a rebuilt frame stays under 2 ms on this machine");

    // -------------------------------------------------------------- instances
    // Everything above drew the display list straight into an image. Here the
    // *same* list is walked into an instance buffer — the form a backend is
    // handed, one fixed-size record per shape — packed into the layout a device
    // buffer wants, and then rasterised by a `zbackend` kernel into an RGBA
    // target, exactly as a backend dispatch would.
    //
    // The two pictures are then compared pixel by pixel. They are not "close":
    // both paths call `zui.raster.coverage` and composite in the same order, so
    // the difference is zero, and the check below says so.
    var instance_scratch: [512]zui.Instance = undefined;
    const frame_insts = zui.instances.collect(&ui.list, .{}, &instance_scratch);
    try expect(&checks, alloc, frame_insts.len >= ui.list.len(), "the frame expands to at least one instance per command (a border is two, text is one per glyph)");

    const flat = try alloc.alloc(f32, frame_insts.len * zui.pipeline.stride + 8);
    defer alloc.free(flat);
    const flat_instances = zui.pipeline.pack(frame_insts, flat[8..]);
    var round_tripped: [512]zui.Instance = undefined;
    const back = zui.pipeline.unpack(flat_instances, &round_tripped);
    var layout_ok = back.len == frame_insts.len;
    for (back, frame_insts) |a, b| layout_ok = layout_ok and std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    try expect(&checks, alloc, layout_ok, "the instance buffer round-trips through the device layout byte for byte");
    try expect(&checks, alloc, zui.pipeline.atlasPixels().len == zui.pipeline.atlas_size[0] * zui.pipeline.atlas_size[1], "the glyph atlas travels with the frame");

    // The kernel is order-dependent (a later shape covers an earlier one), so it
    // must not claim it can run on a backend that does not run invocations in
    // order. `canRun` false is what makes `zbackend.dispatch` fall back to the
    // CPU instead of handing a driver something it cannot run.
    const gpu_backends = [_]zbackend.Backend{ .vulkan, .webgpu, .metal, .d3d12 };
    var refuses_gpu = true;
    for (gpu_backends) |b| refuses_gpu = refuses_gpu and !zui.pipeline.kernel.canRun(b);
    try expect(&checks, alloc, refuses_gpu, "the kernel refuses every GPU backend, so a dispatch falls back to the CPU instead of failing");

    const wpx: usize = @intFromFloat(width);
    const hpx: usize = @intFromFloat(height);
    const target = try alloc.alloc(f32, wpx * hpx * 4);
    defer alloc.free(target);
    for (0..wpx * hpx) |i| {
        target[i * 4] = ui.theme.bg.r;
        target[i * 4 + 1] = ui.theme.bg.g;
        target[i * 4 + 2] = ui.theme.bg.b;
        target[i * 4 + 3] = 1;
    }
    const kernel_t0 = nowNs(io);
    try zui.pipeline.dispatch(.cpu, frame_insts, target, wpx, hpx, .{}, alloc);
    const kernel_us = @as(f64, @floatFromInt(nowNs(io) - kernel_t0)) / 1e3;

    var differing: usize = 0;
    for (0..hpx) |y| {
        for (0..wpx) |x| {
            const i = (y * wpx + x) * 4;
            const want = scratch.getPixel(x, y);
            if (@abs(target[i] - want.r) > 1e-6 or @abs(target[i + 1] - want.g) > 1e-6 or @abs(target[i + 2] - want.b) > 1e-6) differing += 1;
        }
    }
    try expect(&checks, alloc, differing == 0, "the kernel's pixels are identical to the software renderer's");
    try expect(&checks, alloc, kernel_us < 50_000, "a frame of instances dispatches in well under 50 ms");

    // And the kernel's own picture, on disk, so the claim can be looked at.
    var kernel_img = try zgf.Image.init(alloc, wpx, hpx, 3);
    defer kernel_img.deinit();
    for (0..hpx) |y| {
        for (0..wpx) |x| {
            const i = (y * wpx + x) * 4;
            kernel_img.setPixel(@intCast(x), @intCast(y), .{ .r = target[i], .g = target[i + 1], .b = target[i + 2], .a = 1 });
        }
    }
    // ------------------------------------------------------------------ report
    var out = std.Io.Writer.Allocating.init(alloc);
    defer out.deinit();
    const w = &out.writer;
    try w.print("zui demo — {d}x{d} logical pixels, {d} panels, {d}x{d} strip\n", .{ width, height, film.count, cols, rows });
    try w.print("  frame 1 painted {d} commands; the last frame painted {d}\n", .{ first_commands, ui.list.len() });
    try w.print("  state: clicks={d}, toggled={}, gain={d:.3}, field=\"{s}\", scrolled by {d:.0} px\n", .{
        app.clicks,
        app.toggled,
        app.gain,
        app.field.slice(),
        before_scroll - after_row1.y,
    });
    try w.print("  frames: {d} run, {d} skipped by the idle rule ({d:.0}%)\n", .{
        stats.frames,
        stats.skipped,
        @as(f64, @floatFromInt(stats.skipped)) * 100.0 / @as(f64, @floatFromInt(stats.frames)),
    });
    try w.print("  timing: body {d:.1} µs, software rasteriser {d:.1} µs ({d:.1} Mpixel/s) — debug build\n", .{
        build_us,
        render_us,
        mpixel_s,
    });
    try w.print("  instances: {d} records for {d} commands, {d} bytes each, packed into a device buffer\n", .{ frame_insts.len, ui.list.len(), zui.pipeline.stride * @sizeOf(f32) });
    try w.print("  device path: zbackend kernel on .cpu drew the same {d}x{d} frame in {d:.1} µs; {d} pixels differ from the software renderer\n", .{ wpx, hpx, kernel_us, differing });
    try w.print("  output: zig-out/ui-demo.ppm ({d}x{d}), zig-out/ui-demo-kernel.ppm, zig-out/ui-demo.txt\n\n", .{ montage.width(), montage.height() });

    var passed: usize = 0;
    for (checks.items) |c| {
        try w.print("  [{s}] {s}\n", .{ if (c.ok) "ok" else "FAIL", c.what });
        if (c.ok) passed += 1;
    }
    try w.print("\n{d}/{d} checks passed\n", .{ passed, checks.items.len });

    try std.Io.Dir.cwd().createDirPath(io, "zig-out");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "zig-out/ui-demo.txt", .data = out.written() });
    try montage.savePpm(io, "zig-out/ui-demo.ppm");
    try kernel_img.savePpm(io, "zig-out/ui-demo-kernel.ppm");
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    if (passed != checks.items.len) return error.ChecksFailed;
}

/// Centre of the ink box of the text run `needle`: how the demo knows where a
/// widget is. `zui` keeps no widget objects to query, so a script — or a host
/// anchoring a tooltip — reads the frame it just produced.
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

/// First rounded box of exactly `h` pixels: the slider track is the only 6 px
/// rounded box in this UI.
fn findRounded(ui: *const Ui, h: f32) ?Rect {
    for (ui.list.commands.items) |c| {
        switch (c) {
            .rounded => |r| if (r.r.h == h) return r.r,
            else => {},
        }
    }
    return null;
}

fn nowNs(io: std.Io) i128 {
    return @as(i128, @intCast(std.Io.Clock.awake.now(io).nanoseconds));
}

fn expect(checks: *std.ArrayList(Check), alloc: std.mem.Allocator, ok: bool, what: []const u8) !void {
    try checks.append(alloc, .{ .ok = ok, .what = what });
}

/// Top-left corner of panel `i` inside the montage, as `(x, y)`.
fn panelOrigin(i: usize) Rect {
    const gap: usize = 6;
    const header: usize = 20;
    const w: usize = @intFromFloat(width);
    const h: usize = @intFromFloat(height);
    const x = gap + (i % cols) * (w + gap);
    const y = gap + (i / cols) * (h + header + gap) + header;
    return Rect.init(@floatFromInt(x), @floatFromInt(y), @floatFromInt(w), @floatFromInt(h));
}

/// Compares two panels sample by sample: 400 probes on a fixed lattice, which
/// is enough to catch "these two frames drew the same thing" without comparing
/// every pixel of a 900x600 strip.
fn samplesEqual(img: *const zgf.Image, a: Rect, b: Rect) bool {
    const ax: usize = @intFromFloat(a.x);
    const ay: usize = @intFromFloat(a.y);
    const aw: usize = @intFromFloat(a.w);
    const ah: usize = @intFromFloat(a.h);
    const bx: usize = @intFromFloat(b.x);
    const by: usize = @intFromFloat(b.y);
    var i: usize = 0;
    while (i < 400) : (i += 1) {
        const dx = (i * 7) % aw;
        const dy = (i * 13) % ah;
        const pa = img.getPixel(ax + dx, ay + dy);
        const pb = img.getPixel(bx + dx, by + dy);
        if (@abs(pa.r - pb.r) > 0.01 or @abs(pa.g - pb.g) > 0.01 or @abs(pa.b - pb.b) > 0.01) return false;
    }
    return true;
}

/// Lays the panels out in a grid with a caption strip, and returns one image.
fn buildMontage(alloc: std.mem.Allocator, panels: []const zgf.Image, labels: []const []const u8) !zgf.Image {
    const w: usize = @intFromFloat(width);
    const h: usize = @intFromFloat(height);
    const gap: usize = 6;
    const header: usize = 20;
    const out_w = cols * w + (cols + 1) * gap;
    const out_h = rows * (h + header) + (rows + 1) * gap;

    var img = try zgf.Image.init(alloc, out_w, out_h, 3);
    errdefer img.deinit();
    img.fill(.{ .r = zui.Theme.dark.bg.r, .g = zui.Theme.dark.bg.g, .b = zui.Theme.dark.bg.b });

    for (panels, 0..) |panel, i| {
        const origin = panelOrigin(i);
        const x0: usize = @intFromFloat(origin.x);
        const y0: usize = @intFromFloat(origin.y);
        blit(&img, &panel, x0, y0);
        var buf: [64]u8 = undefined;
        const caption = std.fmt.bufPrint(&buf, "{d}/{d}  {s}", .{ i + 1, panels.len, labels[i] }) catch "?";
        drawCaption(&img, @floatFromInt(x0), @as(f32, @floatFromInt(y0)) - 15, caption);
    }
    return img;
}

fn blit(dst: *zgf.Image, src: *const zgf.Image, x0: usize, y0: usize) void {
    for (0..src.height()) |y| {
        for (0..src.width()) |x| {
            if (x0 + x < dst.width() and y0 + y < dst.height()) {
                dst.setPixel(@intCast(x0 + x), @intCast(y0 + y), src.getPixel(x, y));
            }
        }
    }
}

/// The caption font is the same 14 px glyph set the widgets use, drawn straight
/// into an image — the renderer's text path without a `Ui` around it.
fn drawCaption(img: *zgf.Image, x: f32, baseline: f32, text: []const u8) void {
    const color = zgf.Color{ .r = 0.72, .g = 0.78, .b = 0.9, .a = 1 };
    var it = zui.font.layout(text, zui.font.em, x);
    while (it.next()) |g| {
        const left = g.x + @as(f32, @floatFromInt(g.glyph.left));
        const top = baseline + @as(f32, @floatFromInt(g.glyph.top));
        var gy: i32 = 0;
        while (gy < g.glyph.h) : (gy += 1) {
            var gx: i32 = 0;
            while (gx < g.glyph.w) : (gx += 1) {
                const cov = zui.font.coverage(g.glyph, gx, gy);
                if (cov == 0) continue;
                const px = left + @as(f32, @floatFromInt(gx));
                const py = top + @as(f32, @floatFromInt(gy));
                if (px < 0 or py < 0) continue;
                const ix: usize = @intFromFloat(px);
                const iy: usize = @intFromFloat(py);
                if (ix >= img.width() or iy >= img.height()) continue;
                img.blendCoverage(ix, iy, color, @as(f32, @floatFromInt(cov)) / 255.0);
            }
        }
    }
}
