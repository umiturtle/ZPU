//! The immediate-mode API, the two-pass layout and the widgets.
//!
//! # How a frame is built
//!
//! `ui.run(input, body)` calls `body.run(&ui)` once, and the body is ordinary
//! Zig: `if`, `for`, `switch`, and calls into the program's own state. A widget
//! does two things, in this order — it **reserves** a box in the current
//! container, then it **emits** commands into the display list and reads the
//! interaction state.
//!
//! Reserving is where the two passes come from. A container cannot place a child
//! before it knows how big that child is, and it cannot know that before the
//! child's body has run. So a container runs its body twice:
//!
//! 1. **measure** — widgets compute their natural size and add it up; nothing is
//!    painted and no retained state is touched;
//! 2. **draw** — the same body again, now with the container's own size known,
//!    so `fill` children can be given their share of what is left and every
//!    widget learns its rectangle.
//!
//! Bodies therefore have to be deterministic, which is why widget return values
//! are suppressed while measuring: `if (ui.button(...)) state += 1` cannot fire
//! twice. The cost is one extra traversal per nested container, and the frame
//! benchmark in `bench/` measures it.
//!
//! # Where the state lives
//!
//! Nowhere in the widgets. Interaction state lives in `Context` (keyed by widget
//! id), the picture lives in the `DisplayList`, and the frame's strings live in
//! an arena that is *reset*, not freed, when the next frame begins.
const std = @import("std");
const types = @import("types.zig");
const input_mod = @import("input.zig");
const context_mod = @import("context.zig");
const frame_mod = @import("frame.zig");
const font = @import("font.zig");
const text_mod = @import("text.zig");

const Alignment = types.Alignment;
const Axis = types.Axis;
const Color = types.Color;
const DisplayList = frame_mod.DisplayList;
const Edges = types.Edges;
const Id = types.Id;
const Input = input_mod.Input;
const Rect = types.Rect;
const Size = types.Size;
const Vec2 = types.Vec2;
const TextField = text_mod.TextField;

/// Nesting depth of containers and id scopes: deep enough for hand-written UI,
/// and a fixed array so pushing can never fail.
pub const max_depth = 32;

pub const Theme = struct {
    text: Color = Color.hex(0xe6e9ef, 1),
    text_dim: Color = Color.hex(0x9aa3b2, 1),
    bg: Color = Color.hex(0x171a21, 1),
    surface: Color = Color.hex(0x232838, 1),
    border: Color = Color.hex(0x3a4257, 1),
    track: Color = Color.hex(0x11141b, 1),
    accent: Color = Color.hex(0x5b8cff, 1),
    accent_text: Color = Color.hex(0xffffff, 1),
    widget: Color = Color.hex(0x2c3242, 1),
    widget_hover: Color = Color.hex(0x39415a, 1),
    radio_dot: Color = Color.hex(0x9fb4ff, 1),

    font_size: f32 = 14,
    item_h: f32 = 26,
    radius: f32 = 5,
    gap: f32 = 6,
    pad: f32 = 8,
    /// How fast hover/press glows settle, in 1/seconds.
    anim_rate: f32 = 18,

    pub const dark = Theme{};
    pub const light = Theme{
        .text = Color.hex(0x1b1e26, 1),
        .text_dim = Color.hex(0x5b6273, 1),
        .bg = Color.hex(0xf3f4f7, 1),
        .surface = Color.hex(0xffffff, 1),
        .border = Color.hex(0xc9cfdd, 1),
        .track = Color.hex(0xdfe3ec, 1),
        .accent = Color.hex(0x2f6bff, 1),
        .widget = Color.hex(0xe3e7f0, 1),
        .widget_hover = Color.hex(0xd2d9e8, 1),
        .radio_dot = Color.hex(0x2f6bff, 1),
    };
};

/// What a widget or a container asks for. `w`/`h` are along *screen* axes and are
/// mapped onto the current container's main/cross axis by the layout.
pub const Style = struct {
    w: Size = .auto,
    h: Size = .auto,
    pad: Edges = .{},
    /// Distance to the next sibling (containers: between children).
    gap: f32 = 0,
    dir: Axis = .y,
    /// Cross-axis placement of this container's children.
    cross_align: Alignment = .start,
    /// Main-axis placement of the whole group when there is slack.
    justify: Alignment = .start,
    clip: bool = false,
    bg: Color = Color.transparent,
    border: f32 = 0,
    border_color: Color = Color.hex(0xffffff, 0),
    radius: f32 = 0,
    min_w: f32 = 0,
    min_h: f32 = 0,

    pub fn sizeAlong(self: Style, axis: Axis) Size {
        return if (axis == .x) self.w else self.h;
    }

    /// The minimum on `axis`; a box is never laid out smaller than this.
    pub fn minAlong(self: Style, axis: Axis) f32 {
        return if (axis == .x) self.min_w else self.min_h;
    }
};

/// What a widget reports about itself for this frame. Every field is a result of
/// this frame's input plus the retained interaction state, so a widget never has
/// to be asked twice.
pub const Response = struct {
    hovered: bool = false,
    /// The pointer is down and this widget owns the press.
    active: bool = false,
    /// Press *and* release inside, or a keyboard activation: the edge that fires
    /// an action.
    clicked: bool = false,
    focused: bool = false,
    disabled: bool = false,
    /// Animated `0..1` hover glow and press amount.
    hover_t: f32 = 0,
    press_t: f32 = 0,
    rect: Rect = .{},
};

const Mode = enum { measure, draw };

/// One container's layout state; the top of the stack is where the next widget
/// goes.
const LayoutFrame = struct {
    mode: Mode,
    dir: Axis = .y,
    /// Content box (draw) or the offered space (measure).
    content: Rect = .{},
    gap: f32 = 0,
    cross_align: Alignment = .start,
    /// Main-axis cursor and cross extent, in absolute coordinates.
    cursor: f32 = 0,
    cross: f32 = 0,
    /// Accumulated extents of the non-`fill` children (draw: what is fixed;
    /// measure: the same, plus gaps).
    fixed_main: f32 = 0,
    /// What the `fill` children asked for by content, and their total weight.
    fill_content: f32 = 0,
    weight: f32 = 0,
    gaps: f32 = 0,
    max_cross: f32 = 0,
    children: usize = 0,
    /// Space the `fill` children share (draw mode).
    fill_extent: f32 = 0,

    fn contentMain(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.w else self.content.h;
    }

    fn contentCross(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.h else self.content.w;
    }

    fn cursorStart(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.x else self.content.y;
    }

    /// What is left of the content box's main axis.
    fn remaining(self: *const LayoutFrame) f32 {
        return @max(0, self.contentMain() - (self.cursor - self.cursorStart()));
    }
};

pub const FrameResult = struct {
    rebuilt: bool,
    /// Nothing could have changed and the body was not called: the display list
    /// is last frame's, command for command.
    skipped: bool,
    commands: usize,
    revision: u64,
};

/// A tooltip waiting to be painted at the end of the frame.
const Tip = struct {
    anchor: Rect,
    text: []const u8,
};

pub const Options = struct {
    width: f32 = 320,
    height: f32 = 200,
    theme: Theme = Theme.dark,
    /// Cap on the lines one label may wrap to: the arena is not a bin.
    max_lines: usize = 64,
};

pub const Stats = struct {
    frames: u64,
    rebuilt: u64,
    skipped: u64,
    commands: usize,
    revision: u64,
};

pub const Ui = struct {
    /// The error set of a frame. Widgets allocate (display list, wrapped text),
    /// so a body can only fail by running out of memory — there is no "invalid
    /// widget" state to report, which is what keeps `try` in the caller simple.
    pub const Error = std.mem.Allocator.Error;
    alloc: std.mem.Allocator,
    /// The retained minimum (hot/active/focus/scroll/anims/edit state).
    ctx: context_mod.Context,
    /// The picture.
    list: DisplayList,
    theme: Theme,
    viewport: Vec2,

    input: Input = Input.empty,
    arena: std.heap.ArenaAllocator,

    id_stack: [max_depth]u64 = @splat(0),
    id_top: usize = 0,
    layouts: [max_depth]LayoutFrame = undefined,
    depth: usize = 0,

    measuring: bool = false,
    /// "My own data changed" — the one thing only the program knows.
    dirty: bool = false,
    /// This frame reuses the previous display list.
    skipping: bool = false,
    can_skip: bool = false,
    /// The pointer as of the last frame, so `beginFrame` can tell a pointer that
    /// merely *is* somewhere from one that has *moved*.
    last_pointer: Vec2 = Vec2{ .x = std.math.nan(f32), .y = std.math.nan(f32) },
    /// A tooltip queued during this frame; painted by `endFrame` so it ends up
    /// on top of the picture.
    tip: ?Tip = null,

    /// Focusable widgets, in build order: this frame's and the previous frame's
    /// (the latter drives Tab, which is handled before the body runs).
    focusables: std.ArrayListUnmanaged(Id) = .empty,
    focusables_prev: std.ArrayListUnmanaged(Id) = .empty,

    clip: Rect = .{ .w = std.math.floatMax(f32), .h = std.math.floatMax(f32) },
    max_lines: usize = 64,

    pub fn init(alloc: std.mem.Allocator, opts: Options) Ui {
        return .{
            .alloc = alloc,
            .ctx = context_mod.Context.init(alloc),
            .list = .{},
            .theme = opts.theme,
            .viewport = .{ .x = opts.width, .y = opts.height },
            .arena = std.heap.ArenaAllocator.init(alloc),
            .max_lines = opts.max_lines,
        };
    }

    pub fn deinit(self: *Ui) void {
        self.arena.deinit();
        self.focusables.deinit(self.alloc);
        self.focusables_prev.deinit(self.alloc);
        self.list.deinit(self.alloc);
        self.ctx.deinit();
    }

    pub fn resize(self: *Ui, width: f32, height: f32) void {
        if (self.viewport.x == width and self.viewport.y == height) return;
        self.viewport = .{ .x = width, .y = height };
        self.markDirty();
    }

    pub fn isMeasuring(self: *const Ui) bool {
        return self.measuring;
    }

    pub fn markDirty(self: *Ui) void {
        self.dirty = true;
    }

    pub fn stats(self: *const Ui) Stats {
        return .{
            .frames = self.ctx.frames,
            .rebuilt = self.ctx.rebuilt,
            .skipped = self.ctx.skipped,
            .commands = self.list.len(),
            .revision = self.list.revision,
        };
    }

    // ------------------------------------------------------------------- frame

    /// Runs one frame. `body` is any value with `run(self, ui) Error!void`: a
    /// hand-rolled closure, which is how Zig spells one without allocating.
    pub fn run(self: *Ui, input: Input, body: anytype) Error!FrameResult {
        self.beginFrame(input);
        // `endFrame` must have run before the result is taken (it is what bumps
        // the revision) and must still run if the body fails half way through,
        // or the next frame would start with a half-open layout stack.
        errdefer self.endFrame();
        if (!self.skipping) try body.run(self);
        self.endFrame();
        return self.result();
    }

    pub fn beginFrame(self: *Ui, input: Input) void {
        self.ctx.frames += 1;

        // The one retained decision, and the reason this library is a hybrid:
        // when nothing arrived, nothing is animating and the program did not mark
        // anything dirty, the picture cannot differ from last frame's, so the
        // body is not run and the display list is reused as-is.
        //
        // The pointer position is part of "nothing arrived": a moved pointer can
        // light up a different widget, so the body has to see it. `isIdle` alone
        // cannot know that — it is a property of one frame, and a position needs
        // two — which is why the previous position is retained here.
        const pointer_moved = !input.pointer.eql(self.last_pointer);
        self.skipping = self.can_skip and input.isIdle() and !pointer_moved;

        self.input = input;
        self.last_pointer = input.pointer;
        self.ctx.animating = false;
        self.dirty = false;
        _ = self.arena.reset(.retain_capacity);
        self.id_top = 0;
        self.depth = 0;
        self.clip = .{ .w = std.math.floatMax(f32), .h = std.math.floatMax(f32) };
        // Hotness is "under the pointer *this* frame", so it is recomputed from
        // scratch; `active`, `focus` and the animations are what persist.
        self.ctx.hot = Id.none;
        std.mem.swap(std.ArrayListUnmanaged(Id), &self.focusables, &self.focusables_prev);
        self.focusables.clearRetainingCapacity();

        if (self.skipping) {
            self.ctx.skipped += 1;
            return;
        }
        self.handleFocusKeys();
        self.list.reset();
        self.pushRootLayout();
    }

    pub fn endFrame(self: *Ui) void {
        if (self.skipping) {
            self.list.reused = true;
            return;
        }
        self.list.reused = false;
        self.list.revision += 1;
        while (self.depth > 0) self.popLayout();
        // Last, so nothing in the frame can cover it.
        if (self.tip) |tip| self.paintTooltip(tip) catch {};
        self.tip = null;
        self.ctx.rebuilt += 1;
        self.can_skip = self.input.isIdle() and !self.dirty and !self.ctx.animating;
        self.ctx.focusables.clearRetainingCapacity();
        self.ctx.focusables.appendSlice(self.ctx.alloc, self.focusables.items) catch {};
    }

    /// Draws the queued tooltip: a bordered panel under the anchor, flipped
    /// above it and clamped inside the window when there is no room below.
    fn paintTooltip(self: *Ui, tip: Tip) !void {
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

    fn result(self: *const Ui) FrameResult {
        return .{
            .rebuilt = !self.skipping,
            .skipped = self.skipping,
            .commands = self.list.len(),
            .revision = self.list.revision,
        };
    }

    fn pushRootLayout(self: *Ui) void {
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

    // ---------------------------------------------------------------- painting

    pub fn pushClip(self: *Ui, r: Rect) Error!void {
        self.clip = self.clip.intersect(r);
        try self.list.push(self.alloc, .{ .push_clip = self.clip });
    }

    pub fn popClip(self: *Ui, previous: Rect) Error!void {
        self.clip = previous;
        try self.list.push(self.alloc, .{ .pop_clip = {} });
    }

    pub fn paintRect(self: *Ui, r: Rect, color: Color) Error!void {
        if (self.measuring or color.a <= 0 or r.isEmpty()) return;
        try self.list.push(self.alloc, .{ .rect = .{ .r = r, .color = color } });
    }

    pub fn paintRounded(self: *Ui, r: Rect, radius: f32, color: Color, border: f32, border_color: Color) Error!void {
        if (self.measuring or r.isEmpty()) return;
        if (color.a <= 0 and (border <= 0 or border_color.a <= 0)) return;
        try self.list.push(self.alloc, .{
            .rounded = .{ .r = r, .radius = radius, .color = color, .border = border, .border_color = border_color },
        });
    }

    pub fn paintLine(self: *Ui, a: Vec2, b: Vec2, width: f32, color: Color) Error!void {
        if (self.measuring or color.a <= 0) return;
        try self.list.push(self.alloc, .{ .line = .{ .a = a, .b = b, .width = width, .color = color } });
    }

    pub fn paintTriangle(self: *Ui, a: Vec2, b: Vec2, c: Vec2, color: Color) Error!void {
        if (self.measuring or color.a <= 0) return;
        try self.list.push(self.alloc, .{ .triangle = .{ .a = a, .b = b, .c = c, .color = color } });
    }

    /// One line of text, baseline at `y`. The bytes are copied into the frame
    /// arena, so the caller's string may die immediately.
    pub fn paintText(self: *Ui, x: f32, baseline: f32, size: f32, color: Color, content: []const u8) Error!void {
        if (self.measuring or color.a <= 0 or content.len == 0) return;
        const owned = try self.arena.allocator().dupe(u8, content);
        try self.list.push(self.alloc, .{
            .text = .{ .pos = .{ .x = x, .y = baseline }, .size = size, .color = color, .text = owned },
        });
    }

    // ------------------------------------------------------------------ layout

    fn top(self: *Ui) *LayoutFrame {
        return &self.layouts[self.depth - 1];
    }

    fn pushLayout(self: *Ui, frame: LayoutFrame) void {
        std.debug.assert(self.depth < max_depth);
        self.layouts[self.depth] = frame;
        self.depth += 1;
    }

    fn popLayout(self: *Ui) void {
        std.debug.assert(self.depth > 0);
        self.depth -= 1;
    }

    /// Space a child may use for wrapping: the container's cross extent.
    pub fn available(self: *Ui) Vec2 {
        const lf = self.top();
        return .{ .x = lf.contentCross(), .y = lf.contentMain() };
    }

    pub fn contentWidth(self: *Ui) f32 {
        return self.top().contentCross();
    }

    /// Reserves a box: accumulates in the measure pass, returns the placement
    /// and advances the cursor in the draw pass.
    fn occupy(self: *Ui, style: Style, content_size: Vec2) Rect {
        const lf = self.top();
        const main = lf.dir;
        const cross = main.other();

        const min_main = style.minAlong(main);
        const min_cross = style.minAlong(cross);
        const wanted_main = @max(min_main, content_size.index(main) + style.pad.sum(main));
        const wanted_cross = @max(min_cross, content_size.index(cross) + style.pad.sum(cross));
        const main_req = style.sizeAlong(main);
        const cross_req = style.sizeAlong(cross);

        if (lf.mode == .measure) {
            switch (main_req) {
                .auto => lf.fixed_main += wanted_main,
                .px => |v| lf.fixed_main += @max(v, min_main),
                .fill => |w| {
                    lf.fill_content += wanted_main;
                    lf.weight += w;
                },
            }
            if (lf.children > 0) {
                lf.fixed_main += lf.gap;
                lf.fill_content += lf.gap;
                lf.gaps += lf.gap;
            }
            const want_cross: f32 = switch (cross_req) {
                .auto, .fill => wanted_cross,
                .px => |v| @max(v, min_cross),
            };
            lf.max_cross = @max(lf.max_cross, want_cross);
            lf.children += 1;
            return Rect.fromSize(0, 0);
        }

        const main_extent: f32 = @max(min_main, switch (main_req) {
            .auto => wanted_main,
            .px => |v| v,
            .fill => |w| if (lf.weight > 0) lf.fill_extent * (w / lf.weight) else lf.remaining(),
        });
        const cross_extent: f32 = @max(min_cross, switch (cross_req) {
            .px => |v| v,
            .fill => lf.cross,
            .auto => if (lf.cross_align == .stretch) lf.cross else wanted_cross,
        });

        const start = if (lf.children == 0) lf.cursor else lf.cursor + lf.gap;
        lf.cursor = start + main_extent;
        lf.children += 1;

        const slack = @max(0, lf.cross - cross_extent);
        const cross_off: f32 = switch (cross_req) {
            .fill => 0,
            .px, .auto => switch (lf.cross_align) {
                .start, .stretch => 0,
                .center => slack * 0.5,
                .end => slack,
            },
        };

        return if (main == .x)
            Rect.init(start, lf.content.y + cross_off, main_extent, cross_extent)
        else
            Rect.init(lf.content.x + cross_off, start, cross_extent, main_extent);
    }

    // --------------------------------------------------------------------- ids

    fn scopeHash(self: *const Ui) u64 {
        return if (self.id_top == 0) 0 else self.id_stack[self.id_top - 1];
    }

    pub fn idOf(self: *const Ui, name: []const u8) Id {
        return Id.hash(self.scopeHash(), name);
    }

    /// Scopes an id by index instead of by name: row `i` of a list is a widget
    /// id like any other, and formatting a label for each of a million rows
    /// would be work for nothing.
    pub fn pushIdIndex(self: *Ui, i: usize) void {
        self.id_stack[self.id_top] = Id.index(self.scopeHash(), i).h;
        self.id_top += 1;
    }

    pub fn pushId(self: *Ui, name: []const u8) void {
        self.id_stack[self.id_top] = Id.hash(self.scopeHash(), name).h;
        self.id_top += 1;
    }

    pub fn popId(self: *Ui) void {
        self.id_top -= 1;
    }

    // -------------------------------------------------------------- interaction

    fn respond(self: *Ui, id: Id, r: Rect, focusable: bool, enabled: bool) Response {
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

    pub fn requestFocus(self: *Ui, id: Id) void {
        if (!self.measuring) self.ctx.focus = id;
    }

    pub fn clearFocus(self: *Ui) void {
        if (!self.measuring) self.ctx.focus = Id.none;
    }

    /// Tab / shift-Tab, against the focus ring the previous frame built. Done
    /// before the body runs, so the widget it lands on sees the focus this frame.
    fn handleFocusKeys(self: *Ui) void {
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

    // --------------------------------------------------------------- containers

    /// A container: measures its body, places itself, then draws its body.
    ///
    /// `body` is any value with `run(self, ui) Error!void` — the struct instance
    /// *is* the captured state.
    pub fn container(self: *Ui, id: []const u8, style: Style, body: anytype) Error!void {
        const avail = self.available();
        self.pushId(id);
        defer self.popId();

        const measured = try self.measureContainer(style, avail, body);
        const r = self.occupy(style, measured.content);
        if (self.measuring) return;
        if (r.isEmpty()) return;

        if (style.bg.a > 0 or (style.border > 0 and style.border_color.a > 0) or style.radius > 0) {
            try self.paintRounded(r, style.radius, style.bg, style.border, style.border_color);
        }

        const content_rect = r.inset(style.pad);
        const previous_clip = self.clip;
        if (style.clip) try self.pushClip(r);
        defer if (style.clip) {
            self.popClip(previous_clip) catch {};
        };

        var frame = drawFrame(style, content_rect, measured);
        if (frame.weight == 0 and style.justify != .start) {
            const slack = @max(0, frame.contentMain() - measured.fixed_main);
            const off: f32 = switch (style.justify) {
                .start, .stretch => 0,
                .center => slack * 0.5,
                .end => slack,
            };
            frame.cursor += off;
        }
        self.pushLayout(frame);
        defer self.popLayout();
        try body.run(self);
    }

    const Measured = struct {
        content: Vec2,
        fixed_main: f32 = 0,
        fill_content: f32 = 0,
        weight: f32 = 0,
        gaps: f32 = 0,
        max_cross: f32 = 0,
    };

    fn drawFrame(style: Style, content_rect: Rect, measured: Measured) LayoutFrame {
        var frame = LayoutFrame{
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

    fn measureContainer(self: *Ui, style: Style, avail: Vec2, body: anytype) Error!Measured {
        const was_measuring = self.measuring;
        self.measuring = true;
        self.pushLayout(.{
            .mode = .measure,
            .dir = style.dir,
            .content = Rect.init(0, 0, avail.x, avail.y),
            .gap = style.gap,
            .cross_align = style.cross_align,
            .cross = avail.x,
        });
        try body.run(self);
        const frame = self.layouts[self.depth - 1];
        self.popLayout();
        self.measuring = was_measuring;

        const main = frame.fixed_main + frame.fill_content;
        return .{
            .content = if (style.dir == .x)
                Vec2{ .x = main, .y = frame.max_cross }
            else
                Vec2{ .x = frame.max_cross, .y = main },
            .fixed_main = frame.fixed_main,
            .fill_content = frame.fill_content,
            .weight = frame.weight,
            .gaps = frame.gaps,
            .max_cross = frame.max_cross,
        };
    }

    // ----------------------------------------------------------------- widgets

    pub const LabelOptions = struct {
        color: ?Color = null,
        size: ?f32 = null,
        wrap: bool = true,
    };

    /// A run of text. Wraps to the container's width unless told otherwise.
    pub fn label(self: *Ui, content: []const u8, opts: LabelOptions) Error!void {
        const size = opts.size orelse self.theme.font_size;
        const color = opts.color orelse self.theme.text;
        const max_w = if (opts.wrap) self.contentWidth() else std.math.floatMax(f32);
        const lines = self.wrapText(content, size, max_w);
        const line_h = font.measure("", size).height;

        var widest: f32 = 0;
        for (lines) |line| widest = @max(widest, font.measure(content[line.start .. line.start + line.len], size).width);

        const r = self.occupy(.{}, .{ .x = widest, .y = line_h * @as(f32, @floatFromInt(lines.len)) });
        if (self.measuring) return;
        var baseline = r.y + font.measure("", size).ascent;
        for (lines) |line| {
            try self.paintText(r.x, baseline, size, color, content[line.start .. line.start + line.len]);
            baseline += line_h;
        }
    }

    pub const ButtonOptions = struct {
        w: Size = .auto,
        h: ?f32 = null,
        pad: ?Edges = null,
        enabled: bool = true,
        radius: ?f32 = null,
        /// Draw a border.
        border: bool = true,
    };

    /// A push button. `true` on the frame it was activated — by a click, or by
    /// `enter`/`space` while focused.
    pub fn button(self: *Ui, content: []const u8, opts: ButtonOptions) Error!bool {
        return self.buttonId(self.idOf(content), content, opts);
    }

    /// A button whose id is explicit: for labels that change (an id that moves
    /// loses the press) or for rows of identical labels.
    pub fn buttonId(self: *Ui, id: Id, content: []const u8, opts: ButtonOptions) Error!bool {
        const size = self.theme.font_size;
        const m = font.measure(content, size);
        const pad = opts.pad orelse Edges.xy(self.theme.pad + 2, (self.theme.item_h - m.height) * 0.5);
        const style = Style{
            .w = opts.w,
            .h = Size{ .px = opts.h orelse @max(self.theme.item_h, m.height + pad.sum(.y)) },
            .pad = pad,
            .min_w = m.width + pad.sum(.x),
        };
        const r = self.occupy(style, .{ .x = m.width, .y = m.height });
        const resp = self.respond(id, r, true, opts.enabled);
        if (self.measuring) return false;

        const radius = opts.radius orelse self.theme.radius;
        var fill = self.theme.widget;
        if (!opts.enabled) {
            fill = Color.lerp(self.theme.widget, self.theme.bg, 0.5);
        } else {
            fill = Color.lerp(self.theme.widget, self.theme.widget_hover, resp.hover_t);
            fill = Color.lerp(fill, self.theme.accent, resp.press_t * 0.85);
        }
        try self.paintRounded(r, radius, fill, if (opts.border) 1 else 0, self.theme.border);
        const text_color = if (!opts.enabled)
            self.theme.text_dim.scale(0.7)
        else if (resp.press_t > 0.5)
            self.theme.accent_text
        else
            self.theme.text;
        const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
        try self.paintText(r.x + (r.w - m.width) * 0.5, baseline, size, text_color, content);
        if (resp.focused) try self.paintFocusRing(r, radius);
        return resp.clicked and opts.enabled;
    }

    fn paintFocusRing(self: *Ui, r: Rect, radius: f32) Error!void {
        try self.paintRounded(r.shrink(-1.5), radius + 1.5, Color.transparent, 1.5, self.theme.accent.withAlpha(0.9));
    }

    pub const ToggleOptions = struct {
        enabled: bool = true,
        w: Size = .auto,
    };

    /// A checkbox. `true` on the frame it was toggled.
    pub fn checkbox(self: *Ui, content: []const u8, value: *bool, opts: ToggleOptions) Error!bool {
        const id = self.idOf(content);
        const size = self.theme.font_size;
        const m = font.measure(content, size);
        const box = @max(15, size);
        const wanted = Vec2{ .x = box + self.theme.pad + m.width, .y = @max(self.theme.item_h, m.height) };
        const r = self.occupy(.{ .w = opts.w, .h = Size{ .px = wanted.y }, .min_w = wanted.x }, wanted);
        const resp = self.respond(id, r, true, opts.enabled);
        if (self.measuring) return false;

        const bx = Rect.init(r.x, r.y + (r.h - box) * 0.5, box, box);
        var fill = Color.lerp(self.theme.widget, self.theme.widget_hover, resp.hover_t);
        if (value.*) fill = Color.lerp(fill, self.theme.accent, 1.0);
        try self.paintRounded(bx, self.theme.radius * 0.6, fill, 1, self.theme.border);
        if (value.*) {
            // A tick in two strokes: no glyph in a 95-character font is as
            // recognisable at 15 px.
            const c = bx.center();
            const w = box * 0.2;
            try self.paintLine(.{ .x = c.x - w * 1.4, .y = c.y }, .{ .x = c.x - w * 0.2, .y = c.y + w * 1.2 }, 2, self.theme.accent_text);
            try self.paintLine(.{ .x = c.x - w * 0.2, .y = c.y + w * 1.2 }, .{ .x = c.x + w * 1.5, .y = c.y - w * 1.1 }, 2, self.theme.accent_text);
        }
        const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
        try self.paintText(r.x + box + self.theme.pad, baseline, size, if (opts.enabled) self.theme.text else self.theme.text_dim, content);
        if (resp.focused) try self.paintFocusRing(bx, self.theme.radius * 0.6);

        if (resp.clicked and opts.enabled) {
            value.* = !value.*;
            self.markDirty();
            return true;
        }
        return false;
    }

    /// A radio button; `selected` is the index of the chosen item in the group.
    pub fn radio(self: *Ui, content: []const u8, selected: *usize, index: usize, opts: ToggleOptions) Error!bool {
        const id = self.idOf(content);
        const size = self.theme.font_size;
        const m = font.measure(content, size);
        const box = @max(15, size);
        const wanted = Vec2{ .x = box + self.theme.pad + m.width, .y = @max(self.theme.item_h, m.height) };
        const r = self.occupy(.{ .w = opts.w, .h = Size{ .px = wanted.y }, .min_w = wanted.x }, wanted);
        const resp = self.respond(id, r, true, opts.enabled);
        if (self.measuring) return false;

        const cy = r.y + r.h * 0.5;
        const ring = Rect.init(r.x, cy - box * 0.5, box, box);
        const is_on = selected.* == index;
        var fill = Color.lerp(self.theme.widget, self.theme.widget_hover, resp.hover_t);
        if (is_on) fill = self.theme.track;
        // A circle is a rounded rectangle whose radius is half its side: the
        // display list needs no circle command for this.
        try self.paintRounded(ring, box * 0.5, fill, 1, self.theme.border);
        if (is_on) {
            const d = box * 0.44;
            try self.paintRounded(Rect.init(r.x + (box - d) * 0.5, cy - d * 0.5, d, d), d * 0.5, self.theme.radio_dot, 0, Color.transparent);
        }
        const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
        try self.paintText(r.x + box + self.theme.pad, baseline, size, self.theme.text, content);
        if (resp.focused) try self.paintFocusRing(ring, box * 0.5);

        if (resp.clicked and opts.enabled and !is_on) {
            selected.* = index;
            self.markDirty();
            return true;
        }
        return false;
    }

    pub const SliderOptions = struct {
        min: f32 = 0,
        max: f32 = 1,
        w: Size = .{ .px = 160 },
        enabled: bool = true,
        /// Shown next to the label.
        show_value: bool = true,
    };

    /// A slider: dragging anywhere on the track moves the value (no "grab the
    /// knob" friction), and the pixels map linearly onto the range.
    pub fn slider(self: *Ui, content: []const u8, value: *f32, opts: SliderOptions) Error!bool {
        const id = self.idOf(content);
        const size = self.theme.font_size;
        const m = font.measure(content, size);
        var value_buf: [32]u8 = undefined;
        const value_text = if (opts.show_value)
            std.fmt.bufPrint(&value_buf, "{d:.2}", .{value.*}) catch ""
        else
            "";
        const vm = font.measure(value_text, size);
        const label_w = m.width + self.theme.pad;
        const min_w = label_w + @max(opts.w.resolve(0, 0), 40) + vm.width;

        const style = Style{ .w = opts.w, .h = Size{ .px = @max(self.theme.item_h, m.height) }, .min_w = min_w };
        const r = self.occupy(style, .{ .x = min_w, .y = m.height });
        const resp = self.respond(id, r, true, opts.enabled);
        if (self.measuring) return false;

        const range = if (opts.max > opts.min) opts.max - opts.min else 1;
        var changed = false;
        const track = Rect.init(
            r.x + label_w,
            r.y + r.h * 0.5 - 3,
            @max(20, r.w - label_w - (if (opts.show_value) vm.width + self.theme.pad else 0)),
            6,
        );
        if (resp.active and opts.enabled) {
            const t = std.math.clamp((self.input.pointer.x - track.x) / @max(1, track.w), 0, 1);
            const next = opts.min + t * range;
            if (next != value.*) {
                value.* = next;
                changed = true;
            }
        }
        if (resp.focused and opts.enabled) {
            const step = range / 100;
            if (self.input.keyPressed(.left)) {
                value.* = std.math.clamp(value.* - step, opts.min, opts.max);
                changed = true;
            }
            if (self.input.keyPressed(.right)) {
                value.* = std.math.clamp(value.* + step, opts.min, opts.max);
                changed = true;
            }
        }
        if (changed) self.markDirty();

        const t = std.math.clamp((value.* - opts.min) / range, 0, 1);
        try self.paintRounded(track, 3, self.theme.track, 1, self.theme.border);
        if (t > 0) {
            try self.paintRounded(Rect.init(track.x, track.y, track.w * t, track.h), 3, self.theme.accent, 0, Color.transparent);
        }
        const knob = Rect.init(track.x + track.w * t - 6, r.y + r.h * 0.5 - 6, 12, 12);
        try self.paintRounded(knob, 6, Color.lerp(self.theme.text, self.theme.accent_text, resp.hover_t * 0.5), 1, self.theme.border);

        const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
        try self.paintText(r.x, baseline, size, self.theme.text_dim, content);
        if (opts.show_value) {
            const vx = r.x + r.w - vm.width;
            try self.paintText(vx, baseline, size, self.theme.text, value_text);
        }
        if (resp.focused) try self.paintFocusRing(track, 4);
        return changed;
    }

    pub const ProgressOptions = struct {
        w: f32 = 160,
        h: f32 = 8,
    };

    /// A progress bar: display only, so no id and no retained state.
    pub fn progress(self: *Ui, value: f32, opts: ProgressOptions) Error!void {
        const r = self.occupy(.{ .w = Size{ .px = opts.w }, .h = Size{ .px = opts.h } }, .{ .x = opts.w, .y = opts.h });
        if (self.measuring) return;
        const t = std.math.clamp(value, 0, 1);
        try self.paintRounded(r, r.h * 0.5, self.theme.track, 1, self.theme.border);
        if (t > 0) {
            try self.paintRounded(Rect.init(r.x, r.y, r.w * t, r.h), r.h * 0.5, self.theme.accent, 0, Color.transparent);
        }
    }

    /// A hairline across the container.
    pub fn separator(self: *Ui) Error!void {
        const r = self.occupy(.{ .w = .full, .h = Size{ .px = 1 } }, .{ .x = 0, .y = 1 });
        if (self.measuring) return;
        try self.paintRect(r, self.theme.border);
    }

    /// Empty space along the container's main axis.
    pub fn spacer(self: *Ui, extent: f32) Error!void {
        _ = self.occupy(.{ .w = Size{ .px = extent }, .h = Size{ .px = extent } }, .{ .x = extent, .y = extent });
    }

    // ------------------------------------------------------------ scroll view

    pub const ScrollOptions = struct {
        style: Style = .{},
        bar: f32 = 8,
    };

    /// A container that scrolls its children. The offset is retained (it cannot
    /// be recomputed from a single frame's input), the clipping is the
    /// painter's, and the bar only appears when the content does not fit.
    pub fn scrollable(self: *Ui, id: []const u8, opts: ScrollOptions, body: anytype) Error!void {
        const avail = self.available();
        self.pushId(id);
        defer self.popId();
        const key = self.idOf("scroll");

        const measured = try self.measureContainer(opts.style, .{ .x = @max(0, avail.x - opts.bar), .y = std.math.floatMax(f32) }, body);
        const r = self.occupy(opts.style, measured.content);
        if (self.measuring) return;
        if (r.isEmpty()) return;

        // Unlike a plain container, a scroll region paints itself even with a
        // transparent style: it owns the pixels that the bar and the clipped
        // children are drawn against.
        try self.paintRounded(r, opts.style.radius, opts.style.bg, opts.style.border, opts.style.border_color);

        const view = r.inset(.{ .l = opts.style.pad.l, .t = opts.style.pad.t, .r = opts.style.pad.r + opts.bar, .b = opts.style.pad.b });
        const content_h = measured.content.y;
        const scroll = try self.scrollRegion(key, r, view, content_h, opts.bar, opts.style.pad);

        // Children are laid out in an unclipped column at `-scroll` and then
        // clipped: layout never has to know about scrolling.
        var frame = LayoutFrame{
            .mode = .draw,
            .dir = opts.style.dir,
            .content = Rect.init(view.x, view.y - scroll, view.w, content_h),
            .gap = opts.style.gap,
            .cross_align = opts.style.cross_align,
            .cursor = view.y - scroll,
            .cross = view.w,
            .fixed_main = measured.fixed_main,
            .fill_content = measured.fill_content,
            .weight = measured.weight,
            .gaps = measured.gaps,
            .max_cross = measured.max_cross,
        };
        frame.fill_extent = @max(0, content_h - measured.fixed_main);

        const previous_clip = self.clip;
        try self.pushClip(view);
        self.pushLayout(frame);
        defer {
            self.popLayout();
            self.popClip(previous_clip) catch {};
        }
        try body.run(self);
    }

    // ------------------------------------------------------------- scrolling

    /// The scrolling half of a scroll region: wheel input, the bar (drag and
    /// paint), clamping, and storing the offset in the retained state.
    ///
    /// `scrollable` and `virtualList` both go through here, so a list and a
    /// viewport cannot disagree about what a wheel event does or where the thumb
    /// sits. Returns the offset to lay the children out at.
    fn scrollRegion(self: *Ui, key: Id, r: Rect, view: Rect, content_h: f32, bar: f32, pad: Edges) Error!f32 {
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

    // ------------------------------------------------------------- virtual list

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
        comptime rowFn: fn (@TypeOf(ctx), *Ui, usize, Rect) Error!void,
    ) Error!void {
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

    // ---------------------------------------------------------------- tooltip

    /// Queues a tooltip for the widget `id` when it is hot.
    ///
    /// The tip is painted by `endFrame`, *after* the body, so it lands on top of
    /// everything the frame drew and cannot be covered by a later widget or by
    /// the container it belongs to. That is the whole reason it is deferred:
    /// a popup needs a second pass over the whole window, a tooltip does not.
    pub fn tooltip(self: *Ui, id: Id, anchor: Rect, text: []const u8) Error!void {
        if (self.measuring or self.skipping) return;
        if (!self.ctx.hot.eql(id) or text.len == 0) return;
        self.tip = .{ .anchor = anchor, .text = text };
    }

    // ------------------------------------------------------- collapsing header

    pub const CollapsingOptions = struct {
        /// State on the first frame; after that the retained flag wins.
        open: bool = false,
    };

    /// A header that shows or hides a section. The open flag is retained — it is
    /// exactly the kind of state a body cannot recompute.
    pub fn collapsing(self: *Ui, id: []const u8, title: []const u8, opts: CollapsingOptions, body: anytype) Error!bool {
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
        var frame = drawFrame(inner_style, box.inset(inner_style.pad), measured);
        frame.cursor = box.y + inner_style.pad.t;
        self.pushLayout(frame);
        defer self.popLayout();
        try body.run(self);
        return open;
    }

    // -------------------------------------------------------------- text field

    pub const TextEditOptions = struct {
        placeholder: []const u8 = "",
        w: Size = .full,
        enabled: bool = true,
    };

    /// The one genuinely retained widget: a caret is a cursor, and a cursor is
    /// not a function of the string. Everything a *caller* sees — the value and
    /// the change flag — stays immediate.
    pub fn textEdit(self: *Ui, id: []const u8, field: *TextField, opts: TextEditOptions) Error!bool {
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

    // ---------------------------------------------------------------- helpers

    fn wrapText(self: *Ui, content: []const u8, size: f32, max_width: f32) []font.Line {
        const a = self.arena.allocator();
        if (content.len == 0) {
            const one = a.alloc(font.Line, 1) catch return &.{};
            one[0] = .{ .start = 0, .len = 0 };
            return one;
        }
        const buf = a.alloc(font.Line, @min(self.max_lines, content.len + 1)) catch return &.{};
        return font.wrap(content, size, max_width, buf);
    }
};

/// Salts so hover and press animations of the same widget are different ids.
const hover_salt: u64 = 0x9E3779B97F4A7C15;
const press_salt: u64 = 0xC2B2AE3D27D4EB4F;

test "style size resolution" {
    const s = Style{ .w = .{ .px = 40 }, .h = .full };
    try std.testing.expectEqual(@as(f32, 40), s.sizeAlong(.x).resolve(10, 100));
    try std.testing.expectEqual(@as(f32, 100), s.sizeAlong(.y).resolve(10, 100));
    try std.testing.expectEqual(@as(f32, 10), (Style{}).sizeAlong(.x).resolve(10, 100));
}

test "ids are scoped by pushId" {
    var ui = Ui.init(std.testing.allocator, .{});
    defer ui.deinit();
    const a = ui.idOf("ok");
    ui.pushId("panel");
    const b = ui.idOf("ok");
    ui.pushId("inner");
    const c = ui.idOf("ok");
    ui.popId();
    const d = ui.idOf("ok");
    try std.testing.expect(!a.eql(b));
    try std.testing.expect(!b.eql(c));
    try std.testing.expect(b.eql(d));
}
