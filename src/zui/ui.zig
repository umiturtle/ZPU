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
const config_mod = @import("config.zig");

// Widget modules - extracted for modularity
const label_widget = @import("widgets/label.zig");
const button_widget = @import("widgets/button.zig");
const separator_widget = @import("widgets/separator.zig");
const checkbox_widget = @import("widgets/checkbox.zig");
const radio_widget = @import("widgets/radio.zig");
const slider_widget = @import("widgets/slider.zig");
const progress_widget = @import("widgets/progress.zig");
const spacer_widget = @import("widgets/spacer.zig");
const tooltip_widget = @import("widgets/tooltip.zig");
const collapsing_widget = @import("widgets/collapsing.zig");
const scrollable_widget = @import("widgets/scrollable.zig");
const virtual_list_widget = @import("widgets/virtual_list.zig");
const text_edit_widget = @import("widgets/text_edit.zig");
const container_widget = @import("widgets/container.zig");
const scroll_region_widget = @import("widgets/scroll_region.zig");
const occupy_widget = @import("widgets/occupy.zig");
const respond_widget = @import("widgets/respond.zig");
const measure_container_widget = @import("widgets/measure_container.zig");
const frame_widget = @import("widgets/frame.zig");
const helpers_widget = @import("widgets/helpers.zig");
const paint_widget = @import("widgets/paint.zig");
const lifecycle_widget = @import("widgets/lifecycle.zig");

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

// Re-export config types for convenience
pub const Quality = config_mod.Quality;
pub const Config = config_mod.Config;
pub const SoftwareBackend = config_mod.SoftwareBackend;
pub const MockBackend = config_mod.MockBackend;

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
pub const LayoutFrame = struct {
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

    pub fn contentMain(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.w else self.content.h;
    }

    fn contentCross(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.h else self.content.w;
    }

    fn cursorStart(self: *const LayoutFrame) f32 {
        return if (self.dir == .x) self.content.x else self.content.y;
    }

    /// What is left of the content box's main axis.
    pub fn remaining(self: *const LayoutFrame) f32 {
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
pub const Tip = struct {
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
        return lifecycle_widget.init(alloc, opts);
    }

    pub fn deinit(self: *Ui) void {
        return lifecycle_widget.deinit(self);
    }

    pub fn resize(self: *Ui, width: f32, height: f32) void {
        return lifecycle_widget.resize(self, width, height);
    }

    pub fn isMeasuring(self: *const Ui) bool {
        return lifecycle_widget.isMeasuring(self);
    }

    pub fn markDirty(self: *Ui) void {
        return lifecycle_widget.markDirty(self);
    }

    pub fn stats(self: *const Ui) Stats {
        return lifecycle_widget.stats(self);
    }

    // ------------------------------------------------------------------- frame

    /// Runs one frame. `body` is any value with `run(self, ui) Error!void`: a
    /// hand-rolled closure, which is how Zig spells one without allocating.
    pub fn run(self: *Ui, input: Input, body: anytype) Error!FrameResult {
        return lifecycle_widget.run(self, input, body);
    }

    pub fn beginFrame(self: *Ui, input: Input) void {
        return frame_widget.beginFrame(self, input);
    }

    pub fn endFrame(self: *Ui) void {
        return frame_widget.endFrame(self);
    }

    /// Draws the queued tooltip: a bordered panel under the anchor, flipped
    /// above it and clamped inside the window when there is no room below.
    pub fn paintTooltip(self: *Ui, tip: Tip) !void {
        return helpers_widget.paintTooltip(self, tip);
    }

    fn result(self: *const Ui) FrameResult {
        return lifecycle_widget.result(self);
    }

    pub fn pushRootLayout(self: *Ui) void {
        return helpers_widget.pushRootLayout(self);
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
        return paint_widget.paintRect(self, r, color);
    }

    pub fn paintRounded(self: *Ui, r: Rect, radius: f32, color: Color, border: f32, border_color: Color) Error!void {
        return paint_widget.paintRounded(self, r, radius, color, border, border_color);
    }

    pub fn paintLine(self: *Ui, a: Vec2, b: Vec2, width: f32, color: Color) Error!void {
        return paint_widget.paintLine(self, a, b, width, color);
    }

    pub fn paintTriangle(self: *Ui, a: Vec2, b: Vec2, c: Vec2, color: Color) Error!void {
        return paint_widget.paintTriangle(self, a, b, c, color);
    }

    /// One line of text, baseline at `y`. The bytes are copied into the frame
    /// arena, so the caller's string may die immediately.
    pub fn paintText(self: *Ui, x: f32, baseline: f32, size: f32, color: Color, content: []const u8) Error!void {
        return paint_widget.paintText(self, x, baseline, size, color, content);
    }

    // ------------------------------------------------------------------ layout

    pub fn top(self: *Ui) *LayoutFrame {
        return &self.layouts[self.depth - 1];
    }

    pub fn pushLayout(self: *Ui, frame: LayoutFrame) void {
        std.debug.assert(self.depth < max_depth);
        self.layouts[self.depth] = frame;
        self.depth += 1;
    }

    pub fn popLayout(self: *Ui) void {
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
    pub fn occupy(self: *Ui, style: Style, content_size: Vec2) Rect {
        return occupy_widget.occupy(self, style, content_size);
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

    pub fn respond(self: *Ui, id: Id, r: Rect, focusable: bool, enabled: bool) Response {
        return respond_widget.respond(self, id, r, focusable, enabled);
    }

    pub fn requestFocus(self: *Ui, id: Id) void {
        if (!self.measuring) self.ctx.focus = id;
    }

    pub fn clearFocus(self: *Ui) void {
        if (!self.measuring) self.ctx.focus = Id.none;
    }

    /// Tab / shift-Tab, against the focus ring the previous frame built. Done
    /// before the body runs, so the widget it lands on sees the focus this frame.
    pub fn handleFocusKeys(self: *Ui) void {
        return helpers_widget.handleFocusKeys(self);
    }

    // --------------------------------------------------------------- containers

    /// A container: measures its body, places itself, then draws its body.
    ///
    /// `body` is any value with `run(self, ui) Error!void` — the struct instance
    /// *is* the captured state.
    pub fn container(self: *Ui, id: []const u8, style: Style, body: anytype) Error!void {
        return container_widget.container(self, id, style, body);
    }

    pub const Measured = struct {
        content: Vec2,
        fixed_main: f32 = 0,
        fill_content: f32 = 0,
        weight: f32 = 0,
        gaps: f32 = 0,
        max_cross: f32 = 0,
    };

    pub fn drawFrame(style: Style, content_rect: Rect, measured: Measured) LayoutFrame {
        return helpers_widget.drawFrame(style, content_rect, measured);
    }

    pub fn measureContainer(self: *Ui, style: Style, avail: Vec2, body: anytype) Error!Measured {
        return measure_container_widget.measureContainer(self, style, avail, body);
    }

    // ----------------------------------------------------------------- widgets

    // Widget option types - re-exported from widget modules for backward compatibility
    pub const LabelOptions = label_widget.LabelOptions;

    /// A run of text. Wraps to the container's width unless told otherwise.
    pub fn label(self: *Ui, content: []const u8, opts: LabelOptions) Error!void {
        return label_widget.label(self, content, opts);
    }

    pub const ButtonOptions = button_widget.ButtonOptions;

    /// A push button. `true` on the frame it was activated — by a click, or by
    /// `enter`/`space` while focused.
    pub fn button(self: *Ui, content: []const u8, opts: ButtonOptions) Error!bool {
        return button_widget.button(self, content, opts);
    }

    /// A button whose id is explicit: for labels that change (an id that moves
    /// loses the press) or for rows of identical labels.
    pub fn buttonId(self: *Ui, id: Id, content: []const u8, opts: ButtonOptions) Error!bool {
        return button_widget.buttonId(self, id, content, opts);
    }

    pub const ToggleOptions = checkbox_widget.ToggleOptions;

    /// A checkbox. `true` on the frame it was toggled.
    pub fn checkbox(self: *Ui, content: []const u8, value: *bool, opts: ToggleOptions) Error!bool {
        return checkbox_widget.checkbox(self, content, value, opts);
    }

    /// A radio button; `selected` is the index of the chosen item in the group.
    pub fn radio(self: *Ui, content: []const u8, selected: *usize, index: usize, opts: ToggleOptions) Error!bool {
        return radio_widget.radio(self, content, selected, index, opts);
    }

    pub const SliderOptions = slider_widget.SliderOptions;

    /// A slider: dragging anywhere on the track moves the value (no "grab the
    /// knob" friction), and the pixels map linearly onto the range.
    pub fn slider(self: *Ui, content: []const u8, value: *f32, opts: SliderOptions) Error!bool {
        return slider_widget.slider(self, content, value, opts);
    }

    pub const ProgressOptions = progress_widget.ProgressOptions;

    /// A progress bar: display only, so no id and no retained state.
    pub fn progress(self: *Ui, value: f32, opts: ProgressOptions) Error!void {
        return progress_widget.progress(self, value, opts);
    }

    /// A hairline across the container.
    pub fn separator(self: *Ui) Error!void {
        return separator_widget.separator(self);
    }

    /// Empty space along the container's main axis.
    pub fn spacer(self: *Ui, extent: f32) void {
        spacer_widget.spacer(self, extent);
    }

    // ------------------------------------------------------------ scroll view

    pub const ScrollOptions = scrollable_widget.ScrollOptions;

    /// A container that scrolls its children. The offset is retained (it cannot
    /// be recomputed from a single frame's input), the clipping is the
    /// painter's, and the bar only appears when the content does not fit.
    pub fn scrollable(self: *Ui, id: []const u8, opts: ScrollOptions, body: anytype) Error!void {
        return scrollable_widget.scrollable(self, id, opts, body);
    }

    // ------------------------------------------------------------- scrolling

    /// The scrolling half of a scroll region: wheel input, the bar (drag and
    /// paint), clamping, and storing the offset in the retained state.
    ///
    /// `scrollable` and `virtualList` both go through here, so a list and a
    /// viewport cannot disagree about what a wheel event does or where the thumb
    /// sits. Returns the offset to lay the children out at.
    pub fn scrollRegion(self: *Ui, key: Id, r: Rect, view: Rect, content_h: f32, bar: f32, pad: Edges) Error!f32 {
        return scroll_region_widget.scrollRegion(self, key, r, view, content_h, bar, pad);
    }

    // ------------------------------------------------------------- virtual list

    pub const VirtualOptions = virtual_list_widget.VirtualOptions;

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
        return virtual_list_widget.virtualList(self, id, rows, opts, ctx, rowFn);
    }

    // ---------------------------------------------------------------- tooltip

    /// Queues a tooltip for the widget `id` when it is hot.
    ///
    /// The tip is painted by `endFrame`, *after* the body, so it lands on top of
    /// everything the frame drew and cannot be covered by a later widget or by
    /// the container it belongs to. That is the whole reason it is deferred:
    /// a popup needs a second pass over the whole window, a tooltip does not.
    pub fn tooltip(self: *Ui, id: Id, anchor: Rect, text: []const u8) Error!void {
        return tooltip_widget.tooltip(self, id, anchor, text);
    }

    // ------------------------------------------------------- collapsing header

    pub const CollapsingOptions = collapsing_widget.CollapsingOptions;

    /// A header that shows or hides a section. The open flag is retained — it is
    /// exactly the kind of state a body cannot recompute.
    pub fn collapsing(self: *Ui, id: []const u8, title: []const u8, opts: CollapsingOptions, body: anytype) Error!bool {
        return collapsing_widget.collapsing(self, id, title, opts, body);
    }

    // -------------------------------------------------------------- text field

    pub const TextEditOptions = text_edit_widget.TextEditOptions;

    /// The one genuinely retained widget: a caret is a cursor, and a cursor is
    /// not a function of the string. Everything a *caller* sees — the value and
    /// the change flag — stays immediate.
    pub fn textEdit(self: *Ui, id: []const u8, field: *TextField, opts: TextEditOptions) Error!bool {
        return text_edit_widget.textEdit(self, id, field, opts);
    }

    // ---------------------------------------------------------------- helpers

    pub fn wrapText(self: *Ui, content: []const u8, size: f32, max_width: f32) []font.Line {
        return helpers_widget.wrapText(self, content, size, max_width);
    }
};

/// Ui with comptime configuration for specialization.
/// Different configs generate different code at compile time.
///
/// Example:
/// ```zig
/// const MyUi = UiWith(.{ .quality = .high, .animations = false });
/// var ui = MyUi.init(alloc, .{});
/// ```
pub fn UiWith(comptime cfg: Config) type {
    return struct {
        pub const config = cfg;
        pub const Inner = Ui;

        inner: Ui,

        pub fn init(alloc: std.mem.Allocator, opts: Options) @This() {
            return .{
                .inner = Ui.init(alloc, opts),
            };
        }

        pub fn deinit(self: *@This()) void {
            self.inner.deinit();
        }

        /// Check if animations are enabled for this config (comptime-known)
        pub fn animationsEnabled() bool {
            return cfg.animations;
        }

        /// Check if tooltips are enabled for this config (comptime-known)
        pub fn tooltipsEnabled() bool {
            return cfg.tooltips;
        }

        /// Check if keyboard navigation is enabled for this config (comptime-known)
        pub fn keyboardNavEnabled() bool {
            return cfg.keyboard_nav;
        }

        /// Get the quality level for this config (comptime-known)
        pub fn qualityLevel() Quality {
            return cfg.quality;
        }

        /// Get the antialiasing level based on quality (comptime-known)
        pub fn antialiasLevel() u2 {
            return cfg.quality.antialiasLevel();
        }

        /// Access the underlying Ui for full API
        pub fn ui(self: *@This()) *Ui {
            return &self.inner;
        }

        pub const ui_ptr = ui;
    };
}

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

test "UiWith comptime specialization" {
    // Test default config
    const DefaultUi = UiWith(.{});
    try std.testing.expect(DefaultUi.animationsEnabled());
    try std.testing.expect(DefaultUi.tooltipsEnabled());
    try std.testing.expect(DefaultUi.keyboardNavEnabled());
    try std.testing.expectEqual(Quality.medium, DefaultUi.qualityLevel());

    // Test custom config with animations disabled
    const NoAnimUi = UiWith(.{ .animations = false });
    try std.testing.expect(!NoAnimUi.animationsEnabled());
    try std.testing.expect(NoAnimUi.tooltipsEnabled()); // default

    // Test custom config with low quality
    const LowQualityUi = UiWith(.{ .quality = .low });
    try std.testing.expectEqual(Quality.low, LowQualityUi.qualityLevel());
    try std.testing.expectEqual(@as(u2, 0), LowQualityUi.antialiasLevel());

    // Test high quality
    const HighQualityUi = UiWith(.{ .quality = .high });
    try std.testing.expectEqual(Quality.high, HighQualityUi.qualityLevel());
    try std.testing.expectEqual(@as(u2, 2), HighQualityUi.antialiasLevel());
}

test "UiWith with different configs generates different types" {
    const Ui1 = UiWith(.{ .quality = .low });
    const Ui2 = UiWith(.{ .quality = .high });

    // Different configs should produce different types
    try std.testing.expect(Ui1 != Ui2);

    // But they should have the same inner type
    try std.testing.expect(Ui1.Inner == Ui2.Inner);
    try std.testing.expect(Ui1.Inner == Ui);
}

test "UiWith runtime usage" {
    var ui_wrapper = UiWith(.{ .quality = .high, .animations = false }).init(std.testing.allocator, .{});
    defer ui_wrapper.deinit();

    // Access underlying Ui
    const u = ui_wrapper.ui();
    try std.testing.expect(u.viewport.x > 0 or u.viewport.x == 0); // Just verify it works

    // Verify comptime config is accessible
    try std.testing.expect(!@TypeOf(ui_wrapper).animationsEnabled());
    try std.testing.expectEqual(Quality.high, @TypeOf(ui_wrapper).qualityLevel());
}
