//! The retained minimum.
//!
//! An immediate-mode API recomputes the picture every frame, so almost nothing
//! needs to be remembered. The exceptions are exact and short, and they all live
//! here, in one struct, keyed by widget [`Id`]:
//!
//! * **interaction** — which widget the pointer is over (`hot`), which one owns
//!   the press (`active`), which one has keyboard focus (`focus`), and the focus
//!   ring for Tab;
//! * **scroll** — an offset can only be *derived* from input over time, so it is
//!   stored per scrollable id;
//! * **state flags** — collapsing headers open/closed, a checkbox's "was it
//!   just toggled" memory, anything a caller would otherwise keep outside;
//! * **animation** — a value that approaches a target over time (hover glows,
//!   the caret blink) cannot be a pure function of the current frame;
//! * **text edits** — carets, selections and the recent-edit history of a field.
//!
//! Two rules make this tractable: nothing here is ever read by a widget before
//! it is written for the first time (defaults are values, not missing entries),
//! and nothing here is allocated per frame (the maps only grow the first time an
//! id is seen).
const std = @import("std");
const types = @import("types.zig");

const Id = types.Id;
const Vec2 = types.Vec2;

/// Per-field editing state: the one place where "remember it" is not an
/// optimisation but a requirement — a caret is a cursor into a string, and a
/// cursor cannot be recomputed from the string alone.
pub const EditState = struct {
    /// Byte offset of the caret into the field's buffer.
    caret: usize = 0,
    /// Anchor of a selection; equal to `caret` when there is none.
    anchor: usize = 0,
    /// First visible byte, so a long line scrolls instead of overflowing.
    scroll: f32 = 0,
    /// Seconds since the caret was last moved, for the blink.
    caret_age: f32 = 0,
    /// Set when the field gained focus, so the first frame selects nothing.
    focused: bool = false,

    pub fn selection(self: EditState) [2]usize {
        return .{ @min(self.caret, self.anchor), @max(self.caret, self.anchor) };
    }

    pub fn hasSelection(self: EditState) bool {
        return self.caret != self.anchor;
    }

    pub fn moveTo(self: *EditState, at: usize) void {
        self.caret = at;
        self.anchor = at;
        self.caret_age = 0;
    }

    pub fn extendTo(self: *EditState, at: usize) void {
        self.caret = at;
        self.caret_age = 0;
    }
};

pub const Context = struct {
    alloc: std.mem.Allocator,

    /// The widget under the pointer as of the last frame.
    hot: Id = Id.none,
    /// The widget that owns the press: while it is set, no other widget reacts.
    active: Id = Id.none,
    /// Keyboard focus.
    focus: Id = Id.none,
    /// Set to a widget's id by `Ui.requestFocus` and consumed by that widget on
    /// its next call, so a program can move focus from outside the widget.
    focus_request: Id = Id.none,
    /// Focus ring for Tab: ids in the order the widgets were built last frame.
    focusables: std.ArrayListUnmanaged(Id) = .empty,
    /// Where `focus` sits in `focusables`, so Tab moves *relative* to it.
    focus_index: usize = 0,

    scroll: std.AutoHashMapUnmanaged(Id, Vec2) = .empty,
    flags: std.AutoHashMapUnmanaged(Id, bool) = .empty,
    anims: std.AutoHashMapUnmanaged(Id, f32) = .empty,
    edits: std.AutoHashMapUnmanaged(Id, EditState) = .empty,

    /// Set by `animate` while a value has not reached its target yet: the
    /// second half of the idle rule (input, animations, an explicit
    /// `markDirty`), because a moving value changes the picture with no input
    /// at all.
    animating: bool = false,

    /// Counters, for the demo and the benchmark to report and the tests to
    /// assert on.
    frames: u64 = 0,
    rebuilt: u64 = 0,
    skipped: u64 = 0,

    pub fn init(alloc: std.mem.Allocator) Context {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *Context) void {
        self.focusables.deinit(self.alloc);
        self.scroll.deinit(self.alloc);
        self.flags.deinit(self.alloc);
        self.anims.deinit(self.alloc);
        self.edits.deinit(self.alloc);
        self.* = .{ .alloc = self.alloc };
    }

    /// Drops everything that is keyed by id. Called when the widget tree is
    /// torn down (a dialog closes, a list is replaced), so ids from the old tree
    /// cannot leak state into the new one.
    pub fn forgetAll(self: *Context) void {
        self.animating = false;
        self.hot = Id.none;
        self.active = Id.none;
        self.focus = Id.none;
        self.focus_request = Id.none;
        self.focusables.clearRetainingCapacity();
        self.focus_index = 0;
        self.scroll.clearRetainingCapacity();
        self.flags.clearRetainingCapacity();
        self.anims.clearRetainingCapacity();
        self.edits.clearRetainingCapacity();
    }

    /// Forgets one widget's retained state.
    pub fn forget(self: *Context, id: Id) void {
        _ = self.scroll.remove(id);
        _ = self.flags.remove(id);
        _ = self.anims.remove(id);
        _ = self.edits.remove(id);
        if (self.hot.eql(id)) self.hot = Id.none;
        if (self.active.eql(id)) self.active = Id.none;
        if (self.focus.eql(id)) self.focus = Id.none;
    }

    // ------------------------------------------------------------------ scroll

    pub fn scrollOf(self: *const Context, id: Id) Vec2 {
        return self.scroll.get(id) orelse Vec2.zero;
    }

    pub fn setScroll(self: *Context, id: Id, v: Vec2) void {
        self.scroll.put(self.alloc, id, v) catch {};
    }

    // ------------------------------------------------------------------- flags

    pub fn flag(self: *const Context, id: Id, default: bool) bool {
        return self.flags.get(id) orelse default;
    }

    pub fn setFlag(self: *Context, id: Id, v: bool) void {
        self.flags.put(self.alloc, id, v) catch {};
    }

    // -------------------------------------------------------------- animations

    /// A value that walks towards `target` at `rate` (larger is snappier), with
    /// the exponential approach that is stable for any `dt`:
    ///
    ///     v += (target - v) * (1 - exp(-rate * dt))
    ///
    /// Written this way rather than as `v += (target - v) * rate * dt` so that a
    /// frame hitch cannot overshoot. A value within `epsilon` snaps, so the
    /// value really does stop changing and the idle rule can fire.
    pub fn animate(self: *Context, id: Id, target: f32, rate: f32, dt: f32) f32 {
        const cur = self.anims.get(id) orelse target;
        if (dt <= 0) return cur;
        const k = 1.0 - @exp(-rate * dt);
        var next = cur + (target - cur) * k;
        if (@abs(target - next) < 1.0 / 512.0) next = target;
        if (next != target) self.animating = true;
        self.anims.put(self.alloc, id, next) catch {};
        return next;
    }

    pub fn animValue(self: *const Context, id: Id) ?f32 {
        return self.anims.get(id);
    }

    // --------------------------------------------------------------- text edit

    pub fn edit(self: *Context, id: Id) *EditState {
        const gop = self.edits.getOrPut(self.alloc, id) catch null;
        if (gop) |entry| {
            if (!entry.found_existing) entry.value_ptr.* = .{};
            return entry.value_ptr;
        }
        // Out of memory: a static fallback keeps the library sound, the caller's
        // edit is simply not persisted (the field still works this frame).
        return &fallback_edit;
    }

    pub fn editIfPresent(self: *const Context, id: Id) ?*EditState {
        return self.edits.getPtr(id);
    }
};

/// Used only when the edit map cannot grow. A UI that cannot allocate is
/// broken, but it should not be able to corrupt memory on the way down.
var fallback_edit: EditState = .{};

test "animation approaches its target and stops" {
    var ctx = Context.init(std.testing.allocator);
    defer ctx.deinit();
    const id = Id.hash(0, "hover");

    // The first sighting of an id adopts its target (nothing fades in on the
    // frame a widget appears), so the value has to be seeded.
    const seed = ctx.animate(id, 0.0, 20.0, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 0), seed);

    var frames: usize = 0;
    var v: f32 = 0;
    while (frames < 1000) : (frames += 1) {
        v = ctx.animate(id, 1.0, 20.0, 1.0 / 60.0);
        if (v == 1.0) break;
    }
    try std.testing.expect(ctx.animating);
    // Exponential approach: within a couple of tenths of a second at rate 20,
    // and then *exactly* 1.0, so the frame can be skipped.
    try std.testing.expect(v == 1.0);
    try std.testing.expect(frames < 60);
    try std.testing.expect(ctx.animValue(id) != null);
}

test "flags and scroll default to zero and persist" {
    var ctx = Context.init(std.testing.allocator);
    defer ctx.deinit();
    const a = Id.hash(0, "a");
    const b = Id.hash(0, "b");

    try std.testing.expect(!ctx.flag(a, false));
    try std.testing.expect(ctx.flag(a, true));
    ctx.setFlag(a, true);
    try std.testing.expect(ctx.flag(a, false));
    try std.testing.expect(!ctx.flag(b, false));

    try std.testing.expect(ctx.scrollOf(a).eql(Vec2.zero));
    ctx.setScroll(a, .{ .x = 0, .y = 42 });
    try std.testing.expectEqual(@as(f32, 42), ctx.scrollOf(a).y);
    try std.testing.expectEqual(@as(f32, 0), ctx.scrollOf(b).y);

    ctx.forget(a);
    try std.testing.expectEqual(@as(f32, 0), ctx.scrollOf(a).y);
    try std.testing.expect(!ctx.flag(a, false));
}

test "focus bookkeeping is forgotten wholesale" {
    var ctx = Context.init(std.testing.allocator);
    defer ctx.deinit();
    ctx.hot = Id.hash(0, "x");
    ctx.active = Id.hash(0, "y");
    ctx.focus = Id.hash(0, "z");
    try ctx.focusables.append(ctx.alloc, Id.hash(0, "z"));
    ctx.forgetAll();
    try std.testing.expect(ctx.hot.isNone() and ctx.active.isNone() and ctx.focus.isNone());
    try std.testing.expectEqual(@as(usize, 0), ctx.focusables.items.len);
}

test "edit state selection maths" {
    var e = EditState{};
    e.moveTo(5);
    try std.testing.expect(!e.hasSelection());
    e.extendTo(2);
    try std.testing.expect(e.hasSelection());
    try std.testing.expectEqual([2]usize{ 2, 5 }, e.selection());
    e.caret = 0;
    try std.testing.expectEqual([2]usize{ 0, 5 }, e.selection());
}
