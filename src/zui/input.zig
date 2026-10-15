//! Input as a *value*.
//!
//! A UI library that owns the window, the event loop and the IME is a UI library
//! that cannot be tested. `zui` takes the opposite route: one frame's worth of
//! input is a plain struct — pointer position, edges, scroll, key edges, typed
//! codepoints, time — and the host (an SDL/Wayland/win32 program, a test, a
//! scripted demo) fills it in. Everything the library does is then a pure
//! function of `(Input, previous state)`.
const std = @import("std");
const types = @import("types.zig");

const Vec2 = types.Vec2;

/// Keys the library actually reads. Anything else a host sends is ignored, so a
/// host never has to translate its whole keyboard.
pub const KeyCode = enum {
    tab,
    enter,
    escape,
    backspace,
    delete,
    space,
    left,
    right,
    up,
    down,
    home,
    end,
    page_up,
    page_down,
};

pub const Key = struct {
    code: KeyCode,
    /// `false` is a release edge, which the library needs to tell a tap from a
    /// held key (`repeat`).
    down: bool = true,
    /// The host's auto-repeat: the same as a fresh press for text and navigation.
    repeat: bool = false,
};

pub const Input = struct {
    /// Position in logical pixels, relative to the UI's top-left corner.
    pointer: Vec2 = Vec2.zero,
    pointer_down: bool = false,
    /// `false -> true` on this frame. A host that only reports levels still
    /// works: `zui` derives the edges it needs from `pointer_down` when the
    /// frame's input is fed through `deriveEdges`.
    pointer_pressed: bool = false,
    pointer_released: bool = false,
    /// Wheel or trackpad: in logical pixels (positive `y` scrolls down).
    scroll: Vec2 = Vec2.zero,

    keys: []const Key = &.{},
    /// Codepoints typed this frame, in order (UTF-32; the font only has Latin,
    /// but the input layer does not care what the host typed).
    text: []const u21 = &.{},

    shift: bool = false,
    ctrl: bool = false,
    alt: bool = false,

    /// Monotonic seconds, and the time since the previous frame: animations read
    /// both, and they are *inputs* rather than a clock read so that a test can
    /// drive time by hand.
    time: f64 = 0,
    dt: f32 = 1.0 / 60.0,

    pub const empty: Input = .{};

    /// A press or release edge for `code`, which is what widget activation and
    /// focus navigation look at.
    pub fn keyPressed(self: Input, code: KeyCode) bool {
        for (self.keys) |k| {
            if (k.code == code and k.down) return true;
        }
        return false;
    }

    pub fn keyDown(self: Input, code: KeyCode) bool {
        for (self.keys) |k| {
            if (k.code == code and k.down) return true;
        }
        return false;
    }

    /// True when this frame carries anything that could change the picture.
    ///
    /// This is the input half of the idle rule: a frame with no movement, no
    /// click, no key, no scroll and no typed text cannot change the UI by
    /// itself, so the body does not have to run again (see `Ui.beginFrame`).
    pub fn isIdle(self: Input) bool {
        return !self.pointer_down and
            !self.pointer_pressed and
            !self.pointer_released and
            self.scroll.eql(Vec2.zero) and
            self.keys.len == 0 and
            self.text.len == 0;
    }

    /// Fills in `pointer_pressed`/`pointer_released` from the level, for hosts
    /// that only push the current button state. `prev_down` is the host's own
    /// previous level (not the library's), so this stays a pure function.
    pub fn deriveEdges(self: *Input, prev_down: bool) void {
        self.pointer_pressed = self.pointer_down and !prev_down;
        self.pointer_released = !self.pointer_down and prev_down;
    }
};

/// A scripted host, used by tests and by the demo: it plays back a list of
/// input snapshots, one per frame, so a whole interaction — click, drag, type —
/// becomes data instead of a manual test.
pub const Script = struct {
    frames: []const Input,
    index: usize = 0,

    pub fn next(self: *Script) Input {
        if (self.index >= self.frames.len) return Input.empty;
        const f = self.frames[self.index];
        self.index += 1;
        return f;
    }

    pub fn done(self: *const Script) bool {
        return self.index >= self.frames.len;
    }
};

test "edge derivation from a level" {
    var i = Input{ .pointer_down = false };
    i.deriveEdges(false);
    try std.testing.expect(!i.pointer_pressed and !i.pointer_released);

    i.pointer_down = true;
    i.deriveEdges(false);
    try std.testing.expect(i.pointer_pressed and !i.pointer_released);

    i.deriveEdges(true);
    try std.testing.expect(!i.pointer_pressed and !i.pointer_released);

    i.pointer_down = false;
    i.deriveEdges(true);
    try std.testing.expect(!i.pointer_pressed and i.pointer_released);
}

test "idle detection" {
    try std.testing.expect(Input.empty.isIdle());
    var busy = Input{ .pointer = .{ .x = 3, .y = 4 } };
    try std.testing.expect(busy.isIdle()); // moving the pointer alone is not input
    busy.pointer_down = true;
    try std.testing.expect(!busy.isIdle());
    var typed = Input{ .text = &.{'a'} };
    try std.testing.expect(!typed.isIdle());
    const keys = [_]Key{.{ .code = .left }};
    var nav = Input{ .keys = &keys };
    try std.testing.expect(!nav.isIdle());
    try std.testing.expect(nav.keyPressed(.left));
    try std.testing.expect(!nav.keyPressed(.right));
}
