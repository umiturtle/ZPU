//! A text buffer and the string operations a field needs.
//!
//! The buffer is caller-owned and fixed (`[]u8`), because that is the honest
//! shape of UI text in Zig: no hidden growth, no allocator in the widget, no
//! surprise. `TextField` is a `len` next to that slice, and every edit is a
//! memmove — the operations here are the ones a one-line field actually uses.
const std = @import("std");
const font = @import("font.zig");

pub const TextField = struct {
    /// Capacity. The field never allocates and never writes past this.
    buf: []u8,
    len: usize = 0,

    pub fn init(buf: []u8) TextField {
        return .{ .buf = buf, .len = 0 };
    }

    pub fn fromString(buf: []u8, text: []const u8) TextField {
        const n = @min(buf.len, text.len);
        @memcpy(buf[0..n], text[0..n]);
        return .{ .buf = buf, .len = n };
    }

    pub fn slice(self: *const TextField) []const u8 {
        return self.buf[0..self.len];
    }

    pub fn capacity(self: *const TextField) usize {
        return self.buf.len;
    }

    pub fn isFull(self: *const TextField) bool {
        return self.len >= self.buf.len;
    }

    /// Inserts one byte at `at`, if there is room. Returns the offset after the
    /// inserted byte, or `at` when the buffer is full.
    pub fn insert(self: *TextField, at: usize, c: u8) usize {
        const p = @min(at, self.len);
        if (self.isFull()) return p;
        std.mem.copyBackwards(u8, self.buf[p + 1 .. self.len + 1], self.buf[p..self.len]);
        self.buf[p] = c;
        self.len += 1;
        return p + 1;
    }

    pub fn insertSlice(self: *TextField, at: usize, text: []const u8) usize {
        var p = @min(at, self.len);
        for (text) |c| p = self.insert(p, c);
        return p;
    }

    /// Deletes `[from, to)`, returning the new caret position (`from`).
    pub fn deleteRange(self: *TextField, from: usize, to: usize) usize {
        const a = @min(from, self.len);
        const b = @min(@max(to, a), self.len);
        if (b == a) return a;
        std.mem.copyForwards(u8, self.buf[a .. self.len - (b - a)], self.buf[b..self.len]);
        self.len -= b - a;
        return a;
    }

    pub fn clear(self: *TextField) void {
        self.len = 0;
    }

    /// Byte offset of the caret when the pointer is at `x` pixels from the
    /// text's left edge: the midpoint rule (nearest boundary) that every text
    /// field uses.
    pub fn offsetAtX(self: *const TextField, size: f32, x: f32) usize {
        const s = font.scaleOf(size);
        var pen: f32 = 0;
        for (self.slice(), 0..) |c, i| {
            const w = font.advance(c, s);
            if (x < pen + w * 0.5) return i;
            pen += w;
        }
        return self.len;
    }

    /// Caret x offset for a byte offset, for drawing the caret.
    pub fn xOfOffset(self: *const TextField, size: f32, offset: usize) f32 {
        const s = font.scaleOf(size);
        var pen: f32 = 0;
        for (self.slice()[0..@min(offset, self.len)]) |c| pen += font.advance(c, s);
        return pen;
    }

    /// Moves one byte left/right, which is one character in this font's ASCII
    /// world. (`utf8` stepping belongs with a real shaper; the model here is
    /// bytes, stated plainly.)
    pub fn stepLeft(at: usize) usize {
        return if (at == 0) 0 else at - 1;
    }

    pub fn stepRight(self: *const TextField, at: usize) usize {
        return if (at >= self.len) self.len else at + 1;
    }
};

test "insert and delete keep the buffer consistent" {
    var buf: [8]u8 = undefined;
    var f = TextField.init(&buf);
    try std.testing.expectEqual(@as(usize, 0), f.len);

    _ = f.insert(0, 'a');
    _ = f.insert(1, 'c');
    try std.testing.expectEqual(@as(usize, 2), f.insert(1, 'b'));
    try std.testing.expectEqualStrings("abc", f.slice());
    try std.testing.expectEqual(@as(usize, 2), f.insert(1, 'x'));
    try std.testing.expectEqualStrings("axbc", f.slice());

    // Deleting a range returns the new caret position and closes the gap.
    try std.testing.expectEqual(@as(usize, 1), f.deleteRange(1, 3));
    try std.testing.expectEqualStrings("ac", f.slice());

    // Backspace at the start, and delete past the end, are no-ops.
    try std.testing.expectEqual(@as(usize, 0), f.deleteRange(0, 0));
    try std.testing.expectEqual(@as(usize, 2), f.deleteRange(2, 99));
    try std.testing.expectEqualStrings("ac", f.slice());
}

test "the buffer refuses to overflow" {
    var buf: [3]u8 = undefined;
    var f = TextField.fromString(&buf, "abcd");
    try std.testing.expectEqualStrings("abc", f.slice());
    try std.testing.expect(f.isFull());
    const at = f.insert(1, 'z');
    try std.testing.expectEqual(@as(usize, 1), at);
    try std.testing.expectEqualStrings("abc", f.slice());
}

test "caret movement and offset hit testing" {
    var buf: [16]u8 = undefined;
    var f = TextField.fromString(&buf, "hello");
    try std.testing.expectEqual(@as(usize, 0), TextField.stepLeft(0));
    try std.testing.expectEqual(@as(usize, 4), TextField.stepLeft(5));
    try std.testing.expectEqual(@as(usize, 5), f.stepRight(5));
    try std.testing.expectEqual(@as(usize, 1), f.stepRight(0));

    // Offsets and x positions are inverses of each other, up to the midpoint.
    for (0..f.len + 1) |i| {
        const x = f.xOfOffset(14, i);
        try std.testing.expectEqual(i, f.offsetAtX(14, x));
    }
    try std.testing.expectEqual(f.len, f.offsetAtX(14, 10_000));
    try std.testing.expectEqual(@as(usize, 0), f.offsetAtX(14, -5));
}
