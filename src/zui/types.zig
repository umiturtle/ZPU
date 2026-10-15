//! Geometry, colour and identity for `zui`.
//!
//! Deliberately dependency-free: nothing here knows what an image, a tensor or a
//! window is. `zui/render.zig` is the only file in the library that imports
//! `zgf`, and it converts at that boundary — the same rule the rest of the
//! project follows for backends.
const std = @import("std");

/// A point or a size in logical pixels. `y` grows downwards, like every screen.
pub const Vec2 = struct {
    x: f32 = 0,
    y: f32 = 0,

    pub const zero: Vec2 = .{ .x = 0, .y = 0 };

    pub fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }

    pub fn splat(v: f32) Vec2 {
        return .{ .x = v, .y = v };
    }

    pub fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }

    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }

    pub fn scale(a: Vec2, s: f32) Vec2 {
        return .{ .x = a.x * s, .y = a.y * s };
    }

    pub fn max(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = @max(a.x, b.x), .y = @max(a.y, b.y) };
    }

    pub fn index(self: Vec2, axis: Axis) f32 {
        return if (axis == .x) self.x else self.y;
    }

    /// Flips one axis, for code that has to be written once and used for both
    /// a row and a column.
    pub fn along(self: Vec2, axis: Axis) Vec2 {
        return if (axis == .x) self else .{ .x = self.y, .y = self.x };
    }

    pub fn eql(a: Vec2, b: Vec2) bool {
        return a.x == b.x and a.y == b.y;
    }
};

/// Which axis a layout runs along.
pub const Axis = enum {
    x,
    y,

    pub fn other(self: Axis) Axis {
        return if (self == .x) .y else .x;
    }
};

/// A rectangle in logical pixels: `x`/`y` is the top-left corner.
pub const Rect = struct {
    x: f32 = 0,
    y: f32 = 0,
    w: f32 = 0,
    h: f32 = 0,

    pub fn fromSize(w: f32, h: f32) Rect {
        return .{ .w = w, .h = h };
    }

    pub fn init(x: f32, y: f32, w: f32, h: f32) Rect {
        return .{ .x = x, .y = y, .w = w, .h = h };
    }

    pub fn min(self: Rect) Vec2 {
        return .{ .x = self.x, .y = self.y };
    }

    pub fn max(self: Rect) Vec2 {
        return .{ .x = self.x + self.w, .y = self.y + self.h };
    }

    pub fn size(self: Rect) Vec2 {
        return .{ .x = self.w, .y = self.h };
    }

    pub fn center(self: Rect) Vec2 {
        return .{ .x = self.x + self.w * 0.5, .y = self.y + self.h * 0.5 };
    }

    pub fn contains(self: Rect, p: Vec2) bool {
        return p.x >= self.x and p.x < self.x + self.w and p.y >= self.y and p.y < self.y + self.h;
    }

    pub fn overlaps(self: Rect, o: Rect) bool {
        return self.x < o.x + o.w and o.x < self.x + self.w and self.y < o.y + o.h and o.y < self.y + self.h;
    }

    /// The intersection, which may be empty (zero-sized) when they miss.
    pub fn intersect(self: Rect, o: Rect) Rect {
        const x0 = @max(self.x, o.x);
        const y0 = @max(self.y, o.y);
        const x1 = @min(self.x + self.w, o.x + o.w);
        const y1 = @min(self.y + self.h, o.y + o.h);
        return .{ .x = x0, .y = y0, .w = @max(0, x1 - x0), .h = @max(0, y1 - y0) };
    }

    pub fn isEmpty(self: Rect) bool {
        return !(self.w > 0 and self.h > 0);
    }

    pub fn inset(self: Rect, e: Edges) Rect {
        return .{
            .x = self.x + e.l,
            .y = self.y + e.t,
            .w = @max(0, self.w - e.l - e.r),
            .h = @max(0, self.h - e.t - e.b),
        };
    }

    /// x, y, w, h as a plain array: the form an instance carries a box in.
    pub fn toArray(self: Rect) [4]f32 {
        return .{ self.x, self.y, self.w, self.h };
    }

    pub fn fromArray(v: [4]f32) Rect {
        return .{ .x = v[0], .y = v[1], .w = v[2], .h = v[3] };
    }

    /// The same rectangle grown by `by` on every side (the inverse of `shrink`).
    /// Used to pad a bounding box by the one pixel an antialiased edge can
    /// reach into.
    pub fn outset(self: Rect, by: f32) Rect {
        return .{ .x = self.x - by, .y = self.y - by, .w = self.w + by * 2, .h = self.h + by * 2 };
    }

    pub fn shrink(self: Rect, by: f32) Rect {
        return self.inset(.all(by));
    }

    pub fn offset(self: Rect, d: Vec2) Rect {
        return .{ .x = self.x + d.x, .y = self.y + d.y, .w = self.w, .h = self.h };
    }

    /// Grows to include `o`.
    pub fn merged(self: Rect, o: Rect) Rect {
        if (self.isEmpty()) return o;
        if (o.isEmpty()) return self;
        const x0 = @min(self.x, o.x);
        const y0 = @min(self.y, o.y);
        const x1 = @max(self.x + self.w, o.x + o.w);
        const y1 = @max(self.y + self.h, o.y + o.h);
        return .{ .x = x0, .y = y0, .w = x1 - x0, .h = y1 - y0 };
    }

    /// A box of `size`, placed inside `self` according to `align`.
    pub fn place(self: Rect, inner: Vec2, a: Alignment) Rect {
        const free = Vec2.sub(self.size(), inner);
        const off = switch (a) {
            .start => Vec2.zero,
            .center => free.scale(0.5),
            .end => free,
            .stretch => Vec2.zero,
        };
        const s = switch (a) {
            .stretch => self.size(),
            else => inner,
        };
        return .{ .x = self.x + off.x, .y = self.y + off.y, .w = s.x, .h = s.y };
    }
};

/// Distances from the four edges of a box.
pub const Edges = struct {
    l: f32 = 0,
    t: f32 = 0,
    r: f32 = 0,
    b: f32 = 0,

    pub fn all(v: f32) Edges {
        return .{ .l = v, .t = v, .r = v, .b = v };
    }

    pub fn xy(x: f32, y: f32) Edges {
        return .{ .l = x, .t = y, .r = x, .b = y };
    }

    pub fn sum(self: Edges, axis: Axis) f32 {
        return if (axis == .x) self.l + self.r else self.t + self.b;
    }

    pub fn start(self: Edges, axis: Axis) f32 {
        return if (axis == .x) self.l else self.t;
    }

    pub fn end(self: Edges, axis: Axis) f32 {
        return if (axis == .x) self.r else self.b;
    }
};

/// Colour, straight alpha, components in `0..1` like the rest of the project.
pub const Color = struct {
    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,
    a: f32 = 1,

    pub const transparent: Color = .{ .a = 0 };

    pub fn rgb(r: f32, g: f32, b: f32) Color {
        return .{ .r = r, .g = g, .b = b };
    }

    pub fn rgba(r: f32, g: f32, b: f32, a: f32) Color {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    /// `0xRRGGBB` with an alpha, so a palette reads like a palette.
    pub fn hex(rgb_value: u32, a: f32) Color {
        return .{
            .r = @as(f32, @floatFromInt((rgb_value >> 16) & 0xff)) / 255.0,
            .g = @as(f32, @floatFromInt((rgb_value >> 8) & 0xff)) / 255.0,
            .b = @as(f32, @floatFromInt(rgb_value & 0xff)) / 255.0,
            .a = a,
        };
    }

    pub fn fromRect(r: Rect) Color {
        return .{ .r = r.x, .g = r.y, .b = r.w, .a = r.h };
    }

    /// The four components as a plain array: the form a GPU buffer wants, and
    /// the form an instance carries.
    pub fn toArray(self: Color) [4]f32 {
        return .{ self.r, self.g, self.b, self.a };
    }

    pub fn fromArray(v: [4]f32) Color {
        return .{ .r = v[0], .g = v[1], .b = v[2], .a = v[3] };
    }

    pub fn withAlpha(self: Color, a: f32) Color {
        var c = self;
        c.a = a;
        return c;
    }

    pub fn lerp(a: Color, b: Color, t: f32) Color {
        return .{
            .r = a.r + (b.r - a.r) * t,
            .g = a.g + (b.g - a.g) * t,
            .b = a.b + (b.b - a.b) * t,
            .a = a.a + (b.a - a.a) * t,
        };
    }

    pub fn scale(self: Color, s: f32) Color {
        return .{ .r = self.r * s, .g = self.g * s, .b = self.b * s, .a = self.a };
    }
};

/// Cross-axis placement inside a row or a column.
pub const Alignment = enum { start, center, end, stretch };

/// What a widget or a container asks for along one axis.
///
/// `auto` means "as small as my content allows"; `fill` means "a share of what
/// is left over" (a share of exactly the remainder when the total share is 1).
pub const Size = union(enum) {
    auto,
    px: f32,
    fill: f32,

    pub const full: Size = .{ .fill = 1.0 };

    pub fn resolve(self: Size, content: f32, avail: f32) f32 {
        return switch (self) {
            .auto => content,
            .px => |v| v,
            .fill => |w| @max(0, avail) * w,
        };
    }
};

/// A widget's identity across frames: a hash of its label and of the labels of
/// the containers it sits in.
///
/// Nothing is stored under an `Id` except interaction state (which widget is
/// hot, pressed, focused) and the handful of things that cannot be recomputed
/// (scroll offsets, open flags, carets, animation values). Two widgets with the
/// same label in different containers therefore stay distinct — and that is the
/// only way a user can make them collide, so the rule is easy to state: *one
/// label per container*.
pub const Id = struct {
    h: u64 = 0,

    pub const none: Id = .{};

    pub fn eql(a: Id, b: Id) bool {
        return a.h == b.h;
    }

    pub fn isNone(self: Id) bool {
        return self.h == 0;
    }

    /// Hashes a label. `seed` is 0 at the root and the parent's hash below it,
    /// so the same label in two containers gives two ids (Wyhash, 64 bit: the
    /// collision chance is not something a UI will ever meet).
    pub fn hash(seed: u64, label: []const u8) Id {
        return .{ .h = std.hash.Wyhash.hash(seed, label) };
    }

    /// Integer labels cost nothing at all: no hashing, no string, and a `for`
    /// loop can name its rows with them.
    pub fn index(seed: u64, i: usize) Id {
        return .{ .h = seed ^ (0x9E3779B97F4A7C15 *% (@as(u64, @intCast(i)) +% 1)) };
    }
};

test "rect maths" {
    const r = Rect.init(10, 20, 100, 50);
    try std.testing.expectEqual(@as(f32, 110), r.max().x);
    try std.testing.expectEqual(@as(f32, 45), r.center().y);
    try std.testing.expect(r.contains(.{ .x = 10, .y = 20 }));
    try std.testing.expect(!r.contains(.{ .x = 110, .y = 20 }));
    try std.testing.expect(r.contains(.{ .x = 109, .y = 69.9 }));
    try std.testing.expect(!r.contains(.{ .x = 110, .y = 70 }));
    try std.testing.expect(r.inset(.all(5)).w == 90);
    const clipped = r.intersect(Rect.init(0, 0, 20, 30));
    try std.testing.expectEqual(Rect.init(10, 20, 10, 10), clipped);
    try std.testing.expect(Rect.init(0, 0, 1, 1).intersect(Rect.init(5, 5, 1, 1)).isEmpty());
}

test "placement of a box inside another" {
    // Screen-axis alignment: `start` is the top-left corner, `end` the
    // bottom-right, `stretch` grows the box to the container. (Cross-axis
    // alignment *inside a layout* is the container's job, in `ui.occupy`.)
    const outer = Rect.init(0, 0, 100, 40);
    try std.testing.expectEqual(Rect.init(0, 0, 20, 20), outer.place(.{ .x = 20, .y = 20 }, .start));
    try std.testing.expectEqual(Rect.init(40, 10, 20, 20), outer.place(.{ .x = 20, .y = 20 }, .center));
    try std.testing.expectEqual(Rect.init(80, 20, 20, 20), outer.place(.{ .x = 20, .y = 20 }, .end));
    try std.testing.expectEqual(Rect.init(0, 0, 100, 40), outer.place(.{ .x = 20, .y = 20 }, .stretch));

    // The intersection of two boxes never exceeds either of them.
    const a = Rect.init(0, 0, 10, 10);
    const b = Rect.init(5, 5, 10, 10);
    try std.testing.expectEqual(Rect.init(5, 5, 5, 5), a.intersect(b));
    try std.testing.expectEqual(@as(f32, 15), a.merged(b).w);
    try std.testing.expectEqual(@as(f32, 15), a.merged(b).h);
}

test "ids are scoped by their container" {
    const root = Id.hash(0, "panel");
    const a = Id.hash(root.h, "ok");
    const b = Id.hash(root.h, "cancel");
    const other = Id.hash(Id.hash(0, "dialog").h, "ok");
    try std.testing.expect(!a.eql(b));
    try std.testing.expect(!a.eql(other));
    try std.testing.expect(a.eql(Id.hash(root.h, "ok")));
    try std.testing.expect(!Id.index(0, 1).eql(Id.index(0, 2)));
    try std.testing.expect(Id.none.isNone());
}
