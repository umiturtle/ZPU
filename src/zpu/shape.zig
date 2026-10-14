//! Shapes: a small, allocation free, comptime friendly value type.
//!
//! ZPU aims to make array shapes part of the type system where it pays off
//! (`Shape.initComptime(&.{2, 3}, f32)` produces a compile time constant), but a
//! runtime `Shape` value is used everywhere internally.
const std = @import("std");

pub const MAX_RANK: usize = 8;

pub const Error = error{ InvalidShape, RankTooLarge };

pub const Shape = struct {
    dims: [MAX_RANK]usize = @splat(0),
    len: u8 = 0,

    pub const empty = Shape{};

    pub fn init(dims: []const usize) Error!Shape {
        if (dims.len > MAX_RANK) return Error.RankTooLarge;
        var s = Shape{};
        for (dims, 0..) |d, i| s.dims[i] = d;
        s.len = @intCast(dims.len);
        return s;
    }

    /// A comptime-checked shape; rejects ranks above `MAX_RANK` at compile time.
    pub fn initComptime(comptime dims: []const usize) Shape {
        comptime {
            if (dims.len > MAX_RANK) @compileError("rank too large");
        }
        var s = Shape{};
        inline for (dims, 0..) |d, i| s.dims[i] = d;
        s.len = dims.len;
        return s;
    }

    /// Builds a shape from a full `MAX_RANK`-long dims array plus a length.
    pub fn fromParts(dims: [MAX_RANK]usize, len: usize) Shape {
        var s = Shape{};
        s.dims = dims;
        s.len = @intCast(len);
        return s;
    }

    pub fn scalar() Shape {
        return Shape{};
    }

    pub fn rank(self: Shape) usize {
        return self.len;
    }

    pub fn numel(self: Shape) usize {
        var n: usize = 1;
        for (0..self.rank()) |i| n *= self.dims[i];
        return n;
    }

    pub fn stridesContiguous(self: Shape) [MAX_RANK]isize {
        var st: [MAX_RANK]isize = @splat(0);
        var acc: isize = 1;
        var i = self.rank();
        while (i > 0) {
            i -= 1;
            st[i] = acc;
            acc *= @intCast(self.dims[i]);
        }
        return st;
    }

    /// NumPy style right-aligned broadcast of two shapes.
    pub fn broadcastShape(a: Shape, b: Shape) Error!Shape {
        return broadcastShapeRanked(a, b, @max(a.rank(), b.rank()));
    }

    /// Broadcasts `a` to `b`, but the result keeps at least `min_rank` dims
    /// (used by ZPU to preserve the `vmap` batch axes).
    pub fn broadcastShapeRanked(a: Shape, b: Shape, min_rank: usize) Error!Shape {
        const r = @max(@max(a.rank(), b.rank()), min_rank);
        if (r > MAX_RANK) return Error.RankTooLarge;
        var out: [MAX_RANK]usize = @splat(1);
        var i: usize = 0;
        while (i < a.rank()) : (i += 1) {
            const d = a.dims[a.rank() - 1 - i];
            const oi = r - 1 - i;
            out[oi] = d;
        }
        i = 0;
        while (i < b.rank()) : (i += 1) {
            const d = b.dims[b.rank() - 1 - i];
            const oi = r - 1 - i;
            if (out[oi] == 1 or d == 1) {
                out[oi] = @max(out[oi], d);
            } else if (out[oi] != d) {
                return Error.InvalidShape;
            }
        }
        return fromParts(out, r);
    }

    pub fn concat(a: Shape, b: Shape) Error!Shape {
        if (a.rank() + b.rank() > MAX_RANK) return Error.RankTooLarge;
        var out = a;
        for (0..b.rank()) |i| out.dims[a.rank() + i] = b.dims[i];
        out.len = a.len + b.len;
        return out;
    }

    /// Inserts a size-1 dim at `axis` (`axis` is the index in the result).
    pub fn withInserted(self: Shape, axis: usize) Error!Shape {
        if (self.rank() + 1 > MAX_RANK) return Error.RankTooLarge;
        if (axis > self.rank()) return Error.InvalidShape;
        var out = Shape{};
        var i: usize = 0;
        while (i < axis) : (i += 1) out.dims[i] = self.dims[i];
        out.dims[axis] = 1;
        i = axis;
        while (i < self.rank()) : (i += 1) out.dims[i + 1] = self.dims[i];
        out.len = self.len + 1;
        return out;
    }

    /// Removes dim `axis` (which must be 1).
    pub fn without(self: Shape, axis: usize) Error!Shape {
        if (axis >= self.rank()) return Error.InvalidShape;
        if (self.dims[axis] != 1) return Error.InvalidShape;
        var out = Shape{};
        var n: usize = 0;
        for (0..self.rank()) |i| {
            if (i == axis) continue;
            out.dims[n] = self.dims[i];
            n += 1;
        }
        out.len = @intCast(n);
        return out;
    }

    /// Sets `dims[axis] = v`, building a new shape.
    pub fn withDim(self: Shape, axis: usize, v: usize) Error!Shape {
        if (axis >= self.rank()) return Error.InvalidShape;
        var out = self;
        out.dims[axis] = v;
        return out;
    }

    /// Normalises a (possibly negative) axis against the rank, erroring if the
    /// resulting index is out of range.
    pub fn normAxis(self: Shape, axis: i64) Error!usize {
        const r: i64 = @intCast(self.rank());
        const a = if (axis < 0) axis + r else axis;
        if (a < 0 or a >= r) return Error.InvalidShape;
        return @intCast(a);
    }

    pub fn eql(a: Shape, b: Shape) bool {
        if (a.rank() != b.rank()) return false;
        for (0..a.rank()) |i| {
            if (a.dims[i] != b.dims[i]) return false;
        }
        return true;
    }

    pub fn format(
        self: Shape,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("(");
        for (0..self.rank()) |i| {
            if (i != 0) try writer.writeAll(", ");
            try writer.print("{d}", .{self.dims[i]});
        }
        if (self.rank() == 1) try writer.writeAll(",");
        try writer.writeAll(")");
    }
};

test "shape broadcasting" {
    const a = try Shape.init(&.{ 2, 3 });
    const b = try Shape.init(&.{3});
    const c = try Shape.broadcastShape(a, b);
    try std.testing.expect(Shape.eql(c, a));
    try std.testing.expectEqual(@as(usize, 6), c.numel());

    const d = try Shape.init(&.{ 1, 3, 1 });
    const e = try Shape.init(&.{ 2, 3, 4 });
    const f = try Shape.broadcastShape(d, e);
    try std.testing.expectEqual(@as(usize, 3), f.rank());
    try std.testing.expectEqual(@as(usize, 2), f.dims[0]);
}

test "shape helpers" {
    const s = Shape.initComptime(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), s.numel());
    const t = try s.withInserted(1);
    try std.testing.expectEqual(@as(usize, 4), t.rank());
    try std.testing.expectEqual(@as(usize, 1), t.dims[1]);
    const u = try t.without(1);
    try std.testing.expect(Shape.eql(u, s));
    try std.testing.expectEqual(@as(usize, 2), try s.normAxis(-1));
}
