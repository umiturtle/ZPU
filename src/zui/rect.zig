//! Rectangle type for ZUI
//!
//! Dependency-free, fully testable rectangle operations.

const std = @import("std");

/// A point or size in 2D space (minimal version for rect operations)
pub const Vec2 = struct {
    x: f32 = 0,
    y: f32 = 0,
};

/// An axis-aligned rectangle.
/// The coordinate system has the origin at the top-left.
pub const Rect = struct {
    x: f32 = 0,
    y: f32 = 0,
    w: f32 = 0,
    h: f32 = 0,

    /// Create a rectangle from position and size
    pub fn fromPositionSize(pos: Vec2, dimensions: Vec2) Rect {
        return .{ .x = pos.x, .y = pos.y, .w = dimensions.x, .h = dimensions.y };
    }

    pub fn fromEdges(x1: f32, y1: f32, x2: f32, y2: f32) Rect {
        return .{ .x = x1, .y = y1, .w = x2 - x1, .h = y2 - y1 };
    }

    pub fn zero() Rect {
        return .{ .x = 0, .y = 0, .w = 0, .h = 0 };
    }

    pub fn left(self: Rect) f32 {
        return self.x;
    }

    pub fn top(self: Rect) f32 {
        return self.y;
    }

    pub fn right(self: Rect) f32 {
        return self.x + self.w;
    }

    pub fn bottom(self: Rect) f32 {
        return self.y + self.h;
    }

    pub fn topLeft(self: Rect) Vec2 {
        return .{ .x = self.x, .y = self.y };
    }

    pub fn bottomRight(self: Rect) Vec2 {
        return .{ .x = self.right(), .y = self.bottom() };
    }

    pub fn center(self: Rect) Vec2 {
        return .{ .x = self.x + self.w / 2, .y = self.y + self.h / 2 };
    }

    pub fn size(self: Rect) Vec2 {
        return .{ .x = self.w, .y = self.h };
    }

    pub fn position(self: Rect) Vec2 {
        return .{ .x = self.x, .y = self.y };
    }

    pub fn area(self: Rect) f32 {
        return self.w * self.h;
    }

    pub fn contains(self: Rect, p: Vec2) bool {
        return p.x >= self.x and p.x < self.right() and p.y >= self.y and p.y < self.bottom();
    }

    pub fn containsRect(self: Rect, other: Rect) bool {
        return self.x <= other.x and
            self.y <= other.y and
            self.right() >= other.right() and
            self.bottom() >= other.bottom();
    }

    pub fn overlaps(self: Rect, other: Rect) bool {
        return self.x < other.right() and
            self.right() > other.x and
            self.y < other.bottom() and
            self.bottom() > other.y;
    }

    pub fn intersection(self: Rect, other: Rect) ?Rect {
        const x1 = @max(self.x, other.x);
        const y1 = @max(self.y, other.y);
        const x2 = @min(self.right(), other.right());
        const y2 = @min(self.bottom(), other.bottom());

        if (x1 < x2 and y1 < y2) {
            return .{ .x = x1, .y = y1, .w = x2 - x1, .h = y2 - y1 };
        }
        return null;
    }

    pub fn merge(self: Rect, other: Rect) Rect {
        const x1 = @min(self.x, other.x);
        const y1 = @min(self.y, other.y);
        const x2 = @max(self.right(), other.right());
        const y2 = @max(self.bottom(), other.bottom());
        return .{ .x = x1, .y = y1, .w = x2 - x1, .h = y2 - y1 };
    }

    pub fn expand(self: Rect, amount: f32) Rect {
        return .{
            .x = self.x - amount,
            .y = self.y - amount,
            .w = self.w + amount * 2,
            .h = self.h + amount * 2,
        };
    }

    pub fn shrink(self: Rect, amount: f32) Rect {
        return self.expand(-amount);
    }

    pub fn translate(self: Rect, offset: Vec2) Rect {
        return .{ .x = self.x + offset.x, .y = self.y + offset.y, .w = self.w, .h = self.h };
    }

    pub fn clampVec2(self: Rect, p: Vec2) Vec2 {
        return .{
            .x = std.math.clamp(p.x, self.x, self.right() - 1),
            .y = std.math.clamp(p.y, self.y, self.bottom() - 1),
        };
    }

    pub fn eql(a: Rect, b: Rect) bool {
        return a.x == b.x and a.y == b.y and a.w == b.w and a.h == b.h;
    }

    pub fn approxEql(a: Rect, b: Rect, epsilon: f32) bool {
        return @abs(a.x - b.x) < epsilon and
            @abs(a.y - b.y) < epsilon and
            @abs(a.w - b.w) < epsilon and
            @abs(a.h - b.h) < epsilon;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "rect - fromPositionSize" {
    const r = Rect.fromPositionSize(.{ .x = 10, .y = 20 }, .{ .x = 100, .y = 50 });
    try std.testing.expectEqual(@as(f32, 10), r.x);
    try std.testing.expectEqual(@as(f32, 20), r.y);
    try std.testing.expectEqual(@as(f32, 100), r.w);
    try std.testing.expectEqual(@as(f32, 50), r.h);
}

test "rect - fromEdges" {
    const r = Rect.fromEdges(10, 20, 110, 70);
    try std.testing.expectEqual(@as(f32, 10), r.x);
    try std.testing.expectEqual(@as(f32, 20), r.y);
    try std.testing.expectEqual(@as(f32, 100), r.w);
    try std.testing.expectEqual(@as(f32, 50), r.h);
}

test "rect - edges" {
    const r = Rect{ .x = 10, .y = 20, .w = 100, .h = 50 };
    try std.testing.expectEqual(@as(f32, 10), r.left());
    try std.testing.expectEqual(@as(f32, 20), r.top());
    try std.testing.expectEqual(@as(f32, 110), r.right());
    try std.testing.expectEqual(@as(f32, 70), r.bottom());
}

test "rect - center" {
    const r = Rect{ .x = 0, .y = 0, .w = 100, .h = 50 };
    const c = r.center();
    try std.testing.expectEqual(@as(f32, 50), c.x);
    try std.testing.expectEqual(@as(f32, 25), c.y);
}

test "rect - area" {
    const r = Rect{ .x = 0, .y = 0, .w = 100, .h = 50 };
    try std.testing.expectApproxEqAbs(@as(f32, 5000), r.area(), 0.001);
}

test "rect - contains point" {
    const r = Rect{ .x = 10, .y = 10, .w = 100, .h = 100 };
    try std.testing.expect(r.contains(.{ .x = 50, .y = 50 }));
    try std.testing.expect(!r.contains(.{ .x = 5, .y = 5 }));
    try std.testing.expect(!r.contains(.{ .x = 200, .y = 50 }));
}

test "rect - containsRect" {
    const outer = Rect{ .x = 0, .y = 0, .w = 100, .h = 100 };
    const inner = Rect{ .x = 10, .y = 10, .w = 50, .h = 50 };
    try std.testing.expect(outer.containsRect(inner));
    try std.testing.expect(!inner.containsRect(outer));
}

test "rect - overlaps" {
    const a = Rect{ .x = 0, .y = 0, .w = 100, .h = 100 };
    const b = Rect{ .x = 50, .y = 50, .w = 100, .h = 100 };
    const c = Rect{ .x = 200, .y = 200, .w = 50, .h = 50 };
    try std.testing.expect(a.overlaps(b));
    try std.testing.expect(!a.overlaps(c));
}

test "rect - intersection" {
    const a = Rect{ .x = 0, .y = 0, .w = 100, .h = 100 };
    const b = Rect{ .x = 50, .y = 50, .w = 100, .h = 100 };
    const i = a.intersection(b).?;
    try std.testing.expectEqual(@as(f32, 50), i.x);
    try std.testing.expectEqual(@as(f32, 50), i.y);
    try std.testing.expectEqual(@as(f32, 50), i.w);
    try std.testing.expectEqual(@as(f32, 50), i.h);
}

test "rect - intersection none" {
    const a = Rect{ .x = 0, .y = 0, .w = 50, .h = 50 };
    const b = Rect{ .x = 100, .y = 100, .w = 50, .h = 50 };
    try std.testing.expect(a.intersection(b) == null);
}

test "rect - merge" {
    const a = Rect{ .x = 10, .y = 10, .w = 50, .h = 50 };
    const b = Rect{ .x = 40, .y = 40, .w = 50, .h = 50 };
    const u = a.merge(b);
    try std.testing.expectEqual(@as(f32, 10), u.x);
    try std.testing.expectEqual(@as(f32, 10), u.y);
    try std.testing.expectEqual(@as(f32, 80), u.w);
    try std.testing.expectEqual(@as(f32, 80), u.h);
}

test "rect - expand" {
    const r = Rect{ .x = 10, .y = 10, .w = 100, .h = 100 };
    const e = r.expand(5);
    try std.testing.expectEqual(@as(f32, 5), e.x);
    try std.testing.expectEqual(@as(f32, 5), e.y);
    try std.testing.expectEqual(@as(f32, 110), e.w);
    try std.testing.expectEqual(@as(f32, 110), e.h);
}

test "rect - translate" {
    const r = Rect{ .x = 10, .y = 20, .w = 100, .h = 50 };
    const t = r.translate(.{ .x = 5, .y = -10 });
    try std.testing.expectEqual(@as(f32, 15), t.x);
    try std.testing.expectEqual(@as(f32, 10), t.y);
    try std.testing.expectEqual(@as(f32, 100), t.w);
}

test "rect - eql" {
    const a = Rect{ .x = 10, .y = 20, .w = 100, .h = 50 };
    const b = Rect{ .x = 10, .y = 20, .w = 100, .h = 50 };
    const c = Rect{ .x = 11, .y = 20, .w = 100, .h = 50 };
    try std.testing.expect(Rect.eql(a, b));
    try std.testing.expect(!Rect.eql(a, c));
}
