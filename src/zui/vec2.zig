//! 2D vector type for ZUI
//!
//! Dependency-free, fully testable vector operations.

const std = @import("std");

/// A point or size in 2D space (logical pixels)
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

    pub fn min(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = @min(a.x, b.x), .y = @min(a.y, b.y) };
    }

    pub fn length(self: Vec2) f32 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }

    pub fn lengthSquared(self: Vec2) f32 {
        return self.x * self.x + self.y * self.y;
    }

    pub fn normalize(self: Vec2) Vec2 {
        const len = self.length();
        if (len < 1e-6) return .zero;
        return self.scale(1.0 / len);
    }

    pub fn dot(a: Vec2, b: Vec2) f32 {
        return a.x * b.x + a.y * b.y;
    }

    pub fn eql(a: Vec2, b: Vec2) bool {
        return a.x == b.x and a.y == b.y;
    }

    pub fn approxEql(a: Vec2, b: Vec2, epsilon: f32) bool {
        return @abs(a.x - b.x) < epsilon and @abs(a.y - b.y) < epsilon;
    }

    /// Axis type for index/along methods
    pub const Axis = enum {
        x,
        y,

        pub fn other(self: Axis) Axis {
            return if (self == .x) .y else .x;
        }
    };

    /// Get component by axis
    pub fn index(self: Vec2, axis: Axis) f32 {
        return if (axis == .x) self.x else self.y;
    }

    /// Flip axes for code that works with both row and column layouts
    pub fn along(self: Vec2, axis: Axis) Vec2 {
        return if (axis == .x) self else .{ .x = self.y, .y = self.x };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "vec2 - init" {
    const v = Vec2.init(3, 4);
    try std.testing.expectEqual(@as(f32, 3), v.x);
    try std.testing.expectEqual(@as(f32, 4), v.y);
}

test "vec2 - zero" {
    try std.testing.expectEqual(@as(f32, 0), Vec2.zero.x);
    try std.testing.expectEqual(@as(f32, 0), Vec2.zero.y);
}

test "vec2 - splat" {
    const v = Vec2.splat(5);
    try std.testing.expectEqual(@as(f32, 5), v.x);
    try std.testing.expectEqual(@as(f32, 5), v.y);
}

test "vec2 - add" {
    const a = Vec2.init(1, 2);
    const b = Vec2.init(3, 4);
    const sum = Vec2.add(a, b);
    try std.testing.expectEqual(@as(f32, 4), sum.x);
    try std.testing.expectEqual(@as(f32, 6), sum.y);
}

test "vec2 - sub" {
    const a = Vec2.init(5, 7);
    const b = Vec2.init(2, 3);
    const diff = Vec2.sub(a, b);
    try std.testing.expectEqual(@as(f32, 3), diff.x);
    try std.testing.expectEqual(@as(f32, 4), diff.y);
}

test "vec2 - scale" {
    const v = Vec2.init(3, 4);
    const scaled = Vec2.scale(v, 2);
    try std.testing.expectEqual(@as(f32, 6), scaled.x);
    try std.testing.expectEqual(@as(f32, 8), scaled.y);
}

test "vec2 - length" {
    const v = Vec2.init(3, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 5), v.length(), 0.0001);
}

test "vec2 - lengthSquared" {
    const v = Vec2.init(3, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 25), v.lengthSquared(), 0.0001);
}

test "vec2 - normalize" {
    const v = Vec2.init(3, 4);
    const n = v.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1), n.length(), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), n.x, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), n.y, 0.0001);
}

test "vec2 - normalize zero vector" {
    const v = Vec2.zero;
    const n = v.normalize();
    try std.testing.expectEqual(Vec2.zero, n);
}

test "vec2 - dot" {
    const a = Vec2.init(1, 2);
    const b = Vec2.init(3, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 11), Vec2.dot(a, b), 0.0001);
}

test "vec2 - max" {
    const a = Vec2.init(1, 5);
    const b = Vec2.init(3, 2);
    const m = Vec2.max(a, b);
    try std.testing.expectEqual(@as(f32, 3), m.x);
    try std.testing.expectEqual(@as(f32, 5), m.y);
}

test "vec2 - min" {
    const a = Vec2.init(1, 5);
    const b = Vec2.init(3, 2);
    const m = Vec2.min(a, b);
    try std.testing.expectEqual(@as(f32, 1), m.x);
    try std.testing.expectEqual(@as(f32, 2), m.y);
}

test "vec2 - eql" {
    const a = Vec2.init(1, 2);
    const b = Vec2.init(1, 2);
    const c = Vec2.init(1, 3);
    try std.testing.expect(Vec2.eql(a, b));
    try std.testing.expect(!Vec2.eql(a, c));
}

test "vec2 - approxEql" {
    const a = Vec2.init(1.0, 2.0);
    const b = Vec2.init(1.001, 2.001);
    try std.testing.expect(Vec2.approxEql(a, b, 0.01));
    try std.testing.expect(!Vec2.approxEql(a, b, 0.0001));
}
