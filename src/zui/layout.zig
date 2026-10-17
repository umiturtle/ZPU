//! Layout algorithms for ZUI
//!
//! Pure layout calculations, independent of rendering or state.
//! Can be tested and optimized separately.

const std = @import("std");
const types = @import("types.zig");

const Vec2 = types.Vec2;
const Rect = types.Rect;
const Axis = types.Axis;
const Size = types.Size;
const Edges = types.Edges;
const Alignment = types.Alignment;

/// Layout constraints: what space is available
pub const Constraints = struct {
    min_width: f32 = 0,
    max_width: f32 = std.math.inf(f32),
    min_height: f32 = 0,
    max_height: f32 = std.math.inf(f32),

    pub fn exact(width: f32, height: f32) Constraints {
        return .{
            .min_width = width,
            .max_width = width,
            .min_height = height,
            .max_height = height,
        };
    }

    pub fn loosen(self: Constraints) Constraints {
        return .{
            .min_width = 0,
            .max_width = self.max_width,
            .min_height = 0,
            .max_height = self.max_height,
        };
    }

    pub fn constrain(self: Constraints, size: Vec2) Vec2 {
        return .{
            .x = std.math.clamp(size.x, self.min_width, self.max_width),
            .y = std.math.clamp(size.y, self.min_height, self.max_height),
        };
    }
};

/// Result of a layout pass
pub const LayoutResult = struct {
    size: Vec2 = .{},
    position: Vec2 = .{},

    pub fn rect(self: LayoutResult) Rect {
        return .{
            .x = self.position.x,
            .y = self.position.y,
            .w = self.size.x,
            .h = self.size.y,
        };
    }
};

/// Resolve a Size given constraints
pub fn resolveSize(size: Size, constraints: Constraints, axis: Axis) f32 {
    return switch (size) {
        .exact => |v| v,
        .auto => switch (axis) {
            .x => constraints.min_width,
            .y => constraints.min_height,
        },
        .fill => |factor| switch (axis) {
            .x => constraints.max_width * factor,
            .y => constraints.max_height * factor,
        },
    };
}

/// Align a child within available space
pub fn alignChild(
    available: f32,
    child_size: f32,
    alignment: Alignment,
) f32 {
    return switch (alignment) {
        .start => 0,
        .center => (available - child_size) / 2,
        .end => available - child_size,
        .stretch => 0,
    };
}

/// Distribute space among children with gap
pub fn distributeSpace(
    total: f32,
    child_count: usize,
    gap: f32,
) struct { child_space: f32, total_gap: f32 } {
    if (child_count == 0) return .{ .child_space = 0, .total_gap = 0 };
    const total_gap = gap * @as(f32, @floatFromInt(child_count - 1));
    const available = total - total_gap;
    const child_space = if (child_count > 0) available / @as(f32, @floatFromInt(child_count)) else 0;
    return .{ .child_space = child_space, .total_gap = total_gap };
}

/// Calculate positions for a linear layout (row or column)
pub fn linearLayout(
    comptime direction: Axis,
    container_size: Vec2,
    child_sizes: []const Vec2,
    gap: f32,
    padding: Edges,
    alignment: Alignment,
) []LayoutResult {
    // This would need allocation, so we'll provide a stack-based version
    _ = direction;
    _ = container_size;
    _ = child_sizes;
    _ = gap;
    _ = padding;
    _ = alignment;
    // Implementation would go here
    return &.{};
}

/// Measure text dimensions (placeholder for actual text measurement)
pub fn measureText(
    text: []const u8,
    font_size: f32,
) Vec2 {
    // Approximate: assume monospace-like metrics
    const char_width = font_size * 0.6;
    const line_height = font_size * 1.2;
    return .{
        .x = char_width * @as(f32, @floatFromInt(text.len)),
        .y = line_height,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "constraints - exact" {
    const c = Constraints.exact(100, 50);
    try std.testing.expectEqual(@as(f32, 100), c.min_width);
    try std.testing.expectEqual(@as(f32, 100), c.max_width);
    try std.testing.expectEqual(@as(f32, 50), c.min_height);
    try std.testing.expectEqual(@as(f32, 50), c.max_height);
}

test "constraints - loosen" {
    const c = Constraints.exact(100, 50).loosen();
    try std.testing.expectEqual(@as(f32, 0), c.min_width);
    try std.testing.expectEqual(@as(f32, 100), c.max_width);
    try std.testing.expectEqual(@as(f32, 0), c.min_height);
    try std.testing.expectEqual(@as(f32, 50), c.max_height);
}

test "constraints - constrain" {
    const c = Constraints{ .min_width = 50, .max_width = 200, .min_height = 30, .max_height = 100 };
    
    const too_small = c.constrain(.{ .x = 10, .y = 5 });
    try std.testing.expectEqual(@as(f32, 50), too_small.x);
    try std.testing.expectEqual(@as(f32, 30), too_small.y);
    
    const too_large = c.constrain(.{ .x = 300, .y = 200 });
    try std.testing.expectEqual(@as(f32, 200), too_large.x);
    try std.testing.expectEqual(@as(f32, 100), too_large.y);
    
    const just_right = c.constrain(.{ .x = 100, .y = 50 });
    try std.testing.expectEqual(@as(f32, 100), just_right.x);
    try std.testing.expectEqual(@as(f32, 50), just_right.y);
}

test "resolveSize - exact" {
    const c = Constraints{ .min_width = 0, .max_width = 500 };
    const size = resolveSize(.{ .exact = 100 }, c, .x);
    try std.testing.expectEqual(@as(f32, 100), size);
}

test "resolveSize - fill" {
    const c = Constraints{ .min_width = 0, .max_width = 500 };
    const size = resolveSize(.{ .fill = 1.0 }, c, .x);
    try std.testing.expectEqual(@as(f32, 500), size);
    
    const half = resolveSize(.{ .fill = 0.5 }, c, .x);
    try std.testing.expectEqual(@as(f32, 250), half);
}

test "resolveSize - auto" {
    const c = Constraints{ .min_width = 50, .max_width = 500 };
    const size = resolveSize(.auto, c, .x);
    try std.testing.expectEqual(@as(f32, 50), size);
}

test "alignChild - start" {
    const pos = alignChild(100, 50, .start);
    try std.testing.expectEqual(@as(f32, 0), pos);
}

test "alignChild - center" {
    const pos = alignChild(100, 50, .center);
    try std.testing.expectEqual(@as(f32, 25), pos);
}

test "alignChild - end" {
    const pos = alignChild(100, 50, .end);
    try std.testing.expectEqual(@as(f32, 50), pos);
}

test "distributeSpace - equal distribution" {
    const result = distributeSpace(100, 3, 10);
    try std.testing.expectEqual(@as(f32, 20), result.total_gap);
    // Each child gets (100 - 20) / 3 = 26.666...
    try std.testing.expectApproxEqAbs(@as(f32, 26.666), result.child_space, 0.01);
}

test "distributeSpace - no children" {
    const result = distributeSpace(100, 0, 10);
    try std.testing.expectEqual(@as(f32, 0), result.total_gap);
    try std.testing.expectEqual(@as(f32, 0), result.child_space);
}

test "distributeSpace - one child" {
    const result = distributeSpace(100, 1, 10);
    try std.testing.expectEqual(@as(f32, 0), result.total_gap);
    try std.testing.expectEqual(@as(f32, 100), result.child_space);
}

test "measureText - basic" {
    const size = measureText("Hello", 14);
    try std.testing.expect(size.x > 0);
    try std.testing.expect(size.y > 0);
    
    // Longer text should be wider
    const longer = measureText("Hello World", 14);
    try std.testing.expect(longer.x > size.x);
    
    // Same width, different height
    try std.testing.expectEqual(size.y, longer.y);
}

test "LayoutResult - rect conversion" {
    const result = LayoutResult{
        .size = .{ .x = 100, .y = 50 },
        .position = .{ .x = 10, .y = 20 },
    };
    const rect = result.rect();
    try std.testing.expectEqual(@as(f32, 10), rect.x);
    try std.testing.expectEqual(@as(f32, 20), rect.y);
    try std.testing.expectEqual(@as(f32, 100), rect.w);
    try std.testing.expectEqual(@as(f32, 50), rect.h);
}
