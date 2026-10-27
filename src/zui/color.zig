//! Color type for ZUI
//!
//! RGBA color with float components (0.0 - 1.0).
//! Dependency-free and fully testable.

const std = @import("std");

/// RGBA color, each component in [0, 1]
pub const Color = struct {
    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,
    a: f32 = 1,

    /// Create a color from RGB hex value (e.g., 0xff8040)
    pub fn hex(hex_value: u24, alpha: f32) Color {
        const r = @as(f32, @floatFromInt((hex_value >> 16) & 0xff)) / 255.0;
        const g = @as(f32, @floatFromInt((hex_value >> 8) & 0xff)) / 255.0;
        const b = @as(f32, @floatFromInt(hex_value & 0xff)) / 255.0;
        return .{ .r = r, .g = g, .b = b, .a = alpha };
    }

    /// Create a grayscale color
    pub fn gray(value: f32) Color {
        return .{ .r = value, .g = value, .b = value, .a = 1 };
    }

    /// Create a color from RGB components
    pub fn rgb(red: f32, green: f32, blue: f32) Color {
        return .{ .r = red, .g = green, .b = blue, .a = 1 };
    }

    /// Linear interpolation between two colors
    pub fn lerp(a: Color, b: Color, t: f32) Color {
        return .{
            .r = a.r + (b.r - a.r) * t,
            .g = a.g + (b.g - a.g) * t,
            .b = a.b + (b.b - a.b) * t,
            .a = a.a + (b.a - a.a) * t,
        };
    }

    /// Multiply alpha
    pub fn withAlpha(self: Color, alpha: f32) Color {
        return .{ .r = self.r, .g = self.g, .b = self.b, .a = alpha };
    }

    /// Scale RGB components
    pub fn scale(self: Color, s: f32) Color {
        return .{ .r = self.r * s, .g = self.g * s, .b = self.b * s, .a = self.a };
    }

    /// Check if two colors are approximately equal
    pub fn approxEql(a: Color, b: Color, epsilon: f32) bool {
        return @abs(a.r - b.r) < epsilon and
            @abs(a.g - b.g) < epsilon and
            @abs(a.b - b.b) < epsilon and
            @abs(a.a - b.a) < epsilon;
    }

    /// Convert to array [r, g, b, a]
    pub fn toArray(self: Color) [4]f32 {
        return .{ self.r, self.g, self.b, self.a };
    }

    /// Create from array [r, g, b, a]
    pub fn fromArray(arr: [4]f32) Color {
        return .{ .r = arr[0], .g = arr[1], .b = arr[2], .a = arr[3] };
    }

    /// Transparent color (alpha = 0)
    pub const transparent: Color = .{ .r = 0, .g = 0, .b = 0, .a = 0 };
};

// ============================================================================
// Tests
// ============================================================================

test "color - hex conversion" {
    const red = Color.hex(0xff0000, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), red.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), red.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), red.b, 0.01);

    const green = Color.hex(0x00ff00, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), green.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), green.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), green.b, 0.01);

    const blue = Color.hex(0x0000ff, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), blue.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), blue.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), blue.b, 0.01);
}

test "color - hex with alpha" {
    const color = Color.hex(0xffffff, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), color.a, 0.01);
}

test "color - gray" {
    const white = Color.gray(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), white.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), white.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), white.b, 0.01);

    const black = Color.gray(0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), black.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), black.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), black.b, 0.01);

    const mid = Color.gray(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.r, 0.01);
}

test "color - lerp" {
    const black = Color{ .r = 0, .g = 0, .b = 0, .a = 1 };
    const white = Color{ .r = 1, .g = 1, .b = 1, .a = 1 };

    const mid = Color.lerp(black, white, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.g, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.b, 0.01);

    const quarter = Color.lerp(black, white, 0.25);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), quarter.r, 0.01);
}

test "color - withAlpha" {
    const color = Color{ .r = 1, .g = 0, .b = 0, .a = 1 };
    const transparent = color.withAlpha(0.5);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), transparent.r, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), transparent.a, 0.01);
}

test "color - approxEql" {
    const a = Color{ .r = 1.0, .g = 0.5, .b = 0.0, .a = 1.0 };
    const b = Color{ .r = 1.001, .g = 0.501, .b = 0.001, .a = 1.0 };

    try std.testing.expect(Color.approxEql(a, b, 0.01));
    try std.testing.expect(!Color.approxEql(a, b, 0.0001));
}
