//! Animation utilities for ZUI
//!
//! Pure animation math, independent of rendering or state.
//! Can be tested and optimized separately.

const std = @import("std");

/// Easing functions: map t in [0, 1] to output in [0, 1]
pub const Easing = enum {
    linear,
    ease_in,
    ease_out,
    ease_in_out,
    cubic_in,
    cubic_out,
    cubic_in_out,
    elastic_out,
    bounce_out,

    pub fn apply(self: Easing, t: f32) f32 {
        const clamped = std.math.clamp(t, 0, 1);
        return switch (self) {
            .linear => clamped,
            .ease_in => clamped * clamped,
            .ease_out => 1 - (1 - clamped) * (1 - clamped),
            .ease_in_out => if (clamped < 0.5)
                2 * clamped * clamped
            else
                1 - 2 * (1 - clamped) * (1 - clamped),
            .cubic_in => clamped * clamped * clamped,
            .cubic_out => 1 - std.math.pow(f32, 1 - clamped, 3),
            .cubic_in_out => if (clamped < 0.5)
                4 * clamped * clamped * clamped
            else
                1 - std.math.pow(f32, 2 - 2 * clamped, 3) / 2,
            .elastic_out => elasticOut(clamped),
            .bounce_out => bounceOut(clamped),
        };
    }

    fn elasticOut(t: f32) f32 {
        if (t == 0 or t == 1) return t;
        const p: f32 = 0.3;
        const s = p / 4;
        return std.math.pow(f32, 2, -10 * t) * std.math.sin((t - s) * 2 * std.math.pi / p) + 1;
    }

    fn bounceOut(t: f32) f32 {
        if (t < 1 / 2.75) {
            return 7.5625 * t * t;
        } else if (t < 2 / 2.75) {
            const t2 = t - 1.5 / 2.75;
            return 7.5625 * t2 * t2 + 0.75;
        } else if (t < 2.5 / 2.75) {
            const t2 = t - 2.25 / 2.75;
            return 7.5625 * t2 * t2 + 0.9375;
        } else {
            const t2 = t - 2.625 / 2.75;
            return 7.5625 * t2 * t2 + 0.984375;
        }
    }
};

/// Animation state: tracks progress from 0 to 1
pub const Animation = struct {
    start_time: f64 = 0,
    duration: f32 = 0.3,
    easing: Easing = .ease_out,
    running: bool = false,

    /// Get current progress (0 to 1)
    pub fn progress(self: Animation, current_time: f64) f32 {
        if (!self.running or self.duration <= 0) return 1;
        const elapsed = @as(f32, @floatCast(current_time - self.start_time));
        const t = std.math.clamp(elapsed / self.duration, 0, 1);
        return self.easing.apply(t);
    }

    /// Check if animation is complete
    pub fn isComplete(self: Animation, current_time: f64) bool {
        if (!self.running) return true;
        const elapsed = @as(f32, @floatCast(current_time - self.start_time));
        return elapsed >= self.duration;
    }

    /// Start animation at given time
    pub fn start(self: *Animation, time: f64, duration: f32) void {
        self.start_time = time;
        self.duration = duration;
        self.running = true;
    }

    /// Stop animation
    pub fn stop(self: *Animation) void {
        self.running = false;
    }
};

/// Spring animation: physics-based
pub const Spring = struct {
    position: f32 = 0,
    velocity: f32 = 0,
    target: f32 = 0,
    stiffness: f32 = 100,
    damping: f32 = 10,
    mass: f32 = 1,

    /// Update spring physics for one timestep
    pub fn update(self: *Spring, dt: f32) void {
        const displacement = self.position - self.target;
        const spring_force = -self.stiffness * displacement;
        const damping_force = -self.damping * self.velocity;
        const acceleration = (spring_force + damping_force) / self.mass;

        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;
    }

    /// Check if spring is at rest
    pub fn isAtRest(self: Spring, threshold: f32) bool {
        return @abs(self.position - self.target) < threshold and
            @abs(self.velocity) < threshold;
    }

    /// Set target position
    pub fn setTarget(self: *Spring, target: f32) void {
        self.target = target;
    }

    /// Snap to target immediately
    pub fn snap(self: *Spring) void {
        self.position = self.target;
        self.velocity = 0;
    }
};

/// Smooth interpolation (exponential decay)
pub fn smoothDamp(current: f32, target: f32, rate: f32, dt: f32) f32 {
    const t = 1 - std.math.exp(-rate * dt);
    return current + (target - current) * t;
}

/// Linear interpolation
pub fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

/// Inverse linear interpolation: given a value between a and b, return t
pub fn inverseLerp(a: f32, b: f32, value: f32) f32 {
    if (@abs(b - a) < 1e-6) return 0;
    return (value - a) / (b - a);
}

/// Remap value from one range to another
pub fn remap(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) f32 {
    const t = inverseLerp(from_min, from_max, value);
    return lerp(to_min, to_max, t);
}

// ============================================================================
// Tests
// ============================================================================

test "easing - linear" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), Easing.linear.apply(0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), Easing.linear.apply(0.5), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), Easing.linear.apply(1), 0.001);
}

test "easing - ease_in" {
    // Starts slow, ends fast
    const mid = Easing.ease_in.apply(0.5);
    try std.testing.expect(mid < 0.5);
}

test "easing - ease_out" {
    // Starts fast, ends slow
    const mid = Easing.ease_out.apply(0.5);
    try std.testing.expect(mid > 0.5);
}

test "easing - clamping" {
    // Should clamp values outside [0, 1]
    try std.testing.expectApproxEqAbs(@as(f32, 0), Easing.linear.apply(-0.5), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), Easing.linear.apply(1.5), 0.001);
}

test "animation - progress" {
    var anim = Animation{};
    anim.start(0, 1.0);
    
    try std.testing.expectApproxEqAbs(@as(f32, 0), anim.progress(0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), anim.progress(0.5), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1), anim.progress(1.0), 0.01);
}

test "animation - isComplete" {
    var anim = Animation{};
    anim.start(0, 0.5);
    
    try std.testing.expect(!anim.isComplete(0.25));
    try std.testing.expect(anim.isComplete(0.5));
    try std.testing.expect(anim.isComplete(1.0));
}

test "spring - basic movement" {
    var spring = Spring{
        .position = 0,
        .target = 100,
        .stiffness = 200,
        .damping = 20,
    };
    
    // After many updates, should approach target
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        spring.update(0.016);
    }
    
    try std.testing.expectApproxEqAbs(@as(f32, 100), spring.position, 0.1);
}

test "spring - isAtRest" {
    var spring = Spring{
        .position = 50,
        .target = 50,
        .velocity = 0,
    };
    
    try std.testing.expect(spring.isAtRest(0.01));
    
    spring.velocity = 100;
    try std.testing.expect(!spring.isAtRest(0.01));
}

test "spring - snap" {
    var spring = Spring{
        .position = 0,
        .target = 100,
        .velocity = 50,
    };
    
    spring.snap();
    try std.testing.expectEqual(@as(f32, 100), spring.position);
    try std.testing.expectEqual(@as(f32, 0), spring.velocity);
}

test "smoothDamp - approaches target" {
    var value: f32 = 0;
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        value = smoothDamp(value, 100, 5, 0.016);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 100), value, 1);
}

test "lerp - basic" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), lerp(0, 100, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50), lerp(0, 100, 0.5), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 100), lerp(0, 100, 1), 0.001);
}

test "inverseLerp - basic" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), inverseLerp(0, 100, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), inverseLerp(0, 100, 50), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), inverseLerp(0, 100, 100), 0.001);
}

test "remap - basic" {
    const result = remap(50, 0, 100, 0, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result, 0.001);
    
    const reversed = remap(50, 0, 100, 1, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), reversed, 0.001);
}
