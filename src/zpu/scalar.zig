//! Scalar math kernels.
//!
//! Every element-wise primitive is defined here once, on scalars. The tensor
//! engines (`ops.zig`, `autodiff.zig`) only deal with memory layout.
const std = @import("std");

pub inline fn toF64(x: anytype) f64 {
    const T = @TypeOf(x);
    return switch (@typeInfo(T)) {
        .float => @floatCast(x),
        .int => @floatFromInt(x),
        .bool => if (x) 1 else 0,
        else => @compileError("toF64: unsupported type " ++ @typeName(T)),
    };
}

pub inline fn fromF64(comptime T: type, v: f64) T {
    if (T == f64) return v;
    if (T == bool) return v != 0;
    if (@typeInfo(T) == .int) {
        if (std.math.isNan(v)) return 0;
        const lo: f64 = @floatFromInt(std.math.minInt(T));
        const hi: f64 = @floatFromInt(std.math.maxInt(T));
        return @intFromFloat(std.math.clamp(v, lo, hi));
    }
    return @floatCast(v);
}

inline fn isUnsignedT(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => |i| i.signedness == .unsigned,
        else => false,
    };
}

inline fn isFloatT(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float => true,
        else => false,
    };
}

inline fn cut(comptime T: type, v: f64) T {
    return if (T == f64) v else @floatCast(v);
}

// --- f64 helpers (also used for f32/f16 through `cut`) ----------------------

fn h_expm1(x: f64) f64 {
    return std.math.expm1(x);
}
fn h_log1p(x: f64) f64 {
    return std.math.log1p(x);
}
fn h_cbrt(x: f64) f64 {
    return std.math.cbrt(x);
}
fn h_asinh(x: f64) f64 {
    return std.math.asinh(x);
}
fn h_acosh(x: f64) f64 {
    return std.math.acosh(x);
}
fn h_atanh(x: f64) f64 {
    return std.math.atanh(x);
}
/// `erf` is not part of `std.math`, so ZPU ships its own implementation,
/// accurate to ~1e-16:
///
/// * `|x| < 1`: the Maclaurin series. The largest term is `O(1)` there, so the
///   alternating sum loses almost no digits.
/// * `|x| >= 1`: `erfc` from the continued fraction of the regularised upper
///   incomplete gamma function `Q(1/2, x^2)`, evaluated with Lentz's algorithm.
pub fn erf(x: f64) f64 {
    if (std.math.isNan(x)) return x;
    const ax = @abs(x);
    if (ax < 1.0) {
        const v = erfSeries(ax);
        return if (x < 0) -v else v;
    }
    const v = 1.0 - erfcCF(ax);
    return if (x < 0) -v else v;
}

/// Complementary error function.
pub fn erfc(x: f64) f64 {
    if (std.math.isNan(x)) return x;
    if (x < 0) return 2.0 - erfc(-x);
    if (x < 1.0) return 1.0 - erfSeries(x);
    return erfcCF(x);
}

fn erfSeries(x: f64) f64 {
    const two_over_sqrt_pi = 1.1283791670955125738961589031215;
    var sum: f64 = 0;
    var term: f64 = x;
    const x2 = x * x;
    var n: u32 = 0;
    while (n < 200) : (n += 1) {
        const fn_ = @as(f64, @floatFromInt(n));
        const add = term / (2.0 * fn_ + 1.0);
        sum += add;
        term *= -x2 / (fn_ + 1.0);
        if (@abs(add) < 1e-19) break;
    }
    return two_over_sqrt_pi * sum;
}

/// `erfc(x)` for `x >= 1`, via `Q(1/2, x^2)` (Lentz's continued fraction).
fn erfcCF(z: f64) f64 {
    const a = 0.5;
    const x = z * z;
    const fpmin: f64 = 1e-300;
    var b: f64 = x + 1.0 - a;
    var c: f64 = 1.0 / fpmin;
    var d: f64 = 1.0 / b;
    var h: f64 = d;
    var i: u32 = 1;
    while (i <= 300) : (i += 1) {
        const fi: f64 = @floatFromInt(i);
        const an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if (@abs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const delta = d * c;
        h *= delta;
        if (@abs(delta - 1.0) < 1e-16) break;
    }
    // log(sqrt(pi)) = lgamma(0.5)
    return @exp(-x + a * @log(x) - 0.5723649429247000871) * h;
}

fn h_erf(x: f64) f64 {
    return erf(x);
}
fn h_sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}
fn h_softplus(x: f64) f64 {
    // log(1 + e^x), stable
    return @max(x, 0.0) + std.math.log1p(@exp(-@abs(x)));
}
fn h_sign(x: f64) f64 {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
fn h_pow(a: f64, b: f64) f64 {
    return std.math.pow(f64, a, b);
}
fn h_max(a: f64, b: f64) f64 {
    if (std.math.isNan(a)) return a;
    if (std.math.isNan(b)) return b;
    return if (a > b) a else b;
}
fn h_min(a: f64, b: f64) f64 {
    if (std.math.isNan(a)) return a;
    if (std.math.isNan(b)) return b;
    return if (a < b) a else b;
}
fn h_mod(a: f64, b: f64) f64 {
    if (b == 0) return std.math.nan(f64);
    var r = @rem(a, b);
    if (r != 0 and ((r < 0) != (b < 0))) r += b;
    return r;
}
fn h_hypot(a: f64, b: f64) f64 {
    return std.math.hypot(a, b);
}
fn h_atan2(a: f64, b: f64) f64 {
    return std.math.atan2(a, b);
}

/// Unary primitive on f64.
pub fn unaryF(comptime p: anytype, x: f64) f64 {
    return switch (p) {
        .neg => -x,
        .abs => @abs(x),
        .exp => @exp(x),
        .expm1 => h_expm1(x),
        .log => @log(x),
        .log1p => h_log1p(x),
        .sqrt => @sqrt(x),
        .rsqrt => 1.0 / @sqrt(x),
        .square => x * x,
        .recip => 1.0 / x,
        .cbrt => h_cbrt(x),
        .sin => @sin(x),
        .cos => @cos(x),
        .tan => @tan(x),
        .asin => std.math.asin(x),
        .acos => std.math.acos(x),
        .atan => std.math.atan(x),
        .sinh => std.math.sinh(x),
        .cosh => std.math.cosh(x),
        .tanh => std.math.tanh(x),
        .asinh => h_asinh(x),
        .acosh => h_acosh(x),
        .atanh => h_atanh(x),
        .erf => h_erf(x),
        .sigmoid => h_sigmoid(x),
        .softplus => h_softplus(x),
        .relu => if (x > 0) x else 0,
        .floor => @floor(x),
        .ceil => @ceil(x),
        .round => @round(x),
        .trunc => @trunc(x),
        .sign => h_sign(x),
        .copy => x,
        else => @compileError("not a float unary primitive: " ++ @tagName(p)),
    };
}

/// Binary primitive on f64.
pub fn binaryF(comptime p: anytype, a: f64, b: f64) f64 {
    return switch (p) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .pow => h_pow(a, b),
        .maximum => h_max(a, b),
        .minimum => h_min(a, b),
        .atan2 => h_atan2(a, b),
        .hypot => h_hypot(a, b),
        .mod_op => h_mod(a, b),
        .relu => if (a > 0) a else 0,
        else => @compileError("not a float binary primitive: " ++ @tagName(p)),
    };
}

/// Comparison / logical primitive on f64.
pub fn pred2F(comptime p: anytype, a: f64, b: f64) bool {
    return switch (p) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
        .logical_and => (a != 0) and (b != 0),
        .logical_or => (a != 0) or (b != 0),
        .logical_xor => (a != 0) != (b != 0),
        else => @compileError("not a predicate: " ++ @tagName(p)),
    };
}

/// Unary predicate on f64.
pub fn pred1F(comptime p: anytype, x: f64) bool {
    return switch (p) {
        .isnan => std.math.isNan(x),
        .isinf => std.math.isInf(x),
        .isfinite => std.math.isFinite(x),
        else => @compileError("not a unary predicate: " ++ @tagName(p)),
    };
}

/// Unary primitive on a concrete element type.
pub fn unary(comptime T: type, comptime p: anytype, x: T) T {
    if (comptime isFloatT(T)) {
        if (T == f64) return unaryF(p, x);
        return cut(T, unaryF(p, toF64(x)));
    } else {
        if (p == .relu) {
            if (comptime T == bool) return x;
            return if (x > 0) x else 0;
        }
        if (p == .copy) return x;
        if (p == .abs) {
            if (comptime isUnsignedT(T) or T == bool) return x;
            return if (x < 0) -x else x;
        }
        if (p == .sign) {
            if (comptime isUnsignedT(T) or T == bool) return x;
            return if (x > 0) 1 else if (x < 0) -1 else 0;
        }
        return fromF64(T, unaryF(p, toF64(x)));
    }
}

/// Binary primitive on a concrete element type.
pub fn binary(comptime T: type, comptime p: anytype, a: T, b: T) T {
    if (comptime isFloatT(T)) {
        if (T == f64) return binaryF(p, a, b);
        return cut(T, binaryF(p, toF64(a), toF64(b)));
    } else {
        switch (p) {
            .add => return a + b,
            .sub => return a - b,
            .mul => return a * b,
            .div => return @divTrunc(a, b),
            .mod_op => return if (b == 0) a else @mod(a, b),
            .maximum => return if (a > b) a else b,
            .minimum => return if (a < b) a else b,
            else => {},
        }
        return fromF64(T, binaryF(p, toF64(a), toF64(b)));
    }
}

/// Comparison of two values of the same concrete type.
pub fn pred2(comptime T: type, comptime p: anytype, a: T, b: T) bool {
    switch (p) {
        .eq => return a == b,
        .ne => return a != b,
        .lt => return a < b,
        .le => return a <= b,
        .gt => return a > b,
        .ge => return a >= b,
        .logical_and => return (a != zeroOf(T)) and (b != zeroOf(T)),
        .logical_or => return (a != zeroOf(T)) or (b != zeroOf(T)),
        .logical_xor => return (a != zeroOf(T)) != (b != zeroOf(T)),
        else => {},
    }
    return pred2F(p, toF64(a), toF64(b));
}

fn zeroOf(comptime T: type) T {
    if (T == bool) return false;
    if (@typeInfo(T) == .int) return 0;
    return 0;
}

test "scalar kernels" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), unaryF(.sigmoid, 0.0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), unaryF(.relu, -2.0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), unaryF(.relu, 2.0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), binaryF(.div, 1.0, 2.0), 1e-12);
    try std.testing.expect(pred2(i64, .lt, 3, 4));
    try std.testing.expectEqual(@as(i64, 7), binary(i64, .add, 3, 4));
    try std.testing.expectEqual(@as(i32, 3), unary(i32, .abs, -3));
    try std.testing.expect(pred2(f32, .gt, 1.5, 0.5));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), binaryF(.mod_op, 4.0, 3.0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), binaryF(.mod_op, -1.0, 3.0), 1e-12);
}

test "erf accuracy" {
    const cases = [_][2]f64{
        .{ 0.0, 0.0 },
        .{ 0.5, 0.5204998778130465377 },
        .{ 1.0, 0.8427007929497148693 },
        .{ 2.0, 0.9953222650189527342 },
        .{ 2.5, 0.9995930479825550411 },
        .{ 3.0, 0.9999779095030014146 },
        .{ 4.0, 0.9999999845827420997 },
        .{ -1.5, -0.9661051464753107271 },
    };
    for (cases) |c| {
        const got = erf(c[0]);
        try std.testing.expectApproxEqAbs(c[1], got, 1e-15);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.541725790028002e-8), erfc(4.0), 1e-22);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5374597944280349e-12), erfc(5.0), 1e-26);
}
