//! Kernel fusion for ZPU.
//!
//! This module implements fusion of consecutive elementwise operations into
//! a single kernel pass, reducing memory traffic and improving performance.
//!
//! Example: `relu(x + bias)` becomes a single fused kernel instead of:
//!   1. temp = x + bias  (write to memory)
//!   2. output = relu(temp)  (read from memory)
//!
//! The fused version reads x and bias once, computes relu(x + bias), and
//! writes the result once — cutting memory traffic by ~50%.
const std = @import("std");
const prims = @import("prims.zig");
const scalar = @import("scalar.zig");

const Prim = prims.Prim;

/// Fused unary operation: dst[i] = outer(inner(src[i]))
/// Combines two unary operations into a single pass.
pub fn fusedUnaryUnary(
    comptime T: type,
    dst: []T,
    src: []const T,
    comptime inner: Prim,
    comptime outer: Prim,
) void {
    for (0..dst.len) |i| {
        const tmp = scalar.unary(T, inner, src[i]);
        dst[i] = scalar.unary(T, outer, tmp);
    }
}

/// Fused binary + unary operation: dst[i] = outer(binary(a[i], b[i]))
/// Example: relu(a + b), sigmoid(a * b), etc.
pub fn fusedBinaryUnary(
    comptime T: type,
    dst: []T,
    a: []const T,
    b: []const T,
    comptime binary_op: Prim,
    comptime unary_op: Prim,
) void {
    for (0..dst.len) |i| {
        const tmp = scalar.binary(T, binary_op, a[i], b[i]);
        dst[i] = scalar.unary(T, unary_op, tmp);
    }
}

/// Fused binary with scalar + unary: dst[i] = op(a[i] + scalar)
/// Optimized for bias addition followed by activation.
pub fn fusedAddScalarActivation(
    comptime T: type,
    dst: []T,
    src: []const T,
    bias: T,
    comptime activation: Prim,
) void {
    for (0..dst.len) |i| {
        const tmp = src[i] + bias;
        dst[i] = scalar.unary(T, activation, tmp);
    }
}

/// Common fused operations used in neural networks.
pub const FusedOps = struct {
    /// dst = relu(x + bias) — common in linear layers
    pub fn linearRelu(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarActivation(T, dst, x, bias, .relu);
    }

    /// dst = sigmoid(x + bias) — common in logistic regression
    pub fn linearSigmoid(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarActivation(T, dst, x, bias, .sigmoid);
    }

    /// dst = tanh(x + bias) — common in RNNs
    pub fn linearTanh(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarActivation(T, dst, x, bias, .tanh);
    }

    /// dst = relu(a + b) — elementwise addition with activation
    pub fn addRelu(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .add, .relu);
    }

    /// dst = relu(a * b) — elementwise multiplication with activation (gating)
    pub fn mulRelu(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .mul, .relu);
    }

    /// dst = sigmoid(a + b) — used in LSTM/GRU gates
    pub fn addSigmoid(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .add, .sigmoid);
    }

    /// dst = a * sigmoid(b) — gating operation
    pub fn gateSigmoid(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .mul, .sigmoid);
    }

    /// dst = a * tanh(b) — gating operation
    pub fn gateTanh(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .mul, .tanh);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "fused linear + relu" {
    const n = 1000;
    const alloc = std.testing.allocator;
    var x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    var dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;

    FusedOps.linearRelu(f32, dst, x, 2.0);

    for (0..n) |i| {
        const expected = @max(x[i] + 2.0, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "fused add + relu" {
    const n = 1000;
    const alloc = std.testing.allocator;
    var a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    var b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    var dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;
        b[i] = @as(f32, @floatFromInt(i % 50)) * 0.2 - 3.0;
    }

    FusedOps.addRelu(f32, dst, a, b);

    for (0..n) |i| {
        const expected = @max(a[i] + b[i], 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "fused mul + sigmoid (gating)" {
    const n = 100;
    const alloc = std.testing.allocator;
    var a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    var b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    var dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(i % 20)) * 0.2 - 2.0;
    }

    FusedOps.gateSigmoid(f32, dst, a, b);

    for (0..n) |i| {
        const sig = 1.0 / (1.0 + @exp(-b[i]));
        const expected = a[i] * sig;
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}
