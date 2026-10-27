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
//!
//! All fused kernels are SIMD-vectorized for maximum throughput.
const std = @import("std");
const prims = @import("prims.zig");
const scalar = @import("scalar.zig");
const kernels = @import("kernels.zig");
const parallel = @import("parallel.zig");

const Prim = prims.Prim;

// ============================================================================
// SIMD-vectorized fused kernels
// ============================================================================

/// Check if a unary op has a fused vector form
fn hasFusedVectorForm(comptime p: Prim) bool {
    return switch (p) {
        .relu, .sigmoid, .neg, .abs, .exp, .log, .sqrt, .square => true,
        else => false, // tanh and others use scalar fallback
    };
}

/// Fused binary + unary operation with SIMD: dst[i] = outer(binary(a[i], b[i]))
/// Example: relu(a + b), sigmoid(a * b), etc.
pub fn fusedBinaryUnary(
    comptime T: type,
    dst: []T,
    a: []const T,
    b: []const T,
    comptime binary_op: Prim,
    comptime unary_op: Prim,
) void {
    // Use vectorized path only if both ops have vector forms
    if (comptime !kernels.hasVectorBinary(binary_op) or !hasFusedVectorForm(unary_op)) {
        // Fallback to scalar
        for (0..dst.len) |i| {
            const tmp = scalar.binary(T, binary_op, a[i], b[i]);
            dst[i] = scalar.unary(T, unary_op, tmp);
        }
        return;
    }

    const L = comptime kernels.lanes(T);
    const V = kernels.Vec(T);

    var i: usize = 0;
    while (i + L <= dst.len) : (i += L) {
        const va: V = a[i..][0..L].*;
        const vb: V = b[i..][0..L].*;
        const sum = kernels.applyBinary(T, binary_op, va, vb);
        const result = tryApplyFusedUnary(T, unary_op, sum) orelse unreachable;
        dst[i..][0..L].* = result;
    }

    // Tail
    const rem = dst.len - i;
    if (rem > 0) {
        var ta: [L]T = undefined;
        var tb: [L]T = undefined;
        for (0..L) |k| {
            ta[k] = a[i + @min(k, rem - 1)];
            tb[k] = b[i + @min(k, rem - 1)];
        }
        const va: V = ta;
        const vb: V = tb;
        const sum = kernels.applyBinary(T, binary_op, va, vb);
        const result = tryApplyFusedUnary(T, unary_op, sum) orelse unreachable;
        const out: [L]T = result;
        @memcpy(dst[i..], out[0..rem]);
    }
}

/// Fused add + unary with scalar: dst[i] = unary(a[i] + bias)
pub fn fusedAddScalarUnary(
    comptime T: type,
    dst: []T,
    src: []const T,
    bias: T,
    comptime unary_op: Prim,
) void {
    if (comptime !hasFusedVectorForm(unary_op)) {
        for (0..dst.len) |i| {
            dst[i] = scalar.unary(T, unary_op, src[i] + bias);
        }
        return;
    }

    const L = comptime kernels.lanes(T);
    const V = kernels.Vec(T);
    const vb: V = @splat(bias);

    var i: usize = 0;
    while (i + L <= dst.len) : (i += L) {
        const va: V = src[i..][0..L].*;
        const sum = va + vb;
        const result = tryApplyFusedUnary(T, unary_op, sum) orelse unreachable;
        dst[i..][0..L].* = result;
    }

    const rem = dst.len - i;
    if (rem > 0) {
        var ta: [L]T = undefined;
        for (0..L) |k| ta[k] = src[i + @min(k, rem - 1)];
        const va: V = ta;
        const sum = va + vb;
        const result = tryApplyFusedUnary(T, unary_op, sum) orelse unreachable;
        const out: [L]T = result;
        @memcpy(dst[i..], out[0..rem]);
    }
}

/// Fused mul + unary with scalar: dst[i] = unary(a[i] * scale)
pub fn fusedMulScalarUnary(
    comptime T: type,
    dst: []T,
    src: []const T,
    scale: T,
    comptime unary_op: Prim,
) void {
    if (comptime !hasFusedVectorForm(unary_op)) {
        for (0..dst.len) |i| {
            dst[i] = scalar.unary(T, unary_op, src[i] * scale);
        }
        return;
    }

    const L = comptime kernels.lanes(T);
    const V = kernels.Vec(T);
    const vs: V = @splat(scale);

    var i: usize = 0;
    while (i + L <= dst.len) : (i += L) {
        const va: V = src[i..][0..L].*;
        const prod = va * vs;
        const result = tryApplyFusedUnary(T, unary_op, prod) orelse unreachable;
        dst[i..][0..L].* = result;
    }

    const rem = dst.len - i;
    if (rem > 0) {
        var ta: [L]T = undefined;
        for (0..L) |k| ta[k] = src[i + @min(k, rem - 1)];
        const va: V = ta;
        const prod = va * vs;
        const result = tryApplyFusedUnary(T, unary_op, prod) orelse unreachable;
        const out: [L]T = result;
        @memcpy(dst[i..], out[0..rem]);
    }
}

/// Apply unary op using vector form — returns null if not vectorizable
inline fn tryApplyFusedUnary(comptime T: type, comptime p: Prim, x: kernels.Vec(T)) ?kernels.Vec(T) {
    const V = kernels.Vec(T);
    const one: V = @splat(1);
    const zero: V = @splat(0);
    return switch (p) {
        .relu => @max(x, zero),
        .sigmoid => one / (one + @exp(-x)),
        .neg => -x,
        .abs => @abs(x),
        .exp => @exp(x),
        .log => @log(x),
        .sqrt => @sqrt(x),
        .square => x * x,
        else => null, // tanh and others fall back to scalar
    };
}

/// Apply scalar unary for non-vectorizable ops
inline fn applyScalarUnary(comptime T: type, comptime p: Prim, x: T) T {
    return scalar.unary(T, p, x);
}

// ============================================================================
// Parallel fused kernels (for large arrays)
// ============================================================================

const ParallelFusedCtx = struct {
    dst_ptr: [*]u8,
    a_ptr: [*]const u8,
    b_ptr: [*]const u8,
    dst_len: usize,
    item_size: usize,
    binary_op: Prim,
    unary_op: Prim,
};

/// Parallel fused binary+unary for large tensors
pub fn fusedBinaryUnaryParallel(
    comptime T: type,
    dst: []T,
    a: []const T,
    b: []const T,
    comptime binary_op: Prim,
    comptime unary_op: Prim,
) void {
    if (dst.len < parallel.parallel_threshold) {
        fusedBinaryUnary(T, dst, a, b, binary_op, unary_op);
        return;
    }

    const WorkerCtx = struct {
        dst_slice: []T,
        a_slice: []const T,
        b_slice: []const T,
    };

    const workerFn = (struct {
        fn run(ctx_param: WorkerCtx, start: usize, end: usize) void {
            fusedBinaryUnary(T, ctx_param.dst_slice[start..end], ctx_param.a_slice[start..end], ctx_param.b_slice[start..end], binary_op, unary_op);
        }
    }).run;

    const ctx = WorkerCtx{ .dst_slice = dst, .a_slice = a, .b_slice = b };
    parallel.parallelFor(std.heap.c_allocator, dst.len, WorkerCtx, ctx, workerFn) catch {
        // Fallback to sequential on error
        fusedBinaryUnary(T, dst, a, b, binary_op, unary_op);
    };
}

/// Parallel fused add+scalar+unary for large tensors
pub fn fusedAddScalarUnaryParallel(
    comptime T: type,
    dst: []T,
    src: []const T,
    bias: T,
    comptime unary_op: Prim,
) void {
    if (dst.len < parallel.parallel_threshold) {
        fusedAddScalarUnary(T, dst, src, bias, unary_op);
        return;
    }

    const WorkerCtx = struct {
        dst_slice: []T,
        src_slice: []const T,
        bias_val: T,
    };

    const workerFn = (struct {
        fn run(ctx_param: WorkerCtx, start: usize, end: usize) void {
            fusedAddScalarUnary(T, ctx_param.dst_slice[start..end], ctx_param.src_slice[start..end], ctx_param.bias_val, unary_op);
        }
    }).run;

    const ctx = WorkerCtx{ .dst_slice = dst, .src_slice = src, .bias_val = bias };
    parallel.parallelFor(std.heap.c_allocator, dst.len, WorkerCtx, ctx, workerFn) catch {
        fusedAddScalarUnary(T, dst, src, bias, unary_op);
    };
}

// ============================================================================
// Common fused operations used in neural networks (high-level API)
// ============================================================================

pub const FusedOps = struct {
    /// dst = relu(x + bias) — common in linear layers
    pub fn linearRelu(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarUnary(T, dst, x, bias, .relu);
    }

    /// dst = sigmoid(x + bias) — common in logistic regression
    pub fn linearSigmoid(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarUnary(T, dst, x, bias, .sigmoid);
    }

    /// dst = tanh(x + bias) — common in RNNs
    pub fn linearTanh(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarUnary(T, dst, x, bias, .tanh);
    }

    /// dst = exp(x + bias) — used in softmax
    pub fn linearExp(comptime T: type, dst: []T, x: []const T, bias: T) void {
        fusedAddScalarUnary(T, dst, x, bias, .exp);
    }

    /// dst = relu(a + b) — elementwise addition with activation
    pub fn addRelu(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .add, .relu);
    }

    /// dst = sigmoid(a + b) — used in LSTM/GRU gates
    pub fn addSigmoid(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .add, .sigmoid);
    }

    /// dst = tanh(a + b) — used in LSTM/GRU gates
    pub fn addTanh(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .add, .tanh);
    }

    /// dst = relu(a * b) — elementwise multiplication with activation (gating)
    pub fn mulRelu(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        fusedBinaryUnary(T, dst, a, b, .mul, .relu);
    }

    /// dst = a * sigmoid(b) — gating operation (LSTM/GRU)
    /// This is NOT fusedBinaryUnary(.mul, .sigmoid) which would be sigmoid(a*b).
    /// Instead it's a special gate pattern: multiply a by sigmoid of b.
    pub fn gateSigmoid(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        const L = comptime kernels.lanes(T);
        const V = kernels.Vec(T);
        const one: V = @splat(1);

        if (comptime kernels.hasVectorBinary(.mul)) {
            var i: usize = 0;
            while (i + L <= dst.len) : (i += L) {
                const va: V = a[i..][0..L].*;
                const vb: V = b[i..][0..L].*;
                const sig = one / (one + @exp(-vb));
                dst[i..][0..L].* = va * sig;
            }
            const rem = dst.len - i;
            if (rem > 0) {
                for (i..dst.len) |j| {
                    const sig: T = 1.0 / (1.0 + @exp(-b[j]));
                    dst[j] = a[j] * sig;
                }
            }
        } else {
            for (0..dst.len) |i| {
                const sig: T = 1.0 / (1.0 + @exp(-b[i]));
                dst[i] = a[i] * sig;
            }
        }
    }

    /// dst = a * tanh(b) — gating operation
    pub fn gateTanh(comptime T: type, dst: []T, a: []const T, b: []const T) void {
        for (0..dst.len) |i| {
            dst[i] = a[i] * @as(T, @floatCast(std.math.tanh(@as(f64, @floatCast(b[i])))));
        }
    }

    /// dst = relu(x * scale) — scaled activation
    pub fn scaleRelu(comptime T: type, dst: []T, x: []const T, scale: T) void {
        fusedMulScalarUnary(T, dst, x, scale, .relu);
    }

    /// dst = sigmoid(x * scale) — scaled activation
    pub fn scaleSigmoid(comptime T: type, dst: []T, x: []const T, scale: T) void {
        fusedMulScalarUnary(T, dst, x, scale, .sigmoid);
    }

    /// Batch normalization with pre-computed statistics (inference mode)
    /// dst = (x - mean) / sqrt(var + eps) * gamma + beta
    /// This fuses normalize + scale + shift into a single vectorized pass
    pub fn batchNormInference(
        comptime T: type,
        dst: []T,
        x: []const T,
        mean: T,
        variance: T,
        gamma: T,
        beta: T,
        eps: T,
    ) void {
        const n = x.len;
        if (n == 0) return;

        const inv_std = gamma / @sqrt(variance + eps);
        const shift = beta - mean * inv_std;

        // Single fused pass: dst = x * inv_std + shift
        const L = comptime kernels.lanes(T);
        const V = kernels.Vec(T);

        if (comptime kernels.hasVectorBinary(.mul) and kernels.hasVectorBinary(.add)) {
            const v_inv_std: V = @splat(inv_std);
            const v_shift: V = @splat(shift);

            var i: usize = 0;
            while (i + L <= n) : (i += L) {
                const vx: V = x[i..][0..L].*;
                dst[i..][0..L].* = vx * v_inv_std + v_shift;
            }

            const rem = n - i;
            if (rem > 0) {
                for (i..n) |j| {
                    dst[j] = x[j] * inv_std + shift;
                }
            }
        } else {
            for (0..n) |i| {
                dst[i] = x[i] * inv_std + shift;
            }
        }
    }

    /// Fused residual connection: dst = activation(x + residual)
    /// Common in ResNet: y = relu(x + F(x))
    pub fn residualActivation(
        comptime T: type,
        dst: []T,
        x: []const T,
        residual: []const T,
        comptime activation: Prim,
    ) void {
        fusedBinaryUnary(T, dst, x, residual, .add, activation);
    }

    /// dst = layer_norm(x): (x - mean) / sqrt(var + eps)
    /// Fused mean + variance + normalize in a single pass pattern
    pub fn layerNorm(
        comptime T: type,
        dst: []T,
        x: []const T,
        gamma: []const T,
        beta: []const T,
        eps: T,
    ) void {
        const n = x.len;
        if (n == 0) return;

        // Pass 1: compute mean
        var sum: T = 0;
        for (x) |v| sum += v;
        const mean = sum / @as(T, @floatFromInt(n));

        // Pass 2: compute variance
        var var_sum: T = 0;
        for (x) |v| {
            const diff = v - mean;
            var_sum += diff * diff;
        }
        const variance = var_sum / @as(T, @floatFromInt(n));
        const inv_std = 1.0 / @sqrt(variance + eps);

        // Pass 3: normalize + scale + shift (vectorized)
        const L = comptime kernels.lanes(T);
        const V = kernels.Vec(T);

        if (comptime kernels.hasVectorBinary(.add) and kernels.hasVectorBinary(.mul)) {
            const v_mean: V = @splat(mean);
            const v_inv_std: V = @splat(inv_std);

            var i: usize = 0;
            while (i + L <= n) : (i += L) {
                const vx: V = x[i..][0..L].*;
                const vg: V = gamma[i..][0..L].*;
                const vbeta: V = beta[i..][0..L].*;
                const normalized = (vx - v_mean) * v_inv_std;
                dst[i..][0..L].* = normalized * vg + vbeta;
            }

            const rem = n - i;
            if (rem > 0) {
                for (i..n) |j| {
                    dst[j] = (x[j] - mean) * inv_std * gamma[j] + beta[j];
                }
            }
        } else {
            for (0..n) |j| {
                dst[j] = (x[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "fused linear + relu (SIMD)" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;

    FusedOps.linearRelu(f32, dst, x, 2.0);

    for (0..n) |i| {
        const expected = @max(x[i] + 2.0, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "fused linear + sigmoid (SIMD)" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;

    FusedOps.linearSigmoid(f32, dst, x, 1.0);

    for (0..n) |i| {
        const tmp = x[i] + 1.0;
        const expected = 1.0 / (1.0 + @exp(-tmp));
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "fused linear + tanh (SIMD)" {
    const n = 500;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 50)) * 0.1 - 2.5;

    FusedOps.linearTanh(f32, dst, x, 0.5);

    for (0..n) |i| {
        const tmp = x[i] + 0.5;
        const expected: f32 = @floatCast(std.math.tanh(@as(f64, @floatCast(tmp))));
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-4);
    }
}

test "fused add + relu (SIMD)" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
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

test "fused add + sigmoid (SIMD)" {
    const n = 500;
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(i % 30)) * 0.1 - 1.5;
    }

    FusedOps.addSigmoid(f32, dst, a, b);

    for (0..n) |i| {
        const tmp = a[i] + b[i];
        const expected = 1.0 / (1.0 + @exp(-tmp));
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "fused mul + sigmoid (gating, SIMD)" {
    const n = 100;
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(i % 20)) * 0.2 - 2.0;
    }

    FusedOps.gateSigmoid(f32, dst, a, b);

    for (0..n) |i| {
        const sig: f32 = 1.0 / (1.0 + @exp(-b[i]));
        const expected = a[i] * sig;
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "fused mul + relu (SIMD)" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;
        b[i] = @as(f32, @floatFromInt(i % 50)) * 0.2 - 3.0;
    }

    FusedOps.mulRelu(f32, dst, a, b);

    for (0..n) |i| {
        const expected = @max(a[i] * b[i], 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "fused scale + relu (SIMD)" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;

    FusedOps.scaleRelu(f32, dst, x, 2.0);

    for (0..n) |i| {
        const expected = @max(x[i] * 2.0, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "fused layer norm (SIMD)" {
    const n = 256;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const gamma = try alloc.alloc(f32, n);
    defer alloc.free(gamma);
    const beta = try alloc.alloc(f32, n);
    defer alloc.free(beta);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        x[i] = @as(f32, @floatFromInt(i)) * 0.01;
        gamma[i] = 1.0;
        beta[i] = 0.0;
    }

    FusedOps.layerNorm(f32, dst, x, gamma, beta, 1e-5);

    // With gamma=1, beta=0, output should be normalized (mean~0, var~1)
    var mean: f32 = 0;
    for (dst) |v| mean += v;
    mean /= @as(f32, @floatFromInt(n));
    try std.testing.expectApproxEqAbs(0.0, mean, 1e-4);

    var variance: f32 = 0;
    for (dst) |v| {
        const diff = v - mean;
        variance += diff * diff;
    }
    variance /= @as(f32, @floatFromInt(n));
    try std.testing.expectApproxEqAbs(1.0, variance, 1e-2);
}

test "fused add + relu parallel (large array)" {
    const n = 200_000;
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i % 1000)) * 0.01 - 5.0;
        b[i] = @as(f32, @floatFromInt(i % 500)) * 0.02 - 3.0;
    }

    fusedBinaryUnaryParallel(f32, dst, a, b, .add, .relu);

    for (0..n) |i| {
        const expected = @max(a[i] + b[i], 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "fused linear + relu parallel (large array)" {
    const n = 200_000;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 1000)) * 0.01 - 5.0;

    fusedAddScalarUnaryParallel(f32, dst, x, 2.5, .relu);

    for (0..n) |i| {
        const expected = @max(x[i] + 2.5, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "fused operations work with f64" {
    const n = 100;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f64, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f64, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f64, @floatFromInt(i)) * 0.1 - 5.0;

    FusedOps.linearRelu(f64, dst, x, 2.0);

    for (0..n) |i| {
        const expected = @max(x[i] + 2.0, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-10);
    }
}

test "fused unary+unary: exp then relu" {
    // Test fusing two unary operations via binary op workaround
    const n = 100;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| x[i] = @as(f32, @floatFromInt(i % 20)) * 0.5 - 5.0;

    // Manually fuse: relu(exp(x) - 1) which is like softplus approximation
    // Use addRelu pattern: dst = relu(a + b) where a = exp(x), b = -1
    const exp_x = try alloc.alloc(f32, n);
    defer alloc.free(exp_x);
    const ones = try alloc.alloc(f32, n);
    defer alloc.free(ones);

    for (0..n) |i| {
        exp_x[i] = @exp(x[i]);
        ones[i] = -1.0;
    }

    FusedOps.addRelu(f32, dst, exp_x, ones);

    for (0..n) |i| {
        const expected = @max(@exp(x[i]) - 1.0, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-4);
    }
}

test "batch norm inference" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    // Create data with known mean and variance
    for (0..n) |i| {
        x[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    const mean = 5.0;
    const variance = 2.0;
    const gamma = 1.5;
    const beta = 0.5;
    const eps = 1e-5;

    FusedOps.batchNormInference(f32, dst, x, mean, variance, gamma, beta, eps);

    // Verify: dst = (x - mean) / sqrt(var + eps) * gamma + beta
    const inv_std = gamma / @sqrt(variance + eps);
    const shift = beta - mean * inv_std;
    for (0..n) |i| {
        const expected = x[i] * inv_std + shift;
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "residual connection with relu" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const residual = try alloc.alloc(f32, n);
    defer alloc.free(residual);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        x[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;
        residual[i] = @as(f32, @floatFromInt(i % 50)) * 0.2 - 3.0;
    }

    FusedOps.residualActivation(f32, dst, x, residual, .relu);

    for (0..n) |i| {
        const expected = @max(x[i] + residual[i], 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "residual connection with sigmoid" {
    const n = 500;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const residual = try alloc.alloc(f32, n);
    defer alloc.free(residual);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        x[i] = @as(f32, @floatFromInt(i)) * 0.01;
        residual[i] = @as(f32, @floatFromInt(i % 30)) * 0.1 - 1.5;
    }

    FusedOps.residualActivation(f32, dst, x, residual, .sigmoid);

    for (0..n) |i| {
        const tmp = x[i] + residual[i];
        const expected = 1.0 / (1.0 + @exp(-tmp));
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}
