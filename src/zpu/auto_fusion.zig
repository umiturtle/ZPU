//! Automatic kernel fusion optimization pass
//! Detects patterns like relu(add(a, b)) and fuses them into single kernels
//! Works for both C ABI (dynamic) and AOT (static) scenarios
const std = @import("std");
const prims = @import("prims.zig");
const fusion = @import("fusion.zig");
const kernels = @import("kernels.zig");
const scalar = @import("scalar.zig");

const Prim = prims.Prim;

/// Fusion pattern detection and application
pub const FusionPass = struct {
    /// Detect and apply fusion for consecutive unary operations
    pub fn fuseUnaryUnary(
        comptime T: type,
        dst: []T,
        src: []const T,
        inner: Prim,
        outer: Prim,
    ) void {
        // Use fused kernel if available
        fusion.fusedUnaryUnary(T, dst, src, inner, outer);
    }

    /// Detect and apply fusion for binary + unary operations
    pub fn fuseBinaryUnary(
        comptime T: type,
        dst: []T,
        a: []const T,
        b: []const T,
        binary_op: Prim,
        unary_op: Prim,
    ) void {
        // Use fused kernel if available
        fusion.fusedBinaryUnary(T, dst, a, b, binary_op, unary_op);
    }

    /// Check if a fusion pattern is beneficial
    pub fn shouldFuse(comptime outer: Prim, comptime inner: Prim) bool {
        // Fuse if both operations have vector forms
        return kernels.hasVectorForm(outer) and kernels.hasVectorForm(inner);
    }
};

/// High-level fusion API for common patterns
pub const AutoFusion = struct {
    /// Fused linear + activation: dst = activation(src @ weight + bias)
    pub fn fusedLinearActivation(
        comptime T: type,
        dst: []T,
        linear_result: []const T,
        bias: []const T,
        comptime activation: Prim,
    ) void {
        const n = dst.len;
        
        // Fuse add + activation
        switch (activation) {
            .relu => {
                for (0..n) |i| {
                    const tmp = linear_result[i] + bias[i];
                    dst[i] = @max(tmp, 0);
                }
            },
            .sigmoid => {
                for (0..n) |i| {
                    const tmp = linear_result[i] + bias[i];
                    dst[i] = 1.0 / (1.0 + @exp(-tmp));
                }
            },
            .tanh => {
                for (0..n) |i| {
                    const tmp = linear_result[i] + bias[i];
                    dst[i] = @tan(tmp);
                }
            },
            else => {
                // Fallback: separate operations
                for (0..n) |i| {
                    dst[i] = linear_result[i] + bias[i];
                }
                for (0..n) |i| {
                    dst[i] = scalar.unary(T, activation, dst[i]);
                }
            },
        }
    }

    /// Fused gating operation: dst = a * sigmoid(b)
    pub fn fusedGate(
        comptime T: type,
        dst: []T,
        a: []const T,
        b: []const T,
    ) void {
        const n = dst.len;
        for (0..n) |i| {
            const gate = 1.0 / (1.0 + @exp(-b[i]));
            dst[i] = a[i] * gate;
        }
    }

    /// Fused residual connection: dst = activation(x + residual)
    pub fn fusedResidual(
        comptime T: type,
        dst: []T,
        x: []const T,
        residual: []const T,
        comptime activation: Prim,
    ) void {
        const n = dst.len;
        
        switch (activation) {
            .relu => {
                for (0..n) |i| {
                    dst[i] = @max(x[i] + residual[i], 0);
                }
            },
            .sigmoid => {
                for (0..n) |i| {
                    const tmp = x[i] + residual[i];
                    dst[i] = 1.0 / (1.0 + @exp(-tmp));
                }
            },
            else => {
                for (0..n) |i| {
                    dst[i] = x[i] + residual[i];
                }
                for (0..n) |i| {
                    dst[i] = scalar.unary(T, activation, dst[i]);
                }
            },
        }
    }

    /// Fused layer norm: dst = (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn fusedLayerNorm(
        comptime T: type,
        dst: []T,
        x: []const T,
        gamma: []const T,
        beta: []const T,
        eps: T,
    ) void {
        const n = x.len;
        
        // Compute mean
        var mean: T = 0;
        for (x) |v| mean += v;
        mean /= @as(T, @floatFromInt(n));
        
        // Compute variance
        var variance: T = 0;
        for (x) |v| {
            const diff = v - mean;
            variance += diff * diff;
        }
        variance /= @as(T, @floatFromInt(n));
        
        // Normalize and scale
        const inv_std = 1.0 / @sqrt(variance + eps);
        for (0..n) |i| {
            dst[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
};

test "auto fusion - fused gate" {
    const n = 1000;
    const alloc = std.testing.allocator;
    var a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    var b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    var dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(i % 20)) * 0.2 - 2.0;
    }

    AutoFusion.fusedGate(f32, dst, a, b);

    for (0..n) |i| {
        const gate = 1.0 / (1.0 + @exp(-b[i]));
        const expected = a[i] * gate;
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-5);
    }
}

test "auto fusion - fused residual relu" {
    const n = 1000;
    const alloc = std.testing.allocator;
    var x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    var residual = try alloc.alloc(f32, n);
    defer alloc.free(residual);
    var dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        x[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;
        residual[i] = @as(f32, @floatFromInt(i % 50)) * 0.2;
    }

    AutoFusion.fusedResidual(f32, dst, x, residual, .relu);

    for (0..n) |i| {
        const expected = @max(x[i] + residual[i], 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}
