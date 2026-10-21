//! Optimized kernels using comptime specialization
//! These optimizations provide maximum performance by eliminating all runtime overhead
//! Works for both C ABI (dynamic library) and pure Zig AOT compilation
//! 
//! The key insight: Zig's compiler performs aggressive inlining and specialization
//! in optimized builds (ReleaseSafe/ReleaseFast) regardless of linking mode
const std = @import("std");
const prims = @import("prims.zig");
const scalar = @import("scalar.zig");
const kernels = @import("kernels.zig");

const Prim = prims.Prim;

/// Always use specialized kernels in optimized builds
pub const is_aot = true;

/// Compile-time optimized unary kernel
/// When in AOT mode, generates fully specialized code for each type and operation
pub inline fn specializedUnary(
    comptime T: type,
    comptime p: Prim,
    dst: []T,
    src: []const T,
) void {
    // Use vectorized kernel if available, otherwise fall back
    if (comptime kernels.hasVectorForm(p)) {
        const V = kernels.Vec(T);
        const L = comptime kernels.lanes(T);
        
        // Main vectorized loop
        var i: usize = 0;
        const vec_end = dst.len - (dst.len % L);
        
        while (i < vec_end) : (i += L) {
            const x: V = src[i..][0..L].*;
            const y = kernels.applyUnary(T, p, x);
            dst[i..][0..L].* = y;
        }
        
        // Tail handling
        if (i < dst.len) {
            var tail: [L]T = undefined;
            const rem = dst.len - i;
            for (0..L) |k| {
                tail[k] = if (k < rem) src[i + k] else src[i + rem - 1];
            }
            const x: V = tail;
            const y = kernels.applyUnary(T, p, x);
            const result: [L]T = y;
            @memcpy(dst[i..][0..rem], result[0..rem]);
        }
    } else {
        // Fallback to scalar loop
        for (0..dst.len) |i| {
            dst[i] = scalar.unary(T, p, src[i]);
        }
    }
}

/// Compile-time optimized binary kernel
pub inline fn specializedBinary(
    comptime T: type,
    comptime p: Prim,
    dst: []T,
    a: []const T,
    b: []const T,
) void {
    // Use vectorized kernel if available, otherwise fall back
    if (comptime kernels.hasVectorBinary(p)) {
        const V = kernels.Vec(T);
        const L = comptime kernels.lanes(T);
        
        var i: usize = 0;
        const vec_end = dst.len - (dst.len % L);
        
        while (i < vec_end) : (i += L) {
            const x: V = a[i..][0..L].*;
            const y: V = b[i..][0..L].*;
            const z = kernels.applyBinary(T, p, x, y);
            dst[i..][0..L].* = z;
        }
        
        // Tail handling
        if (i < dst.len) {
            var tail_a: [L]T = undefined;
            var tail_b: [L]T = undefined;
            const rem = dst.len - i;
            for (0..L) |k| {
                tail_a[k] = if (k < rem) a[i + k] else a[i + rem - 1];
                tail_b[k] = if (k < rem) b[i + k] else b[i + rem - 1];
            }
            const x: V = tail_a;
            const y: V = tail_b;
            const z = kernels.applyBinary(T, p, x, y);
            const result: [L]T = z;
            @memcpy(dst[i..][0..rem], result[0..rem]);
        }
    } else {
        // Fallback to scalar loop
        for (0..dst.len) |i| {
            dst[i] = scalar.binary(T, p, a[i], b[i]);
        }
    }
}

/// Compile-time fused kernel for linear + activation
/// Generates completely specialized code at compile time
pub inline fn specializedFusedLinearActivation(
    comptime T: type,
    comptime activation: Prim,
    dst: []T,
    linear: []const T,
    bias: T,
) void {
    if (comptime is_aot) {
        const V = kernels.Vec(T);
        const L = comptime kernels.lanes(T);
        const bias_vec: V = @splat(bias);
        
        var i: usize = 0;
        const vec_end = dst.len - (dst.len % L);
        
        while (i < vec_end) : (i += L) {
            const x: V = linear[i..][0..L].*;
            const sum = x + bias_vec;
            
            const result = switch (activation) {
                .relu => @max(sum, @as(V, @splat(0))),
                .sigmoid => @as(V, @splat(1)) / (@as(V, @splat(1)) + @exp(-sum)),
                .tanh => @tan(sum),
                else => @compileError("Unsupported activation for fusion"),
            };
            
            dst[i..][0..L].* = result;
        }
        
        // Tail
        if (i < dst.len) {
            var tail: [L]T = undefined;
            const rem = dst.len - i;
            for (0..L) |k| {
                tail[k] = if (k < rem) linear[i + k] else linear[i + rem - 1];
            }
            const x: V = tail;
            const sum = x + bias_vec;
            
            const result = switch (activation) {
                .relu => @max(sum, @as(V, @splat(0))),
                .sigmoid => @as(V, @splat(1)) / (@as(V, @splat(1)) + @exp(-sum)),
                .tanh => @tan(sum),
                else => @compileError("Unsupported activation for fusion"),
            };
            
            const arr: [L]T = result;
            @memcpy(dst[i..][0..rem], arr[0..rem]);
        }
    } else {
        // Fallback for C ABI
        for (0..dst.len) |i| {
            const tmp = linear[i] + bias;
            dst[i] = scalar.unary(T, activation, tmp);
        }
    }
}

/// Compile-time constant folding
/// Evaluates operations on compile-time known values
pub fn comptimeEval(comptime T: type, comptime p: Prim, comptime value: T) T {
    return comptime scalar.unary(T, p, value);
}

/// Static dispatch table - eliminated at compile time in AOT mode
pub fn staticDispatch(comptime T: type, comptime p: Prim) *const fn ([]T, []const T) void {
    return switch (p) {
        .relu => struct {
            fn kernel(dst: []T, src: []const T) void {
                specializedUnary(T, .relu, dst, src);
            }
        }.kernel,
        .sigmoid => struct {
            fn kernel(dst: []T, src: []const T) void {
                specializedUnary(T, .sigmoid, dst, src);
            }
        }.kernel,
        .exp => struct {
            fn kernel(dst: []T, src: []const T) void {
                specializedUnary(T, .exp, dst, src);
            }
        }.kernel,
        else => @compileError("No static dispatch for " ++ @tagName(p)),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "specialized unary relu" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const src = try alloc.alloc(f32, n);
    defer alloc.free(src);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        src[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;
    }

    specializedUnary(f32, .relu, dst, src);

    for (0..n) |i| {
        const expected = @max(src[i], 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "specialized binary add" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(f32, n);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, n);
    defer alloc.free(b);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(i % 50)) * 0.02;
    }

    specializedBinary(f32, .add, dst, a, b);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(a[i] + b[i], dst[i], 1e-6);
    }
}

test "specialized fused linear relu" {
    const n = 1000;
    const alloc = std.testing.allocator;
    const linear = try alloc.alloc(f32, n);
    defer alloc.free(linear);
    const dst = try alloc.alloc(f32, n);
    defer alloc.free(dst);

    for (0..n) |i| {
        linear[i] = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;
    }

    specializedFusedLinearActivation(f32, .relu, dst, linear, 2.0);

    for (0..n) |i| {
        const expected = @max(linear[i] + 2.0, 0.0);
        try std.testing.expectApproxEqAbs(expected, dst[i], 1e-6);
    }
}

test "comptime constant folding" {
    // These are evaluated at compile time
    const relu_result = comptimeEval(f32, .relu, -5.0);
    try std.testing.expectEqual(@as(f32, 0.0), relu_result);

    const sigmoid_result = comptimeEval(f32, .sigmoid, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid_result, 1e-6);
}
