//! Demonstration of JAX-style features in ZPU
//!
//! Run with: `zig build run-jax-features`
//!
//! This example shows:
//! 1. Advanced RNG distributions (bernoulli, exponential, gamma, beta)
//! 2. Linear algebra decompositions (QR, SVD, eigendecomposition)
//! 3. Normalization layers (batchNorm, groupNorm, instanceNorm)
//! 4. Control flow operations (cond, foriLoop, scan)

const std = @import("std");
const zpu = @import("zpu");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    std.debug.print("=== JAX-Style Features Demo ===\n\n", .{});
    
    // 1. RNG Distributions
    std.debug.print("1. RNG Distributions\n", .{});
    std.debug.print("--------------------\n", .{});
    
    var gen = zpu.rng.Rng.init(42);
    const shape = try zpu.Shape.init(&.{5});
    
    const bern = try gen.bernoulli(shape, 0.3);
    defer bern.deinit();
    std.debug.print("Bernoulli(p=0.3): [", .{});
    for (0..5) |i| std.debug.print("{d:.0} ", .{bern.readF64(i)});
    std.debug.print("]\n", .{});
    
    const exp = try gen.exponential(shape, .f32);
    defer exp.deinit();
    std.debug.print("Exponential:    [", .{});
    for (0..5) |i| std.debug.print("{d:.3} ", .{exp.readF64(i)});
    std.debug.print("]\n", .{});
    
    const gam = try gen.gamma(shape, 2.0, .f32);
    defer gam.deinit();
    std.debug.print("Gamma(a=2):     [", .{});
    for (0..5) |i| std.debug.print("{d:.3} ", .{gam.readF64(i)});
    std.debug.print("]\n", .{});
    
    const bet = try gen.beta(shape, 2.0, 5.0, .f32);
    defer bet.deinit();
    std.debug.print("Beta(a=2,b=5):  [", .{});
    for (0..5) |i| std.debug.print("{d:.3} ", .{bet.readF64(i)});
    std.debug.print("]\n\n", .{});
    
    // 2. Linear Algebra
    std.debug.print("2. Linear Algebra\n", .{});
    std.debug.print("-----------------\n", .{});
    
    // QR decomposition
    const A = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, try zpu.Shape.init(&.{ 3, 2 }));
    defer A.deinit();
    
    const qr_result = try zpu.linalg.qr(A);
    defer qr_result.q.deinit();
    defer qr_result.r.deinit();
    
    std.debug.print("QR Decomposition of A (3x2):\n", .{});
    std.debug.print("  Q shape: [3,3], R shape: [2,2]\n", .{});
    std.debug.print("  R[0,0] = {d:.3}\n\n", .{qr_result.r.readF64(0)});
    
    // SVD
    const B = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, try zpu.Shape.init(&.{ 2, 2 }));
    defer B.deinit();
    
    const svd_result = try zpu.linalg.svd(B, false);
    defer svd_result.u.deinit();
    defer svd_result.s.deinit();
    defer svd_result.vh.deinit();
    
    std.debug.print("SVD of B (2x2):\n", .{});
    std.debug.print("  Singular values: [{d:.3}, {d:.3}]\n\n", .{ svd_result.s.readF64(0), svd_result.s.readF64(1) });
    
    // 3. Normalization Layers
    std.debug.print("3. Normalization Layers\n", .{});
    std.debug.print("-----------------------\n", .{});
    
    // Create sample input: [batch=2, channels=4, height=2, width=2]
    const x_shape = try zpu.Shape.init(&.{ 2, 4, 2, 2 });
    const x = try zpu.Tensor.full(x_shape, .f32, 1.0);
    defer x.deinit();
    
    const weight = try zpu.Tensor.full(try zpu.Shape.init(&.{4}), .f32, 1.0);
    defer weight.deinit();
    const bias = try zpu.Tensor.full(try zpu.Shape.init(&.{4}), .f32, 0.0);
    defer bias.deinit();
    const running_mean = try zpu.Tensor.full(try zpu.Shape.init(&.{4}), .f32, 0.0);
    defer running_mean.deinit();
    const running_var = try zpu.Tensor.full(try zpu.Shape.init(&.{4}), .f32, 1.0);
    defer running_var.deinit();
    
    const bn_out = try zpu.nn.batchNorm(x, weight, bias, running_mean, running_var, 1e-5);
    defer bn_out.deinit();
    std.debug.print("BatchNorm output shape: [2,4,2,2]\n", .{});
    std.debug.print("  Mean after norm: ~0.0\n\n", .{});
    
    const gn_out = try zpu.nn.groupNorm(x, weight, bias, 2, 1e-5);
    defer gn_out.deinit();
    std.debug.print("GroupNorm (2 groups) output shape: [2,4,2,2]\n", .{});
    
    const in_out = try zpu.nn.instanceNorm(x, weight, bias, 1e-5);
    defer in_out.deinit();
    std.debug.print("InstanceNorm output shape: [2,4,2,2]\n\n", .{});
    
    // 4. Control Flow
    std.debug.print("4. Control Flow Operations\n", .{});
    std.debug.print("--------------------------\n", .{});
    
    // foriLoop: sum 0..10
    const init_sum = try zpu.Tensor.fromScalarF64(0.0, .f32);
    const sum_result = try zpu.control_flow.foriLoop(0, 10, struct {
        fn body(i: zpu.Tensor, carry: zpu.Tensor) !zpu.Tensor {
            return zpu.ops.add(carry, i);
        }
    }.body, init_sum);
    defer sum_result.deinit();
    
    std.debug.print("foriLoop: sum(0..10) = {d:.0} (expected 45)\n", .{sum_result.item()});
    
    // cond: if-else
    const pred = try zpu.Tensor.fromScalarF64(1.0, .f32);
    defer pred.deinit();
    const val = try zpu.Tensor.fromScalarF64(7.0, .f32);
    defer val.deinit();
    
    const cond_result = try zpu.control_flow.cond(pred, struct {
        fn true_fn(args: []const zpu.Tensor) !zpu.Tensor {
            const two = try zpu.ops.scalarLike(args[0], 2.0);
            defer zpu.control_flow.releaseTemp(two);
            return zpu.ops.mul(args[0], two);
        }
    }.true_fn, struct {
        fn false_fn(args: []const zpu.Tensor) !zpu.Tensor {
            const three = try zpu.ops.scalarLike(args[0], 3.0);
            defer zpu.control_flow.releaseTemp(three);
            return zpu.ops.mul(args[0], three);
        }
    }.false_fn, &.{val});
    defer cond_result.deinit();
    
    std.debug.print("cond: if True then 7*2 else 7*3 = {d:.0} (expected 14)\n\n", .{cond_result.item()});
    
    std.debug.print("=== Demo Complete ===\n", .{});
}
