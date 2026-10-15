//! Advanced edge case audit
const std = @import("std");
const zpu = @import("zpu");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    try zpu.init(alloc, .{});
    defer zpu.deinit();

    var issues: u32 = 0;

    // Test 1: Transpose high-rank tensors
    std.debug.print("Test 1: High-rank transpose...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{ 2, 3, 4 });
        const tensor = try zpu.Tensor.full(shape, .f32, 1.0);
        defer tensor.deinit();
        
        const transposed = try zpu.transpose(tensor, &.{ 2, 0, 1 });
        defer transposed.deinit();
        
        const expected = try zpu.Shape.init(&.{ 4, 2, 3 });
        if (!transposed.shape.eql(expected)) {
            std.debug.print("  ❌ High-rank transpose failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ High-rank transpose works\n", .{});
        }
    }

    // Test 2: Reshape
    std.debug.print("\nTest 2: Reshape...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{ 2, 3, 4 });
        const tensor = try zpu.Tensor.full(shape, .f32, 1.0);
        defer tensor.deinit();
        
        const reshaped = try zpu.reshape(tensor, try zpu.Shape.init(&.{ 2, 12 }));
        defer reshaped.deinit();
        
        const expected = try zpu.Shape.init(&.{ 2, 12 });
        if (!reshaped.shape.eql(expected)) {
            std.debug.print("  ❌ Reshape failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Reshape works\n", .{});
        }
    }

    // Test 3: Slicing
    std.debug.print("\nTest 3: Slicing...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{10});
        const tensor = try zpu.Tensor.fromArray(f32, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, shape);
        defer tensor.deinit();
        
        const sliced = try zpu.narrow(tensor, 0, 7, 3);
        defer sliced.deinit();
        
        if (sliced.readF64(0) != 7.0 or sliced.readF64(2) != 9.0) {
            std.debug.print("  ❌ Slicing failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Slicing works\n", .{});
        }
    }

    // Test 4: Division by zero handling
    std.debug.print("\nTest 4: Division by zero...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{3});
        const a = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0 }, shape);
        defer a.deinit();
        const b = try zpu.Tensor.fromArray(f32, &.{ 1.0, 0.0, 3.0 }, shape);
        defer b.deinit();
        
        const result = try zpu.div(a, b);
        defer result.deinit();
        
        const val = result.readF64(1);
        if (!std.math.isInf(val) and !std.math.isNan(val)) {
            std.debug.print("  ❌ Division by zero not handled correctly: {d}\n", .{val});
            issues += 1;
        } else {
            std.debug.print("  ✓ Division by zero handled\n", .{});
        }
    }

    // Test 5: Power operations
    std.debug.print("\nTest 5: Power operations...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{3});
        const base = try zpu.Tensor.fromArray(f32, &.{ 2.0, 3.0, 4.0 }, shape);
        defer base.deinit();
        const exp = try zpu.Tensor.fromArray(f32, &.{ 2.0, 2.0, 2.0 }, shape);
        defer exp.deinit();
        
        const result = try zpu.pow(base, exp);
        defer result.deinit();
        
        if (result.readF64(0) != 4.0 or result.readF64(1) != 9.0 or result.readF64(2) != 16.0) {
            std.debug.print("  ❌ Power failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Power works\n", .{});
        }
    }

    // Test 6: Sqrt of negative numbers
    std.debug.print("\nTest 6: Sqrt of negative numbers...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{2});
        const tensor = try zpu.Tensor.fromArray(f32, &.{ -1.0, -4.0 }, shape);
        defer tensor.deinit();
        
        const result = try zpu.sqrt(tensor);
        defer result.deinit();
        
        if (!std.math.isNan(result.readF64(0))) {
            std.debug.print("  ❌ Sqrt of negative not handled\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Sqrt of negative handled\n", .{});
        }
    }

    // Test 7: Log of zero and negative
    std.debug.print("\nTest 7: Log of zero/negative...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{3});
        const tensor = try zpu.Tensor.fromArray(f32, &.{ 0.0, -1.0, 1.0 }, shape);
        defer tensor.deinit();
        
        const result = try zpu.log(tensor);
        defer result.deinit();
        
        const v0 = result.readF64(0);
        const v1 = result.readF64(1);
        const v2 = result.readF64(2);
        
        if (!std.math.isInf(v0) or !std.math.isNan(v1) or @abs(v2) > 1e-6) {
            std.debug.print("  ❌ Log edge cases not handled\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Log edge cases handled\n", .{});
        }
    }

    // Test 8: Large tensor allocation
    std.debug.print("\nTest 8: Large tensor...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{ 1000, 1000 });
        const tensor = try zpu.Tensor.full(shape, .f32, 1.0);
        defer tensor.deinit();
        
        const sum = try zpu.sumAll(tensor);
        defer sum.deinit();
        
        if (sum.item() != 1_000_000.0) {
            std.debug.print("  ❌ Large tensor sum wrong: {d}\n", .{sum.item()});
            issues += 1;
        } else {
            std.debug.print("  ✓ Large tensor works\n", .{});
        }
    }

    // Test 9: Cumulative sum
    std.debug.print("\nTest 9: Cumulative sum...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{5});
        const tensor = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0, 4.0, 5.0 }, shape);
        defer tensor.deinit();
        
        const cumsum = try zpu.cumsum(tensor, 0, false);
        defer cumsum.deinit();
        
        if (cumsum.readF64(0) != 1.0 or cumsum.readF64(4) != 15.0) {
            std.debug.print("  ❌ Cumsum failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Cumsum works\n", .{});
        }
    }

    // Test 10: Clip operation
    std.debug.print("\nTest 10: Clip operation...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{5});
        const tensor = try zpu.Tensor.fromArray(f32, &.{ -5.0, -1.0, 0.0, 3.0, 10.0 }, shape);
        defer tensor.deinit();
        
        const clipped = try zpu.clamp(tensor, 0.0, 5.0);
        defer clipped.deinit();
        
        if (clipped.readF64(0) != 0.0 or clipped.readF64(4) != 5.0) {
            std.debug.print("  ❌ Clip failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Clip works\n", .{});
        }
    }

    std.debug.print("\n=== Advanced Audit Summary ===\n", .{});
    if (issues == 0) {
        std.debug.print("✓ All advanced tests passed\n", .{});
    } else {
        std.debug.print("❌ Found {d} issue(s)\n", .{issues});
    }
}
