//! Edge case audit for zpu operations
const std = @import("std");
const zpu = @import("zpu");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;

    try zpu.init(alloc, .{});
    defer zpu.deinit();

    var issues: u32 = 0;

    // Test 1: Empty tensor operations (FIXED)
    std.debug.print("Test 1: Empty tensor operations...\n", .{});
    {
        const empty_shape = try zpu.Shape.init(&.{0});
        const empty = try zpu.Tensor.zeros(empty_shape, .f32);
        defer empty.deinit();
        
        // Sum of empty should be 0
        const sum = try zpu.sumAll(empty);
        defer sum.deinit();
        if (sum.item() != 0) {
            std.debug.print("  ❌ Sum of empty tensor is not 0: {d}\n", .{sum.item()});
            issues += 1;
        } else {
            std.debug.print("  ✓ Sum of empty tensor is 0\n", .{});
        }
    }

    // Test 2: Scalar tensor operations
    std.debug.print("\nTest 2: Scalar tensor operations...\n", .{});
    {
        const scalar_shape = try zpu.Shape.init(&.{});
        const scalar = try zpu.Tensor.full(scalar_shape, .f32, 5.0);
        defer scalar.deinit();
        
        const doubled = try zpu.mulScalar(scalar, 2.0);
        defer doubled.deinit();
        if (doubled.item() != 10.0) {
            std.debug.print("  ❌ Scalar mul failed: expected 10, got {d}\n", .{doubled.item()});
            issues += 1;
        } else {
            std.debug.print("  ✓ Scalar operations work\n", .{});
        }
    }

    // Test 3: Broadcasting edge cases
    std.debug.print("\nTest 3: Broadcasting edge cases...\n", .{});
    {
        // [3, 1] + [1, 4] should give [3, 4]
        const a_shape = try zpu.Shape.init(&.{ 3, 1 });
        const b_shape = try zpu.Shape.init(&.{ 1, 4 });
        const a = try zpu.Tensor.full(a_shape, .f32, 1.0);
        defer a.deinit();
        const b = try zpu.Tensor.full(b_shape, .f32, 2.0);
        defer b.deinit();
        
        const c = try zpu.add(a, b);
        defer c.deinit();
        
        const expected_shape = try zpu.Shape.init(&.{ 3, 4 });
        if (!c.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Broadcast shape wrong\n", .{});
            issues += 1;
        } else if (c.readF64(0) != 3.0) {
            std.debug.print("  ❌ Broadcast value wrong: {d}\n", .{c.readF64(0)});
            issues += 1;
        } else {
            std.debug.print("  ✓ Broadcasting works\n", .{});
        }
    }

    // Test 4: Type casting
    std.debug.print("\nTest 4: Type casting...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{3});
        const f64_tensor = try zpu.Tensor.fromArray(f64, &.{ 1.5, 2.7, 3.9 }, shape);
        defer f64_tensor.deinit();
        
        const f32_tensor = try zpu.cast(f64_tensor, .f32);
        defer f32_tensor.deinit();
        
        if (f32_tensor.dtype() != .f32) {
            std.debug.print("  ❌ Cast to f32 failed\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Type casting works\n", .{});
        }
    }

    // Test 5: Reduction on single element
    std.debug.print("\nTest 5: Reduction on single element...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{1});
        const tensor = try zpu.Tensor.fromArray(f32, &.{42.0}, shape);
        defer tensor.deinit();
        
        const sum = try zpu.sumAll(tensor);
        defer sum.deinit();
        if (sum.item() != 42.0) {
            std.debug.print("  ❌ Single element sum wrong: {d}\n", .{sum.item()});
            issues += 1;
        } else {
            std.debug.print("  ✓ Single element reduction works\n", .{});
        }
    }

    // Test 6: Matrix operations
    std.debug.print("\nTest 6: Matrix multiplication...\n", .{});
    {
        const a_shape = try zpu.Shape.init(&.{ 2, 3 });
        const b_shape = try zpu.Shape.init(&.{ 3, 2 });
        const a = try zpu.Tensor.full(a_shape, .f32, 1.0);
        defer a.deinit();
        const b = try zpu.Tensor.full(b_shape, .f32, 2.0);
        defer b.deinit();
        
        const c = try zpu.matmul(a, b);
        defer c.deinit();
        
        // [2,3] @ [3,2] = [2,2], each element = 3 * 1 * 2 = 6
        const expected_shape = try zpu.Shape.init(&.{ 2, 2 });
        if (!c.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Matmul shape wrong\n", .{});
            issues += 1;
        } else if (c.readF64(0) != 6.0) {
            std.debug.print("  ❌ Matmul value wrong: {d}\n", .{c.readF64(0)});
            issues += 1;
        } else {
            std.debug.print("  ✓ Matrix multiplication works\n", .{});
        }
    }

    // Test 7: Argmax/Argmin
    std.debug.print("\nTest 7: Argmax/Argmin...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{5});
        const tensor = try zpu.Tensor.fromArray(f32, &.{ 1.0, 5.0, 3.0, 2.0, 4.0 }, shape);
        defer tensor.deinit();
        
        const argmax = try zpu.argmax(tensor, 0);
        defer argmax.deinit();
        if (argmax.readF64(0) != 1.0) {
            std.debug.print("  ❌ Argmax wrong: expected 1, got {d}\n", .{argmax.readF64(0)});
            issues += 1;
        } else {
            std.debug.print("  ✓ Argmax works\n", .{});
        }
    }

    // Test 8: Variance/Stddev
    std.debug.print("\nTest 8: Variance/Stddev...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{4});
        const tensor = try zpu.Tensor.fromArray(f64, &.{ 2.0, 4.0, 4.0, 4.0 }, shape);
        defer tensor.deinit();
        
        const mean = try zpu.meanAll(tensor);
        defer mean.deinit();
        // mean = (2+4+4+4)/4 = 3.5
        if (@abs(mean.item() - 3.5) > 1e-6) {
            std.debug.print("  ❌ Mean wrong: {d}\n", .{mean.item()});
            issues += 1;
        } else {
            std.debug.print("  ✓ Mean works\n", .{});
        }
    }

    // Test 9: Concat/Stack
    std.debug.print("\nTest 9: Concat/Stack...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{2});
        const a = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0 }, shape);
        defer a.deinit();
        const b = try zpu.Tensor.fromArray(f32, &.{ 3.0, 4.0 }, shape);
        defer b.deinit();
        
        const concat = try zpu.concat(a, b, 0);
        defer concat.deinit();
        
        const expected_shape = try zpu.Shape.init(&.{4});
        if (!concat.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Concat shape wrong\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Concat works\n", .{});
        }
    }

    // Test 10: Where operation
    std.debug.print("\nTest 10: Where operation...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{5});
        const cond = try zpu.Tensor.fromArray(f32, &.{ 1.0, 0.0, 1.0, 0.0, 1.0 }, shape);
        defer cond.deinit();
        const a = try zpu.Tensor.full(shape, .f32, 10.0);
        defer a.deinit();
        const b = try zpu.Tensor.full(shape, .f32, 20.0);
        defer b.deinit();
        
        const result = try zpu.where(cond, a, b);
        defer result.deinit();
        
        if (result.readF64(0) != 10.0 or result.readF64(1) != 20.0) {
            std.debug.print("  ❌ Where operation wrong\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Where operation works\n", .{});
        }
    }

    std.debug.print("\n=== Audit Summary ===\n", .{});
    if (issues == 0) {
        std.debug.print("✓ All edge case tests passed\n", .{});
    } else {
        std.debug.print("❌ Found {d} issue(s)\n", .{issues});
    }
}
