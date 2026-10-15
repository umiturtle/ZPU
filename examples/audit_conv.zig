//! Conv2d and NN operation audit
const std = @import("std");
const zpu = @import("zpu");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    try zpu.init(alloc, .{});
    defer zpu.deinit();

    var issues: u32 = 0;

    // Test 1: Conv2d basic
    std.debug.print("Test 1: Conv2d basic...\n", .{});
    {
        // Input: [1, 1, 5, 5] (batch, channels, height, width)
        const input_shape = try zpu.Shape.init(&.{ 1, 1, 5, 5 });
        const input = try zpu.Tensor.full(input_shape, .f32, 1.0);
        defer input.deinit();
        
        // Kernel: [1, 1, 3, 3]
        const kernel_shape = try zpu.Shape.init(&.{ 1, 1, 3, 3 });
        const kernel = try zpu.Tensor.full(kernel_shape, .f32, 1.0);
        defer kernel.deinit();
        
        const result = try zpu.nn.conv2d(input, kernel, 1, 0);
        defer result.deinit();
        
        // Output should be [1, 1, 3, 3] with each value = 9
        const expected_shape = try zpu.Shape.init(&.{ 1, 1, 3, 3 });
        if (!result.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Conv2d shape wrong: ", .{});
            issues += 1;
        } else if (result.readF64(0) != 9.0) {
            std.debug.print("  ❌ Conv2d value wrong: {d}\n", .{result.readF64(0)});
            issues += 1;
        } else {
            std.debug.print("  ✓ Conv2d works\n", .{});
        }
    }

    // Test 2: Conv2d with stride
    std.debug.print("\nTest 2: Conv2d with stride...\n", .{});
    {
        const input_shape = try zpu.Shape.init(&.{ 1, 1, 6, 6 });
        const input = try zpu.Tensor.full(input_shape, .f32, 1.0);
        defer input.deinit();
        
        const kernel_shape = try zpu.Shape.init(&.{ 1, 1, 3, 3 });
        const kernel = try zpu.Tensor.full(kernel_shape, .f32, 1.0);
        defer kernel.deinit();
        
        const result = try zpu.nn.conv2d(input, kernel, 2, 0);
        defer result.deinit();
        
        // With stride 2, output should be [1, 1, 2, 2]
        const expected_shape = try zpu.Shape.init(&.{ 1, 1, 2, 2 });
        if (!result.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Conv2d stride shape wrong: ", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Conv2d with stride works\n", .{});
        }
    }

    // Test 3: Conv2d with padding
    std.debug.print("\nTest 3: Conv2d with padding...\n", .{});
    {
        const input_shape = try zpu.Shape.init(&.{ 1, 1, 5, 5 });
        const input = try zpu.Tensor.full(input_shape, .f32, 1.0);
        defer input.deinit();
        
        const kernel_shape = try zpu.Shape.init(&.{ 1, 1, 3, 3 });
        const kernel = try zpu.Tensor.full(kernel_shape, .f32, 1.0);
        defer kernel.deinit();
        
        const result = try zpu.nn.conv2d(input, kernel, 1, 1);
        defer result.deinit();
        
        // With padding 1, output should be [1, 1, 5, 5]
        const expected_shape = try zpu.Shape.init(&.{ 1, 1, 5, 5 });
        if (!result.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Conv2d padding shape wrong: ", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Conv2d with padding works\n", .{});
        }
    }

    // Test 4: MaxPool2d
    std.debug.print("\nTest 4: MaxPool2d...\n", .{});
    {
        const input_shape = try zpu.Shape.init(&.{ 1, 1, 4, 4 });
        const input = try zpu.Tensor.full(input_shape, .f32, 1.0);
        defer input.deinit();
        
        const result = try zpu.nn.maxPool2d(input, 2);
        defer result.deinit();
        
        const expected_shape = try zpu.Shape.init(&.{ 1, 1, 2, 2 });
        if (!result.shape.eql(expected_shape)) {
            std.debug.print("  ❌ MaxPool2d shape wrong: ", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ MaxPool2d works\n", .{});
        }
    }

    // Test 5: Dense layer
    std.debug.print("\nTest 5: Dense layer...\n", .{});
    {
        const x_shape = try zpu.Shape.init(&.{ 2, 3 });
        const x = try zpu.Tensor.full(x_shape, .f32, 1.0);
        defer x.deinit();
        
        const w_shape = try zpu.Shape.init(&.{ 3, 4 });
        const w = try zpu.Tensor.full(w_shape, .f32, 1.0);
        defer w.deinit();
        
        const b_shape = try zpu.Shape.init(&.{4});
        const b = try zpu.Tensor.full(b_shape, .f32, 0.0);
        defer b.deinit();
        
        const result = try zpu.nn.dense(x, w, b);
        defer result.deinit();
        
        // [2,3] @ [3,4] + [4] = [2,4], each val = 3
        const expected_shape = try zpu.Shape.init(&.{ 2, 4 });
        if (!result.shape.eql(expected_shape)) {
            std.debug.print("  ❌ Dense shape wrong\n", .{});
            issues += 1;
        } else if (result.readF64(0) != 3.0) {
            std.debug.print("  ❌ Dense value wrong: {d}\n", .{result.readF64(0)});
            issues += 1;
        } else {
            std.debug.print("  ✓ Dense works\n", .{});
        }
    }

    // Test 6: Softmax
    std.debug.print("\nTest 6: Softmax...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{ 2, 3 });
        const tensor = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 }, shape);
        defer tensor.deinit();
        
        const result = try zpu.softmax(tensor, 1);
        defer result.deinit();
        
        // Sum along axis 1 should be 1.0 for each row
        const sum0 = result.readF64(0) + result.readF64(1) + result.readF64(2);
        if (@abs(sum0 - 1.0) > 1e-5) {
            std.debug.print("  ❌ Softmax sum wrong: {d}\n", .{sum0});
            issues += 1;
        } else {
            std.debug.print("  ✓ Softmax works\n", .{});
        }
    }

    // Test 7: Log softmax
    std.debug.print("\nTest 7: Log softmax...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{ 1, 3 });
        const tensor = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0 }, shape);
        defer tensor.deinit();
        
        const result = try zpu.logSoftmax(tensor, 1);
        defer result.deinit();
        
        // All values should be negative
        if (result.readF64(0) > 0 or result.readF64(1) > 0 or result.readF64(2) > 0) {
            std.debug.print("  ❌ Log softmax values not negative\n", .{});
            issues += 1;
        } else {
            std.debug.print("  ✓ Log softmax works\n", .{});
        }
    }

    // Test 8: One-hot encoding
    std.debug.print("\nTest 8: One-hot...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{3});
        const indices = try zpu.Tensor.fromArray(i32, &.{ 0, 2, 1 }, shape);
        defer indices.deinit();
        
        const result = try zpu.nn.oneHot(indices, 4, .f32);
        defer result.deinit();
        
        const expected_shape = try zpu.Shape.init(&.{ 3, 4 });
        if (!result.shape.eql(expected_shape)) {
            std.debug.print("  ❌ One-hot shape wrong\n", .{});
            issues += 1;
        } else {
            // Check: [0,2,1] with depth 4
            // Row 0: [1,0,0,0], Row 1: [0,0,1,0], Row 2: [0,1,0,0]
            if (result.readF64(0) != 1.0) {
                std.debug.print("  ❌ One-hot value [0,0] wrong: {d}\n", .{result.readF64(0)});
                issues += 1;
            } else if (result.readF64(6) != 1.0) { // Row 1, col 2
                std.debug.print("  ❌ One-hot value [1,2] wrong: {d}\n", .{result.readF64(6)});
                issues += 1;
            } else {
                std.debug.print("  ✓ One-hot works\n", .{});
            }
        }
    }

    // Test 9: Layer norm
    std.debug.print("\nTest 9: Layer norm...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{ 2, 4 });
        const x = try zpu.Tensor.full(shape, .f32, 3.0);
        defer x.deinit();
        
        const w = try zpu.Tensor.full(try zpu.Shape.init(&.{4}), .f32, 1.0);
        defer w.deinit();
        const b = try zpu.Tensor.full(try zpu.Shape.init(&.{4}), .f32, 0.0);
        defer b.deinit();
        
        const result = try zpu.nn.layerNorm(x, w, b, 1e-5, -1);
        defer result.deinit();
        
        // All constant input should normalize to 0
        if (@abs(result.readF64(0)) > 1e-4) {
            std.debug.print("  ❌ Layer norm value wrong: {d}\n", .{result.readF64(0)});
            issues += 1;
        } else {
            std.debug.print("  ✓ Layer norm works\n", .{});
        }
    }

    // Test 10: MSE loss
    std.debug.print("\nTest 10: MSE loss...\n", .{});
    {
        const shape = try zpu.Shape.init(&.{4});
        const pred = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, shape);
        defer pred.deinit();
        const target = try zpu.Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, shape);
        defer target.deinit();
        
        const result = try zpu.nn.mse(pred, target);
        defer result.deinit();
        
        if (@abs(result.item()) > 1e-6) {
            std.debug.print("  ❌ MSE should be 0, got {d}\n", .{result.item()});
            issues += 1;
        } else {
            std.debug.print("  ✓ MSE works\n", .{});
        }
    }

    std.debug.print("\n=== Conv/NN Audit Summary ===\n", .{});
    if (issues == 0) {
        std.debug.print("✓ All NN tests passed\n", .{});
    } else {
        std.debug.print("❌ Found {d} issue(s)\n", .{issues});
    }
}
