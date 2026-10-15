//! Demonstration of JAX-style advanced indexing operations
//!
//! Run with: `zig build run-indexing`
//!
//! This example shows:
//! 1. dynamic_slice - Extract slices with runtime-computed indices
//! 2. dynamic_update_slice - Update slices with runtime-computed indices
//! 3. gather - Advanced indexing with flexible dimension mappings

const std = @import("std");
const zpu = @import("zpu");

fn printTensor(name: []const u8, t: zpu.Tensor) void {
    std.debug.print("{s} shape: [", .{name});
    for (t.shape.dims[0..t.rank()]) |d| std.debug.print("{d}", .{d});
    std.debug.print("]\n", .{});
    std.debug.print("  data: [", .{});
    const n = @min(t.numel(), 20);
    for (0..n) |i| {
        std.debug.print("{d:.2}", .{t.readF64(i)});
        if (i < n - 1) std.debug.print(", ", .{});
    }
    if (t.numel() > 20) std.debug.print(", ...", .{});
    std.debug.print("]\n\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    std.debug.print("=== Advanced Indexing Operations Demo ===\n\n", .{});
    
    // 1. dynamic_slice
    std.debug.print("1. Dynamic Slice\n", .{});
    std.debug.print("----------------\n", .{});
    
    // Create a 4x4 matrix
    const matrix_data = [_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    const matrix = try zpu.Tensor.fromArray(f64, &matrix_data, try zpu.Shape.init(&.{ 4, 4 }));
    defer matrix.deinit();
    printTensor("Matrix", matrix);
    
    // Extract 2x2 slice starting at (1, 1)
    const start_row = try zpu.Tensor.fromScalarF64(1.0, .i32);
    defer start_row.deinit();
    const start_col = try zpu.Tensor.fromScalarF64(1.0, .i32);
    defer start_col.deinit();
    
    const slice = try zpu.ops.dynamicSlice(matrix, &.{ start_row, start_col }, &.{ 2, 2 });
    defer slice.deinit();
    printTensor("Slice at (1,1) size (2,2)", slice);
    
    // Extract with different start positions
    const start_row2 = try zpu.Tensor.fromScalarF64(2.0, .i32);
    defer start_row2.deinit();
    const start_col2 = try zpu.Tensor.fromScalarF64(0.0, .i32);
    defer start_col2.deinit();
    
    const slice2 = try zpu.ops.dynamicSlice(matrix, &.{ start_row2, start_col2 }, &.{ 2, 3 });
    defer slice2.deinit();
    printTensor("Slice at (2,0) size (2,3)", slice2);
    
    // 2. dynamic_update_slice
    std.debug.print("2. Dynamic Update Slice\n", .{});
    std.debug.print("-----------------------\n", .{});
    
    // Update the 2x2 block at (0, 0) with new values
    const update_data = [_]f64{ 100, 200, 300, 400 };
    const update = try zpu.Tensor.fromArray(f64, &update_data, try zpu.Shape.init(&.{ 2, 2 }));
    defer update.deinit();
    
    const update_start_row = try zpu.Tensor.fromScalarF64(0.0, .i32);
    defer update_start_row.deinit();
    const update_start_col = try zpu.Tensor.fromScalarF64(0.0, .i32);
    defer update_start_col.deinit();
    
    const updated = try zpu.ops.dynamicUpdateSlice(matrix, update, &.{ update_start_row, update_start_col });
    defer updated.deinit();
    printTensor("Matrix after update at (0,0)", updated);
    
    // Update at different position
    const update_start_row2 = try zpu.Tensor.fromScalarF64(1.0, .i32);
    defer update_start_row2.deinit();
    const update_start_col2 = try zpu.Tensor.fromScalarF64(2.0, .i32);
    defer update_start_col2.deinit();
    
    const update2 = try zpu.Tensor.fromArray(f64, &.{ 50, 60, 70, 80 }, try zpu.Shape.init(&.{ 2, 2 }));
    defer update2.deinit();
    
    const updated2 = try zpu.ops.dynamicUpdateSlice(matrix, update2, &.{ update_start_row2, update_start_col2 });
    defer updated2.deinit();
    printTensor("Matrix after update at (1,2)", updated2);
    
    // 3. gather
    std.debug.print("3. Gather\n", .{});
    std.debug.print("---------\n", .{});
    
    // Create a 3D tensor
    const tensor3d_data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const tensor3d = try zpu.Tensor.fromArray(f64, &tensor3d_data, try zpu.Shape.init(&.{ 2, 2, 2 }));
    defer tensor3d.deinit();
    printTensor("3D Tensor", tensor3d);
    
    // Gather multiple slices
    // start_indices shape: [batch_size, num_dimensions]
    const gather_indices_data = [_]f64{
        0, 0, 0, // Start at (0,0,0)
        1, 0, 0, // Start at (1,0,0)
        0, 1, 0, // Start at (0,1,0)
    };
    const gather_indices = try zpu.Tensor.fromArray(f64, &gather_indices_data, try zpu.Shape.init(&.{ 3, 3 }));
    defer gather_indices.deinit();
    
    const gathered = try zpu.ops.gather(tensor3d, gather_indices, &.{ 1, 1, 2 });
    defer gathered.deinit();
    printTensor("Gathered slices", gathered);
    
    // 4. Practical example: Sliding window
    std.debug.print("4. Practical Example: Sliding Window\n", .{});
    std.debug.print("------------------------------------\n", .{});
    
    // Create a 1D signal
    const signal_data = [_]f64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const signal = try zpu.Tensor.fromArray(f64, &signal_data, try zpu.Shape.init(&.{10}));
    defer signal.deinit();
    printTensor("Signal", signal);
    
    // Extract sliding windows of size 3
    const window_size = 3;
    const num_windows = 8;
    
    std.debug.print("Sliding windows of size 3:\n", .{});
    for (0..num_windows) |i| {
        const start_idx = try zpu.Tensor.fromScalarF64(@floatFromInt(i), .i32);
        const window = try zpu.ops.dynamicSlice(signal, &.{start_idx}, &.{window_size});
        defer window.deinit();
        
        std.debug.print("  Window {}: [", .{i});
        for (0..window.numel()) |j| {
            std.debug.print("{d:.0}", .{window.readF64(j)});
            if (j < window.numel() - 1) std.debug.print(", ", .{});
        }
        std.debug.print("]\n", .{});
        start_idx.deinit();
    }
    
    std.debug.print("\n=== Demo Complete ===\n", .{});
}
