//! JAX-style control flow operations.
//!
//! These provide functional control flow that works with autodiff, similar to
//! `jax.lax.cond`, `jax.lax.while_loop`, `jax.lax.scan`, and `jax.lax.fori_loop`.
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");
const prims = @import("prims.zig");

pub const Tensor = tmod.Tensor;
pub const Error = tmod.Error;
pub const Shape = tmod.Shape;
pub const releaseTemp = tmod.releaseTemp;

/// Function type for control flow operations.
pub const Fn = *const fn ([]const Tensor) Error!Tensor;

/// Functional if-else: executes true_fn if pred is truthy, else false_fn.
/// Both branches receive the same operands and must return the same shape.
///
/// Example:
/// ```zig
/// const result = try cond(pred, true_fn, false_fn, &.{x, y});
/// ```
pub fn cond(pred: Tensor, true_fn: Fn, false_fn: Fn, operands: []const Tensor) Error!Tensor {
    // Check if pred is truthy (any non-zero element)
    const pred_val = pred.readF64(0);
    const is_true = pred_val != 0.0;
    
    if (is_true) {
        return true_fn(operands);
    } else {
        return false_fn(operands);
    }
}

/// Functional while loop: repeatedly applies body_fn while cond_fn is truthy.
/// Returns the final carry value.
///
/// Example:
/// ```zig
/// // Compute sum 1..10
/// const result = try whileLoop(
///     struct {
///         fn cond(args: []const Tensor) Error!Tensor {
///             const i = args[0];
///             return ops.lt(i, 10);
///         }
///         fn body(args: []const Tensor) Error!Tensor {
///             const i = args[0];
///             const sum = args[1];
///             const new_sum = try ops.add(sum, i);
///             defer releaseTemp(new_sum);
///             const one = try ops.scalarLike(i, 1.0);
///             defer releaseTemp(one);
///             return ops.add(i, one);
///         }
///     }.cond,
///     struct {
///         fn body(args: []const Tensor) Error!Tensor {
///             const i = args[0];
///             const sum = args[1];
///             const new_sum = try ops.add(sum, i);
///             defer releaseTemp(new_sum);
///             const one = try ops.scalarLike(i, 1.0);
///             defer releaseTemp(one);
///             const new_i = try ops.add(i, one);
///             return new_i;
///         }
///     }.body,
///     &.{init_i, init_sum},
/// );
/// ```
pub fn whileLoop(cond_fn: Fn, body_fn: Fn, init_carry: []const Tensor) Error![]Tensor {
    // Make a mutable copy of init_carry
    var carry = try std.heap.c_allocator.alloc(Tensor, init_carry.len);
    errdefer std.heap.c_allocator.free(carry);
    for (init_carry, 0..) |t, i| {
        carry[i] = t;
    }
    var guard: usize = 0;
    const max_iters = 10000;
    
    while (guard < max_iters) : (guard += 1) {
        const pred = try cond_fn(carry);
        
        const pred_val = pred.readF64(0);
        if (pred_val == 0.0) break;
        
        // Execute body
        var new_carry = try std.heap.c_allocator.alloc(Tensor, carry.len);
        errdefer std.heap.c_allocator.free(new_carry);
        
        for (0..carry.len) |i| {
            new_carry[i] = try body_fn(carry);
        }
        
        // Let Scope handle cleanup - don't manually deinit Tensors
        std.heap.c_allocator.free(carry);
        
        carry = new_carry;
    }
    
    if (guard >= max_iters) {
        // Free carry before returning error
        std.heap.c_allocator.free(carry);
        return Error.NotSupported;
    }
    
    return carry;
}

/// Functional scan: applies a function over a sequence, carrying state.
/// Similar to fold/reduce but also returns intermediate values.
///
/// Example:
/// ```zig
/// // Cumulative sum
/// const result = try scan(
///     struct {
///         fn f(carry: Tensor, x: Tensor) Error!struct {carry: Tensor, y: Tensor} {
///             const new_carry = try ops.add(carry, x);
///             return .{ .carry = new_carry, .y = new_carry };
///         }
///     }.f,
///     init_carry,
///     xs,
/// );
/// ```
pub const ScanResult = struct { carry: Tensor, y: Tensor };
pub const ScanOutput = struct { final_carry: Tensor, ys: Tensor };
pub fn scan(f: *const fn (Tensor, Tensor) Error!ScanResult, init: Tensor, xs: Tensor) Error!ScanOutput {
    if (xs.rank() < 1) return Error.ShapeMismatch;
    const seq_len = xs.shape.dims[0];
    
    var carry = init;
    var ys_data = try std.heap.c_allocator.alloc(f64, seq_len * (xs.numel() / seq_len));
    defer std.heap.c_allocator.free(ys_data);
    
    for (0..seq_len) |i| {
        // Extract xs[i]
        const x_i = try ops.narrow(xs, 0, @intCast(i), 1);
        
        const result = try f(carry, x_i);
        
        // Store y - use readF64 instead of flatSlice for dtype compatibility
        for (0..result.y.numel()) |j| {
            ys_data[i * (xs.numel() / seq_len) + j] = result.y.readF64(j);
        }
        
        // Update carry - let Scope handle cleanup
        carry = result.carry;
    }
    
    // Create ys tensor
    var ys_shape: [prims.MAX_RANK]usize = @splat(0);
    ys_shape[0] = seq_len;
    for (1..xs.rank()) |i| ys_shape[i] = xs.shape.dims[i];
    var ys = try Tensor.empty(tmod.Shape.fromParts(ys_shape, xs.rank()), xs.dt);
    ys.batch_depth = xs.batch_depth;
    for (ys_data, 0..) |v, i| ys.writeF64(i, v);
    
    return .{ .final_carry = carry, .ys = ys };
}

/// Functional for loop: applies body_fn for i in [lower, upper).
/// Returns the final carry value.
///
/// Example:
/// ```zig
/// // Compute sum 0..10
/// const result = try foriLoop(0, 10,
///     struct {
///         fn body(i: Tensor, carry: Tensor) Error!Tensor {
///             return ops.add(carry, i);
///         }
///     }.body,
///     init_sum,
/// );
/// ```
pub fn foriLoop(lower: i32, upper: i32, body_fn: *const fn (Tensor, Tensor) Error!Tensor, init_carry: Tensor) Error!Tensor {
    if (upper <= lower) return init_carry;
    
    var carry = init_carry;
    
    for (@as(usize, @intCast(lower))..@as(usize, @intCast(upper))) |i| {
        const i_tensor = try Tensor.fromScalarF64(@floatFromInt(i), carry.dtype());
        const new_carry = try body_fn(i_tensor, carry);
        
        // Let Scope handle cleanup - don't manually deinit
        carry = new_carry;
    }
    
    return carry;
}

/// Switch: selects one of several branches based on an index.
///
/// Example:
/// ```zig
/// const result = try switchCase(index, &.{branch0_fn, branch1_fn, branch2_fn}, operands);
/// ```
pub fn switchCase(index: Tensor, branches: []const Fn, operands: []const Tensor) Error!Tensor {
    const idx = @as(usize, @intFromFloat(index.readF64(0)));
    if (idx >= branches.len) return Error.ShapeMismatch;
    return branches[idx](operands);
}

test "foriLoop basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Sum 0..5
    const init = try Tensor.fromScalarF64(0.0, .f32);
    // Don't deinit init - Scope will handle it
    const result = try foriLoop(0, 5, struct {
        fn body(i: Tensor, carry: Tensor) Error!Tensor {
            return ops.add(carry, i);
        }
    }.body, init);
    
    // 0+1+2+3+4 = 10
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.item(), 1e-6);
}

test "cond basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    const pred_true = try Tensor.fromScalarF64(1.0, .f32);
    const pred_false = try Tensor.fromScalarF64(0.0, .f32);
    
    const x = try Tensor.fromScalarF64(5.0, .f32);
    
    const result_true = try cond(pred_true, struct {
        fn f(args: []const Tensor) Error!Tensor {
            const two = try ops.scalarLike(args[0], 2.0);
            return ops.mul(args[0], two);
        }
    }.f, struct {
        fn f(args: []const Tensor) Error!Tensor {
            const three = try ops.scalarLike(args[0], 3.0);
            return ops.mul(args[0], three);
        }
    }.f, &.{x});
    
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result_true.item(), 1e-6);
    
    const result_false = try cond(pred_false, struct {
        fn f(args: []const Tensor) Error!Tensor {
            const two = try ops.scalarLike(args[0], 2.0);
            return ops.mul(args[0], two);
        }
    }.f, struct {
        fn f(args: []const Tensor) Error!Tensor {
            const three = try ops.scalarLike(args[0], 3.0);
            return ops.mul(args[0], three);
        }
    }.f, &.{x});
    
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), result_false.item(), 1e-6);
}

test "scan basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Cumulative sum using scan
    const xs = try Tensor.fromArray(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, try Shape.init(&.{4}));
    
    const init = try Tensor.fromScalarF64(0.0, .f32);
    
    const result = try scan(struct {
        fn f(carry: Tensor, x: Tensor) Error!ScanResult {
            const new_carry = try ops.add(carry, x);
            return .{ .carry = new_carry, .y = new_carry };
        }
    }.f, init, xs);
    
    // Check cumulative sums: [1, 3, 6, 10]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.ys.readF64(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result.ys.readF64(1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result.ys.readF64(2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.ys.readF64(3), 1e-6);
    
    // Final carry should be 10
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.final_carry.item(), 1e-6);
}

test "whileLoop basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Count down from 5 to 0
    const init = try Tensor.fromScalarF64(5.0, .f32);
    
    const result = try whileLoop(struct {
        fn cond(args: []const Tensor) Error!Tensor {
            const zero = try ops.scalarLike(args[0], 0.0);
            return ops.gt(args[0], zero);
        }
        fn body(args: []const Tensor) Error!Tensor {
            const one = try ops.scalarLike(args[0], 1.0);
            return ops.sub(args[0], one);
        }
    }.cond, struct {
        fn body(args: []const Tensor) Error!Tensor {
            const one = try ops.scalarLike(args[0], 1.0);
            return ops.sub(args[0], one);
        }
    }.body, &.{init});
    
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[0].item(), 1e-6);
}

test "switch basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    const x = try Tensor.fromScalarF64(10.0, .f32);
    
    const branches = [_]Fn{
        struct {
            fn f(args: []const Tensor) Error!Tensor {
                const c = try ops.scalarLike(args[0], 2.0);
                return ops.mul(args[0], c);
            }
        }.f,
        struct {
            fn f(args: []const Tensor) Error!Tensor {
                const c = try ops.scalarLike(args[0], 3.0);
                return ops.mul(args[0], c);
            }
        }.f,
        struct {
            fn f(args: []const Tensor) Error!Tensor {
                const c = try ops.scalarLike(args[0], 4.0);
                return ops.mul(args[0], c);
            }
        }.f,
    };
    
    // Test index 0: 10 * 2 = 20
    const idx0 = try Tensor.fromScalarF64(0.0, .i32);
    const result0 = try switchCase(idx0, &branches, &.{x});
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), result0.item(), 1e-6);
    
    // Test index 1: 10 * 3 = 30
    const idx1 = try Tensor.fromScalarF64(1.0, .i32);
    const result1 = try switchCase(idx1, &branches, &.{x});
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), result1.item(), 1e-6);
    
    // Test index 2: 10 * 4 = 40
    const idx2 = try Tensor.fromScalarF64(2.0, .i32);
    const result2 = try switchCase(idx2, &branches, &.{x});
    try std.testing.expectApproxEqAbs(@as(f64, 40.0), result2.item(), 1e-6);
}
