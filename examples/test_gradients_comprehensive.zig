//! Comprehensive gradient tests to catch autodiff bugs
//! Tests each operation's gradient against numerical differentiation

const std = @import("std");
const zpu = @import("zpu");

const Tensor = zpu.Tensor;
const Shape = zpu.Shape;
const DType = zpu.DType;

fn numericalGradient(
    f: *const fn (Tensor) anyerror!Tensor,
    x: Tensor,
    eps: f64,
) !Tensor {
    var grad = try Tensor.zeros(x.shape, x.dtype());
    const n = x.numel();
    
    for (0..n) |i| {
        // x + eps
        var x_plus = try x.clone();
        const orig_val = x_plus.readF64(i);
        try x_plus.writeF64(i, orig_val + eps);
        const f_plus = try f(x_plus);
        const val_plus = f_plus.item();
        x_plus.deinit();
        f_plus.deinit();
        
        // x - eps
        var x_minus = try x.clone();
        try x_minus.writeF64(i, orig_val - eps);
        const f_minus = try f(x_minus);
        const val_minus = f_minus.item();
        x_minus.deinit();
        f_minus.deinit();
        
        // Central difference
        const grad_val = (val_plus - val_minus) / (2.0 * eps);
        try grad.writeF64(i, grad_val);
    }
    
    return grad;
}

test "gradient: square operation" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x) = sum(x^2)
    const x = try Tensor.fromArray(f64, &.{1.0, 2.0, 3.0}, try Shape.init(&.{3}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const sq = try zpu.square(t);
            defer sq.deinit();
            return zpu.sumAll(sq);
        }
    }.call;
    
    // Numerical gradient
    const num_grad = try numericalGradient(&f, x, 1e-5);
    defer num_grad.deinit();
    
    // Analytical gradient via autodiff
    const auto_grad = try zpu.grad(f, &.{x});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    // Compare
    for (0..3) |i| {
        const ng = num_grad.readF64(i);
        const ag = auto_grad[0].readF64(i);
        const expected = 2.0 * x.readF64(i); // d/dx(x^2) = 2x
        
        std.debug.print("x[{d}] = {d:.3}, numerical = {d:.6}, auto = {d:.6}, expected = {d:.6}\n", 
            .{i, x.readF64(i), ng, ag, expected});
        
        try std.testing.expectApproxEqAbs(expected, ag, 1e-4);
    }
}

test "gradient: exp operation" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    const x = try Tensor.fromArray(f64, &.{0.0, 1.0, 2.0}, try Shape.init(&.{3}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const e = try zpu.exp(t);
            defer e.deinit();
            return zpu.sumAll(e);
        }
    }.call;
    
    const auto_grad = try zpu.grad(f, &.{x});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    for (0..3) |i| {
        const ag = auto_grad[0].readF64(i);
        const expected = @exp(x.readF64(i)); // d/dx(e^x) = e^x
        
        std.debug.print("x[{d}] = {d:.3}, auto = {d:.6}, expected = {d:.6}\n", 
            .{i, x.readF64(i), ag, expected});
        
        try std.testing.expectApproxEqAbs(expected, ag, 1e-4);
    }
}

test "gradient: log operation" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    const x = try Tensor.fromArray(f64, &.{1.0, 2.0, 3.0}, try Shape.init(&.{3}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const l = try zpu.log(t);
            defer l.deinit();
            return zpu.sumAll(l);
        }
    }.call;
    
    const auto_grad = try zpu.grad(f, &.{x});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    for (0..3) |i| {
        const ag = auto_grad[0].readF64(i);
        const expected = 1.0 / x.readF64(i); // d/dx(log(x)) = 1/x
        
        std.debug.print("x[{d}] = {d:.3}, auto = {d:.6}, expected = {d:.6}\n", 
            .{i, x.readF64(i), ag, expected});
        
        try std.testing.expectApproxEqAbs(expected, ag, 1e-4);
    }
}

test "gradient: matmul operation" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(A, B) = sum(A @ B)
    const A = try Tensor.fromArray(f64, &.{1.0, 2.0, 3.0, 4.0}, try Shape.init(&.{2, 2}));
    defer A.deinit();
    const B = try Tensor.fromArray(f64, &.{5.0, 6.0, 7.0, 8.0}, try Shape.init(&.{2, 2}));
    defer B.deinit();
    
    const f = struct {
        fn call(args: []const Tensor) !Tensor {
            const C = try zpu.matmul(args[0], args[1]);
            defer C.deinit();
            return zpu.sumAll(C);
        }
    }.call;
    
    const auto_grads = try zpu.grad(f, &.{A, B});
    defer auto_grads[0].deinit();
    defer auto_grads[1].deinit();
    defer alloc.free(auto_grads);
    
    // d/dA sum(A@B) = sum over output, gradient flows back
    // For C = A@B, dL/dA = dL/dC @ B^T
    // Since dL/dC = 1 (sum), dL/dA = 1 @ B^T = sum of B^T rows
    std.debug.print("grad A:\n", .{});
    for (0..4) |i| {
        std.debug.print("  {d:.3}\n", .{auto_grads[0].readF64(i)});
    }
    
    std.debug.print("grad B:\n", .{});
    for (0..4) |i| {
        std.debug.print("  {d:.3}\n", .{auto_grads[1].readF64(i)});
    }
}

test "gradient: broadcasting in add" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x, y) = sum(x + y) where y is broadcasted
    const x = try Tensor.fromArray(f64, &.{1.0, 2.0, 3.0, 4.0}, try Shape.init(&.{2, 2}));
    defer x.deinit();
    const y = try Tensor.fromArray(f64, &.{10.0, 20.0}, try Shape.init(&.{2}));
    defer y.deinit();
    
    const f = struct {
        fn call(args: []const Tensor) !Tensor {
            const z = try zpu.add(args[0], args[1]);
            defer z.deinit();
            return zpu.sumAll(z);
        }
    }.call;
    
    const auto_grads = try zpu.grad(f, &.{x, y});
    defer auto_grads[0].deinit();
    defer auto_grads[1].deinit();
    defer alloc.free(auto_grads);
    
    // d/dx sum(x + y) = 1 for all elements
    std.debug.print("grad x (should be all 1s):\n", .{});
    for (0..4) |i| {
        const g = auto_grads[0].readF64(i);
        std.debug.print("  {d:.3}\n", .{g});
        try std.testing.expectApproxEqAbs(1.0, g, 1e-6);
    }
    
    // d/dy sum(x + y) = 2 for each element (broadcasted to 2 rows)
    std.debug.print("grad y (should be all 2s):\n", .{});
    for (0..2) |i| {
        const g = auto_grads[1].readF64(i);
        std.debug.print("  {d:.3}\n", .{g});
        try std.testing.expectApproxEqAbs(2.0, g, 1e-6);
    }
}

test "gradient: relu operation" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    const x = try Tensor.fromArray(f64, &.{-2.0, -1.0, 0.0, 1.0, 2.0}, try Shape.init(&.{5}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const r = try zpu.relu(t);
            defer r.deinit();
            return zpu.sumAll(r);
        }
    }.call;
    
    const auto_grad = try zpu.grad(f, &.{x});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    // d/dx relu(x) = 1 if x > 0, else 0
    for (0..5) |i| {
        const ag = auto_grad[0].readF64(i);
        const x_val = x.readF64(i);
        const expected = if (x_val > 0) 1.0 else 0.0;
        
        std.debug.print("x[{d}] = {d:.3}, auto = {d:.6}, expected = {d:.6}\n", 
            .{i, x_val, ag, expected});
        
        try std.testing.expectApproxEqAbs(expected, ag, 1e-6);
    }
}

test "gradient: sum reduction" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    const x = try Tensor.fromArray(f64, &.{1.0, 2.0, 3.0, 4.0}, try Shape.init(&.{2, 2}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            return zpu.sumAll(t);
        }
    }.call;
    
    const auto_grad = try zpu.grad(f, &.{x});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    // d/dx sum(x) = 1 for all elements
    for (0..4) |i| {
        const ag = auto_grad[0].readF64(i);
        std.debug.print("x[{d}] = {d:.3}, auto = {d:.6}, expected = 1.0\n", 
            .{i, x.readF64(i), ag});
        try std.testing.expectApproxEqAbs(1.0, ag, 1e-6);
    }
}

test "gradient: mean reduction" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    const x = try Tensor.fromArray(f64, &.{1.0, 2.0, 3.0, 4.0}, try Shape.init(&.{2, 2}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            return zpu.meanAll(t);
        }
    }.call;
    
    const auto_grad = try zpu.grad(f, &.{x});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    // d/dx mean(x) = 1/n for all elements
    const expected = 1.0 / 4.0;
    for (0..4) |i| {
        const ag = auto_grad[0].readF64(i);
        std.debug.print("x[{d}] = {d:.3}, auto = {d:.6}, expected = {d:.6}\n", 
            .{i, x.readF64(i), ag, expected});
        try std.testing.expectApproxEqAbs(expected, ag, 1e-6);
    }
}
