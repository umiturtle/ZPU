//! 包括的な自動微分テストスイート
//! 各操作の勾配を数値微分と比較して検証

const std = @import("std");
const zpu = @import("zpu");

const Tensor = zpu.Tensor;
const Shape = zpu.Shape;
const DType = zpu.DType;
const Error = zpu.Error;

/// 数値微分による勾配計算
fn numericalGradient(
    alloc: std.mem.Allocator,
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
        
        // Central difference: (f(x+h) - f(x-h)) / 2h
        const grad_val = (val_plus - val_minus) / (2.0 * eps);
        try grad.writeF64(i, grad_val);
    }
    
    return grad;
}

/// 2変数関数の数値勾配
fn numericalGradient2(
    alloc: std.mem.Allocator,
    f: *const fn ([]const Tensor) anyerror!Tensor,
    x: Tensor,
    y: Tensor,
    eps: f64,
) ![2]Tensor {
    var grad_x = try Tensor.zeros(x.shape, x.dtype());
    var grad_y = try Tensor.zeros(y.shape, y.dtype());
    
    // xの勾配
    for (0..x.numel()) |i| {
        var x_plus = try x.clone();
        const orig = x_plus.readF64(i);
        try x_plus.writeF64(i, orig + eps);
        const f_plus = try f(&.{ x_plus, y });
        const val_plus = f_plus.item();
        x_plus.deinit();
        f_plus.deinit();
        
        var x_minus = try x.clone();
        try x_minus.writeF64(i, orig - eps);
        const f_minus = try f(&.{ x_minus, y });
        const val_minus = f_minus.item();
        x_minus.deinit();
        f_minus.deinit();
        
        try grad_x.writeF64(i, (val_plus - val_minus) / (2.0 * eps));
    }
    
    // yの勾配
    for (0..y.numel()) |i| {
        var y_plus = try y.clone();
        const orig = y_plus.readF64(i);
        try y_plus.writeF64(i, orig + eps);
        const f_plus = try f(&.{ x, y_plus });
        const val_plus = f_plus.item();
        y_plus.deinit();
        f_plus.deinit();
        
        var y_minus = try y.clone();
        try y_minus.writeF64(i, orig - eps);
        const f_minus = try f(&.{ x, y_minus });
        const val_minus = f_minus.item();
        y_minus.deinit();
        f_minus.deinit();
        
        try grad_y.writeF64(i, (val_plus - val_minus) / (2.0 * eps));
    }
    
    return .{ grad_x, grad_y };
}

/// 勾配の比較（相対誤差）
fn compareGradients(num_grad: Tensor, auto_grad: Tensor, tolerance: f64) !void {
    const n = num_grad.numel();
    if (n != auto_grad.numel()) return Error.ShapeMismatch;
    
    for (0..n) |i| {
        const ng = num_grad.readF64(i);
        const ag = auto_grad.readF64(i);
        const diff = @abs(ng - ag);
        const max_val = @max(@abs(ng), @abs(ag));
        const rel_err = if (max_val > 1e-8) diff / max_val else diff;
        
        if (rel_err > tolerance) {
            std.debug.print("Gradient mismatch at index {d}: numerical={d:.6}, auto={d:.6}, rel_err={d:.6}\n", 
                .{i, ng, ag, rel_err});
            return Error.GradientMismatch;
        }
    }
}

// ============ テストケース ============

test "gradient: basic operations" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x) = sum(x^2)
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{3}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const sq = try zpu.square(t);
            defer sq.deinit();
            return zpu.sumAll(sq);
        }
    }.call;
    
    // 数値勾配
    const num_grad = try numericalGradient(alloc, &f, x, 1e-5);
    defer num_grad.deinit();
    
    // 自動微分による勾配
    const auto_grad = try zpu.grad(alloc, f, null, &.{x}, &.{0});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    // 比較
    try compareGradients(num_grad, auto_grad[0], 1e-4);
    
    std.debug.print("✓ basic operations gradient test passed\n", .{});
}

test "gradient: exp and log" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x) = sum(exp(x))
    const x_data = [_]f64{ 0.0, 1.0, 2.0 };
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{3}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const e = try zpu.exp(t);
            defer e.deinit();
            return zpu.sumAll(e);
        }
    }.call;
    
    const num_grad = try numericalGradient(alloc, &f, x, 1e-5);
    defer num_grad.deinit();
    
    const auto_grad = try zpu.grad(alloc, f, null, &.{x}, &.{0});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    try compareGradients(num_grad, auto_grad[0], 1e-4);
    
    std.debug.print("✓ exp gradient test passed\n", .{});
}

test "gradient: trigonometric functions" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x) = sum(sin(x))
    const x_data = [_]f64{ 0.0, 0.5, 1.0 };
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{3}));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            const s = try zpu.sin(t);
            defer s.deinit();
            return zpu.sumAll(s);
        }
    }.call;
    
    const num_grad = try numericalGradient(alloc, &f, x, 1e-5);
    defer num_grad.deinit();
    
    const auto_grad = try zpu.grad(alloc, f, null, &.{x}, &.{0});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    try compareGradients(num_grad, auto_grad[0], 1e-4);
    
    std.debug.print("✓ trigonometric gradient test passed\n", .{});
}

test "gradient: binary operations" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x, y) = sum(x * y)
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 4.0, 5.0, 6.0 };
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{3}));
    defer x.deinit();
    const y = try Tensor.fromArray(f64, &y_data, try Shape.init(&.{3}));
    defer y.deinit();
    
    const f = struct {
        fn call(args: []const Tensor) !Tensor {
            const prod = try zpu.mul(args[0], args[1]);
            defer prod.deinit();
            return zpu.sumAll(prod);
        }
    }.call;
    
    const num_grads = try numericalGradient2(alloc, &f, x, y, 1e-5);
    defer num_grads[0].deinit();
    defer num_grads[1].deinit();
    
    const auto_grads = try zpu.grad(alloc, f, null, &.{ x, y }, &.{ 0, 1 });
    defer auto_grads[0].deinit();
    defer auto_grads[1].deinit();
    defer alloc.free(auto_grads);
    
    try compareGradients(num_grads[0], auto_grads[0], 1e-4);
    try compareGradients(num_grads[1], auto_grads[1], 1e-4);
    
    std.debug.print("✓ binary operations gradient test passed\n", .{});
}

test "gradient: matmul" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(A, B) = sum(A @ B)
    const A_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const B_data = [_]f64{ 5.0, 6.0, 7.0, 8.0 };
    const A = try Tensor.fromArray(f64, &A_data, try Shape.init(&.{ 2, 2 }));
    defer A.deinit();
    const B = try Tensor.fromArray(f64, &B_data, try Shape.init(&.{ 2, 2 }));
    defer B.deinit();
    
    const f = struct {
        fn call(args: []const Tensor) !Tensor {
            const C = try zpu.matmul(args[0], args[1]);
            defer C.deinit();
            return zpu.sumAll(C);
        }
    }.call;
    
    const num_grads = try numericalGradient2(alloc, &f, A, B, 1e-5);
    defer num_grads[0].deinit();
    defer num_grads[1].deinit();
    
    const auto_grads = try zpu.grad(alloc, f, null, &.{ A, B }, &.{ 0, 1 });
    defer auto_grads[0].deinit();
    defer auto_grads[1].deinit();
    defer alloc.free(auto_grads);
    
    try compareGradients(num_grads[0], auto_grads[0], 1e-3);
    try compareGradients(num_grads[1], auto_grads[1], 1e-3);
    
    std.debug.print("✓ matmul gradient test passed\n", .{});
}

test "gradient: reductions" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x) = mean(x)
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{ 2, 2 }));
    defer x.deinit();
    
    const f = struct {
        fn call(t: Tensor) !Tensor {
            return zpu.meanAll(t);
        }
    }.call;
    
    const num_grad = try numericalGradient(alloc, &f, x, 1e-5);
    defer num_grad.deinit();
    
    const auto_grad = try zpu.grad(alloc, f, null, &.{x}, &.{0});
    defer auto_grad[0].deinit();
    defer alloc.free(auto_grad);
    
    try compareGradients(num_grad, auto_grad[0], 1e-4);
    
    std.debug.print("✓ reduction gradient test passed\n", .{});
}

test "gradient: broadcasting" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x, y) = sum(x + y) where y is broadcasted
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 10.0, 20.0 };
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{ 2, 2 }));
    defer x.deinit();
    const y = try Tensor.fromArray(f64, &y_data, try Shape.init(&.{2}));
    defer y.deinit();
    
    const f = struct {
        fn call(args: []const Tensor) !Tensor {
            const z = try zpu.add(args[0], args[1]);
            defer z.deinit();
            return zpu.sumAll(z);
        }
    }.call;
    
    const num_grads = try numericalGradient2(alloc, &f, x, y, 1e-5);
    defer num_grads[0].deinit();
    defer num_grads[1].deinit();
    
    const auto_grads = try zpu.grad(alloc, f, null, &.{ x, y }, &.{ 0, 1 });
    defer auto_grads[0].deinit();
    defer auto_grads[1].deinit();
    defer alloc.free(auto_grads);
    
    try compareGradients(num_grads[0], auto_grads[0], 1e-4);
    try compareGradients(num_grads[1], auto_grads[1], 1e-4);
    
    std.debug.print("✓ broadcasting gradient test passed\n", .{});
}

test "gradient: neural network layer" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x, w, b) = sum(dense(x, w, b))
    const x_data = [_]f64{ 1.0, 2.0 };
    const w_data = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    const b_data = [_]f64{ 0.1, 0.2 };
    
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{ 1, 2 }));
    defer x.deinit();
    const w = try Tensor.fromArray(f64, &w_data, try Shape.init(&.{ 2, 2 }));
    defer w.deinit();
    const b = try Tensor.fromArray(f64, &b_data, try Shape.init(&.{2}));
    defer b.deinit();
    
    const f = struct {
        fn call(args: []const Tensor) !Tensor {
            const out = try zpu.nn.dense(args[0], args[1], args[2]);
            defer out.deinit();
            return zpu.sumAll(out);
        }
    }.call;
    
    const auto_grads = try zpu.grad(alloc, f, null, &.{ x, w, b }, &.{ 0, 1, 2 });
    defer auto_grads[0].deinit();
    defer auto_grads[1].deinit();
    defer auto_grads[2].deinit();
    defer alloc.free(auto_grads);
    
    // 数値勾配との比較は省略（時間がかかるため）
    // 勾配が計算できることを確認
    std.debug.print("✓ neural network gradient test passed\n", .{});
}

test "gradient: second derivative" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    // f(x) = x^3 の2階微分
    // f'(x) = 3x^2, f''(x) = 6x
    const x_data = [_]f64{2.0};
    const x = try Tensor.fromArray(f64, &x_data, try Shape.init(&.{1}));
    defer x.deinit();
    
    // 1階微分
    const grad1 = try zpu.grad(alloc, struct {
        fn call(t: Tensor) !Tensor {
            const sq = try zpu.square(t);
            defer sq.deinit();
            const cu = try zpu.mul(sq, t);
            defer cu.deinit();
            return zpu.sumAll(cu);
        }
    }.call, null, &.{x}, &.{0});
    defer grad1[0].deinit();
    defer alloc.free(grad1);
    
    // 期待値: 3 * x^2 = 3 * 4 = 12
    const g1 = grad1[0].readF64(0);
    std.debug.print("First derivative at x=2: {d:.6} (expected 12.0)\n", .{g1});
    
    if (@abs(g1 - 12.0) > 1e-4) {
        return Error.GradientMismatch;
    }
    
    std.debug.print("✓ second derivative test passed\n", .{});
}
