//! Performance benchmarks for ZPU operations.
//! Measures wall-clock time for common operations at various sizes.
const std = @import("std");
const zpu = @import("../src/zpu.zig");
const Timer = std.time.Timer;

const Tensor = zpu.Tensor;
const Shape = zpu.Shape;
const DType = zpu.DType;

fn bench(alloc: std.mem.Allocator, name: []const u8, comptime Fn: type, f: Fn, iters: usize) !void {
    var timer = try Timer.start();
    const start = timer.read();
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        try f();
    }
    const elapsed_ns = timer.read() - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iters));
    std.debug.print("{s}: {d:.3} ms total, {d:.3} ms/iter ({d} iters)\n", .{ name, elapsed_ms, per_iter_ms, iters });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("=== ZPU Performance Benchmarks ===\n\n", .{});

    // --- Unary ops ---
    std.debug.print("--- Unary Operations ---\n", .{});

    const sizes = [_]usize{ 1000, 10000, 100000, 1000000 };
    for (sizes) |n| {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();

        const shape = try Shape.init(&.{n});
        const x = try Tensor.full(shape, .f32, 0.5);

        const iters = @max(1, 1000 / (n / 1000));

        try bench(alloc, try std.fmt.allocPrint(alloc, "exp(f32, n={d})", .{n}), struct {
            fn run() anyerror!void {
                const r = try zpu.exp(x);
                _ = r;
            }
        }.run, iters);

        try bench(alloc, try std.fmt.allocPrint(alloc, "relu(f32, n={d})", .{n}), struct {
            fn run() anyerror!void {
                const r = try zpu.relu(x);
                _ = r;
            }
        }.run, iters);

        try bench(alloc, try std.fmt.allocPrint(alloc, "sigmoid(f32, n={d})", .{n}), struct {
            fn run() anyerror!void {
                const r = try zpu.sigmoid(x);
                _ = r;
            }
        }.run, iters);
    }

    // --- Binary ops ---
    std.debug.print("\n--- Binary Operations ---\n", .{});

    for (sizes) |n| {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();

        const shape = try Shape.init(&.{n});
        const a = try Tensor.full(shape, .f32, 0.5);
        const b = try Tensor.full(shape, .f32, 0.3);

        const iters = @max(1, 1000 / (n / 1000));

        try bench(alloc, try std.fmt.allocPrint(alloc, "add(f32, n={d})", .{n}), struct {
            fn run() anyerror!void {
                const r = try zpu.add(a, b);
                _ = r;
            }
        }.run, iters);

        try bench(alloc, try std.fmt.allocPrint(alloc, "mul(f32, n={d})", .{n}), struct {
            fn run() anyerror!void {
                const r = try zpu.mul(a, b);
                _ = r;
            }
        }.run, iters);
    }

    // --- Matmul ---
    std.debug.print("\n--- Matrix Multiplication ---\n", .{});

    const mat_sizes = [_]usize{ 32, 64, 128, 256, 512 };
    for (mat_sizes) |n| {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();

        const shape = try Shape.init(&.{ n, n });
        const a = try Tensor.full(shape, .f32, 0.1);
        const b = try Tensor.full(shape, .f32, 0.2);

        const iters = @max(1, 100 / (n * n / 1000));

        try bench(alloc, try std.fmt.allocPrint(alloc, "matmul(f32, {d}x{d})", .{ n, n }), struct {
            fn run() anyerror!void {
                const r = try zpu.matmul(a, b);
                _ = r;
            }
        }.run, iters);
    }

    // --- Autodiff ---
    std.debug.print("\n--- Autodiff ---\n", .{});

    for (sizes) |n| {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();

        const shape = try Shape.init(&.{n});
        const x = try Tensor.full(shape, .f32, 0.5);

        const iters = @max(1, 100 / (n / 1000));

        try bench(alloc, try std.fmt.allocPrint(alloc, "grad(sum(exp(x))), n={d})", .{n}), struct {
            fn run() anyerror!void {
                var args = [_]Tensor{x};
                const r = try zpu.grad(alloc, struct {
                    fn f(_: ?*anyopaque, a: []const Tensor) anyerror!Tensor {
                        const e = try zpu.exp(a[0]);
                        return try zpu.sumAll(e);
                    }
                }.f, null, &args);
                _ = r;
            }
        }.run, iters);
    }

    std.debug.print("\n=== Done ===\n", .{});
}
