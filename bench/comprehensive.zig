//! Comprehensive benchmark suite for ZPU optimizations
const std = @import("std");
const zpu = @import("zpu");

const Tensor = zpu.Tensor;
const Shape = zpu.Shape;

fn printResult(name: []const u8, elapsed_ns: i96, iters: usize) void {
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iters));
    std.debug.print("{s}: {d:.3} ms total, {d:.4} ms/iter ({d} iters)\n", .{
        name, elapsed_ms, per_iter_ms, iters,
    });
}

pub fn main(init: std.process.Init) !void {
    const alloc = std.heap.c_allocator;
    const clock = std.Io.Clock.awake;
    
    std.debug.print("\n=== ZPU Performance Benchmark ===\n\n", .{});
    
    // Element-wise operations
    std.debug.print("--- Element-wise Operations ---\n", .{});
    
    const sizes = [_]usize{ 10_000, 100_000, 1_000_000, 10_000_000 };
    
    for (sizes) |n| {
        std.debug.print("\n[n = {d}]\n", .{n});
        
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        
        const shape = try Shape.init(&.{n});
        const a = try Tensor.full(shape, .f32, 0.5);
        const b = try Tensor.full(shape, .f32, 0.3);
        
        // Unary operations
        const unary_iters = @max(1, 1_000_000 / n);
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < unary_iters) : (i += 1) {
                const r = try zpu.exp(a);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  exp(f32)", elapsed.nanoseconds, unary_iters);
        }
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < unary_iters) : (i += 1) {
                const r = try zpu.relu(a);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  relu(f32)", elapsed.nanoseconds, unary_iters);
        }
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < unary_iters) : (i += 1) {
                const r = try zpu.sigmoid(a);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  sigmoid(f32)", elapsed.nanoseconds, unary_iters);
        }
        
        // Binary operations
        const binary_iters = @max(1, 500_000 / n);
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < binary_iters) : (i += 1) {
                const r = try zpu.add(a, b);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  add(f32)", elapsed.nanoseconds, binary_iters);
        }
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < binary_iters) : (i += 1) {
                const r = try zpu.mul(a, b);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  mul(f32)", elapsed.nanoseconds, binary_iters);
        }
        
        // Chained operations
        const chain_iters = @max(1, 200_000 / n);
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < chain_iters) : (i += 1) {
                const sum = try zpu.add(a, b);
                const r = try zpu.relu(sum);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  relu(add(a,b))", elapsed.nanoseconds, chain_iters);
        }
        
        // Fused operations (for comparison)
        if (n >= 100_000) {
            std.debug.print("\n  [Fused Operations]\n", .{});
            
            {
                const start = clock.now(init.io);
                var i: usize = 0;
                while (i < chain_iters) : (i += 1) {
                    const r = try zpu.fusedAddRelu(a, b);
                    _ = r;
                }
                const elapsed = start.untilNow(init.io, clock);
                printResult("  fusedAddRelu(a,b)", elapsed.nanoseconds, chain_iters);
            }
            
            {
                const start = clock.now(init.io);
                var i: usize = 0;
                while (i < chain_iters) : (i += 1) {
                    const r = try zpu.fusedAddSigmoid(a, b);
                    _ = r;
                }
                const elapsed = start.untilNow(init.io, clock);
                printResult("  fusedAddSigmoid(a,b)", elapsed.nanoseconds, chain_iters);
            }
            
            {
                const start = clock.now(init.io);
                var i: usize = 0;
                while (i < chain_iters) : (i += 1) {
                    const r = try zpu.fusedGateSigmoid(a, b);
                    _ = r;
                }
                const elapsed = start.untilNow(init.io, clock);
                printResult("  fusedGateSigmoid(a,b)", elapsed.nanoseconds, chain_iters);
            }
            
            {
                const start = clock.now(init.io);
                var i: usize = 0;
                while (i < chain_iters) : (i += 1) {
                    const r = try zpu.fusedMulRelu(a, b);
                    _ = r;
                }
                const elapsed = start.untilNow(init.io, clock);
                printResult("  fusedMulRelu(a,b)", elapsed.nanoseconds, chain_iters);
            }
        }
    }
    
    // Matrix operations
    std.debug.print("\n\n--- Matrix Operations ---\n", .{});
    
    const mat_sizes = [_]usize{ 64, 128, 256, 512 };
    
    for (mat_sizes) |n| {
        std.debug.print("\n[{d}x{d}]\n", .{ n, n });
        
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        
        const shape = try Shape.init(&.{ n, n });
        const a = try Tensor.full(shape, .f32, 0.1);
        const b = try Tensor.full(shape, .f32, 0.2);
        
        const matmul_iters = @max(1, 1000 / (n * n / 1000));
        
        {
            const start = clock.now(init.io);
            var i: usize = 0;
            while (i < matmul_iters) : (i += 1) {
                const r = try zpu.matmul(a, b);
                _ = r;
            }
            const elapsed = start.untilNow(init.io, clock);
            printResult("  matmul(f32)", elapsed.nanoseconds, matmul_iters);
        }
    }
    
    std.debug.print("\n=== Benchmark Complete ===\n\n", .{});
}
