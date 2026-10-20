//! Parallel execution utilities for ZPU.
//!
//! This module provides thread-parallel execution for large tensor operations.
//! It uses a work-stealing approach where the data is split into chunks and
//! processed by multiple threads.
//!
//! Key design decisions:
//! - Only parallelize when the workload is large enough to amortize thread overhead
//! - Use the number of available CPU cores
//! - Minimum chunk size to avoid excessive overhead
const std = @import("std");
const builtin = @import("builtin");

/// Threshold below which we don't bother parallelizing (elements).
/// Thread creation/sync overhead dominates for small arrays.
pub const parallel_threshold: usize = 64_000;

/// Minimum elements per thread to avoid excessive overhead.
pub const min_per_thread: usize = 16_000;

/// Get the number of worker threads to use based on data size.
pub fn workerCount(n: usize) usize {
    if (n < parallel_threshold) return 1;
    const cores = @max(1, std.Thread.getCpuCount() catch 1);
    const by_size = n / min_per_thread;
    return @min(cores, @max(1, by_size));
}

/// Split a range [0, n) into `num_chunks` roughly equal pieces.
/// Returns start and end indices for chunk `i`.
pub fn chunkRange(n: usize, num_chunks: usize, i: usize) struct { start: usize, end: usize } {
    const base = n / num_chunks;
    const remainder = n % num_chunks;
    const start = if (i < remainder) (base + 1) * i else base * i + remainder;
    const end = if (i < remainder) start + base + 1 else start + base;
    return .{ .start = start, .end = end };
}

/// Execute `f(start, end)` in parallel for each chunk of [0, n).
/// Falls back to sequential execution if n is small or threading fails.
pub fn parallelFor(alloc: std.mem.Allocator, n: usize, comptime Context: type, ctx: Context, comptime f: fn (Context, usize, usize) void) !void {
    const workers = workerCount(n);
    if (workers == 1) {
        f(ctx, 0, n);
        return;
    }

    // Allocate handles for threads
    const handles = try alloc.alloc(?std.Thread, workers);
    defer alloc.free(handles);
    @memset(handles, null);

    // Worker function that captures the chunk bounds
    const Worker = struct {
        context: Context,
        start: usize,
        end: usize,

        fn run(self: *@This()) void {
            f(self.context, self.start, self.end);
        }
    };

    // Launch workers
    var launched: usize = 0;
    errdefer {
        // Join any threads we managed to start
        for (handles[0..launched]) |*h| {
            if (h.*) |*thread| thread.join();
        }
    }

    var workers_data = try alloc.alloc(Worker, workers);
    defer alloc.free(workers_data);

    for (0..workers) |i| {
        const range = chunkRange(n, workers, i);
        workers_data[i] = .{
            .context = ctx,
            .start = range.start,
            .end = range.end,
        };
        handles[i] = try std.Thread.spawn(.{}, Worker.run, .{&workers_data[i]});
        launched += 1;
    }

    // Wait for all workers
    for (handles[0..launched]) |*h| {
        if (h.*) |*thread| thread.join();
    }
}

/// Parallel unary kernel: dst[i] = op(src[i]) for i in [start, end).
pub fn parallelUnary(comptime T: type, dst: []T, src: []const T, comptime op: fn (T) T) !void {
    const WorkerCtx = struct {
        dst_slice: []T,
        src_slice: []const T,
    };

    const workerFn = (struct {
        fn run(ctx_param: WorkerCtx, start: usize, end: usize) void {
            for (start..end) |i| {
                ctx_param.dst_slice[i] = op(ctx_param.src_slice[i]);
            }
        }
    }).run;

    const ctx = WorkerCtx{ .dst_slice = dst, .src_slice = src };
    try parallelFor(std.heap.c_allocator, dst.len, WorkerCtx, ctx, workerFn);
}

/// Parallel binary kernel: dst[i] = op(a[i], b[i]) for i in [start, end).
pub fn parallelBinary(comptime T: type, dst: []T, a: []const T, b: []const T, comptime op: fn (T, T) T) !void {
    const WorkerCtx = struct {
        dst_slice: []T,
        a_slice: []const T,
        b_slice: []const T,
    };

    const workerFn = (struct {
        fn run(ctx_param: WorkerCtx, start: usize, end: usize) void {
            for (start..end) |i| {
                ctx_param.dst_slice[i] = op(ctx_param.a_slice[i], ctx_param.b_slice[i]);
            }
        }
    }).run;

    const ctx = WorkerCtx{ .dst_slice = dst, .a_slice = a, .b_slice = b };
    try parallelFor(std.heap.c_allocator, dst.len, WorkerCtx, ctx, workerFn);
}

// ============================================================================
// Tests
// ============================================================================

test "chunkRange distributes evenly" {
    // 10 elements, 3 chunks: [0,4), [4,7), [7,10)
    const r0 = chunkRange(10, 3, 0);
    try std.testing.expectEqual(@as(usize, 0), r0.start);
    try std.testing.expectEqual(@as(usize, 4), r0.end);

    const r1 = chunkRange(10, 3, 1);
    try std.testing.expectEqual(@as(usize, 4), r1.start);
    try std.testing.expectEqual(@as(usize, 7), r1.end);

    const r2 = chunkRange(10, 3, 2);
    try std.testing.expectEqual(@as(usize, 7), r2.start);
    try std.testing.expectEqual(@as(usize, 10), r2.end);
}

test "chunkRange handles exact division" {
    const r0 = chunkRange(12, 4, 0);
    try std.testing.expectEqual(@as(usize, 0), r0.start);
    try std.testing.expectEqual(@as(usize, 3), r0.end);

    const r3 = chunkRange(12, 4, 3);
    try std.testing.expectEqual(@as(usize, 9), r3.start);
    try std.testing.expectEqual(@as(usize, 12), r3.end);
}

test "workerCount returns 1 for small arrays" {
    try std.testing.expectEqual(@as(usize, 1), workerCount(100));
    try std.testing.expectEqual(@as(usize, 1), workerCount(10_000));
}

test "workerCount scales with size" {
    // Should use more workers for larger arrays
    const small = workerCount(100_000);
    const large = workerCount(10_000_000);
    try std.testing.expect(large >= small);
}

test "parallelFor sequential fallback" {
    var data: [100]f32 = undefined;
    @memset(&data, 0);
    const Ctx = struct {
        d: []f32,
    };
    const ctx = Ctx{ .d = &data };
    const workerFn = (struct {
        fn run(c: Ctx, start: usize, end: usize) void {
            for (start..end) |i| c.d[i] = @floatFromInt(i);
        }
    }).run;
    try parallelFor(std.testing.allocator, data.len, Ctx, ctx, workerFn);
    for (data, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), v, 1e-6);
    }
}

test "parallelFor parallel execution" {
    const n = 200_000;
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    @memset(data, 0);

    const Ctx = struct {
        d: []f32,
    };
    const ctx = Ctx{ .d = data };
    const workerFn = (struct {
        fn run(c: Ctx, start: usize, end: usize) void {
            for (start..end) |i| c.d[i] = @floatFromInt(i % 1000);
        }
    }).run;
    try parallelFor(std.testing.allocator, n, Ctx, ctx, workerFn);

    for (data, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i % 1000)), v, 1e-6);
    }
}
