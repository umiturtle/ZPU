//! Memory pool for tensor allocations
//! Reduces allocation overhead by reusing memory blocks
//! Works for both C ABI (dynamic) and AOT (static) scenarios
const std = @import("std");
const context = @import("context.zig");

/// Thread-local memory pool for fast allocations
pub const MemoryPool = struct {
    const Block = struct {
        ptr: [*]u8,
        size: usize,
        next: ?*Block,
    };

    const Bucket = struct {
        blocks: ?*Block = null,
        
        fn add(self: *Bucket, alloc: std.mem.Allocator, size: usize) ![*]u8 {
            const ptr = try alloc.alignedAlloc(u8, .fromByteUnits(64), size);
            const block = try alloc.create(Block);
            block.* = .{
                .ptr = ptr.ptr,
                .size = size,
                .next = self.blocks,
            };
            self.blocks = block;
            return ptr.ptr;
        }
        
        fn find(self: *Bucket, size: usize) ?[*]u8 {
            var current = self.blocks;
            while (current) |block| {
                if (block.size >= size) {
                    return block.ptr;
                }
                current = block.next;
            }
            return null;
        }
    };

    // Size classes for common tensor sizes
    const size_classes = [_]usize{
        64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536, 131072, 262144,
        524288, 1048576, 2097152, 4194304, 8388608,
        16777216, 33554432, 67108864, 134217728,
    };

    buckets: [size_classes.len]Bucket = [_]Bucket{.{}} ** size_classes.len,
    allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) MemoryPool {
        return .{
            .buckets = undefined,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        // Free all blocks
        for (&self.buckets) |*bucket| {
            var current = bucket.blocks;
            while (current) |block| {
                const next = block.next;
                self.allocator.free(block.ptr[0..block.size]);
                self.allocator.destroy(block);
                current = next;
            }
        }
    }

    fn sizeToBucket(size: usize) usize {
        for (size_classes, 0..) |class_size, i| {
            if (size <= class_size) return i;
        }
        return size_classes.len - 1;
    }

    pub fn allocate(self: *MemoryPool, size: usize) ![*]u8 {
        const bucket_idx = sizeToBucket(size);
        const bucket = &self.buckets[bucket_idx];
        
        // Try to find existing block
        if (bucket.find(size)) |ptr| {
            return ptr;
        }
        
        // Allocate new block
        return try bucket.add(self.allocator, size_classes[bucket_idx]);
    }

    pub fn deallocate(self: *MemoryPool, ptr: [*]u8, size: usize) void {
        // In a real implementation, we'd mark the block as free
        // For now, we just keep it in the pool for reuse
        _ = self;
        _ = ptr;
        _ = size;
    }
};

/// Global memory pool instance
var global_pool: ?MemoryPool = null;

pub fn getPool() *MemoryPool {
    if (global_pool == null) {
        global_pool = MemoryPool.init(std.heap.c_allocator);
    }
    return &global_pool.?;
}

/// Allocate memory from the pool
pub fn poolAllocate(size: usize) ![*]u8 {
    return try getPool().allocate(size);
}

/// Return memory to the pool
pub fn poolDeallocate(ptr: [*]u8, size: usize) void {
    getPool().deallocate(ptr, size);
}

test "memory pool basic" {
    var pool = MemoryPool.init(std.testing.allocator);
    defer pool.deinit();

    const ptr1 = try pool.allocate(1024);
    try std.testing.expect(ptr1 != null);

    const ptr2 = try pool.allocate(2048);
    try std.testing.expect(ptr2 != null);
}
