//! Reference counted, 64 byte aligned storage for tensor elements.
//!
//! Buffers are allocated through the *installed* allocator
//! (`context.Ctx.allocator`, see `zpu.init`) rather than through a hidden
//! global, which keeps ZPU usable inside test harnesses, comptime analysis and
//! embedded targets alike.
const std = @import("std");
const context = @import("context.zig");

pub const ALIGN: usize = 64;

pub const Error = error{OutOfMemory};

pub const Buffer = struct {
    /// Number of *owners*. A buffer is freed exactly when this hits zero.
    rc: std.atomic.Value(u32),
    alloc: std.mem.Allocator,
    /// Raw allocation backing `mem`. `mem.ptr` may be offset inside it when the
    /// buffer was created from a pointer that is not `ALIGN` aligned.
    raw: []align(ALIGN) u8,
    mem: []align(ALIGN) u8,
    /// Number of live views (debug aid only).
    views: u32 = 0,

    /// Allocates `nbytes` uninitialised bytes through the installed allocator.
    pub fn create(nbytes: usize) Error!*Buffer {
        return createWith(context.allocatorRef(), nbytes);
    }

    pub fn createWith(alloc: std.mem.Allocator, nbytes: usize) Error!*Buffer {
        const n = @max(nbytes, 1);
        const raw = try alloc.alignedAlloc(u8, .fromByteUnits(ALIGN), n);
        const buf = try alloc.create(Buffer);
        buf.* = .{
            .rc = std.atomic.Value(u32).init(1),
            .alloc = alloc,
            .raw = raw,
            .mem = raw,
        };
        return buf;
    }

    /// Zero filled variant.
    pub fn createZeroed(nbytes: usize) Error!*Buffer {
        const b = try create(nbytes);
        @memset(b.mem, 0);
        return b;
    }

    /// Wraps caller owned memory. The caller guarantees it stays alive (and
    /// aligned) for the lifetime of the buffer; releasing the buffer does *not*
    /// free it.
    pub fn wrapConst(alloc: std.mem.Allocator, bytes: []align(ALIGN) const u8) Error!*Buffer {
        const buf = try alloc.create(Buffer);
        const mut = @constCast(bytes);
        buf.* = .{
            .rc = std.atomic.Value(u32).init(1),
            .alloc = alloc,
            .raw = &.{},
            .mem = mut,
        };
        return buf;
    }

    pub fn retain(self: *Buffer) *Buffer {
        _ = self.rc.fetchAdd(1, .monotonic);
        return self;
    }

    pub fn release(self: *Buffer) void {
        const prev = self.rc.fetchSub(1, .acq_rel);
        if (prev > 1) return;
        const alloc = self.alloc;
        if (self.raw.len > 0) alloc.free(self.raw);
        self.* = undefined;
        alloc.destroy(self);
    }

    pub fn byteLen(self: *const Buffer) usize {
        return self.mem.len;
    }

    pub fn zero(self: *Buffer) void {
        @memset(self.mem, 0);
    }

    /// Typed read-only view of the whole buffer.
    pub fn data(self: *const Buffer, comptime T: type) []const T {
        std.debug.assert(self.mem.len % @sizeOf(T) == 0);
        return std.mem.bytesAsSlice(T, self.mem);
    }

    /// Typed mutable view of the whole buffer.
    pub fn dataMut(self: *Buffer, comptime T: type) []T {
        std.debug.assert(self.mem.len % @sizeOf(T) == 0);
        return std.mem.bytesAsSlice(T, self.mem);
    }

    pub fn eqlBytes(a: *const Buffer, b: *const Buffer) bool {
        return std.mem.eql(u8, a.mem, b.mem);
    }
};

/// Allocates a zeroed buffer able to hold `n` elements of type `T`.
pub fn zerosOf(comptime T: type, n: usize) Error!*Buffer {
    return Buffer.createZeroed(n * @sizeOf(T));
}

test "buffer refcounting" {
    const b = try Buffer.createZeroed(16 * @sizeOf(f32));
    const v = b.dataMut(f32);
    try std.testing.expectEqual(@as(usize, 16), v.len);
    v[0] = 3.5;
    _ = b.retain();
    b.release();
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), b.data(f32)[0], 1e-6);
    b.release();
}
