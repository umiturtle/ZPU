//! The `Tensor` value type and ZPU's memory model.
//!
//! # Ownership
//!
//! A `Tensor` is a *view*: a pointer to a reference counted `Buffer` plus a
//! shape, strides and an element offset. Copying a `Tensor` value copies the
//! view, never the data.
//!
//! There are two ways to manage the lifetime of the buffers an op creates:
//!
//! 1. **Explicit** (default): every tensor returned to you holds one
//!    reference. Call `deinit()` when you are done with it. Ops release their
//!    internal temporaries themselves.
//! 2. **Scoped** (`Scope`, used by every transform in `transforms.zig`): while
//!    a `Scope` is active every tensor an op creates is adopted by it (no extra
//!    reference is taken) and released when the scope ends. Tensors that must
//!    outlive the scope are promoted with `Scope.keep`.
//!
//! Because a scope owns everything that was created inside it, a transformed
//! function (a gradient, a vmapped body, a JIT trace) can be written in plain
//! op style without a single `defer`.
const std = @import("std");
const dtype_mod = @import("dtype.zig");
const shape_mod = @import("shape.zig");
const buffer_mod = @import("buffer.zig");
const context = @import("context.zig");
const scalar = @import("scalar.zig");

pub const DType = dtype_mod.DType;
pub const Shape = shape_mod.Shape;
pub const Buffer = buffer_mod.Buffer;
pub const MAX_RANK = shape_mod.MAX_RANK;

/// A slice specification for `ops.slice`: `start` may be negative (from the
/// end), `end == null` means "to the end", `step` must not be zero.
pub const Range = struct {
    start: i64 = 0,
    end: ?i64 = null,
    step: isize = 1,

    pub fn all() Range {
        return .{};
    }
    pub fn bounded(start: i64, end: i64) Range {
        return .{ .start = start, .end = end };
    }
    pub fn from(start: i64) Range {
        return .{ .start = start };
    }
    pub fn upto(end: i64) Range {
        return .{ .start = 0, .end = end };
    }
    pub fn strided(start: i64, end: i64, step: isize) Range {
        return .{ .start = start, .end = end, .step = step };
    }
};

pub const Error = error{
    OutOfMemory,
    ShapeMismatch,
    DTypeMismatch,
    RankTooLarge,
    InvalidShape,
    InvalidAxis,
    NonContiguous,
    IndexOutOfRange,
    NotSupported,
    NotDifferentiable,
    NotPositiveDefinite,
    DeviceMismatch,
};

var id_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(1);

fn nextId() u64 {
    return id_counter.fetchAdd(1, .monotonic);
}

pub const Tensor = struct {
    buf: *Buffer,
    dt: DType = .f64,
    shape: Shape = Shape{},
    strides: [MAX_RANK]isize = @splat(0),
    offset: usize = 0,
    requires_grad: bool = false,
    batch_depth: u8 = 0,
    /// Unique identity of this *value* (views included); used by the tape.
    id: u64 = 0,

    // ---------------------------------------------------------------- creation

    /// Uninitialised tensor backed by a fresh buffer.
    pub fn empty(shape: Shape, dt: DType) Error!Tensor {
        const n = shape.numel();
        const b = try Buffer.create(n * dt.itemsize());
        const t = Tensor{
            .buf = b,
            .dt = dt,
            .shape = shape,
            .strides = shape.stridesContiguous(),
            .offset = 0,
            .id = nextId(),
        };
        track(t);
        return t;
    }

    pub fn zeros(shape: Shape, dt: DType) Error!Tensor {
        const t = try empty(shape, dt);
        t.buf.zero();
        return t;
    }

    pub fn ones(shape: Shape, dt: DType) Error!Tensor {
        const t = try empty(shape, dt);
        for (0..t.numel()) |i| t.writeF64(i, 1);
        return t;
    }

    pub fn full(shape: Shape, dt: DType, v: f64) Error!Tensor {
        const t = try empty(shape, dt);
        for (0..t.numel()) |i| t.writeF64(i, v);
        return t;
    }

    pub fn fromScalarF64(v: f64, dt: DType) Error!Tensor {
        return full(Shape{}, dt, v);
    }

    pub fn fromScalar(comptime T: type, v: T) Error!Tensor {
        return fromScalarF64(switch (@typeInfo(T)) {
            .int => @floatFromInt(v),
            .bool => if (v) 1 else 0,
            else => v,
        }, DType.of(T));
    }

    /// Builds a tensor from a slice of values.
    pub fn fromSlice(comptime T: type, values: []const T) Error!Tensor {
        return fromArray(T, values, try Shape.init(&.{values.len}));
    }

    /// Builds a tensor from a flat slice of values plus the logical shape.
    pub fn fromArray(comptime T: type, values: []const T, shape: Shape) Error!Tensor {
        if (values.len != shape.numel()) return Error.ShapeMismatch;
        const t = try empty(shape, DType.of(T));
        if (comptime DType.of(T) == .bool) {
            const dst = t.buf.dataMut(bool);
            for (values, 0..) |v, i| dst[i] = v;
        } else {
            const dst = t.buf.dataMut(T);
            @memcpy(dst[0..values.len], values);
        }
        return t;
    }

    /// `arange(n)` in a float dtype.
    pub fn arange(n: usize, dt: DType) Error!Tensor {
        const t = try empty(try Shape.init(&.{n}), dt);
        for (0..n) |i| t.writeF64(i, @floatFromInt(i));
        return t;
    }

    /// `linspace(lo, hi, n)`.
    pub fn linspace(lo: f64, hi: f64, n: usize, dt: DType) Error!Tensor {
        const t = try empty(try Shape.init(&.{n}), dt);
        if (n == 1) {
            t.writeF64(0, lo);
            return t;
        }
        for (0..n) |i| {
            const f = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
            t.writeF64(i, lo + (hi - lo) * f);
        }
        return t;
    }

    pub fn eye(n: usize, dt: DType) Error!Tensor {
        const t = try zeros(try Shape.init(&.{ n, n }), dt);
        for (0..n) |i| t.writeF64(i * n + i, 1);
        return t;
    }

    // -------------------------------------------------------------- lifetime

    /// Adds a reference; the returned tensor must be freed with `deinit`.
    pub fn retain(self: Tensor) Tensor {
        _ = self.buf.retain();
        return self;
    }

    /// Adds a reference *owned by the active scope* (released when the scope
    /// ends). Outside a scope it behaves exactly like `retain`.
    pub fn retainScoped(self: Tensor) Tensor {
        const t = self.retain();
        trackRef(t);
        return t;
    }

    /// Drops one reference to the backing buffer.
    pub fn deinit(self: Tensor) void {
        self.buf.release();
    }

    // ---------------------------------------------------------------- queries

    pub fn rank(self: Tensor) usize {
        return self.shape.rank();
    }

    pub fn dim(self: Tensor, axis: usize) usize {
        return self.shape.dims[axis];
    }

    pub fn numel(self: Tensor) usize {
        return self.shape.numel();
    }

    pub fn dtype(self: Tensor) DType {
        return self.dt;
    }

    pub fn dtypeOf(self: Tensor) DType {
        return self.dt;
    }

    /// Number of payload axes (rank without the `vmap` batch axes).
    pub fn payloadRank(self: Tensor) usize {
        return self.rank() - self.batch_depth;
    }

    pub fn itemsize(self: Tensor) usize {
        return self.dt.itemsize();
    }

    pub fn isContiguous(self: Tensor) bool {
        const cs = self.shape.stridesContiguous();
        for (0..self.rank()) |i| {
            if (self.strides[i] != cs[i]) return false;
        }
        return true;
    }

    pub fn withRequiresGrad(self: Tensor, v: bool) Tensor {
        var t = self;
        t.requires_grad = v;
        return t;
    }

    pub fn withBatchDepth(self: Tensor, d: u8) Tensor {
        var t = self;
        t.batch_depth = d;
        return t;
    }

    /// `true` when `self` and `other` share the same storage.
    pub fn aliases(self: Tensor, other: Tensor) bool {
        return self.buf == other.buf;
    }

    /// Element access in **memory order** (offset relative). Requires
    /// contiguity; call `toContiguous` first for views.
    pub fn readF64(self: Tensor, i: usize) f64 {
        std.debug.assert(self.isContiguous());
        return switch (self.dt) {
            inline else => |d| switch (d) {
                .bool => if (self.buf.data(bool)[self.offset + i]) 1 else 0,
                inline else => scalar.toF64(self.buf.data(DType.Elt(d))[self.offset + i]),
            },
        };
    }

    pub fn writeF64(self: Tensor, i: usize, v: f64) void {
        std.debug.assert(self.isContiguous());
        switch (self.dt) {
            inline else => |d| switch (d) {
                .bool => self.buf.dataMut(bool)[self.offset + i] = v != 0,
                inline else => {
                    const T = DType.Elt(d);
                    const x = if (@typeInfo(T) == .int)
                        (if (std.math.isNan(v)) 0 else @as(f64, @floatFromInt(@as(T, @intFromFloat(v)))))
                    else
                        v;
                    _ = x;
                    self.buf.dataMut(T)[self.offset + i] = scalar.fromF64(T, v);
                },
            },
        }
    }

    /// Typed read-only slice of the *whole* buffer.
    pub fn flatSlice(self: Tensor, comptime T: type) ?[]const T {
        if (self.dtypeOf() != DType.of(T)) return null;
        return self.buf.data(T)[self.offset .. self.offset + self.numel()];
    }

    /// Typed mutable slice of the whole buffer (contiguous tensors only).
    pub fn flatSliceMut(self: Tensor, comptime T: type) ?[]T {
        if (self.dtypeOf() != DType.of(T)) return null;
        std.debug.assert(self.isContiguous());
        return self.buf.dataMut(T)[self.offset .. self.offset + self.numel()];
    }

    /// `true` when every element is bit-identical.
    pub fn eql(self: Tensor, other: Tensor) bool {
        if (self.rank() != other.rank()) return false;
        if (self.dtypeOf() != other.dtypeOf()) return false;
        if (!self.shape.eql(other.shape)) return false;
        const n = self.numel();
        if (n == 0) return true;
        if (self.isContiguous() and other.isContiguous()) {
            switch (self.dt) {
                inline else => |d| {
                    const T = DType.Elt(d);
                    const a = self.buf.data(T);
                    const b = other.buf.data(T);
                    for (0..n) |i| {
                        if (a[self.offset + i] != b[other.offset + i]) return false;
                    }
                },
            }
            return true;
        }
        // generic (strided) path
        var it = self.iter();
        var it2 = other.iter();
        while (it.nextLogical()) |v| {
            const w = it2.nextLogical().?;
            if (v != w) return false;
        }
        return true;
    }

    /// `|a - b| <= tol` element-wise (same shape required).
    pub fn allClose(self: Tensor, other: Tensor, tol: f64) bool {
        if (!self.shape.eql(other.shape)) return false;
        const n = self.numel();
        var it = self.iter();
        var it2 = other.iter();
        for (0..n) |_| {
            const a = it.nextLogical().?;
            const b = it2.nextLogical().?;
            if (std.math.isNan(a) and std.math.isNan(b)) continue;
            if (!(@abs(a - b) <= tol)) return false;
        }
        return true;
    }

    /// Scalar value of a 1-element tensor.
    pub fn item(self: Tensor) f64 {
        std.debug.assert(self.numel() == 1);
        return self.readF64(0);
    }

    /// Deep copy into a fresh, contiguous buffer.
    pub fn clone(self: Tensor) Error!Tensor {
        const t = try empty(self.shape, self.dtypeOf());
        const n = self.numel();
        for (0..n) |i| t.writeF64(i, self.readLogical(n, i));
        return t;
    }

    // --------------------------------------------------------------- indexing

    /// Element `i` in *logical* (row major) order; works for any strides.
    pub fn readLogical(self: Tensor, n: usize, i: usize) f64 {
        std.debug.assert(i < n);
        if (self.isContiguous()) return self.readF64(i);
        var rem = i;
        var off: isize = 0;
        var a = self.rank();
        while (a > 0) {
            a -= 1;
            const d = self.shape.dims[a];
            const idx = rem % d;
            rem /= d;
            off += @as(isize, @intCast(idx)) * self.strides[a];
        }
        // `flip` and reverse-strided ranges give negative strides, so the walk
        // above accumulates a *negative* delta: it is measured from the view's
        // own offset (for a flipped axis that is the last element in memory, and
        // the walk counts backwards from there). Both terms are isize, and the
        // element always lands inside the buffer.
        const at: usize = @intCast(@as(isize, @intCast(self.offset)) + off);
        return switch (self.dt) {
            inline else => |d| switch (d) {
                .bool => if (self.buf.data(bool)[at]) 1 else 0,
                inline else => scalar.toF64(self.buf.data(DType.Elt(d))[at]),
            },
        };
    }

    /// Materialises a contiguous, offset-0 version of `self`. Contiguous
    /// tensors are returned unchanged (retained).
    pub fn toContiguous(self: Tensor) Error!Tensor {
        if (self.isContiguous()) return self.retainScoped();
        const t = try empty(self.shape, self.dtypeOf());
        const n = self.numel();
        for (0..n) |i| t.writeF64(i, self.readLogical(n, i));
        var out = t;
        out.batch_depth = self.batch_depth;
        out.requires_grad = self.requires_grad;
        return out;
    }

    /// A view sharing `parent`'s buffer (one extra reference is taken).
    pub fn viewOf(parent: Tensor, shape: Shape, strides: [MAX_RANK]isize, offset: usize) Tensor {
        var t = parent;
        t.shape = shape;
        t.strides = strides;
        t.offset = offset;
        t.dt = parent.dt;
        t.batch_depth = parent.batch_depth;
        t.requires_grad = parent.requires_grad;
        t.id = nextId();
        _ = t.buf.retain();
        track(t);
        return t;
    }

    /// Transposes the last two axes (matrix transpose for 2-D).
    pub fn transposeAll(self: Tensor) Error!Tensor {
        if (self.rank() < 2) return self.retain();
        var perm: [MAX_RANK]usize = @splat(0);
        for (0..self.rank()) |i| perm[i] = i;
        const r = self.rank();
        perm[r - 1] = r - 2;
        perm[r - 2] = r - 1;
        const ops = @import("ops.zig");
        return ops.transpose(self, perm[0..r]);
    }

    // ------------------------------------------------------------ iteration

    pub const Iter = struct {
        t: Tensor,
        idx: [MAX_RANK]usize = @splat(0),
        off: isize = 0,
        n: usize,
        i: usize = 0,
        started: bool = false,

        pub fn nextLogical(self: *Iter) ?f64 {
            if (self.i >= self.n) return null;
            const v = self.at();
            self.advance();
            self.i += 1;
            return v;
        }

        fn at(self: *Iter) f64 {
            const off = self.t.offset + @as(usize, @intCast(self.off));
            return switch (self.t.dt) {
                inline else => |d| switch (d) {
                    .bool => if (self.t.buf.data(bool)[off]) 1 else 0,
                    inline else => scalar.toF64(self.t.buf.data(DType.Elt(d))[off]),
                },
            };
        }

        fn advance(self: *Iter) void {
            var a = self.t.rank();
            while (a > 0) {
                a -= 1;
                self.idx[a] += 1;
                self.off += self.t.strides[a];
                if (self.idx[a] < self.t.shape.dims[a]) return;
                self.off -= @as(isize, @intCast(self.t.shape.dims[a])) * self.t.strides[a];
                self.idx[a] = 0;
            }
        }
    };

    /// Logical (row major) iterator over the elements of a strided tensor.
    pub fn iter(self: Tensor) Iter {
        return .{ .t = self, .n = self.numel() };
    }

    /// Sets a single logical element.
    pub fn setLogical(self: Tensor, i: usize, v: f64) void {
        std.debug.assert(self.isContiguous());
        self.writeF64(i, v);
    }

    // ------------------------------------------------------------- printing

    pub fn format(
        self: Tensor,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("tensor({f}, {s})", .{ self.shape, self.dtypeOf().name() });
        const n = self.numel();
        if (n == 0) {
            try writer.writeAll(" []");
            return;
        }
        try writer.writeAll(" [");
        const limit: usize = 96;
        if (n <= limit) {
            var it = self.iter();
            for (0..n) |i| {
                if (i != 0) try writer.writeAll(", ");
                const v = it.nextLogical().?;
                if (self.dtypeOf() == .bool) {
                    try writer.print("{s}", .{if (v != 0) "true" else "false"});
                } else if (self.dtypeOf().isInt()) {
                    try writer.print("{d}", .{@as(i64, @intFromFloat(v))});
                } else {
                    try writer.print("{d:.4}", .{v});
                }
            }
        } else {
            var it = self.iter();
            for (0..4) |i| {
                if (i != 0) try writer.writeAll(", ");
                try writer.print("{d:.4}", .{it.nextLogical().?});
            }
            try writer.print(", ... ({d} elements) ... ", .{n});
            var skip = n - 4;
            while (skip > 0) : (skip -= 1) _ = it.nextLogical();
            for (0..4) |i| {
                if (i != 0) try writer.writeAll(", ");
                try writer.print("{d:.4}", .{it.nextLogical().?});
            }
        }
        try writer.writeAll("]");
    }
};

// ---------------------------------------------------------------------------
// dtype lookups back into the tensor
// ---------------------------------------------------------------------------

/// Scoped, arena-like lifetime management (see the module docs).
pub const Tracker = struct {
    alloc: std.mem.Allocator,
    list: std.ArrayListUnmanaged(Tensor) = .empty,
    prev: ?*Tracker = null,
    installed: bool = false,

    pub fn create(alloc: std.mem.Allocator) Error!*Tracker {
        const tr = try alloc.create(Tracker);
        tr.* = .{ .alloc = alloc };
        return tr;
    }

    /// Destroys the tracker itself (must have been released with `end`).
    pub fn destroy(self: *Tracker) void {
        const alloc = self.alloc;
        self.list.deinit(alloc);
        alloc.destroy(self);
    }

    /// Installs the tracker if no other one is active. Returns `true` when this
    /// tracker is now the active one.
    pub fn begin(self: *Tracker) bool {
        if (context.trackerPtr() != null) return false;
        self.prev = null;
        self.installed = true;
        context.setTracker(@ptrCast(self));
        return true;
    }

    /// Uninstalls the tracker and releases everything it adopted.
    pub fn end(self: *Tracker) void {
        if (!self.installed) return;
        if (context.trackerPtr() == @as(?*anyopaque, @ptrCast(self))) {
            context.setTracker(if (self.prev) |p| @ptrCast(p) else null);
        }
        for (self.list.items) |t| {
            var tt = t;
            tt.deinit();
        }
        self.list.clearRetainingCapacity();
        self.installed = false;
    }

    /// Takes ownership of the reference that `t` already holds (the one taken
    /// when the buffer was created). `end` drops it.
    pub fn adopt(self: *Tracker, t: Tensor) void {
        self.list.append(self.alloc, t) catch {};
    }

    /// Takes `t` off the list, so that `end` no longer releases it: the
    /// creation reference is handed to the caller. This is how long lived state
    /// (optimiser state, module parameters) escapes a scope that merely happens
    /// to be active when it is built.
    pub fn forget(self: *Tracker, t: Tensor) void {
        for (self.list.items, 0..) |it, i| {
            if (it.id == t.id) {
                _ = self.list.swapRemove(i);
                return;
            }
        }
    }

    /// Promotes a tensor so that it survives `end` (adds a reference).
    pub fn keep(self: *Tracker, t: Tensor) Tensor {
        _ = self;
        return t.retain();
    }

    pub fn len(self: *const Tracker) usize {
        return self.list.items.len;
    }
};

fn track(t: Tensor) void {
    const p = context.trackerPtr() orelse return;
    const tr: *Tracker = @ptrCast(@alignCast(p));
    if (!tr.installed) return;
    tr.adopt(t);
}

/// Registers an extra reference taken by a primitive with the active scope.
pub fn trackRef(t: Tensor) void {
    track(t);
}

/// Hands the ownership of `t`'s creation reference to the caller: the active
/// `Scope` (if any) will not release it, so whoever calls this owns `t` and must
/// `deinit` it. Used for state that outlives the scope it was created in.
pub fn untrack(t: Tensor) void {
    const p = context.trackerPtr() orelse return;
    const tr: *Tracker = @ptrCast(@alignCast(p));
    if (!tr.installed) return;
    tr.forget(t);
}

/// Frees a temporary created inside an op, unless a `Scope` owns it.
pub fn releaseTemp(t: Tensor) void {
    if (context.trackerPtr() != null) return;
    var tt = t;
    tt.deinit();
}

/// A scoped lifetime domain: every tensor created while the scope is active is
/// released by `deinit`.
pub const Scope = struct {
    tracker: *Tracker,

    pub fn init(alloc: std.mem.Allocator) Error!Scope {
        if (context.ctx().alloc == null) context.ctx().alloc = alloc;
        const tr = try Tracker.create(alloc);
        _ = tr.begin();
        return .{ .tracker = tr };
    }

    pub fn deinit(self: *Scope) void {
        self.tracker.end();
        self.tracker.destroy();
    }

    /// Promotes a tensor out of the scope's lifetime.
    pub fn keep(self: *Scope, t: Tensor) Tensor {
        return self.tracker.keep(t);
    }

    pub fn count(self: *const Scope) usize {
        return self.tracker.len();
    }
};

test "tensor basics" {
    const alloc = std.testing.allocator;
    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const t = try Tensor.fromArray(f32, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 2, 3 }));
    try std.testing.expectEqual(@as(usize, 2), t.rank());
    try std.testing.expectEqual(@as(usize, 6), t.numel());
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), t.readF64(3), 1e-6);
    try std.testing.expect(t.isContiguous());

    const z = try Tensor.zeros(Shape.initComptime(&.{ 2, 2 }), .f64);
    try std.testing.expect(z.eql(try Tensor.zeros(Shape.initComptime(&.{ 2, 2 }), .f64)));
}

test "tensor views and types" {
    const alloc = std.testing.allocator;
    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const t = try Tensor.fromSlice(i64, &.{ 10, 20, 30, 40 });
    try std.testing.expectEqual(DType.i64, t.dtype());
    const sl = t.flatSlice(i64).?;
    try std.testing.expectEqual(@as(i64, 30), sl[2]);

    const keep = scope.keep(t);
    defer keep.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), keep.readLogical(4, 1), 1e-9);
}
