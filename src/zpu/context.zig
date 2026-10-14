//! Execution context.
//!
//! ZPU keeps its whole dynamic state in a *thread local* context: the
//! allocator, the active tape (autodiff), the active graph recorder (JIT), the
//! active tensor tracker (lifetime management) and the worker pool. Ops look
//! the context up on every call, which is what makes `grad`, `vmap` and `jit`
//! composable without threading extra state through every signature.
const std = @import("std");

pub const Mode = enum(u8) {
    /// Plain eager execution, nothing is recorded.
    eager,
    /// Recording a `graph.Graph` for the JIT.
    trace,
    /// Replaying a traced plan: no tape, no graph, but the JIT's own
    /// interpreter still dispatches through `ops.applyPrim`.
    compiled,
    /// Recording a `tape.Tape` for reverse mode autodiff.
    grad,
};

/// Fallback allocator used when `zpu.init` was never called (so that the
/// library is usable from a single `@import` with zero setup).
var fallback_gpa: std.heap.SafeAllocator = std.heap.SafeAllocator.init(std.heap.page_allocator, .{});
/// Set once `fallback_gpa` has been touched, so that a later `deinit` knows.
var fallback_used: bool = false;

/// How many recorders can be nested (an inner `grad` inside an outer `grad`
/// inside a `jit` …). `grad` of `grad` of `grad` is already exotic; eight
/// levels of nesting is far beyond anything a program needs, and the fixed size
/// keeps the context allocation free.
pub const MAX_TAPE_DEPTH = 8;

pub const Ctx = struct {
    alloc: ?std.mem.Allocator = null,
    mode: Mode = .eager,
    /// Stack of installed tapes. Every op is recorded on *all* of them, which is
    /// what lets an outer `grad`/`jit` differentiate through an inner `grad`:
    /// the outer recorder sees both the inner forward pass and the inner sweep.
    tapes: [MAX_TAPE_DEPTH]?*anyopaque = @splat(null),
    tape_depth: u8 = 0,
    graph: ?*anyopaque = null,
    tracker: ?*anyopaque = null,
    vmap_depth: u8 = 0,
    /// Number of chunks `parallelFor` splits work into (1 = run inline).
    threads: usize = 1,

    pub fn allocator(self: *Ctx) std.mem.Allocator {
        return self.alloc orelse fallbackAllocator();
    }
};

pub fn fallbackAllocator() std.mem.Allocator {
    fallback_used = true;
    return fallback_gpa.allocator();
}

threadlocal var tls: Ctx = .{};

pub fn ctx() *Ctx {
    return &tls;
}

pub fn allocatorRef() std.mem.Allocator {
    return ctx().allocator();
}

pub const InitOptions = struct {
    /// Number of worker threads used by the element-wise kernels (1 = inline).
    threads: usize = 1,
};

/// Installs the allocator (when given) and starts the worker pool.
pub fn init(alloc: ?std.mem.Allocator, opts: InitOptions) !void {
    const c = ctx();
    if (alloc) |a| c.alloc = a;
    c.threads = @max(opts.threads, 1);
}

/// Convenience wrapper: `zpu.initThreaded(alloc, n)`.
pub fn initThreaded(alloc: ?std.mem.Allocator, threads: usize) !void {
    return init(alloc, .{ .threads = threads });
}

/// Tears the context down: stops the pool and resets every field.
pub fn deinit() void {
    const c = ctx();
    const alloc = c.alloc;
    c.* = .{};
    if (alloc == null and fallback_used) {
        fallback_used = false;
        _ = fallback_gpa.deinit();
    }
}

pub fn mode() Mode {
    return ctx().mode;
}

pub fn setMode(m: Mode) void {
    ctx().mode = m;
}

/// Autodiff is recording *and* we are not replaying a compiled plan.
pub fn tapeActive() bool {
    const c = ctx();
    return c.tape_depth > 0 and c.mode == .grad;
}

/// A graph recorder is installed (it records in every mode, so that `grad` of
/// a jitted function still produces a replayable plan).
pub fn graphActive() bool {
    return ctx().graph != null;
}

/// The innermost active tape, if any.
pub fn tapePtr() ?*anyopaque {
    const c = ctx();
    if (c.tape_depth == 0) return null;
    return c.tapes[c.tape_depth - 1];
}

/// Every installed tape, outermost first.
pub fn tapes() []const ?*anyopaque {
    const c = ctx();
    return c.tapes[0..c.tape_depth];
}

/// Installs a tape on top of the stack (keeping the recorders below it alive).
pub fn pushTape(p: ?*anyopaque) void {
    const c = ctx();
    if (c.tape_depth == MAX_TAPE_DEPTH) return;
    c.tapes[c.tape_depth] = p;
    c.tape_depth += 1;
}

/// Removes the innermost tape.
pub fn popTape() void {
    const c = ctx();
    if (c.tape_depth == 0) return;
    c.tape_depth -= 1;
    c.tapes[c.tape_depth] = null;
}

/// Replaces the innermost tape in place (`setTape(null)` disables recording at
/// the current level without disturbing the outer recorders).
pub fn setTape(p: ?*anyopaque) void {
    const c = ctx();
    if (c.tape_depth == 0) {
        if (p != null) pushTape(p);
        return;
    }
    c.tapes[c.tape_depth - 1] = p;
}

pub fn graphPtr() ?*anyopaque {
    return ctx().graph;
}

pub fn setGraph(p: ?*anyopaque) void {
    ctx().graph = p;
}

pub fn trackerPtr() ?*anyopaque {
    return ctx().tracker;
}

pub fn setTracker(p: ?*anyopaque) void {
    ctx().tracker = p;
}

pub fn startVmap() void {
    ctx().vmap_depth += 1;
}

pub fn stopVmap() void {
    ctx().vmap_depth -= 1;
}

pub fn vmapDepth() u8 {
    return ctx().vmap_depth;
}

/// Everything the transforms need to save and restore around a traced call.
/// The whole tape stack travels with it, so an inner transform can unwind
/// without disturbing the recorders that were already installed.
pub const Saved = struct {
    mode: Mode,
    tapes: [MAX_TAPE_DEPTH]?*anyopaque,
    tape_depth: u8,
    graph: ?*anyopaque,
};

pub fn save() Saved {
    const c = ctx();
    return .{ .mode = c.mode, .tapes = c.tapes, .tape_depth = c.tape_depth, .graph = c.graph };
}

pub fn restore(s: Saved) void {
    const c = ctx();
    c.mode = s.mode;
    c.tapes = s.tapes;
    c.tape_depth = s.tape_depth;
    c.graph = s.graph;
}

/// Runs `f(start, end)` over `n` items, split across the worker pool when one
/// is configured, and inline otherwise.
pub fn parallelFor(n: usize, ctxp: *anyopaque, comptime f: anytype) void {
    const c = ctx();
    const threads = @min(c.threads, @max(n, 1));
    if (threads <= 1 or n < 4096) {
        f(ctxp, 0, n);
        return;
    }
    const Task = struct {
        ctp: *anyopaque,
        lo: usize,
        hi: usize,
        fn run(self: @This()) void {
            f(self.ctp, self.lo, self.hi);
        }
    };
    var tasks_buf: [64]Task = undefined;
    const count = @min(threads, tasks_buf.len);
    const chunk = n / count;
    for (0..count) |i| {
        const lo = i * chunk;
        const hi = if (i == count - 1) n else lo + chunk;
        tasks_buf[i] = .{ .ctp = ctxp, .lo = lo, .hi = hi };
    }
    // Fork/join: the last chunk runs on this thread, the rest on workers. No
    // thread pool is kept alive between calls, so `zpu.init`/`deinit` stay
    // allocation free and there is nothing to shut down.
    var spawned: usize = 0;
    var handles: [63]std.Thread = undefined;
    for (tasks_buf[0 .. count - 1]) |t| {
        const h = std.Thread.spawn(.{}, Task.run, .{t}) catch break;
        handles[spawned] = h;
        spawned += 1;
    }
    Task.run(tasks_buf[count - 1]);
    for (handles[0..spawned]) |h| h.join();
}

test "context allocator" {
    const c = ctx();
    var buf: [8]u8 = undefined;
    const a = c.allocator();
    const mem = try a.dupe(u8, &buf);
    a.free(mem);
}
