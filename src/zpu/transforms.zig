//! Function transformations: `grad`, `valueAndGrad`, `vjp`, `jvp`, `vmap` and
//! `jit`.
//!
//! Every transform follows the same recipe:
//!
//! 1. open a `Scope` that owns every tensor the transformed function creates,
//! 2. install the recording machinery (tape and/or graph) it needs,
//! 3. call the function,
//! 4. restore the *outer* recording state so that the work done by the
//!    transform (e.g. the backward sweep) can itself be recorded — this is what
//!    makes `grad(grad(f))`, `jit(grad(f))` and `hvp` work,
//! 5. keep the values that leave the transform.
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");
const prims = @import("prims.zig");
const context = @import("context.zig");
const tape_mod = @import("tape.zig");
const graph_mod = @import("graph.zig");
const autodiff = @import("autodiff.zig");
const jitlib = @import("jit.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = tmod.Shape;
pub const Scope = tmod.Scope;
pub const Tape = tape_mod.Tape;
pub const Graph = graph_mod.Graph;
pub const Error = tmod.Error;
pub const AXIS_NONE = prims.AXIS_NONE;

pub const releaseTemp = tmod.releaseTemp;

/// The shape of a transformed function: user data plus the argument tensors.
pub const Fn = *const fn (ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor;

pub const GradResult = struct {
    value: Tensor,
    grads: []Tensor,
    alloc: std.mem.Allocator,

    pub fn deinit(self: *GradResult) void {
        for (self.grads) |*g| {
            var gg = g.*;
            gg.deinit();
        }
        self.alloc.free(self.grads);
        var v = self.value;
        v.deinit();
    }
};

pub const VjpResult = GradResult;

/// Copies one entry out of a `grad`/`vjp` result and releases every other
/// reference (including the slice itself). A function that wants to *return* a
/// gradient — `hvp`'s trampoline, for instance — uses this instead of leaking
/// the entries it does not need.
pub fn keepGrad(alloc: std.mem.Allocator, grads: []Tensor, i: usize) Tensor {
    const kept = grads[i].retain();
    for (grads) |*t| {
        var tt = t.*;
        tt.deinit();
    }
    alloc.free(grads);
    return kept;
}

/// Copies `args` into a fresh buffer whose entries the transform owns.
fn copyArgs(alloc: std.mem.Allocator, args: []const Tensor, wrt: []const usize) ![]Tensor {
    const buf = try alloc.alloc(Tensor, args.len);
    @memcpy(buf, args);
    for (buf, 0..) |*tt, i| {
        tt.requires_grad = false;
        for (wrt) |w| {
            if (w == i) tt.requires_grad = true;
        }
        _ = tt.retainScoped();
    }
    return buf;
}

/// `valueAndGrad(f)(args)` for the outputs selected by `wrt`.
///
/// The function must produce a *scalar* (a 0-d tensor or one with a single
/// element); use `vjp` for general cotangents.
pub fn valueAndGrad(
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    args: []const Tensor,
    wrt: []const usize,
) !GradResult {
    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const buf = try copyArgs(alloc, args, wrt);
    defer alloc.free(buf);

    var tape = Tape.init(alloc);
    defer tape.deinit();

    const saved = context.save();
    context.setMode(.grad);
    // The forward pass is recorded on this tape (and on every outer recorder,
    // which is what makes `grad` differentiable).
    context.pushTape(&tape);
    const value = f(ud, buf) catch |err| {
        context.popTape();
        context.restore(saved);
        return err;
    };
    if (value.numel() != 1) {
        context.popTape();
        context.restore(saved);
        return Error.NotSupported;
    }

    // Pop this tape before the sweep: everything the sweep does is recorded on
    // the outer recorders only, so the tape itself stays a clean forward graph.
    context.popTape();
    context.restore(saved);

    // The engine takes its own reference to every seed it stores, so the one
    // created here is released again right away.
    const seed = try autodiff.seedOnes(value);
    defer releaseTemp(seed);

    var engine = autodiff.Engine.init(alloc);
    defer engine.deinit();

    var wrt_tensors = try alloc.alloc(Tensor, wrt.len);
    defer alloc.free(wrt_tensors);
    for (wrt, 0..) |w, i| {
        if (w >= args.len) return Error.InvalidShape;
        wrt_tensors[i] = buf[w];
    }

    try engine.run(&tape, &.{value}, &.{seed}, wrt_tensors);

    const grads = try alloc.alloc(Tensor, wrt.len);
    errdefer alloc.free(grads);
    for (0..wrt.len) |i| {
        const g = try engine.get(wrt_tensors[i]);
        grads[i] = g.retain();
    }
    return .{ .value = value.retain(), .grads = grads, .alloc = alloc };
}

/// Vector-Jacobian product: like `valueAndGrad` but with an explicit cotangent
/// for the (possibly non scalar) output.
pub fn vjp(
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    args: []const Tensor,
    wrt: []const usize,
    cotangent: Tensor,
) !GradResult {
    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const buf = try copyArgs(alloc, args, wrt);
    defer alloc.free(buf);

    var tape = Tape.init(alloc);
    defer tape.deinit();

    const saved = context.save();
    context.setMode(.grad);
    context.pushTape(&tape);
    const value = f(ud, buf) catch |err| {
        context.popTape();
        context.restore(saved);
        return err;
    };
    context.popTape();
    context.restore(saved);

    var engine = autodiff.Engine.init(alloc);
    defer engine.deinit();

    var wrt_tensors = try alloc.alloc(Tensor, wrt.len);
    defer alloc.free(wrt_tensors);
    for (wrt, 0..) |w, i| wrt_tensors[i] = buf[w];

    try engine.run(&tape, &.{value}, &.{cotangent}, wrt_tensors);

    const grads = try alloc.alloc(Tensor, wrt.len);
    errdefer alloc.free(grads);
    for (0..wrt.len) |i| {
        const g = try engine.get(wrt_tensors[i]);
        grads[i] = g.retain();
    }
    return .{ .value = value.retain(), .grads = grads, .alloc = alloc };
}

/// The gradient of a scalar function: a slice of one tensor per entry of `wrt`
/// (caller frees the slice and the tensors).
pub fn grad(
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    args: []const Tensor,
    wrt: []const usize,
) ![]Tensor {
    const r = try valueAndGrad(alloc, f, ud, args, wrt);
    const rr = r;
    var v = rr.value;
    v.deinit();
    return rr.grads;
}

/// All argument indices (the common `grad(loss, params)` case).
pub fn allIndices(alloc: std.mem.Allocator, n: usize) ![]usize {
    const out = try alloc.alloc(usize, n);
    for (out, 0..) |*v, i| v.* = i;
    return out;
}

pub const JvpResult = struct {
    value: Tensor,
    tangent: Tensor,
};

/// Jacobian-vector product.
///
/// Implemented as *reverse over reverse*: the gradient is computed with an outer
/// tape installed, which makes the gradient itself a differentiable function of
/// the inputs; contracting it with `tangents` gives `J v` exactly (no finite
/// differences). The price is one reverse sweep per output element, so a scalar
/// output — the case `hvp` needs — costs a single sweep.
pub fn jvp(
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    args: []const Tensor,
    tangents: []const Tensor,
) !JvpResult {
    if (tangents.len != args.len) return Error.ShapeMismatch;

    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const indices = try allIndices(alloc, args.len);
    defer alloc.free(indices);

    // 1. the primal value (nothing recorded)
    const value = try f(ud, args);
    const outs = value.numel();
    if (outs == 0) return Error.ShapeMismatch;
    if (outs > 1024) return Error.NotSupported;

    // 2. the gradient is computed with an outer tape installed, so that the
    //    contraction below lands on it (and, if there is yet another recorder
    //    outside, higher order derivatives keep working).
    var outer = Tape.init(alloc);
    defer outer.deinit();
    const saved = context.save();
    context.setMode(.grad);
    context.pushTape(&outer);
    defer {
        context.popTape();
        context.restore(saved);
    }

    var out_tangent = try Tensor.zeros(value.shape, floatOf(value));

    var j: usize = 0;
    while (j < outs) : (j += 1) {
        const cot = try oneHotCotangent(value, j);
        var r = try vjp(alloc, f, ud, args, indices, cot);
        defer r.deinit();

        // Contract d(value_j)/d(args) with the tangents. `acc` owns exactly one
        // reference (`retain`ed here); everything else is owned by this
        // function's scope.
        var acc: ?Tensor = null;
        defer if (acc) |a| dropOwned(a);
        for (r.grads, 0..) |g, i| {
            if (g.numel() != tangents[i].numel()) return Error.ShapeMismatch;
            const prod = try ops.mul(g, tangents[i]);
            const s = try ops.sumAll(prod);
            if (acc) |a| {
                const sum = try ops.add(a, s);
                dropOwned(a);
                acc = sum.retain();
            } else {
                acc = s.retain();
            }
        }
        const scalar = acc orelse try Tensor.zeros(.{}, floatOf(value));
        out_tangent.writeF64(j, scalar.item());
    }

    // Both results outlive this function's scope, so take a reference for the
    // caller (the scope drops the creation reference).
    return .{ .value = value.retain(), .tangent = out_tangent.retain() };
}

/// Drops a reference the transform itself took with `retain` (as opposed to a
/// scope owned tensor, which must never be deinit'ed by hand).
fn dropOwned(t: Tensor) void {
    var tt = t;
    tt.deinit();
}

fn floatOf(t: Tensor) DType {
    return if (t.dtype().isFloat()) t.dtype() else .f32;
}

fn oneHotCotangent(value: Tensor, j: usize) Error!Tensor {
    var t = try Tensor.zeros(value.shape, floatOf(value));
    t.writeF64(j, 1);
    return t;
}

/// `vmap`: vectorises `f` over the leading (payload) axis of every argument.
///
/// The arguments are marked as *batched* (`Tensor.batch_depth`) instead of being
/// materialised, so the body runs once for the whole batch. Ops keep the batch
/// axes leading: reductions without an axis skip them, `matmul` treats them as
/// batch dimensions, and axis arguments are payload relative.
pub fn vmap(
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    args: []const Tensor,
) !Tensor {
    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const buf = try alloc.alloc(Tensor, args.len);
    defer alloc.free(buf);
    @memcpy(buf, args);
    for (buf) |*tt| {
        if (tt.rank() == 0) return Error.ShapeMismatch;
        tt.* = tt.withBatchDepth(tt.batch_depth + 1);
        _ = tt.retainScoped();
    }

    context.startVmap();
    defer context.stopVmap();

    const out = try f(ud, buf);
    return out.retain();
}

/// `jit`: trace `f` once, then replay the resulting plan.
/// Balances the books on the value a traced function handed back.
///
/// Everything the function created while the trace's scope was active is
/// released by that scope, and every recording node holds one reference while it
/// lives — so a plain op result is fully accounted for. A result that came out
/// of a *transform* (`jit(grad(f))`, `jit(vmap(f))`) additionally carries the
/// owned reference the transform hands its caller; nothing else in the trace
/// will ever release it. Count what is accounted for and drop that one extra
/// reference here, so a traced function can be written exactly like a normal
/// caller of the transform.
fn releaseExtraRef(out: Tensor, graph: *const graph_mod.Graph) void {
    var accounted: usize = 0;
    var tracked = false;
    if (context.trackerPtr()) |tp| {
        const tr: *tmod.Tracker = @ptrCast(@alignCast(tp));
        for (tr.list.items) |it| {
            if (it.id == out.id) {
                accounted += 1;
                tracked = true;
            }
        }
    }
    // an untouched argument must never be released here
    if (!tracked) return;
    for (graph.nodes.items) |nd| {
        if (nd.out.id == out.id) accounted += 1;
    }
    for (context.tapes()) |raw| {
        const tp = raw orelse continue;
        const tape_ptr: *tape_mod.Tape = @ptrCast(@alignCast(tp));
        for (tape_ptr.nodes.items) |nd| {
            if (nd.out.id == out.id) accounted += 1;
        }
    }
    if (out.buf.rc.load(.monotonic) <= accounted) return;
    // a transform adds exactly one owned reference
    var extra = out;
    extra.deinit();
}

pub const Jit = struct {
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    plan: ?jitlib.Plan = null,
    /// How many times the function was traced (1 for the usual "trace once")
    trace_count: usize = 0,

    pub fn init(alloc: std.mem.Allocator, f: Fn, ud: ?*anyopaque) Jit {
        return .{ .alloc = alloc, .f = f, .ud = ud };
    }

    pub fn deinit(self: *Jit) void {
        if (self.plan) |*p| p.destroy();
        self.plan = null;
    }

    /// Runs the jitted function, tracing it on the first call.
    pub fn call(self: *Jit, args: []const Tensor) !Tensor {
        if (self.plan == null) try self.trace(args);
        return self.plan.?.run(self.alloc, args);
    }

    /// Forces a (re)trace with the given argument shapes.
    pub fn trace(self: *Jit, args: []const Tensor) !void {
        if (self.plan) |*p| p.destroy();
        self.plan = null;

        var scope = try Scope.init(self.alloc);
        defer scope.deinit();

        var graph = Graph.init(self.alloc);
        defer graph.deinit();

        const arg_ids = try self.alloc.alloc(u64, args.len);
        defer self.alloc.free(arg_ids);
        for (args, 0..) |a, i| arg_ids[i] = a.id;

        const saved = context.save();
        context.setGraph(&graph);
        context.setMode(.eager);
        defer context.restore(saved);

        const out = try self.f(self.ud, args);
        const out_id = out.id;
        releaseExtraRef(out, &graph);
        const out_ids = [_]u64{out_id};

        self.plan = try jitlib.build(self.alloc, &graph, arg_ids, &out_ids);
        self.trace_count += 1;
    }

    pub fn numSteps(self: *const Jit) usize {
        if (self.plan) |p| return p.numSteps();
        return 0;
    }
};

pub fn jitFn(alloc: std.mem.Allocator, f: Fn, ud: ?*anyopaque) Jit {
    return Jit.init(alloc, f, ud);
}

/// The full Jacobian of `f` at `args`, w.r.t. argument `wrt`.
///
/// For `f: R^n → R^m`, returns an `m × n` matrix (stored row-major, shape
/// `{value.numel(), args[wrt].numel()}`). Each row is the gradient of one
/// output element: `J[i, :] = ∂f_i/∂x`, computed by one VJP per output.
///
/// This is the same strategy JAX uses internally when `jax.jacobian` falls
/// back to per-element VJPs. It is `O(m)` evaluations of the VJP, so it is
/// efficient when the output is small relative to the input (the common case
/// for loss functions); for wide outputs, forward mode (`jvp` per input) is
/// cheaper. A future optimisation can switch automatically.
pub fn jacobian(
    alloc: std.mem.Allocator,
    f: Fn,
    ud: ?*anyopaque,
    args: []const Tensor,
    wrt: usize,
) !Tensor {
    if (wrt >= args.len) return Error.InvalidShape;

    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const indices = try allIndices(alloc, args.len);
    defer alloc.free(indices);

    // Evaluate once to learn the output shape and size.
    // The function returns a scope-owned tensor (via retainScoped).
    const value = try f(ud, args);
    const m = value.numel();
    const n = args[wrt].numel();
    if (m == 0 or n == 0) return Error.ShapeMismatch;

    const out_shape = try Shape.init(&.{ m, n });
    var jac = try Tensor.zeros(out_shape, floatOf(value));

    var j: usize = 0;
    while (j < m) : (j += 1) {
        const cot = try oneHotCotangent(value, j);
        var r = try vjp(alloc, f, ud, args, indices, cot);
        defer r.deinit();

        const g = r.grads[0];
        for (0..n) |k| {
            jac.writeF64(j * n + k, g.readLogical(n, k));
        }
    }

    return jac.retain();
}
