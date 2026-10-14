//! Just-in-time compilation: `graph.Graph` → `Plan`.
//!
//! `build` turns a trace into an executable plan:
//!
//! * **constant folding** — every node whose inputs are all constants (weights,
//!   captured tensors, literals) is evaluated once at build time and stored as a
//!   `Blob`,
//! * **dead code elimination** — only the nodes that the requested outputs
//!   actually depend on are kept.
//!
//! `run` then binds the arguments by position and executes the remaining steps.
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");
const prims = @import("prims.zig");
const context = @import("context.zig");
const graph_mod = @import("graph.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = tmod.Shape;
pub const Error = tmod.Error;
pub const Prim = prims.Prim;
pub const Attrs = prims.Attrs;
pub const Scope = tmod.Scope;
pub const Graph = graph_mod.Graph;

pub const MAX_INPUTS = graph_mod.MAX_INPUTS;

pub const Step = struct {
    prim: Prim,
    ins: [MAX_INPUTS]u64 = @splat(0),
    n_inputs: u8 = 0,
    out: u64,
    attrs: Attrs = .{},
};

/// A value that was folded away at build time.
pub const Blob = struct {
    id: u64,
    tensor: Tensor,
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    arg_ids: []u64,
    out_ids: []u64,
    steps: []Step,
    blobs: []Blob,
    /// statistics, handy for `describe`
    traced: usize = 0,
    folded: usize = 0,
    eliminated: usize = 0,

    pub fn destroy(self: *Plan) void {
        for (self.blobs) |*b| {
            var t = b.tensor;
            t.deinit();
        }
        self.alloc.free(self.blobs);
        self.alloc.free(self.steps);
        self.alloc.free(self.arg_ids);
        self.alloc.free(self.out_ids);
        self.* = undefined;
    }

    pub fn numSteps(self: *const Plan) usize {
        return self.steps.len;
    }

    pub fn describe(self: *const Plan, writer: anytype) !void {
        try writer.print("plan: {d} traced ops -> {d} steps, {d} folded constants, {d} eliminated\n", .{
            self.traced, self.steps.len, self.folded, self.eliminated,
        });
        for (self.steps) |st| {
            try writer.print("  %{d} = {s}(", .{ st.out, @tagName(st.prim) });
            for (0..st.n_inputs) |i| {
                if (i != 0) try writer.writeAll(", ");
                try writer.print("%{d}", .{st.ins[i]});
            }
            try writer.writeAll(")\n");
        }
    }

    /// Executes the plan and returns the first output (caller owns it).
    pub fn run(self: *const Plan, alloc: std.mem.Allocator, args: []const Tensor) !Tensor {
        const outs = try self.runAll(alloc, args);
        defer alloc.free(outs);
        var i: usize = 1;
        while (i < outs.len) : (i += 1) {
            var o = outs[i];
            o.deinit();
        }
        return outs[0];
    }

    /// Executes the plan, returning every requested output (caller frees the
    /// slice *and* the tensors).
    pub fn runAll(self: *const Plan, alloc: std.mem.Allocator, args: []const Tensor) ![]Tensor {
        if (args.len != self.arg_ids.len) return Error.ShapeMismatch;
        var scope = try Scope.init(alloc);
        defer scope.deinit();

        var map: std.AutoHashMapUnmanaged(u64, Tensor) = .empty;
        defer map.deinit(alloc);

        for (args, 0..) |a, i| try map.put(alloc, self.arg_ids[i], a);
        for (self.blobs) |b| try map.put(alloc, b.id, b.tensor);

        for (self.steps) |st| {
            var ins: [MAX_INPUTS]Tensor = undefined;
            for (0..st.n_inputs) |k| {
                ins[k] = map.get(st.ins[k]) orelse return Error.ShapeMismatch;
            }
            const out = try ops.applyPrim(st.prim, ins[0..st.n_inputs], st.attrs);
            try map.put(alloc, st.out, out);
        }

        const outs = try alloc.alloc(Tensor, self.out_ids.len);
        errdefer alloc.free(outs);
        for (self.out_ids, 0..) |id, i| {
            const t = map.get(id) orelse return Error.ShapeMismatch;
            outs[i] = t.retain();
        }
        return outs;
    }
};

/// Builds a plan from a trace.
pub fn build(
    alloc: std.mem.Allocator,
    g: *const Graph,
    arg_ids: []const u64,
    out_ids: []const u64,
) !Plan {
    const n_nodes = g.nodes.items.len;

    // ---- 1. relevance: walk back from the requested outputs -----------------
    var needed = try alloc.alloc(bool, n_nodes);
    defer alloc.free(needed);
    @memset(needed, false);

    var produced: std.AutoHashMapUnmanaged(u64, usize) = .empty;
    defer produced.deinit(alloc);
    for (g.nodes.items, 0..) |*node, i| {
        try produced.put(alloc, node.out.id, i);
    }

    // a work list of ids whose producers still have to be marked
    var pending: std.ArrayListUnmanaged(u64) = .empty;
    defer pending.deinit(alloc);
    for (out_ids) |id| try pending.append(alloc, id);
    while (pending.pop()) |id| {
        const idx = produced.get(id) orelse continue;
        if (needed[idx]) continue;
        needed[idx] = true;
        const node = &g.nodes.items[idx];
        for (0..node.n_inputs) |k| try pending.append(alloc, node.inputs[k].id);
    }

    // ---- 2. constant detection ---------------------------------------------
    var is_arg: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer is_arg.deinit(alloc);
    for (arg_ids) |id| try is_arg.put(alloc, id, {});

    var is_const = try alloc.alloc(bool, n_nodes);
    defer alloc.free(is_const);
    for (g.nodes.items, 0..) |*node, i| {
        var c = true;
        for (0..node.n_inputs) |k| {
            const id = node.inputs[k].id;
            if (is_arg.contains(id)) {
                c = false;
                break;
            }
            if (produced.get(id)) |pi| {
                if (!is_const[pi]) {
                    c = false;
                    break;
                }
            }
        }
        is_const[i] = c and needed[i];
    }

    // ---- 3. fold the constants ---------------------------------------------
    var blobs: std.ArrayListUnmanaged(Blob) = .empty;
    errdefer {
        for (blobs.items) |*b| {
            var t = b.tensor;
            t.deinit();
        }
        blobs.deinit(alloc);
    }
    var const_map: std.AutoHashMapUnmanaged(u64, Tensor) = .empty;
    defer const_map.deinit(alloc);

    for (g.nodes.items, 0..) |*node, i| {
        if (!is_const[i]) continue;
        var ins: [MAX_INPUTS]Tensor = undefined;
        for (0..node.n_inputs) |k| {
            const id = node.inputs[k].id;
            ins[k] = const_map.get(id) orelse node.inputs[k];
        }
        // `evalFolded` hands back one reference; the blob keeps it and the map
        // only borrows it (the map is freed before the blobs are).
        const value = try evalFolded(alloc, node, ins[0..node.n_inputs]);
        try blobs.append(alloc, .{ .id = node.out.id, .tensor = value });
        try const_map.put(alloc, node.out.id, value);
    }

    // ---- 4. captured tensors -----------------------------------------------
    // Anything a kept step reads that is neither an argument nor produced by the
    // trace (weights, literals created inside the function) is a constant.
    for (g.nodes.items, 0..) |*node, i| {
        if (!needed[i] or is_const[i]) continue;
        for (0..node.n_inputs) |k| {
            const id = node.inputs[k].id;
            if (is_arg.contains(id)) continue;
            if (produced.contains(id)) continue;
            if (const_map.contains(id)) continue;
            const held = node.inputs[k].retain();
            try blobs.append(alloc, .{ .id = id, .tensor = held });
            try const_map.put(alloc, id, held);
        }
    }

    // ---- 5. the remaining steps --------------------------------------------
    var steps: std.ArrayListUnmanaged(Step) = .empty;
    errdefer steps.deinit(alloc);
    var folded: usize = 0;
    for (g.nodes.items, 0..) |*node, i| {
        if (!needed[i]) continue;
        if (is_const[i]) {
            folded += 1;
            continue;
        }
        var st = Step{ .prim = node.prim, .out = node.out.id, .attrs = node.attrs };
        st.n_inputs = node.n_inputs;
        for (0..node.n_inputs) |k| st.ins[k] = node.inputs[k].id;
        try steps.append(alloc, st);
    }

    const traced = g.nodes.items.len;
    const n_steps = steps.items.len;
    return .{
        .alloc = alloc,
        .arg_ids = try alloc.dupe(u64, arg_ids),
        .out_ids = try alloc.dupe(u64, out_ids),
        .steps = try steps.toOwnedSlice(alloc),
        .blobs = try blobs.toOwnedSlice(alloc),
        .traced = traced,
        .folded = folded,
        .eliminated = traced - n_steps - folded,
    };
}

/// Evaluates a constant node without recording anything.
fn evalFolded(alloc: std.mem.Allocator, node: *const graph_mod.Node, inputs: []const Tensor) !Tensor {
    const saved = context.save();
    context.setMode(.compiled);
    context.setTape(null);
    context.setGraph(null);
    defer context.restore(saved);

    var scope = try Scope.init(alloc);
    defer scope.deinit();

    const out = try ops.applyPrim(node.prim, inputs, node.attrs);
    return out.retain();
}

test "jit folds constants and eliminates dead code" {
    const alloc = std.testing.allocator;
    const zpu = @import("../zpu.zig");

    const Ctx = struct {
        fn f(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
            _ = ud;
            // a constant that gets folded away
            const c = try zpu.Tensor.ones(args[0].shape, args[0].dtype());
            _ = try zpu.ops.mul(args[1], c);
            // the live computation
            return zpu.ops.add(args[0], args[0]);
        }
    };

    var j = zpu.jit(alloc, Ctx.f, null);
    defer j.deinit();

    const x = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, zpu.Shape.initComptime(&.{4}));
    defer x.deinit();
    const y = try zpu.Tensor.fromArray(f64, &.{ 5, 6, 7, 8 }, zpu.Shape.initComptime(&.{4}));
    defer y.deinit();

    const out = try j.call(&.{ x, y });
    defer out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2), out.readF64(0), 1e-12);
    try std.testing.expectEqual(@as(usize, 1), j.trace_count);
    try std.testing.expect(j.numSteps() <= 2);

    // replaying produces the same result
    const out2 = try j.call(&.{ x, y });
    defer out2.deinit();
    try std.testing.expect(out.allClose(out2, 1e-12));
    try std.testing.expectEqual(@as(usize, 1), j.trace_count);
}
