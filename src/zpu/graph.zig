//! Trace recorder for the JIT.
//!
//! A `Graph` is a flat list of the primitives that were executed while it was
//! installed. Ops always record when a graph is installed (regardless of the
//! execution mode), so `jit(grad(f))` traces the *backward pass* as well.
const std = @import("std");
const tmod = @import("tensor.zig");
const prims = @import("prims.zig");
const context = @import("context.zig");

pub const Tensor = tmod.Tensor;
pub const Prim = prims.Prim;
pub const Attrs = prims.Attrs;
pub const MAX_INPUTS = 3;

pub const Node = struct {
    prim: Prim,
    inputs: [MAX_INPUTS]Tensor = undefined,
    n_inputs: u8 = 0,
    out: Tensor,
    attrs: Attrs = .{},
};

pub const Graph = struct {
    alloc: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(Node) = .empty,

    pub fn init(alloc: std.mem.Allocator) Graph {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *Graph) void {
        for (self.nodes.items) |n| {
            for (0..n.n_inputs) |i| {
                var t = n.inputs[i];
                t.deinit();
            }
            var o = n.out;
            o.deinit();
        }
        self.nodes.deinit(self.alloc);
    }

    pub fn len(self: *const Graph) usize {
        return self.nodes.items.len;
    }

    pub fn append(self: *Graph, node: Node) !void {
        try self.nodes.append(self.alloc, node);
    }

    /// `true` when `id` is produced by some node in the graph.
    pub fn producedBy(self: *const Graph, id: u64) bool {
        for (self.nodes.items) |*n| {
            if (n.out.id == id) return true;
        }
        return false;
    }

    /// Counts the distinct tensor ids that are consumed but never produced:
    /// these are the function's arguments (and its captured constants).
    pub fn externalIds(self: *const Graph, alloc: std.mem.Allocator) ![]u64 {
        var list: std.ArrayListUnmanaged(u64) = .empty;
        errdefer list.deinit(alloc);
        for (self.nodes.items) |*n| {
            for (0..n.n_inputs) |i| {
                const id = n.inputs[i].id;
                if (self.producedBy(id)) continue;
                var seen = false;
                for (list.items) |e| {
                    if (e == id) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) try list.append(alloc, id);
            }
        }
        return list.toOwnedSlice(alloc);
    }
};

pub fn record(prim: Prim, inputs: []const Tensor, out: Tensor, attrs: Attrs) void {
    const g = activeGraph() orelse return;
    var node = Node{ .prim = prim, .out = out, .attrs = attrs };
    const n = @min(inputs.len, MAX_INPUTS);
    for (0..n) |i| node.inputs[i] = inputs[i];
    node.n_inputs = @intCast(n);
    // The graph owns its own references so that a trace survives the scope
    // that produced it.
    _ = node.out.retain();
    for (0..n) |i| {
        node.inputs[i] = node.inputs[i].retain();
    }
    g.append(node) catch {};
}

pub fn activeGraph() ?*Graph {
    const p = context.graphPtr() orelse return null;
    return @ptrCast(@alignCast(p));
}

pub fn setActiveGraph(g: ?*Graph) ?*anyopaque {
    const prev = context.graphPtr();
    context.setGraph(if (g) |gg| @ptrCast(gg) else null);
    return prev;
}

test "graph records and releases" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();

    var g = Graph.init(alloc);
    defer g.deinit();

    const prev = setActiveGraph(&g);
    defer _ = setActiveGraph(if (prev) |p| @ptrCast(@alignCast(p)) else null);

    const a = try Tensor.full(.{}, .f64, 2.0);
    const b = try Tensor.full(.{}, .f64, 3.0);
    const ops = @import("ops.zig");
    const c = try ops.add(a, b);
    try std.testing.expectEqual(@as(usize, 1), g.len());
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), c.item(), 1e-12);
}
