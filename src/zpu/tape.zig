//! The reverse-mode autodiff tape.
//!
//! A tape is a flat, append-only list of `Node`s. Nodes are appended by the op
//! engine whenever `context.tapeActive()` is true, and consumed by
//! `autodiff.zig` for the backward sweep. Because a node stores *copies* of the
//! input tensors and of the output tensor, the traced region owns its own
//! references: the tensors recorded on the tape stay alive as long as the tape
//! does.
const std = @import("std");
const tmod = @import("tensor.zig");
const prims = @import("prims.zig");
const context = @import("context.zig");

pub const Tensor = tmod.Tensor;
pub const Prim = prims.Prim;
pub const Attrs = prims.Attrs;

/// Maximum number of inputs a primitive takes (`where` is the widest).
pub const MAX_INPUTS = 3;

pub const Node = struct {
    prim: Prim,
    inputs: [MAX_INPUTS]Tensor = undefined,
    n_inputs: u8 = 0,
    out: Tensor,
    attrs: Attrs = .{},

    pub fn input(self: *const Node, i: usize) Tensor {
        return self.inputs[i];
    }
};

pub const Tape = struct {
    alloc: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(Node) = .empty,
    /// tensor id -> index into `nodes`, for the sweep
    producer: std.AutoHashMapUnmanaged(u64, usize) = .empty,

    pub fn init(alloc: std.mem.Allocator) Tape {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *Tape) void {
        for (self.nodes.items) |n| {
            for (0..n.n_inputs) |i| {
                var t = n.inputs[i];
                t.deinit();
            }
            var o = n.out;
            o.deinit();
        }
        self.nodes.deinit(self.alloc);
        self.producer.deinit(self.alloc);
    }

    pub fn len(self: *const Tape) usize {
        return self.nodes.items.len;
    }

    /// The node that produced `id`, if any.
    pub fn nodeOf(self: *const Tape, id: u64) ?*const Node {
        const idx = self.producer.get(id) orelse return null;
        return &self.nodes.items[idx];
    }

    fn append(self: *Tape, node: Node) Error!void {
        const idx = self.nodes.items.len;
        try self.producer.put(self.alloc, node.out.id, idx);
        try self.nodes.append(self.alloc, node);
    }
};

pub const Error = error{OutOfMemory};

pub const RecordArgs = struct {
    prim: Prim,
    inputs: []const Tensor,
    out: Tensor,
    attrs: Attrs = .{},
};

/// Appends `args` to the tape installed in the current context. Does nothing
/// (and succeeds) when no tape is active.
pub fn record(prim: Prim, inputs: []const Tensor, out: Tensor, attrs: Attrs) Error!void {
    if (!context.tapeActive()) return;
    // Record on *every* installed tape, not just the innermost one: an outer
    // `grad`/`jit` recorder has to see the whole inner computation (forward
    // pass included) or higher order derivatives lose their graph. Each tape
    // owns its own references, so the stacks stay independent.
    for (context.tapes()) |slot| {
        const p = slot orelse continue;
        const t: *Tape = @ptrCast(@alignCast(p));
        var node = Node{ .prim = prim, .out = out.retain(), .attrs = attrs };
        const n = @min(inputs.len, MAX_INPUTS);
        for (0..n) |i| node.inputs[i] = inputs[i].retain();
        node.n_inputs = @intCast(n);
        try t.append(node);
    }
}

/// The tape installed in the current context, if any.
pub fn activeTape() ?*Tape {
    const p = context.tapePtr() orelse return null;
    return @ptrCast(@alignCast(p));
}

/// Installs (or clears) the active tape, returning the previous one.
pub fn setActiveTape(t: ?*Tape) ?*anyopaque {
    const prev = context.tapePtr();
    context.setTape(if (t) |tt| @ptrCast(tt) else null);
    return prev;
}

test "tape records nodes" {
    const alloc = std.testing.allocator;
    var tape = Tape.init(alloc);
    defer tape.deinit();

    const a = try Tensor.full(.{}, .f64, 1.5);
    defer a.deinit();
    const b = try Tensor.full(.{}, .f64, 2.0);
    defer b.deinit();
    const out = try Tensor.full(.{}, .f64, 3.5);
    defer out.deinit();

    const prev = setActiveTape(&tape);
    defer _ = setActiveTape(if (prev) |p| @ptrCast(@alignCast(p)) else null);
    // recording is gated on `.grad` mode
    context.setMode(.grad);
    defer context.setMode(.eager);
    try record(.add, &.{ a, b }, out, .{});
    try std.testing.expectEqual(@as(usize, 1), tape.len());
    try std.testing.expect(tape.nodeOf(out.id) != null);
}
