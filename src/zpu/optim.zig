//! Optimisers. Every optimiser updates parameter tensors in place from the
//! gradients produced by `zpu.grad`, which is exactly how the JAX ecosystem is
//! used in practice (`grad` + an explicit update rule).
//!
//! ```zig
//! var opt = try zpu.optim.Adam.init(alloc, .{ .lr = 1e-2 });
//! defer opt.deinit();
//! const grads = try zpu.grad(alloc, loss, null, params.items, all_indices);
//! try opt.step(params.items, grads);
//! ```
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Error = tmod.Error;

/// `p -= delta` in place (parameters are user owned and contiguous).
pub fn subInPlace(p: *Tensor, delta: Tensor) Error!void {
    if (!p.isContiguous()) return Error.NonContiguous;
    var tmp: ?Tensor = null;
    defer if (tmp) |tt| ops.releaseTemp(tt);
    const d = if (delta.dtype() != p.dtype()) blk: {
        const c = try ops.cast(delta, p.dtype());
        tmp = c;
        break :blk c;
    } else delta;
    const n = p.numel();
    for (0..n) |i| p.writeF64(i, p.readF64(i) - d.readF64(i));
}

/// `p += delta` in place.
pub fn addInPlace(p: *Tensor, delta: Tensor) Error!void {
    if (!p.isContiguous()) return Error.NonContiguous;
    var tmp: ?Tensor = null;
    defer if (tmp) |tt| ops.releaseTemp(tt);
    const d = if (delta.dtype() != p.dtype()) blk: {
        const c = try ops.cast(delta, p.dtype());
        tmp = c;
        break :blk c;
    } else delta;
    const n = p.numel();
    for (0..n) |i| p.writeF64(i, p.readF64(i) + d.readF64(i));
}

/// Zeroes a tensor in place.
pub fn zeroInPlace(p: *Tensor) Error!void {
    if (!p.isContiguous()) return Error.NonContiguous;
    p.buf.zero();
}

/// A growable list of parameter tensors (with the names used in the examples).
pub const Params = struct {
    items: std.ArrayListUnmanaged(Tensor) = .empty,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) Params {
        return .{ .alloc = alloc };
    }
    pub fn deinit(self: *Params) void {
        self.items.deinit(self.alloc);
    }
    pub fn add(self: *Params, t: Tensor) !void {
        try self.items.append(self.alloc, t);
    }
    /// `wrt` indices for `zpu.grad`, i.e. every parameter.
    pub fn allIndices(self: *const Params, alloc: std.mem.Allocator) ![]usize {
        const out = try alloc.alloc(usize, self.items.items.len);
        for (out, 0..) |*v, i| v.* = i;
        return out;
    }
};

pub const SgdConfig = struct {
    lr: f64 = 1e-2,
    momentum: f64 = 0.0,
    weight_decay: f64 = 0.0,
    nesterov: bool = false,
};

pub const Sgd = struct {
    cfg: SgdConfig,
    /// momentum buffers (allocated on the first step)
    velocity: std.ArrayListUnmanaged(Tensor) = .empty,
    alloc: std.mem.Allocator,
    step_count: usize = 0,

    pub fn init(alloc: std.mem.Allocator, cfg: SgdConfig) Sgd {
        return .{ .alloc = alloc, .cfg = cfg };
    }

    pub fn deinit(self: *Sgd) void {
        for (self.velocity.items) |v| {
            var vv = v;
            vv.deinit();
        }
        self.velocity.deinit(self.alloc);
    }

    pub fn step(self: *Sgd, params: []Tensor, grads: []const Tensor) !void {
        std.debug.assert(params.len == grads.len);
        if (self.cfg.momentum > 0 and self.velocity.items.len == 0) {
            for (params) |p| {
                const zeros = try Tensor.zeros(p.shape, .f32);
                tmod.untrack(zeros); // state outlives any active scope
                try self.velocity.append(self.alloc, zeros);
            }
        }
        var scope = try tmod.Scope.init(self.alloc);
        defer scope.deinit();

        for (params, 0..) |*p, i| {
            var g = grads[i];
            if (self.cfg.weight_decay != 0) {
                const wd = try ops.mulScalar(p.*, self.cfg.weight_decay);
                defer ops.releaseTemp(wd);
                g = try ops.add(g, wd);
            }
            if (self.cfg.momentum > 0) {
                const v = &self.velocity.items[i];
                // v = m v + g
                const scaled = try ops.mulScalar(v.*, self.cfg.momentum);
                defer ops.releaseTemp(scaled);
                const nv = try ops.add(scaled, g);
                defer ops.releaseTemp(nv);
                try copyInto(v, nv);
                if (self.cfg.nesterov) {
                    const nv2 = try ops.add(g, try ops.mulScalar(nv, self.cfg.momentum));
                    defer ops.releaseTemp(nv2);
                    try self.applyDelta(p, nv2);
                } else {
                    try self.applyDelta(p, nv);
                }
            } else {
                try self.applyDelta(p, g);
            }
        }
        self.step_count += 1;
    }

    fn applyDelta(self: *Sgd, p: *Tensor, g: Tensor) !void {
        const delta = try ops.mulScalar(g, self.cfg.lr);
        defer ops.releaseTemp(delta);
        try subInPlace(p, delta);
    }
};

pub const AdamConfig = struct {
    lr: f64 = 1e-3,
    beta1: f64 = 0.9,
    beta2: f64 = 0.999,
    eps: f64 = 1e-8,
    weight_decay: f64 = 0.0,
    /// Decoupled weight decay (AdamW) instead of adding it to the gradient.
    decoupled_decay: bool = false,
};

pub const Adam = struct {
    cfg: AdamConfig,
    m: std.ArrayListUnmanaged(Tensor) = .empty,
    v: std.ArrayListUnmanaged(Tensor) = .empty,
    alloc: std.mem.Allocator,
    t: usize = 0,

    pub fn init(alloc: std.mem.Allocator, cfg: AdamConfig) Adam {
        return .{ .alloc = alloc, .cfg = cfg };
    }

    pub fn deinit(self: *Adam) void {
        for (self.m.items) |x| {
            var xx = x;
            xx.deinit();
        }
        for (self.v.items) |x| {
            var xx = x;
            xx.deinit();
        }
        self.m.deinit(self.alloc);
        self.v.deinit(self.alloc);
    }

    pub fn step(self: *Adam, params: []Tensor, grads: []const Tensor) !void {
        std.debug.assert(params.len == grads.len);
        if (self.m.items.len == 0) {
            for (params) |p| {
                const m0 = try Tensor.zeros(p.shape, .f32);
                const v0 = try Tensor.zeros(p.shape, .f32);
                tmod.untrack(m0); // state outlives any active scope
                tmod.untrack(v0);
                try self.m.append(self.alloc, m0);
                try self.v.append(self.alloc, v0);
            }
        }
        self.t += 1;
        var scope = try tmod.Scope.init(self.alloc);
        defer scope.deinit();

        const bc1 = 1.0 - std.math.pow(f64, self.cfg.beta1, @floatFromInt(self.t));
        const bc2 = 1.0 - std.math.pow(f64, self.cfg.beta2, @floatFromInt(self.t));

        for (params, 0..) |*p, i| {
            const g = grads[i];
            const m = &self.m.items[i];
            const v = &self.v.items[i];

            // m = b1 m + (1 - b1) g
            const m1 = try ops.mulScalar(m.*, self.cfg.beta1);
            defer ops.releaseTemp(m1);
            const m2 = try ops.mulScalar(g, 1.0 - self.cfg.beta1);
            defer ops.releaseTemp(m2);
            const new_m = try ops.add(m1, m2);
            defer ops.releaseTemp(new_m);
            try copyInto(m, new_m);

            // v = b2 v + (1 - b2) g^2
            const sq = try ops.square(g);
            defer ops.releaseTemp(sq);
            const v1 = try ops.mulScalar(v.*, self.cfg.beta2);
            defer ops.releaseTemp(v1);
            const v2 = try ops.mulScalar(sq, 1.0 - self.cfg.beta2);
            defer ops.releaseTemp(v2);
            const new_v = try ops.add(v1, v2);
            defer ops.releaseTemp(new_v);
            try copyInto(v, new_v);

            // p -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
            const mhat = try ops.divScalar(new_m, bc1);
            defer ops.releaseTemp(mhat);
            const vhat = try ops.divScalar(new_v, bc2);
            defer ops.releaseTemp(vhat);
            const den = try ops.add(try ops.sqrt(vhat), try ops.scalarLike(vhat, self.cfg.eps));
            defer ops.releaseTemp(den);
            const upd = try ops.div(mhat, den);
            defer ops.releaseTemp(upd);
            const delta = try ops.mulScalar(upd, self.cfg.lr);
            defer ops.releaseTemp(delta);
            try subInPlace(p, delta);

            if (self.cfg.weight_decay != 0) {
                if (self.cfg.decoupled_decay) {
                    const decay = try ops.mulScalar(p.*, self.cfg.lr * self.cfg.weight_decay);
                    defer ops.releaseTemp(decay);
                    try subInPlace(p, decay);
                } else {
                    // classic L2: added to the gradient before the moments
                    const l2 = try ops.mulScalar(p.*, self.cfg.weight_decay);
                    defer ops.releaseTemp(l2);
                    const gi = try ops.add(g, l2);
                    defer ops.releaseTemp(gi);
                    const c = try ops.mulScalar(gi, self.cfg.lr);
                    defer ops.releaseTemp(c);
                    try subInPlace(p, c);
                }
            }
        }
        _ = &self.t;
    }
};

fn copyInto(dst: *Tensor, src: Tensor) !void {
    const n = dst.numel();
    for (0..n) |i| dst.writeF64(i, src.readF64(i));
}

pub const RmsPropConfig = struct {
    lr: f64 = 1e-3,
    decay: f64 = 0.99,
    eps: f64 = 1e-8,
};

pub const RmsProp = struct {
    cfg: RmsPropConfig,
    v: std.ArrayListUnmanaged(Tensor) = .empty,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, cfg: RmsPropConfig) RmsProp {
        return .{ .alloc = alloc, .cfg = cfg };
    }

    pub fn deinit(self: *RmsProp) void {
        for (self.v.items) |x| {
            var xx = x;
            xx.deinit();
        }
        self.v.deinit(self.alloc);
    }

    pub fn step(self: *RmsProp, params: []Tensor, grads: []const Tensor) !void {
        if (self.v.items.len == 0) {
            for (params) |p| {
                const v0 = try Tensor.zeros(p.shape, .f32);
                tmod.untrack(v0); // state outlives any active scope
                try self.v.append(self.alloc, v0);
            }
        }
        var scope = try tmod.Scope.init(self.alloc);
        defer scope.deinit();
        for (params, 0..) |*p, i| {
            const g = grads[i];
            const v = &self.v.items[i];
            const sq = try ops.square(g);
            defer ops.releaseTemp(sq);
            const a = try ops.mulScalar(v.*, self.cfg.decay);
            defer ops.releaseTemp(a);
            const b = try ops.mulScalar(sq, 1.0 - self.cfg.decay);
            defer ops.releaseTemp(b);
            const nv = try ops.add(a, b);
            defer ops.releaseTemp(nv);
            try copyInto(v, nv);
            const den = try ops.add(try ops.sqrt(nv), try ops.scalarLike(nv, self.cfg.eps));
            defer ops.releaseTemp(den);
            const upd = try ops.div(g, den);
            defer ops.releaseTemp(upd);
            const delta = try ops.mulScalar(upd, self.cfg.lr);
            defer ops.releaseTemp(delta);
            try subInPlace(p, delta);
        }
    }
};

/// A cosine decay schedule with linear warmup.
pub fn cosineLr(base_lr: f64, step_: usize, warmup: usize, total: usize, min_lr: f64) f64 {
    if (step_ < warmup) {
        return base_lr * @as(f64, @floatFromInt(step_ + 1)) / @as(f64, @floatFromInt(@max(warmup, 1)));
    }
    const t = @as(f64, @floatFromInt(step_ - warmup)) / @as(f64, @floatFromInt(@max(total - warmup, 1)));
    const c = 0.5 * (1.0 + @cos(std.math.pi * @min(t, 1.0)));
    return min_lr + (base_lr - min_lr) * c;
}
