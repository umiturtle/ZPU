//! Random number generation (splitmix64 + Box-Muller), producing ZPU tensors.
//!
//! The generator is an explicit value (`Rng`) so results are reproducible and
//! thread independent; `zpu.rng.seed`/`zpu.rng.randn` use a thread-local
//! default for convenience.
const std = @import("std");
const tmod = @import("tensor.zig");
const shape_mod = @import("shape.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = shape_mod.Shape;
pub const Error = tmod.Error;

pub const Rng = struct {
    state: u64,

    pub fn init(seed_value: u64) Rng {
        return .{ .state = seed_value };
    }

    /// splitmix64: fast, well distributed, and trivially reproducible.
    pub fn next(self: *Rng) u64 {
        self.state +%= 0x9E3779B97F4A7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    /// Uniform in [0, 1).
    pub fn nextF64(self: *Rng) f64 {
        const bits = self.next() >> 11;
        return @as(f64, @floatFromInt(bits)) * (1.0 / 9007199254740992.0);
    }

    /// Uniform in [0, 1) as a tensor.
    pub fn uniform(self: *Rng, s: Shape, dt: DType) Error!Tensor {
        var out = try Tensor.empty(s, dt);
        const n = out.numel();
        for (0..n) |i| out.writeF64(i, self.nextF64());
        return out;
    }

    /// Uniform in [low, high).
    pub fn uniformRange(self: *Rng, s: Shape, dt: DType, low: f64, high: f64) Error!Tensor {
        var out = try Tensor.empty(s, dt);
        const n = out.numel();
        for (0..n) |i| out.writeF64(i, low + (high - low) * self.nextF64());
        return out;
    }

    /// Standard normal, sampled per element with the polar Box-Muller method.
    pub fn normal(self: *Rng, s: Shape, dt: DType) Error!Tensor {
        var out = try Tensor.empty(s, dt);
        const n = out.numel();
        var i: usize = 0;
        while (i < n) {
            const a1 = @max(self.nextF64(), 1e-12);
            const a2 = self.nextF64();
            const r = @sqrt(-2.0 * @log(a1));
            const theta = 2.0 * std.math.pi * a2;
            out.writeF64(i, r * @cos(theta));
            i += 1;
            if (i < n) {
                out.writeF64(i, r * @sin(theta));
                i += 1;
            }
        }
        return out;
    }

    /// Truncated normal (used by `nn.init.truncNormal`).
    pub fn truncatedNormal(self: *Rng, s: Shape, dt: DType, stddev: f64, lo: f64, hi: f64) Error!Tensor {
        var out = try Tensor.empty(s, dt);
        const n = out.numel();
        for (0..n) |i| {
            var v: f64 = 0;
            var guard: usize = 0;
            while (guard < 64) : (guard += 1) {
                const b1 = @max(self.nextF64(), 1e-12);
                const b2 = self.nextF64();
                const r = @sqrt(-2.0 * @log(b1));
                v = r * @cos(2.0 * std.math.pi * b2) * stddev;
                if (v >= lo and v <= hi) break;
            }
            out.writeF64(i, v);
        }
        return out;
    }

    /// Integer samples in `[low, high)`.
    pub fn randint(self: *Rng, s: Shape, low: i64, high: i64) Error!Tensor {
        if (high <= low) return Error.ShapeMismatch;
        var out = try Tensor.empty(s, .i64);
        const span: u64 = @intCast(high - low);
        for (0..out.numel()) |i| {
            const v: i64 = low + @as(i64, @intCast(self.next() % span));
            out.writeF64(i, @floatFromInt(v));
        }
        return out;
    }

    /// A random permutation of `0..n`.
    pub fn permutation(self: *Rng, n: usize) Error!Tensor {
        var out = try Tensor.empty(try Shape.init(&.{n}), .i64);
        for (0..n) |i| out.writeF64(i, @floatFromInt(i));
        const data = out.flatSliceMut(i64).?;
        var i = n;
        while (i > 1) {
            i -= 1;
            const j = self.next() % (i + 1);
            const tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
        return out;
    }
};

// ---- thread-local convenience API -------------------------------------------

threadlocal var default_rng: ?Rng = null;

pub fn seed(s: u64) void {
    default_rng = Rng.init(s);
}

fn rng() *Rng {
    if (default_rng == null) default_rng = Rng.init(0x5EED_1234_ABCD_0001);
    return &default_rng.?;
}

pub fn uniform(s: Shape, dt: DType) Error!Tensor {
    return rng().uniform(s, dt);
}
pub fn normal(s: Shape, dt: DType) Error!Tensor {
    return rng().normal(s, dt);
}
pub fn randn(s: Shape, dt: DType) Error!Tensor {
    return rng().normal(s, dt);
}
pub fn randint(s: Shape, low: i64, high: i64) Error!Tensor {
    return rng().randint(s, low, high);
}
pub fn permutation(n: usize) Error!Tensor {
    return rng().permutation(n);
}

test "rng reproducibility" {
    var a = Rng.init(42);
    var b = Rng.init(42);
    try std.testing.expectEqual(a.next(), b.next());
    var c = Rng.init(43);
    try std.testing.expect(a.next() != c.next());

    const alloc = std.testing.allocator;
    var n = try a.normal(try Shape.init(&.{1000}), .f32);
    defer n.deinit();
    // mean ~ 0, stddev ~ 1
    var sum: f64 = 0;
    var sq: f64 = 0;
    for (0..n.numel()) |i| {
        const v = n.readF64(i);
        sum += v;
        sq += v * v;
    }
    const mean = sum / 1000.0;
    const var_ = sq / 1000.0 - mean * mean;
    try std.testing.expect(@abs(mean) < 0.12);
    try std.testing.expect(@abs(var_ - 1.0) < 0.15);
    _ = alloc;
}
