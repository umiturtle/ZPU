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

    /// Bernoulli samples: 1 with probability p, 0 otherwise.
    pub fn bernoulli(self: *Rng, s: Shape, p: f64) Error!Tensor {
        if (p < 0.0 or p > 1.0) return Error.ShapeMismatch;
        var out = try Tensor.empty(s, .f32);
        for (0..out.numel()) |i| {
            const v: f64 = if (self.nextF64() < p) 1.0 else 0.0;
            out.writeF64(i, v);
        }
        return out;
    }

    /// Exponential distribution with rate 1.
    pub fn exponential(self: *Rng, s: Shape, dt: DType) Error!Tensor {
        var out = try Tensor.empty(s, dt);
        for (0..out.numel()) |i| {
            const u = @max(self.nextF64(), 1e-12);
            out.writeF64(i, -@log(u));
        }
        return out;
    }

    /// Gamma distribution with shape parameter `a` and rate 1.
    /// Uses Marsaglia and Tsang's method for a >= 1.
    pub fn gamma(self: *Rng, s: Shape, a: f64, dt: DType) Error!Tensor {
        if (a <= 0) return Error.ShapeMismatch;
        var out = try Tensor.empty(s, dt);
        
        // For a < 1, use the transformation: gamma(a) = gamma(a+1) * U^(1/a)
        const scale_factor = if (a < 1.0) 1.0 else 0.0;
        const a_eff = if (a < 1.0) a + 1.0 else a;
        
        for (0..out.numel()) |i| {
            var v: f64 = 0;
            var guard: usize = 0;
            while (guard < 100) : (guard += 1) {
                // Marsaglia and Tsang's method
                const d = a_eff - 1.0 / 3.0;
                const c = 1.0 / @sqrt(9.0 * d);
                
                var x: f64 = 0;
                var u: f64 = 0;
                while (true) {
                    // Generate standard normal
                    const n1 = @max(self.nextF64(), 1e-12);
                    const n2 = self.nextF64();
                    const r = @sqrt(-2.0 * @log(n1));
                    x = r * @cos(2.0 * std.math.pi * n2);
                    
                    const v_tmp = 1.0 + c * x;
                    if (v_tmp > 0) {
                        v = v_tmp * v_tmp * v_tmp;
                        u = self.nextF64();
                        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) break;
                        if (@log(u) < 0.5 * x * x + d * (1.0 - v + @log(v))) break;
                    }
                }
                
                v = d * v;
                break;
            }
            
            // Apply correction for a < 1
            if (scale_factor > 0.0) {
                const u = self.nextF64();
                v *= @pow(u, 1.0 / a);
            }
            
            out.writeF64(i, v);
        }
        return out;
    }

    /// Beta distribution with parameters a and b.
    /// Uses the relation: beta(a,b) = gamma(a) / (gamma(a) + gamma(b))
    pub fn beta(self: *Rng, s: Shape, a: f64, b: f64, dt: DType) Error!Tensor {
        if (a <= 0 or b <= 0) return Error.ShapeMismatch;
        var out = try Tensor.empty(s, dt);
        
        for (0..out.numel()) |i| {
            // Generate two gamma samples
            const x = try self.gammaScalar(a);
            const y = try self.gammaScalar(b);
            out.writeF64(i, x / (x + y));
        }
        return out;
    }
    
    /// Helper: generate a single gamma sample
    fn gammaScalar(self: *Rng, a: f64) !f64 {
        const scale_factor = if (a < 1.0) 1.0 else 0.0;
        const a_eff = if (a < 1.0) a + 1.0 else a;
        
        var v: f64 = 0;
        const d = a_eff - 1.0 / 3.0;
        const c = 1.0 / @sqrt(9.0 * d);
        
        while (true) {
            const n1 = @max(self.nextF64(), 1e-12);
            const n2 = self.nextF64();
            const r = @sqrt(-2.0 * @log(n1));
            const x = r * @cos(2.0 * std.math.pi * n2);
            
            const v_tmp = 1.0 + c * x;
            if (v_tmp > 0) {
                v = v_tmp * v_tmp * v_tmp;
                const u = self.nextF64();
                if (u < 1.0 - 0.0331 * (x * x) * (x * x)) break;
                if (@log(u) < 0.5 * x * x + d * (1.0 - v + @log(v))) break;
            }
        }
        
        v = d * v;
        if (scale_factor > 0.0) {
            const u = self.nextF64();
            v *= @pow(u, 1.0 / a);
        }
        return v;
    }

    /// Categorical distribution: sample from logits.
    pub fn categorical(self: *Rng, logits: Tensor, axis: i32) Error!Tensor {
        // Softmax to get probabilities, then sample
        const ops = @import("ops.zig");
        const probs = try ops.softmax(logits, axis);
        defer probs.deinit();
        
        const num_categories = logits.shape.dims[@intCast(if (axis < 0) @as(i32, @intCast(logits.rank())) + axis else axis)];
        const batch_size = logits.numel() / num_categories;
        
        var out = try Tensor.empty(try Shape.init(&.{batch_size}), .i64);
        
        for (0..batch_size) |b| {
            const u = self.nextF64();
            var cumsum: f64 = 0;
            var sampled: usize = 0;
            for (0..num_categories) |c| {
                cumsum += probs.readF64(b * num_categories + c);
                if (u < cumsum) {
                    sampled = c;
                    break;
                }
            }
            out.writeF64(b, @floatFromInt(sampled));
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
pub fn bernoulli(s: Shape, p: f64) Error!Tensor {
    return rng().bernoulli(s, p);
}
pub fn exponential(s: Shape, dt: DType) Error!Tensor {
    return rng().exponential(s, dt);
}
pub fn gamma(s: Shape, a: f64, dt: DType) Error!Tensor {
    return rng().gamma(s, a, dt);
}
pub fn beta(s: Shape, a: f64, b: f64, dt: DType) Error!Tensor {
    return rng().beta(s, a, b, dt);
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

test "new distributions" {
    var gen = Rng.init(123);
    const alloc = std.testing.allocator;
    
    // Bernoulli
    const shape = try Shape.init(&.{1000});
    const bern = try gen.bernoulli(shape, 0.3);
    defer bern.deinit();
    var sum: f64 = 0;
    for (0..bern.numel()) |i| sum += bern.readF64(i);
    const mean = sum / 1000.0;
    try std.testing.expect(@abs(mean - 0.3) < 0.05);
    _ = alloc;
    
    // Exponential
    const exp = try gen.exponential(shape, .f32);
    defer exp.deinit();
    sum = 0;
    for (0..exp.numel()) |i| sum += exp.readF64(i);
    const exp_mean = sum / 1000.0;
    try std.testing.expect(@abs(exp_mean - 1.0) < 0.15); // rate=1, mean=1
    
    // Gamma
    const gam = try gen.gamma(shape, 2.0, .f32);
    defer gam.deinit();
    sum = 0;
    for (0..gam.numel()) |i| sum += gam.readF64(i);
    const gam_mean = sum / 1000.0;
    try std.testing.expect(@abs(gam_mean - 2.0) < 0.3); // shape=2, mean=2
    
    // Beta
    const bet = try gen.beta(shape, 2.0, 5.0, .f32);
    defer bet.deinit();
    sum = 0;
    for (0..bet.numel()) |i| sum += bet.readF64(i);
    const bet_mean = sum / 1000.0;
    try std.testing.expect(@abs(bet_mean - 0.2857) < 0.05); // a/(a+b) = 2/7 ≈ 0.2857
}
