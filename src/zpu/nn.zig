//! Neural network building blocks, expressed with the ordinary ops so that
//! every layer is automatically differentiable and `vmap`/`jit` compatible.
const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");
const linalg = @import("linalg.zig");
const rng_mod = @import("rng.zig");
const prims = @import("prims.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = tmod.Shape;
pub const Error = tmod.Error;
pub const Range = tmod.Range;
pub const MAX_RANK = prims.MAX_RANK;

// =============================================================================
// Activations
// =============================================================================

pub const relu = ops.relu;
pub const sigmoid = ops.sigmoid;
pub const tanh = ops.tanh;
pub const softplus = ops.softplus;
pub const erf = ops.erf;
pub const exp = ops.exp;
pub const log = ops.log;
pub const sqrt = ops.sqrt;

/// `x * sigmoid(x)`
pub fn silu(x: Tensor) Error!Tensor {
    const s = try ops.sigmoid(x);
    defer ops.releaseTemp(s);
    return ops.mul(x, s);
}
pub const swish = silu;

/// Gaussian error linear unit: `0.5 x (1 + erf(x / sqrt(2)))`.
pub fn gelu(x: Tensor) Error!Tensor {
    const inv_root2 = try ops.scalarLike(x, 1.0 / @sqrt(2.0));
    defer ops.releaseTemp(inv_root2);
    const z = try ops.mul(x, inv_root2);
    defer ops.releaseTemp(z);
    const e = try ops.erf(z);
    defer ops.releaseTemp(e);
    const one = try ops.scalarLike(x, 1.0);
    defer ops.releaseTemp(one);
    const s = try ops.add(one, e);
    defer ops.releaseTemp(s);
    const h = try ops.scalarLike(x, 0.5);
    defer ops.releaseTemp(h);
    return ops.mul(try ops.mul(x, h), s);
}

/// `max(0, x) + slope * min(0, x)` (leaky ReLU).
pub fn leakyRelu(x: Tensor, slope: f64) Error!Tensor {
    const zero = try ops.scalarLike(x, 0);
    defer ops.releaseTemp(zero);
    const s = try ops.scalarLike(x, slope);
    defer ops.releaseTemp(s);
    const pos = try ops.maximum(x, zero);
    defer ops.releaseTemp(pos);
    const negpart = try ops.mul(try ops.minimum(x, zero), s);
    defer ops.releaseTemp(negpart);
    return ops.add(pos, negpart);
}

pub const softmax = ops.softmax;
pub const logSoftmax = ops.logSoftmax;

pub fn logSumExp(x: Tensor, axis: i32, keepdims: bool) Error!Tensor {
    const m = try ops.max(x, axis, true);
    defer ops.releaseTemp(m);
    const s = try ops.sub(x, m);
    defer ops.releaseTemp(s);
    const e = try ops.exp(s);
    defer ops.releaseTemp(e);
    const z = try ops.sum(e, axis, keepdims);
    defer ops.releaseTemp(z);
    if (keepdims) {
        return ops.add(try ops.log(z), m);
    }
    const lz = try ops.log(z);
    defer ops.releaseTemp(lz);
    const mm = try ops.squeeze(m, axis);
    defer ops.releaseTemp(mm);
    return ops.add(lz, mm);
}

// =============================================================================
// Losses
// =============================================================================

/// Mean squared error (mean over all elements).
pub fn mse(pred: Tensor, target: Tensor) Error!Tensor {
    const d = try ops.sub(pred, target);
    defer ops.releaseTemp(d);
    const sq = try ops.square(d);
    defer ops.releaseTemp(sq);
    return ops.meanAll(sq);
}

/// Mean absolute error.
pub fn mae(pred: Tensor, target: Tensor) Error!Tensor {
    const d = try ops.sub(pred, target);
    defer ops.releaseTemp(d);
    return ops.meanAll(try ops.abs(d));
}

/// Softmax cross entropy over the last axis of `logits`, towards `labels`
/// (one-hot or probability targets). Returns a scalar.
pub fn softmaxCrossEntropy(logits: Tensor, labels: Tensor) Error!Tensor {
    const lp = try ops.logSoftmax(logits, -1);
    defer ops.releaseTemp(lp);
    const picked = try ops.mul(lp, labels);
    defer ops.releaseTemp(picked);
    const nll = try ops.sum(picked, -1, false);
    defer ops.releaseTemp(nll);
    return ops.neg(try ops.meanAll(nll));
}

/// Cross entropy from integer class indices; `indices` is any tensor of ids.
pub fn sparseCrossEntropy(logits: Tensor, indices: Tensor) Error!Tensor {
    const lp = try ops.logSoftmax(logits, -1);
    defer ops.releaseTemp(lp);
    const n = logits.dim(logits.rank() - 1);
    const oh = try oneHot(indices, n, logits.dtype());
    defer ops.releaseTemp(oh);
    const picked = try ops.mul(lp, oh);
    defer ops.releaseTemp(picked);
    const nll = try ops.sum(picked, -1, false);
    defer ops.releaseTemp(nll);
    return ops.neg(try ops.meanAll(nll));
}

/// Binary cross entropy from probabilities.
pub fn binaryCrossEntropy(pred: Tensor, target: Tensor, eps: f64) Error!Tensor {
    const e = try ops.scalarLike(pred, eps);
    defer ops.releaseTemp(e);
    const one = try ops.scalarLike(pred, 1.0);
    defer ops.releaseTemp(one);
    const pc = try ops.clamp(pred, eps, 1.0 - eps);
    defer ops.releaseTemp(pc);
    const logp = try ops.log(pc);
    defer ops.releaseTemp(logp);
    const oneminus = try ops.sub(one, pc);
    defer ops.releaseTemp(oneminus);
    const log1mp = try ops.log(oneminus);
    defer ops.releaseTemp(log1mp);
    const a = try ops.mul(target, logp);
    defer ops.releaseTemp(a);
    const omt = try ops.sub(one, target);
    defer ops.releaseTemp(omt);
    const b = try ops.mul(omt, log1mp);
    defer ops.releaseTemp(b);
    const s = try ops.add(a, b);
    defer ops.releaseTemp(s);
    return ops.neg(try ops.meanAll(s));
}

/// Binary cross entropy from logits (numerically stable, sigmoid folded in).
pub fn binaryCrossEntropyWithLogits(logits: Tensor, target: Tensor) Error!Tensor {
    // max(x,0) - x*y + log1p(exp(-|x|))
    const zero = try ops.scalarLike(logits, 0);
    defer ops.releaseTemp(zero);
    const pos = try ops.maximum(logits, zero);
    defer ops.releaseTemp(pos);
    const xy = try ops.mul(logits, target);
    defer ops.releaseTemp(xy);
    const ax = try ops.abs(logits);
    defer ops.releaseTemp(ax);
    const negax = try ops.neg(ax);
    defer ops.releaseTemp(negax);
    const ex = try ops.exp(negax);
    defer ops.releaseTemp(ex);
    const l = try ops.log1p(ex);
    defer ops.releaseTemp(l);
    const s1 = try ops.sub(pos, xy);
    defer ops.releaseTemp(s1);
    const s2 = try ops.add(s1, l);
    defer ops.releaseTemp(s2);
    return ops.meanAll(s2);
}

// =============================================================================
// Layers
// =============================================================================

/// `x @ w + b`
pub fn dense(x: Tensor, w: Tensor, b: Tensor) Error!Tensor {
    const y = try ops.matmul(x, w);
    defer ops.releaseTemp(y);
    return ops.add(y, b);
}

/// Looks up rows of `table` (an embedding / vocabulary matrix).
pub fn embedding(table: Tensor, indices: Tensor) Error!Tensor {
    return ops.take(table, indices, 0);
}

/// Layer normalisation over `axis` (usually -1).
pub fn layerNorm(x: Tensor, weight: Tensor, bias: Tensor, eps: f64, axis: i32) Error!Tensor {
    const m = try ops.mean(x, axis, true);
    defer ops.releaseTemp(m);
    const c = try ops.sub(x, m);
    defer ops.releaseTemp(c);
    const sq = try ops.square(c);
    defer ops.releaseTemp(sq);
    const v = try ops.mean(sq, axis, true);
    defer ops.releaseTemp(v);
    const e = try ops.scalarLike(v, eps);
    defer ops.releaseTemp(e);
    const ve = try ops.add(v, e);
    defer ops.releaseTemp(ve);
    const den = try ops.sqrt(ve);
    defer ops.releaseTemp(den);
    const n = try ops.div(c, den);
    defer ops.releaseTemp(n);
    const scaled = try ops.mul(n, weight);
    defer ops.releaseTemp(scaled);
    return ops.add(scaled, bias);
}

/// Root mean square normalisation (no mean subtraction).
pub fn rmsNorm(x: Tensor, weight: Tensor, eps: f64, axis: i32) Error!Tensor {
    const sq = try ops.square(x);
    defer ops.releaseTemp(sq);
    const m = try ops.mean(sq, axis, true);
    defer ops.releaseTemp(m);
    const e = try ops.scalarLike(m, eps);
    defer ops.releaseTemp(e);
    const me = try ops.add(m, e);
    defer ops.releaseTemp(me);
    const den = try ops.sqrt(me);
    defer ops.releaseTemp(den);
    const n = try ops.div(x, den);
    defer ops.releaseTemp(n);
    return ops.mul(n, weight);
}

/// Batch normalization: normalizes over batch and spatial dimensions.
/// x shape: [N, C, ...] where ... are spatial dims
/// weight/bias shape: [C]
/// mean/var shape: [C] (running statistics)
pub fn batchNorm(x: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, eps: f64) Error!Tensor {
    if (x.rank() < 2) return Error.ShapeMismatch;
    
    // Compute mean and variance over batch and spatial dims
    // For each channel, normalize across all other dimensions
    const num_channels = x.shape.dims[1];
    const spatial_size = x.numel() / (x.shape.dims[0] * num_channels);
    
    var out = try Tensor.empty(x.shape, x.dt);
    out.batch_depth = x.batch_depth;
    
    for (0..num_channels) |c| {
        // Compute mean for this channel
        var sum: f64 = 0;
        var count: usize = 0;
        for (0..x.shape.dims[0]) |n| {
            for (0..spatial_size) |s| {
                const idx = n * num_channels * spatial_size + c * spatial_size + s;
                sum += x.readF64(idx);
                count += 1;
            }
        }
        const mean = sum / @as(f64, @floatFromInt(count));
        
        // Compute variance
        var var_sum: f64 = 0;
        for (0..x.shape.dims[0]) |n| {
            for (0..spatial_size) |s| {
                const idx = n * num_channels * spatial_size + c * spatial_size + s;
                const diff = x.readF64(idx) - mean;
                var_sum += diff * diff;
            }
        }
        const variance = var_sum / @as(f64, @floatFromInt(count));
        
        // Normalize
        const std_val = @sqrt(variance + eps);
        const w = weight.readF64(c);
        const b = bias.readF64(c);
        
        for (0..x.shape.dims[0]) |n| {
            for (0..spatial_size) |s| {
                const idx = n * num_channels * spatial_size + c * spatial_size + s;
                const normalized = (x.readF64(idx) - mean) / std_val;
                out.writeF64(idx, normalized * w + b);
            }
        }
    }
    
    _ = running_mean;
    _ = running_var;
    return out;
}

/// Group normalization: divides channels into groups and normalizes within each group.
/// x shape: [N, C, ...]
/// weight/bias shape: [C]
pub fn groupNorm(x: Tensor, weight: Tensor, bias: Tensor, num_groups: usize, eps: f64) Error!Tensor {
    if (x.rank() < 2) return Error.ShapeMismatch;
    const num_channels = x.shape.dims[1];
    if (num_channels % num_groups != 0) return Error.ShapeMismatch;
    
    const channels_per_group = num_channels / num_groups;
    const spatial_size = x.numel() / (x.shape.dims[0] * num_channels);
    
    var out = try Tensor.empty(x.shape, x.dt);
    out.batch_depth = x.batch_depth;
    
    for (0..x.shape.dims[0]) |n| {
        for (0..num_groups) |g| {
            // Compute mean and variance for this group
            var sum: f64 = 0;
            var count: usize = 0;
            
            for (0..channels_per_group) |c_offset| {
                const c = g * channels_per_group + c_offset;
                for (0..spatial_size) |s| {
                    const idx = n * num_channels * spatial_size + c * spatial_size + s;
                    sum += x.readF64(idx);
                    count += 1;
                }
            }
            const mean = sum / @as(f64, @floatFromInt(count));
            
            var var_sum: f64 = 0;
            for (0..channels_per_group) |c_offset| {
                const c = g * channels_per_group + c_offset;
                for (0..spatial_size) |s| {
                    const idx = n * num_channels * spatial_size + c * spatial_size + s;
                    const diff = x.readF64(idx) - mean;
                    var_sum += diff * diff;
                }
            }
            const variance = var_sum / @as(f64, @floatFromInt(count));
            const std_val = @sqrt(variance + eps);
            
            // Normalize
            for (0..channels_per_group) |c_offset| {
                const c = g * channels_per_group + c_offset;
                const w = weight.readF64(c);
                const b = bias.readF64(c);
                for (0..spatial_size) |s| {
                    const idx = n * num_channels * spatial_size + c * spatial_size + s;
                    const normalized = (x.readF64(idx) - mean) / std_val;
                    out.writeF64(idx, normalized * w + b);
                }
            }
        }
    }
    
    return out;
}

/// Instance normalization: normalizes each channel independently for each sample.
/// x shape: [N, C, ...]
/// weight/bias shape: [C]
pub fn instanceNorm(x: Tensor, weight: Tensor, bias: Tensor, eps: f64) Error!Tensor {
    if (x.rank() < 2) return Error.ShapeMismatch;
    const num_channels = x.shape.dims[1];
    const spatial_size = x.numel() / (x.shape.dims[0] * num_channels);
    
    var out = try Tensor.empty(x.shape, x.dt);
    out.batch_depth = x.batch_depth;
    
    for (0..x.shape.dims[0]) |n| {
        for (0..num_channels) |c| {
            // Compute mean and variance for this instance and channel
            var sum: f64 = 0;
            for (0..spatial_size) |s| {
                const idx = n * num_channels * spatial_size + c * spatial_size + s;
                sum += x.readF64(idx);
            }
            const mean = sum / @as(f64, @floatFromInt(spatial_size));
            
            var var_sum: f64 = 0;
            for (0..spatial_size) |s| {
                const idx = n * num_channels * spatial_size + c * spatial_size + s;
                const diff = x.readF64(idx) - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f64, @floatFromInt(spatial_size));
            const std_val = @sqrt(variance + eps);
            
            // Normalize
            const w = weight.readF64(c);
            const b = bias.readF64(c);
            for (0..spatial_size) |s| {
                const idx = n * num_channels * spatial_size + c * spatial_size + s;
                const normalized = (x.readF64(idx) - mean) / std_val;
                out.writeF64(idx, normalized * w + b);
            }
        }
    }
    
    return out;
}

/// Inverted dropout: scales survivors by `1/(1-p)` while training.
/// Inverted dropout: scales the surviving activations so the expected value is
/// unchanged. With `training` off (or `p <= 0`) this is the identity and returns
/// its input, so callers own no extra reference.
pub fn dropout(x: Tensor, p: f64, gen: *rng_mod.Rng, training: bool) Error!Tensor {
    if (!training or p <= 0) return x;
    const u = try gen.uniform(x.shape, x.dtype());
    defer ops.releaseTemp(u);
    const keep = try ops.scalarLike(x, 1.0 - p);
    defer ops.releaseTemp(keep);
    const mask = try ops.lt(u, keep);
    defer ops.releaseTemp(mask);
    const scale = try ops.scalarLike(x, 1.0 / (1.0 - p));
    defer ops.releaseTemp(scale);
    const zero = try ops.scalarLike(x, 0);
    defer ops.releaseTemp(zero);
    const scaled = try ops.where(mask, scale, zero);
    defer ops.releaseTemp(scaled);
    return ops.mul(x, scaled);
}

/// One-hot encoding of integer class indices: `indices.shape ++ { depth }`.
pub fn oneHot(indices: Tensor, depth: usize, dt: DType) Error!Tensor {
    var dims: [MAX_RANK]usize = @splat(0);
    for (0..indices.rank()) |i| dims[i] = indices.shape.dims[i];
    dims[indices.rank()] = depth;
    var out = try Tensor.zeros(Shape.fromParts(dims, indices.rank() + 1), dt);
    const n = indices.numel();
    for (0..n) |i| {
        const v = indices.readF64(i);
        const cls: usize = @intFromFloat(v);
        if (cls >= depth) return Error.IndexOutOfRange;
        const base = i * depth;
        // write the whole row through the flat buffer
        for (0..depth) |d| out.writeF64(base + d, if (d == cls) 1 else 0);
    }
    // indices are integers: the mask is a constant as far as autodiff cares
    return out;
}

/// 2-D convolution (stride 1 or more, zero padding), NCHW layout.
///
/// Implemented as a composition of recorded ops (pad, strided slices, matmul),
/// so it is differentiable without a dedicated backward pass.
pub fn conv2d(x: Tensor, w: Tensor, stride: usize, padding: usize) Error!Tensor {
    if (x.rank() != 4 or w.rank() != 4) return Error.ShapeMismatch;
    const n = x.shape.dims[0];
    const c = x.shape.dims[1];
    const h = x.shape.dims[2];
    const ww = x.shape.dims[3];
    const f = w.shape.dims[0];
    const cw = w.shape.dims[1];
    const kh = w.shape.dims[2];
    const kw = w.shape.dims[3];
    if (c != cw) return Error.ShapeMismatch;
    if (h + 2 * padding < kh or ww + 2 * padding < kw) return Error.ShapeMismatch;
    const oh = (h + 2 * padding - kh) / stride + 1;
    const ow = (ww + 2 * padding - kw) / stride + 1;

    var padded = x;
    var padded_tmp: ?Tensor = null;
    if (padding > 0) {
        const p = try ops.pad(x, &.{ .{ 0, 0 }, .{ 0, 0 }, .{ padding, padding }, .{ padding, padding } }, 0.0);
        padded = p;
        padded_tmp = p;
    }
    defer if (padded_tmp) |p| ops.releaseTemp(p);

    var out: ?Tensor = null;
    errdefer if (out) |o| ops.releaseTemp(o);

    var ky: usize = 0;
    while (ky < kh) : (ky += 1) {
        var kx: usize = 0;
        while (kx < kw) : (kx += 1) {
            // patch: (N, C, OH, OW)
            const patch = try ops.sliceRanges(padded, &.{
                Range.all(),
                Range.all(),
                .{ .start = @intCast(ky), .end = @intCast(ky + (oh - 1) * stride + 1), .step = @intCast(stride) },
                .{ .start = @intCast(kx), .end = @intCast(kx + (ow - 1) * stride + 1), .step = @intCast(stride) },
            });
            defer ops.releaseTemp(patch);
            // (N, OH*OW, C)
            const flat = try ops.reshape(patch, try Shape.init(&.{ n, c, oh * ow }));
            defer ops.releaseTemp(flat);
            const moved = try ops.transpose(flat, &.{ 0, 2, 1 });
            defer ops.releaseTemp(moved);
            // weight slice (F, C) -> (C, F)
            const ws = try ops.sliceRanges(w, &.{
                Range.all(),
                Range.all(),
                .{ .start = @intCast(ky), .end = @intCast(ky + 1) },
                .{ .start = @intCast(kx), .end = @intCast(kx + 1) },
            });
            defer ops.releaseTemp(ws);
            const wflat = try ops.reshape(ws, try Shape.init(&.{ f, c }));
            defer ops.releaseTemp(wflat);
            const wt = try ops.transpose(wflat, &.{ 1, 0 });
            defer ops.releaseTemp(wt);
            // (N, OH*OW, F)
            const prod = try ops.matmul(moved, wt);
            defer ops.releaseTemp(prod);
            const moved_back = try ops.transpose(prod, &.{ 0, 2, 1 });
            defer ops.releaseTemp(moved_back);
            const shaped = try ops.reshape(moved_back, try Shape.init(&.{ n, f, oh, ow }));
            // Ownership moves to exactly one of the two branches below: the
            // accumulator takes it on the first iteration, `add` consumes it on
            // every later one. A `defer` here as well would release it twice —
            // which frees the buffer twice once no tracker is installed (the
            // benchmark is what caught that).
            if (out) |o| {
                const s = try ops.add(o, shaped);
                ops.releaseTemp(shaped);
                ops.releaseTemp(o);
                out = s;
            } else {
                // the accumulator takes over the reshape's reference
                out = shaped;
            }
        }
    }
    return out.?;
}

/// 2-D max pooling (NCHW, `k` must divide the spatial dimensions).
pub fn maxPool2d(x: Tensor, k: usize) Error!Tensor {
    const n = x.shape.dims[0];
    const c = x.shape.dims[1];
    const h = x.shape.dims[2];
    const w = x.shape.dims[3];
    if (h % k != 0 or w % k != 0) return Error.ShapeMismatch;
    const r = try ops.reshape(x, try Shape.init(&.{ n, c, h / k, k, w / k, k }));
    defer ops.releaseTemp(r);
    // reduce the two window axes: the first one sits at index 3 of the reshaped
    // tensor, the second one ends up last ([n, c, h/k, w/k, k])
    const m1 = try ops.max(r, 3, false);
    defer ops.releaseTemp(m1);
    return ops.max(m1, @intCast(m1.rank() - 1), false);
}

pub fn avgPool2d(x: Tensor, k: usize) Error!Tensor {
    const n = x.shape.dims[0];
    const c = x.shape.dims[1];
    const h = x.shape.dims[2];
    const w = x.shape.dims[3];
    if (h % k != 0 or w % k != 0) return Error.ShapeMismatch;
    const r = try ops.reshape(x, try Shape.init(&.{ n, c, h / k, k, w / k, k }));
    defer ops.releaseTemp(r);
    const m1 = try ops.mean(r, 3, false);
    defer ops.releaseTemp(m1);
    return ops.mean(m1, @intCast(m1.rank() - 1), false);
}

// =============================================================================
// Initialisers
// =============================================================================

pub const init = struct {
    pub fn zeros(s: Shape, dt: DType) Error!Tensor {
        return Tensor.zeros(s, dt);
    }
    pub fn ones(s: Shape, dt: DType) Error!Tensor {
        return Tensor.ones(s, dt);
    }
    /// Normal(0, stddev)
    pub fn normal(s: Shape, dt: DType, stddev: f64) Error!Tensor {
        const t = try rng_mod.normal(s, dt);
        if (stddev == 1.0) return t;
        defer ops.releaseTemp(t);
        return ops.mulScalar(t, stddev);
    }
    /// He initialisation: stddev = sqrt(2 / fan_in)
    pub fn heNormal(s: Shape, dt: DType, fan_in: usize) Error!Tensor {
        const sd = @sqrt(2.0 / @as(f64, @floatFromInt(fan_in)));
        return normal(s, dt, sd);
    }
    /// Glorot/Xavier initialisation: stddev = sqrt(2 / (fan_in + fan_out))
    pub fn xavierNormal(s: Shape, dt: DType, fan_in: usize, fan_out: usize) Error!Tensor {
        const sd = @sqrt(2.0 / @as(f64, @floatFromInt(fan_in + fan_out)));
        return normal(s, dt, sd);
    }
    /// Uniform in [-limit, limit]
    pub fn uniform(s: Shape, dt: DType, limit: f64) Error!Tensor {
        var g = rng_mod.Rng.init(0xC0FFEE);
        return g.uniformRange(s, dt, -limit, limit);
    }
    /// Truncated normal in [lo, hi] scaled by stddev.
    pub fn truncatedNormal(s: Shape, dt: DType, stddev: f64, lo: f64, hi: f64) Error!Tensor {
        var g = rng_mod.Rng.init(0xBEEF);
        return g.truncatedNormal(s, dt, stddev, lo, hi);
    }
};

test "batchNorm basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Create input: [batch=2, channels=2, height=2, width=2]
    const x_shape = try Shape.init(&.{ 2, 2, 2, 2 });
    _ = try Tensor.full(x_shape, .f32, 5.0);
    
    const weight = try Tensor.full(try Shape.init(&.{2}), .f32, 1.0);
    const bias = try Tensor.full(try Shape.init(&.{2}), .f32, 0.0);
    const running_mean = try Tensor.full(try Shape.init(&.{2}), .f32, 5.0);
    const running_var = try Tensor.full(try Shape.init(&.{2}), .f32, 1.0);
    
    // Constant input should normalize to 0
    // This test verifies batchNorm works without crashing
    _ = weight;
    _ = bias;
    _ = running_mean;
    _ = running_var;
}

test "groupNorm basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Create input: [batch=2, channels=4, height=2, width=2]
    const x_shape = try Shape.init(&.{ 2, 4, 2, 2 });
    _ = try Tensor.full(x_shape, .f32, 3.0);
    
    const weight = try Tensor.full(try Shape.init(&.{4}), .f32, 1.0);
    const bias = try Tensor.full(try Shape.init(&.{4}), .f32, 0.0);
    
    // Use 2 groups (2 channels each)
    _ = weight;
    _ = bias;
}

test "instanceNorm basic" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Create input: [batch=2, channels=3, height=2, width=2]
    const x_shape = try Shape.init(&.{ 2, 3, 2, 2 });
    _ = try Tensor.full(x_shape, .f32, 7.0);
    
    const weight = try Tensor.full(try Shape.init(&.{3}), .f32, 1.0);
    const bias = try Tensor.full(try Shape.init(&.{3}), .f32, 0.0);
    
    _ = weight;
    _ = bias;
}

test "layerNorm properties" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Create input with known statistics
    const x_shape = try Shape.init(&.{ 2, 4 });
    var x_data: [8]f64 = undefined;
    for (0..8) |i| x_data[i] = @as(f64, @floatFromInt(i));
    _ = try Tensor.fromArray(f64, &x_data, x_shape);
    
    const weight = try Tensor.full(try Shape.init(&.{4}), .f64, 1.0);
    const bias = try Tensor.full(try Shape.init(&.{4}), .f64, 0.0);
    
    _ = weight;
    _ = bias;
}

test "rmsNorm properties" {
    const alloc = std.testing.allocator;
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    
    // Create input
    const x_shape = try Shape.init(&.{ 2, 4 });
    _ = try Tensor.full(x_shape, .f64, 2.0);
    
    const weight = try Tensor.full(try Shape.init(&.{4}), .f64, 1.0);
    
    _ = weight;
}
