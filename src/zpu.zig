//! ZPU — a JAX-flavoured numerical computing library for Zig.
//!
//! Layers:
//!
//! * `dtype` / `shape` / `buffer` — fixed-rank metadata and ref counted storage
//! * `tensor`   — the strided N-d array plus the allocating scope (`Tracker`)
//! * `ops`      — eager operations (elementwise, reductions, linalg, indexing)
//! * `tape` / `autodiff`  — reverse and forward mode automatic differentiation
//! * `graph` / `jit`      — traced execution plans with constant folding
//! * `transforms`         — `grad`, `jvp`, `vmap`, `jit`
//! * `nn`, `optim`, `rng` — the neural-network layer built on the above
//!
//! ```zig
//! const zpu = @import("zpu");
//!
//! try zpu.init(alloc, .{});
//! defer zpu.deinit();
//!
//! const x = try zpu.Tensor.fromArray(f32, &.{ 1, 2, 3, 4 }, zpu.Shape.initComptime(&.{ 2, 2 }));
//! const y = try zpu.tanh(try zpu.matmul(x, x));
//! _ = y;
//! ```
const std = @import("std");

pub const dtype = @import("zpu/dtype.zig");
pub const shape = @import("zpu/shape.zig");
pub const buffer = @import("zpu/buffer.zig");
pub const context = @import("zpu/context.zig");
pub const tensor = @import("zpu/tensor.zig");
pub const prims = @import("zpu/prims.zig");
pub const tape = @import("zpu/tape.zig");
pub const graph = @import("zpu/graph.zig");
pub const ops = @import("zpu/ops.zig");
pub const linalg = @import("zpu/linalg.zig");
pub const autodiff = @import("zpu/autodiff.zig");
pub const transforms = @import("zpu/transforms.zig");
pub const nn = @import("zpu/nn.zig");
pub const optim = @import("zpu/optim.zig");
pub const rng = @import("zpu/rng.zig");

pub const DType = dtype.DType;
pub const Shape = shape.Shape;
pub const Buffer = buffer.Buffer;
pub const Tensor = tensor.Tensor;
pub const Tracker = tensor.Tracker;
pub const Scope = tensor.Scope;
pub const Range = tensor.Range;
pub const Prim = prims.Prim;
pub const Attrs = prims.Attrs;
pub const MAX_RANK = shape.MAX_RANK;
pub const AXIS_NONE = ops.AXIS_NONE;
pub const AXIS_LAST = ops.AXIS_LAST;

/// Install the per-thread ZPU context (required before creating tensors).
pub const init = context.init;
pub const InitOptions = context.InitOptions;
pub const initThreaded = context.initThreaded;
pub const deinit = context.deinit;

// ---- transforms -------------------------------------------------------------
/// `zpu.grad(f, args, wrt)` — reverse mode gradients.
pub const grad = transforms.grad;
/// Copies one entry out of a `grad`/`vjp` result and releases the rest.
pub const keepGrad = transforms.keepGrad;
/// `zpu.valueAndGrad(f, args, wrt)` — primal value plus gradients.
pub const valueAndGrad = transforms.valueAndGrad;
/// `zpu.jvp(f, args, tangents)` — forward mode Jacobian-vector product.
pub const jvp = transforms.jvp;
/// `zpu.vmap(f, args)` — vectorise over the leading axis.
pub const vmap = transforms.vmap;
/// `zpu.jit(alloc, f, ud)` — trace once, replay an optimised plan.
pub const Jit = transforms.Jit;

/// Hessian-vector product for a scalar `f`: `H f(x) · v`.
///
/// This is reverse-over-reverse done explicitly, without a trampoline: the
/// gradient `∇f(x)` is computed while an *outer* tape is installed, so both the
/// inner forward pass and the inner backward sweep land on that tape. The outer
/// sweep then differentiates `∇f(x) · v` with respect to `args[wrt]` — which is
/// exactly `H v`.
pub fn hvp(alloc: std.mem.Allocator, f: transforms.Fn, ud: ?*anyopaque, args: []const Tensor, wrt: usize, v: Tensor) !Tensor {
    if (wrt >= args.len) return tensor.Error.InvalidShape;

    var scope = try Scope.init(alloc);
    defer scope.deinit();

    var outer = tape.Tape.init(alloc);
    defer outer.deinit();

    const saved = context.save();
    context.setMode(.grad);
    context.pushTape(&outer);
    defer context.restore(saved);

    // 1. the gradient, recorded on the outer tape
    var r = try transforms.valueAndGrad(alloc, f, ud, args, &.{wrt});
    defer r.deinit();

    // 2. contract it with the direction: `vhat = <grad, v>`
    const vhat = try ops.sumAll(try ops.mul(r.grads[0], v));

    // 3. one more sweep over the outer tape. The tape is detached first: a
    //    sweep builds its own VJP ops, and those must not be appended to the
    //    tape being swept (they belong to whatever recorder was already there).
    context.popTape();
    var engine = autodiff.Engine.init(alloc);
    defer engine.deinit();
    const seed = try autodiff.seedOnes(vhat);
    try engine.run(&outer, &.{vhat}, &.{seed}, &.{args[wrt]});
    const g = try engine.get(args[wrt]);
    return g.retain();
}

/// Builds a JIT-transformed function: `var f = zpu.jit(alloc, loss, ud);`
pub fn jit(alloc: std.mem.Allocator, f: transforms.Fn, ud: ?*anyopaque) Jit {
    return Jit.init(alloc, f, ud);
}

/// The execution-plan machinery used by `jit`.
pub const plan = @import("zpu/jit.zig");

// ---- execution backends -----------------------------------------------------
/// The portable execution backend shared with `zgf`.
pub const zbackend = @import("zbackend");
/// What this machine offers: `zpu.backends().best()`, `.vulkan`, `.detail(.cpu)` …
pub const backends = context.backends;
/// The backend to pick when the caller has no preference.
pub const bestBackend = context.bestBackend;
/// A portability helper: `zpu.Backend`, `zpu.Kernel`, `zpu.dispatchKernel` …
pub const Backend = zbackend.Backend;
pub const Kernel = zbackend.Kernel;
pub const KernelCtx = zbackend.KernelCtx;
pub const dispatchKernel = zbackend.dispatch;
pub const dispatchKernelThreaded = zbackend.dispatchThreaded;

// ---- op surface -------------------------------------------------------------
pub const broadcastPair = ops.broadcastPair;
pub const outDepth = ops.outDepth;
pub const releaseTemp = ops.releaseTemp;
pub const untrack = @import("zpu/tensor.zig").untrack;
pub const zerosLike = ops.zerosLike;
pub const onesLike = ops.onesLike;
pub const zerosLikeT = ops.zerosLikeT;
pub const onesLikeT = ops.onesLikeT;
pub const sealResult = ops.sealResult;
pub const isFloatOnly = ops.isFloatOnly;
pub const isPredicate = ops.isPredicate;
pub const sealOut = ops.sealOut;
pub const neg = ops.neg;
pub const exp = ops.exp;
pub const expm1 = ops.expm1;
pub const log = ops.log;
pub const log1p = ops.log1p;
pub const sqrt = ops.sqrt;
pub const rsqrt = ops.rsqrt;
pub const square = ops.square;
pub const recip = ops.recip;
pub const cbrt = ops.cbrt;
pub const sin = ops.sin;
pub const cos = ops.cos;
pub const tan = ops.tan;
pub const asin = ops.asin;
pub const acos = ops.acos;
pub const atan = ops.atan;
pub const sinh = ops.sinh;
pub const cosh = ops.cosh;
pub const tanh = ops.tanh;
pub const asinh = ops.asinh;
pub const acosh = ops.acosh;
pub const atanh = ops.atanh;
pub const erf = ops.erf;
pub const sigmoid = ops.sigmoid;
pub const softplus = ops.softplus;
pub const relu = ops.relu;
pub const floor = ops.floor;
pub const ceil = ops.ceil;
pub const round = ops.round;
pub const trunc = ops.trunc;
pub const sign = ops.sign;
pub const abs = ops.abs;
pub const isnan = ops.isnan;
pub const isinf = ops.isinf;
pub const isfinite = ops.isfinite;
pub const add = ops.add;
pub const sub = ops.sub;
pub const mul = ops.mul;
pub const div = ops.div;
pub const pow = ops.pow;
pub const maximum = ops.maximum;
pub const minimum = ops.minimum;
pub const atan2 = ops.atan2;
pub const hypot = ops.hypot;
pub const mod = ops.mod;
pub const eq = ops.eq;
pub const ne = ops.ne;
pub const lt = ops.lt;
pub const le = ops.le;
pub const gt = ops.gt;
pub const ge = ops.ge;
pub const logicalAnd = ops.logicalAnd;
pub const logicalOr = ops.logicalOr;
pub const logicalXor = ops.logicalXor;
pub const logicalNot = ops.logicalNot;
pub const addScalar = ops.addScalar;
pub const mulScalar = ops.mulScalar;
pub const divScalar = ops.divScalar;
pub const scalarLike = ops.scalarLike;
pub const where = ops.where;
pub const clamp = ops.clamp;
pub const lerp = ops.lerp;
pub const sum = ops.sum;
pub const prod = ops.prod;
pub const max = ops.max;
pub const min = ops.min;
pub const all = ops.all;
pub const any = ops.any;
pub const sumAll = ops.sumAll;
pub const mean = ops.mean;
pub const meanAll = ops.meanAll;
pub const argmax = ops.argmax;
pub const argmin = ops.argmin;
pub const variance = ops.variance;
pub const stddev = ops.stddev;
pub const softmax = ops.softmax;
pub const logSoftmax = ops.logSoftmax;
pub const matmul = ops.matmul;
pub const reshapeTarget = ops.reshapeTarget;
pub const reshape = ops.reshape;
pub const flatten = ops.flatten;
pub const flattenFrom = ops.flattenFrom;
pub const realAxis = ops.realAxis;
pub const transpose = ops.transpose;
pub const swapAxes = ops.swapAxes;
pub const broadcastTo = ops.broadcastTo;
pub const broadcastTarget = ops.broadcastTarget;
pub const expandDims = ops.expandDims;
pub const squeeze = ops.squeeze;
pub const sliceRanges = ops.sliceRanges;
pub const narrow = ops.narrow;
pub const concat = ops.concat;
pub const stack = ops.stack;
pub const cast = ops.cast;
pub const take = ops.take;
pub const scatterAdd = ops.scatterAdd;
pub const pad = ops.pad;
pub const flip = ops.flip;
pub const cumsum = ops.cumsum;
pub const applyPrim = ops.applyPrim;

test {
    std.testing.refAllDecls(@This());
}

// -----------------------------------------------------------------------------
// Module level unit tests
// -----------------------------------------------------------------------------

// Collects the unit tests written next to each module, so `zig build test`
// runs them together with the numerical suite in `zpu/tests.zig` (which the
// aggregation root imports).
test {
    _ = @import("zpu/autodiff.zig");
    _ = @import("zpu/buffer.zig");
    _ = @import("zpu/context.zig");
    _ = @import("zpu/dtype.zig");
    _ = @import("zpu/graph.zig");
    _ = @import("zpu/jit.zig");
    _ = @import("zpu/linalg.zig");
    _ = @import("zpu/ops.zig");
    _ = @import("zpu/optim.zig");
    _ = @import("zpu/prims.zig");
    _ = @import("zpu/rng.zig");
    _ = @import("zpu/scalar.zig");
    _ = @import("zpu/shape.zig");
    _ = @import("zpu/tape.zig");
    _ = @import("zpu/tensor.zig");
    _ = @import("zpu/transforms.zig");
}
