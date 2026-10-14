# Eager operations (`ops`)

## Usage

```zig
// element-wise (two operands must have the same rank; use broadcastTo otherwise)
const s = try zpu.add(a, b);
const p = try zpu.mul(a, b);
const e = try zpu.exp(a);
const t = try zpu.tanh(a);
const r = try zpu.where(mask, a, b);
const c = try zpu.clamp(a, 0, 1);
const l = try zpu.lerp(a, b, 0.25);

// scalars keep the tensor's dtype
const k = try zpu.mulScalar(a, 2.5);
const q = try zpu.addScalar(a, 1.0);

// reductions: axis is i32, keepdims decides whether the axis survives
const total = try zpu.sum(a, zpu.AXIS_NONE, false);   // everything
const rows = try zpu.sum(a, 0, false);                // along axis 0
const keep = try zpu.mean(a, -1, true);               // last axis, kept
const mx = try zpu.max(a, 1, false);
const which = try zpu.argmax(a, 1);                   // i64 indices

// shape work
const flat = try zpu.reshape(a, zpu.Shape.initComptime(&.{6}));
const tr = try zpu.transpose(a, &.{ 1, 0 });
const big = try zpu.broadcastTo(a, zpu.Shape.initComptime(&.{ 4, 3 }));
const grown = try zpu.expandDims(a, 0);
const cut = try zpu.narrow(a, 1, 0, 2);
const win = try zpu.sliceRanges(a, &.{ zpu.Range.bounded(0, 2), zpu.Range.all() });
const joined = try zpu.concat(a, b, 0);
const picked = try zpu.take(table, indices, 0);
const padded = try zpu.pad(a, &.{ .{ 1, 1 }, .{ 0, 0 } }, 0);
const flipped = try zpu.flip(a, 1);
const cum = try zpu.cumsum(a, 0, false);

// matrix products and the usual suspects
const m = try zpu.matmul(a, b);
const sm = try zpu.softmax(a, -1);
const ls = try zpu.logSoftmax(a, -1);
const v = try zpu.variance(a, -1, false);
```

## Implementation intent

* **Every operation takes one path.** An op finishes with
  `ops.finish(prim, inputs, out, attrs)`, which records the primitive (for the
  tape and the graph) and adopts the result into the active scope. New ops cannot
  forget to do either, and the JIT only ever has to know about `Prim`.

* **Broadcasting is resolved in one place.** `broadcastPair(a, b)` computes the
  output shape and the strides both operands need, so the kernels never think
  about it. `outDepth` keeps the `vmap` depth, which is why a vmapped op
  broadcasts the *payload* axes only.

* **dtype promotion lives in `binaryOutDType`.** Two inputs, one output dtype:
  the wider float wins, integers promote to f64, bool only survives predicates.
  Scalar helpers (`mulScalar`, `addScalar`, …) deliberately keep the tensor's
  dtype, so `mulScalar(i64_tensor, 0.5)` stays i64 exactly as NumPy would.

* **Views are preferred over copies.** `reshape` only materialises when the
  requested shape cannot be expressed as strides; `squeeze`, `expandDims`,
  `transpose`, `flip`, `sliceView` and `narrow` are pure metadata. The
  materialising variants (`sliceRanges`, `concat`, `pad`, …) are named apart so
  the cost is visible at the call site.

* **`matmul` preserves the `vmap` depth.** The last two payload axes are the
  matrix; the leading axes are batch. The `T.f64` CUDA-style tiling is skipped on
  purpose — this is a straightforward kernel that the benchmark records, so a
  faster one can be dropped in without touching the API.

* **Numerically stable forms.** `softmax` subtracts the maximum before `exp`,
  `logSoftmax` goes through log-sum-exp, reductions that produce a mean use a
  fused divide. The forward pass being stable is what keeps recomputed VJPs
  stable.

## Notes

* `add` and friends require **equal rank**: cross-rank broadcasting is done
  internally by `broadcastPair`, and calling `zpu.broadcastTo` first is the
  explicit way.
* Axes are `i32`: `zpu.AXIS_LAST` is `-1`, `zpu.AXIS_NONE` means "reduce
  everything".
* Reductions drop the axis unless `keepdims` is true; inside `vmap` the depth is
  preserved automatically.
* A negative `step` follows NumPy: `end` stays **exclusive** and a negative `end`
  counts from the far end, so `Range.strided(2, -1, -1)` is empty — exactly like
  `x[2:-1:-1]` in NumPy. To walk an axis backwards, `flip` is the natural
  spelling, and it returns a view rather than a copy.
