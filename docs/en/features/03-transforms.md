# Function transforms (`transforms`)

`grad` / `valueAndGrad` / `jvp` / `hvp` / `vmap` / `jit`, all reachable from
`zpu.zig`.

## Usage

```zig
fn f(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
    _ = ud;
    const h = try zpu.matmul(args[0], args[1]);
    defer zpu.releaseTemp(h);
    return zpu.sumAll(zpu.tanh(h));
}

// reverse mode: returns []Tensor, owned by the caller
const grads = try zpu.grad(alloc, f, null, &.{ x, w }, &.{ 0, 1 });
defer alloc.free(grads);
defer zpu.keepGrad(alloc, grads, 0).deinit();   // take one, release the rest

// value and gradients together
var vg = try zpu.valueAndGrad(alloc, f, null, &.{x}, &.{0});
defer vg.deinit();                              // vg.value, vg.grads[0]

// forward mode: hand in a tangent
var jr = try zpu.jvp(alloc, f, null, &.{x}, &.{v});
defer jr.deinit();                              // jr.value, jr.tangent

// H f(x) · v  (reverse over reverse)
const hv = try zpu.hvp(alloc, f, null, &.{x}, 0, v);
defer { var h = hv; h.deinit(); }

// vectorise over the leading axis
const vm = try zpu.vmap(alloc, f, null, &.{batch});
defer { var m = vm; m.deinit(); }

// trace once, replay
var j = zpu.jit(alloc, f, null);
defer j.deinit();
const y = try j.call(&.{ x, w });
_ = j.numSteps();      // size of the plan
_ = j.traceCount;      // how many times it was traced
```

## Implementation intent

Every transform follows the same five steps (documented at the top of
`transforms.zig`):

1. open a `Scope` that owns the tensors the transform creates,
2. install whatever recorder it needs (a tape, a graph),
3. call the function,
4. **restore the previous recording state** — this is what makes
   `grad(grad(f))` and `jit(grad(f))` work, because the transform's own
   arithmetic (the sweep) is recorded by the *outer* recorder,
5. `retain` only the values that escape.

* **`grad` copies its arguments first.** `copyArgs` retains each argument into
  the transform's scope before choosing `wrt`, so a caller's tensor cannot be
  freed while the transform still reads it.

* **`vmap` advances `batch_depth`.** It tags the leading axis as the vectorised
  one and records that depth on the tensors before calling the function. Ops —
  `matmul` in particular — read the depth to decide which axes are matrices and
  which are batches, and `withBatchDepth` puts the depth back on the way out.

* **Reductions inside `vmap`** shift the axis by the depth so the reduced axis is
  the one the user meant rather than the batch axis.

* **`jit` separates tracing from replay.** `Jit.trace` records a `Graph`, and
  `jit.build` turns it into a `Plan`: a list of `Step`s (primitive plus input and
  output ids) and `Blob`s for everything that does not depend on the arguments.
  Replay executes the steps and never looks at the graph again.

* **Ownership across the trace boundary.** Temporaries created while tracing
  belong to the transform's scope; the `Plan` retains exactly what it needs.
  When the traced function returns a *transform result* (a reference the caller
  owns by convention), `Jit.trace` drops that one extra reference in
  `releaseExtraRef`: it counts the references the scope and the recorded nodes
  account for, and returns the one they do not.

* **`hvp` gets no special treatment.** It seeds `<∇f(x), v>` on an outer tape and
  sweeps that tape once more; there is no dedicated kernel.

## Known limitations

* `jit` retraces when the argument **ids** change. Reusing the same tensor with a
  different shape leaves a stale plan, so build a new tensor when the shape
  changes.
* Transform results (`grad`'s `[]Tensor`, `jit.call`'s output) are freed by the
  **caller**; tensors created inside a scope are not. That asymmetry is the one
  rule to remember, and `std.testing.allocator` enforces it in the tests.
* `vmap` vectorises the leading axis only; nested `vmap` is exercised two deep.
