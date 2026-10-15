# Automatic differentiation — `prims` / `tape` / `autodiff` / `transforms`

ZPU's autodiff draws heavily from JAX's design: it offers **function transforms** —
`grad`, `jvp`, `vjp`, `jacobian`, `hessian`, `hvp`, `vmap`, `jit` — that are
**composable** (`jit(grad(f))`, `jacobian(grad(f))` = `hessian(f)`). That is the main
selling point. You do not need this page to use it — `transforms`'s API is enough.
This page is for "how do I add a rule", "why do higher-order derivatives work",
and "how does this differ from JAX".

## The transforms

| Transform | Type | Description |
|-----------|------|-------------|
| `grad(f, args, wrt)` | `(R^n → R) → R^n` | Gradient of a scalar function. Reverse mode. |
| `valueAndGrad(f, args, wrt)` | `→ (R, R^n)` | Function value and gradient together. |
| `vjp(f, args, wrt, cotangent)` | `→ (R, R^n)` | VJP with an explicit cotangent. |
| `jvp(f, args, tangents)` | `→ (R^m, R^m)` | Jacobian-vector product. Forward mode. |
| `jacobian(f, args, wrt)` | `→ R^{m×n}` | The full Jacobian matrix. |
| `hessian(f, args)` | `→ R^{n×n}` | The full Hessian (implemented as `jacobian(grad(f))`). |
| `hvp(f, args, wrt, v)` | `→ R^n` | Hessian-vector product. Reverse-over-reverse. |
| `vmap(f, args)` | `→ R^{k×...}` | Vectorising map over the leading axis. |
| `jit(f)` | `→ Plan` | Trace → execution plan with constant folding. |

The difference from JAX: **`jacobian` and `hessian` are top-level functions**.
In JAX, `jax.jacobian` / `jax.hessian` are defined as compositions of `vjp` /
`jvp`; ZPU's implementation is the same internally (`hessian = jacobian ∘ grad`),
but the common ones are exposed directly.

## Adding a rule

1. Add the op to `prims.Prim`.
2. Write the kernel in `ops` (ending with `ops.finish(prim, inputs, out, attrs)`).
3. Write the **VJP rule** (cotangent w.r.t. the `k`-th input) in
   `autodiff.contribution(node, k, g)`. For forward mode, also add a JVP rule.

```zig
// VJP example: y = exp(x) => dx = g * exp(x)
.exp => blk: {
    const x = node.inputs[0];
    const e = try ops.exp(x);            // recompute the forward value (no save)
    defer ops.releaseTemp(e);
    break :blk try ops.mul(g, e);
},
```

### Rule classification (checked at comptime)

Every `Prim` falls into one of three classes:

* **`.rule`** — has a case in `contribution`. `exp`, `sin`, `matmul`, ...
* **`.zero`** — gradient is always zero. `floor`, `ceil`, `eq`, `lt`, `argmax`, ...
* **`.unimplemented`** — runtime `Error.NotSupported`. `cholesky`, `conv`, ...

This classification is verified at comptime by `classify(prim)`; adding an
unclassified primitive fails the build. The missing rules for `.reverse`
(= `flip`) and `.copy` were found this way.

## Implementation intent

### Recompute instead of saving (checkpointing substitute)

The tape holds no values — only primitives, attributes, and input ids
(`tape.Node`). The VJP recreates intermediate values on the spot.
Memory-friendly and composes well with `jit`'s constant folding.
The cost is recomputation, so ops like `softmax` that have a stable
recomputation form are written that way.

JAX's `jax.checkpoint` / `remat` does not exist yet (under consideration in #8).
In principle, it would flag tape nodes with `checkpoint` and re-run `f`
during the VJP to recreate intermediates.

### The `Engine` owns the gradients

`autodiff.Engine` holds an `id -> Tensor` map, accumulating with `seed(t, g)`.
An existing gradient is summed with `add` and the old reference released;
otherwise the new one is stored. Map values are `retain()`ed and freed
directly by `Engine.deinit` (not `releaseTemp`, because `grad` opens a
scope, and the engine must drop references that are *not* scope-owned).

### Relevance filtering

`run` first walks backwards from `wrt` marking the nodes that need gradients
(the `needed` map). The sweep only looks at marked nodes. Branches unrelated
to `wrt` are not computed. Inside the loop, `tape.nodes.items[idx]` is
re-read every iteration — the cotangent computation itself calls ops that
may append nodes to the same tape, reallocating its storage.

### The tape is a stack

A fixed-length stack of `MAX_TAPE_DEPTH = 8`, with `pushTape` / `popTape` /
`setTape` and a bulk-save `save` / `restore`.

`grad` records the forward computation on an inner tape and sweeps on that
tape after popping it. The pop prevents the sweep's own ops from being
appended to the tape being read.

`hvp` is the opposite: the sweep's ops must go on the **outer tape**, so it
does `saved = context.save()`, then `pushTape(&outer)`, then calls
`valueAndGrad`. This is how higher-order derivatives work: the outer tape
records "the gradient computation itself", and that is differentiated again.

### JVP (forward mode)

`jvp` returns `J · v`, like JAX, but the implementation is **m reverse-mode
calls** (m = output elements):

```
for j in 0..m:
    cotangent_j = one_hot(j)
    grad_j = vjp(f, args, cotangent_j)
    out_tangent[j] = sum(grad_j * tangent for each arg)
```

When the output is small (scalar), reverse mode is faster; when the output
is large, a true forward-mode JVP would be faster. JAX has both; ZPU has
only the reverse-mode implementation.

### `jacobian` and `hessian`

`jacobian(f, args, wrt)` calls VJP once per output element to build the
Jacobian row by row. Cost: `O(m)` VJP evaluations.

`hessian(f, args)` is implemented as `jacobian(grad(f))`: build a function
that returns `grad(f)`, then take its Jacobian. Internally, `valueAndGrad`
is called m times, so cost is `O(n²)`. Hessians are needed for Newton steps,
saddle-point detection, and uncertainty estimation — cases where the
parameter count is usually small, so this is practical.

## Tests

Gradient tests in `src/zpu/tests.zig` use three sources of truth:

* Analytic solutions (`d/dx Σ exp(x²) = 2x·exp(x²)`, etc.)
* Finite-difference comparison (`gradcheck`: functions with many ops)
* Identities (`grad grad sum(x*x) == 2`, `grad grad quad == (3, 4)`, etc.)

A test in `autodiff.zig` verifies that all VJP rules compile; missing cases
in the `switch` become build errors immediately.

`src/zpu/tests_grad.zig` is the dtype × op gradient table: 33 elementwise
unary, 12 binary, 12 reductions, 11 shape/indexing, 6 linalg, 12 nn
layers/losses, compared against finite differences in both f64 and f32.

## JAX correspondence

| JAX | ZPU | Notes |
|-----|-----|-------|
| `jax.grad` | `zpu.grad` | Same |
| `jax.value_and_grad` | `zpu.valueAndGrad` | Same |
| `jax.jacobian` | `zpu.jacobian` | Top-level in ZPU |
| `jax.hessian` | `zpu.hessian` | Implemented as `jacobian(grad(f))` |
| `jax.jvp` | `zpu.jvp` | Internally m VJP calls |
| `jax.vjp` | `zpu.vjp` | Explicit cotangent |
| `jax.vmap` | `zpu.vmap` | Implemented via `batch_depth` |
| `jax.jit` | `zpu.jit` | Trace + constant folding |
| `jax.custom_jvp` | **not implemented** | #6 |
| `jax.custom_vjp` | **not implemented** | #6 |
| `jax.checkpoint` | **not implemented** | #8 |

## Known limitations

* **`conv2d` VJP** — returns `NotSupported`. Needs either a differentiable
  rewrite or a `conv_transpose`-based VJP (#5).
* **`scatter_add` VJP** — the gradient w.r.t. input `x` was `g` but should be
  zero (`x`'s values are not used in the output). Fixed (#3).
* **`prod(axis, keepdims=false)` VJP** — `expandReduced` mishandles the shape,
  producing `ShapeMismatch` (#4).
* **`hessian` memory leak** — the `GradWrapper` returns a tensor whose ownership
  mismatches the scope, leaking 6 allocations. Functionally correct but needs
  an ownership-model fix (#8).
