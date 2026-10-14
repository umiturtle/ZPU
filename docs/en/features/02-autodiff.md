# Inside automatic differentiation (`prims` / `tape` / `autodiff`)

If you only use `grad`, you never need this page. It explains why `grad` can be
both fast and slow, and how a new op gets its derivative.

## Usage (adding a rule)

1. Add the op to `prims.Prim`.
2. Write the kernel in `ops` and finish it with `ops.finish(prim, inputs, out, attrs)`.
3. Add the **VJP** — the cotangent with respect to input `k` — to
   `autodiff.contribution(node, k, g)`. If forward mode should support it, add the
   `jvp` rule too.

```zig
// VJP example: y = exp(x) gives dx = g * exp(x)
.exp => blk: {
    const x = node.inputs[0];
    const e = try ops.exp(x);            // recomputed, not saved
    defer ops.releaseTemp(e);
    break :blk try ops.mul(g, e);
},
```

If you add a primitive and forget to classify it, the build fails. The
classification is `rule` / `zero` / `unimplemented`, and both directions are
checked at compile time: `contribution` refuses to answer with silence, and
`prims.differentiable` has to agree with the classification.

## Implementation intent

* **Recompute, don't store.** A `tape.Node` holds the primitive, its attributes
  and the ids of its inputs — no values. A VJP rebuilds whatever it needs from the
  inputs. Memory stays flat and the JIT can fold whole constant subgraphs; the
  price is extra arithmetic, which is why ops like `softmax` are written in a form
  that can be recomputed stably.

* **The engine owns the gradients.** `autodiff.Engine` keeps an `id -> Tensor`
  map and merges contributions with `seed(t, g)`: an existing gradient is added
  and the old reference released, a missing one is stored. The map holds its own
  reference and `Engine.deinit` drops it directly — not through `releaseTemp`,
  because these references are deliberately outside the scope `grad` opened.

* **Relevance analysis first.** `run` walks the tape backwards from `wrt` and
  marks every node that can reach it; the sweep then visits only marked nodes.
  Branches that cannot influence the result are never differentiated.

* **The tape is a stack.** A fixed `MAX_TAPE_DEPTH = 8` stack with
  `pushTape`/`popTape` and a whole-stack `save`/`restore`. `grad` records the
  forward pass on its own tape and then *pops* it before sweeping, so the
  sweep's own ops are not appended to the tape being read. `hvp` does the
  opposite on purpose: it pushes an outer tape and lets both the inner forward
  pass and the inner sweep land on it.

* **Unimplemented is loud, not silent.** `contribution` returns
  `Error.NotSupported` for primitives whose VJP is not written (`cholesky`,
  `triangular_solve`) and `null` for the ones whose derivative is genuinely zero
  (`floor`, comparisons, `argmax`, …). Zero never stands in for "unknown".

* **`seedOnes` is the identity cotangent.** The reverse sweep starts from ones,
  and `jvp` uses the same helper to seed a tangent of the right shape.

## How the tests argue

The gradient tests in `src/zpu/tests.zig` rest on three kinds of evidence:

* analytic solutions (`d/dx Σ exp(x²) = 2x·exp(x²)` and friends),
* finite differences over functions that touch many ops,
* identities (`grad grad sum(x*x) == 2`, `grad grad quad == (3, 4)`).

On top of that, the `comptime` block in `autodiff.zig` makes gradient coverage a
compile-time property, and the tape/graph lifetimes are checked by the allocator
in every test.
