# Linear algebra (`linalg`)

## Usage

```zig
// decide the layout from the shapes (row/column major, transposes)
const layout = try zpu.linalg.matLayout(a, b);

const x = try zpu.linalg.solve(a, b);        // A x = b
const ai = try zpu.linalg.inv(a);            // A^-1
const d = try zpu.linalg.det(a);             // determinant (a scalar)
const c = try zpu.linalg.cholesky(a, false); // lower triangular
const y = try zpu.linalg.triangularSolve(l, b, false);

var lu = try zpu.linalg.luFactor(a);         // reuse the factorisation
defer lu.deinit();
const s = try zpu.linalg.luSolve(&lu, b);

const dp = try zpu.linalg.dot(u, v);
const nm = try zpu.linalg.norm(x, -1, false);
const tr = try zpu.linalg.trace(m);
const dg = try zpu.linalg.diag(v);
const ot = try zpu.linalg.outer(u, v);
const ct = try zpu.linalg.contract(a, b);    // contract the last axis
```

## Implementation intent

* **Errors are values.** A singular matrix is `Error.Singular`, decided by a
  pivot comparison against `SINGULAR_TOL = 1e-12` — the "close enough to zero"
  threshold lives in exactly one place. Returning NaN and letting the caller
  discover it three operations later is not an option.

* **`det` uses a factorisation.** The product of the LU diagonal, not a cofactor
  expansion: any pivot below the threshold makes it return zero.

* **`solve` is differentiable.** The VJP of `x = A⁻¹b` is written as
  `solve(Aᵀ, g)`, which never forms an explicit inverse and is more stable than
  multiplying by `inv(A)`.

* **Transposes and layout are absorbed.** `matLayout` decides which operand has
  to be read transposed, and `matmul` / `solve` follow that decision by changing
  the traversal order instead of making a contiguous copy.

* **`Lu` is reusable.** For a fixed matrix with many right-hand sides (iterative
  solvers, several objectives) call `luFactor` once and `luSolve` repeatedly;
  `Lu.deinit` releases the workspace it allocated.

## Notes

* Differentiating `solve`, `inv` and `det` works. `cholesky` and
  `triangularSolve` currently report `Error.NotSupported` in a gradient context
  instead of pretending the gradient is zero (see
  [Inside automatic differentiation](02-autodiff.html)).
* Everything here is built on `matmul` and element-wise ops, so it inherits the
  dtypes and the CPU performance of those kernels.

## Tests

`linalg with gradients` in `src/zpu/tests.zig` checks `solve`, `det` and `inv`
against hand-computable values (`det([[3,1],[1,2]]) = 5`,
`solve([[3,1],[1,2]], [9,8]) = [2,3]`) and differentiates `solve` and `det` to
compare against finite differences.
