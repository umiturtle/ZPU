# Optimisers (`optim`)

## Usage

```zig
var opt = zpu.optim.Adam.init(alloc, .{ .lr = 1e-2 });
defer opt.deinit();

var params = [_]zpu.Tensor{ w1, b1, w2, b2 };
for (0..steps) |_| {
    const g = try zpu.grad(alloc, loss, null, &params, &.{ 0, 1, 2, 3 });
    defer {
        for (g) |*t| {
            var tt = t.*;
            tt.deinit();
        }
        alloc.free(g);
    }
    try opt.step(&params, g);
}
```

`Sgd` (with momentum) and `RmsProp` have the same shape:

```zig
var sgd = zpu.optim.Sgd.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
var rms = zpu.optim.RmsProp.init(alloc, .{ .lr = 1e-3 });
```

## Implementation intent

* **State is keyed by parameter id.** Adam's first and second moments, momentum
  and the RMS average all live in an `id -> Tensor` map, so the `step` code does
  not change when a model gains parameters.

* **State is moved out of the scope with `untrack`.** That state is created with
  `Tensor.zeros(...)`, which inside the training loop's scope would be freed when
  the scope ends. Each optimiser calls `zpu.untrack` right after creation and
  becomes the owner; `opt.deinit()` releases it.

* **Updates are written in place.** `subInPlace` performs `p[i] -= delta[i]`
  directly on a contiguous parameter tensor (through `readF64`/`writeF64`), so a
  long training loop allocates nothing new. A gradient of a different dtype is
  cast first.

* **The caller owns the gradients.** `step` borrows `grads` and never frees them;
  that is why the loop above has the `defer` block. Keeping ownership explicit is
  a consistent policy in this library — and the tests detect both leaks and
  double frees through `std.testing.allocator`.

## Tests

`optimisers minimise a quadratic` runs the optimisers for real: Adam (400 steps)
drives `(w-3)² + (b+1)²` to `w=3, b=-1`, and SGD with momentum (200 steps) drives
`(w-3)²` to `w=3`. The learning rates and step counts are pinned inside the test,
so a regression shows up immediately.

## Next

* learning rate schedules (`cosine`, `one_cycle`),
* weight decay and gradient clipping,
* parameter groups (a different learning rate per layer).
