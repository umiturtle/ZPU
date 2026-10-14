# Guide

From an empty file to a trained model, in the order you will actually need it.

## 1. Install the context

Every allocation ZPU makes goes through a per-thread context, so it must be
installed once before the first tensor exists:

```zig
const std = @import("std");
const zpu = @import("zpu");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    try zpu.init(alloc, .{});
    defer zpu.deinit();
}
```

`zpu.init` records the allocator and probes the execution backends once
(`zpu.backends()`, `zpu.bestBackend()`).

## 2. Tensors and scopes

```zig
const a = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, zpu.Shape.initComptime(&.{ 2, 2 }));
const b = try zpu.ones(zpu.Shape.initComptime(&.{ 2, 2 }), .f64);

var scope = try zpu.Scope.init(alloc);
defer scope.deinit();                       // frees everything created inside

const c = try zpu.add(a, b);                // owned by `scope`
const d = try zpu.matmul(c, a);
const e = try zpu.sumAll(zpu.tanh(d));      // a scalar tensor
_ = e.readF64(0);
```

The rule that is worth memorising:

* **created inside a scope → the scope frees it** (never call `deinit` yourself);
* **outside any scope → you free it**;
* **returned by a transform (`grad` / `jvp` / `vmap` / `jit`) → you free it**;
* temporaries inside library code go through `zpu.releaseTemp`, which is a no-op
  while a scope is active.

Because the tests run under `std.testing.allocator`, breaking that rule fails the
build immediately: a leak *and* a double free are both errors.

## 3. A first gradient

```zig
fn loss(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
    _ = ud;
    const h = try zpu.matmul(args[0], args[1]);
    defer zpu.releaseTemp(h);
    return zpu.sumAll(zpu.tanh(h));
}

const grads = try zpu.grad(alloc, loss, null, &.{ x, w }, &.{ 0, 1 });
defer alloc.free(grads);
defer zpu.keepGrad(alloc, grads, 0).deinit();   // keep one, release the rest
```

`grad` returns one tensor per entry of `wrt`; `valueAndGrad` also returns the
primal value, `jvp` the forward-mode tangent, `hvp` a Hessian-vector product, and
`vmap` vectorises the function over the leading axis.

## 4. Training a model

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

Optimiser state is created with `untrack`, so it outlives the scope of one step;
`opt.deinit()` releases it.

## 5. Making it faster

```zig
var j = zpu.jit(alloc, loss, null);
defer j.deinit();
const y = try j.call(&.{ x, w });   // traced once, replayed afterwards
```

`jit` records the first call as a graph, folds the parts that do not depend on
the arguments, and replays the plan. It retraces when the argument *ids* change.

## 6. Pictures

```zig
const zgf = @import("zgf");

var img = try zgf.Image.init(alloc, 256, 256, 3);
defer img.deinit();
img.fill(.{ .r = 0.1, .g = 0.2, .b = 0.3 });
img.drawCircle(128, 128, 60, .{ .r = 1, .g = 1, .b = 0 });

const soft = try img.blur(2);      // a tensor expression, so it is differentiable
const edges = try img.sobel();
try img.savePpm(init.io, "out.ppm");
```

## 7. Which backend will run

```zig
const report = zpu.backends();
std.debug.print("best backend: {s}\n", .{@tagName(report.best())});
for (zpu.Backend.ALL) |b| {
    const info = report.detail(b);
    std.debug.print("{s}: available={} devices={d}\n", .{ @tagName(b), info.available, info.device_count });
}
```

`zig build run-gpu-probe` prints exactly this table for the machine you are on,
plus the same kernel lowered to WGSL, MSL, GLSL and HLSL.
