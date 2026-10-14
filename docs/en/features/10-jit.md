# Execution plans (`jit`)

`zpu.jit` is the "trace once, replay many times" transform. It traces a Zig
function once, turns it into a `Plan` (`zpu.plan`), and every later call only
replays the plan. It nests inside `grad`, so `jit(grad(f))` is a plan that
contains the backward pass.

## Usage

```zig
fn f(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
    _ = ud;
    const h = try zpu.matmul(args[0], args[1]);
    defer zpu.releaseTemp(h);
    return zpu.sumAll(zpu.tanh(h));
}

var j = zpu.jit(alloc, f, null); // nothing traced yet
defer j.deinit();

const y = try j.call(&.{ x, w }); // first call: record, then run
defer y.deinit();

const y2 = try j.call(&.{ x2, w2 }); // later calls: replay only
defer y2.deinit();

_ = j.trace_count; // 1 (the function was traced once)
_ = j.numSteps(); // steps left after folding and elimination
if (j.plan) |p| try p.describe(writer); // human-readable plan
```

* Arguments are bound **by position**. Handing in a different tensor of the same
  shape does not re-trace.
* **Changing shapes or dtypes means calling `j.trace(&args)` explicitly.** Folded
  constants were fixed at the shape they were traced with, so silently feeding
  another shape produces wrong numbers.
* A `Plan` can also be built by hand: `zpu.plan.build(alloc, &graph, arg_ids,
  out_ids)` is the entry point when you recorded the `Graph` yourself.

## What it does

1. **Trace** — install a `Graph` and run `f` once. `ops` record whenever a graph
   is installed, regardless of the execution mode, so the body of a `grad`
   (the backward pass) lands in the trace too.
2. **Build** — `zpu.plan.build` folds the graph in four passes:
   * *dead code elimination*: walk back from the requested outputs and keep only
     what they depend on;
   * *arguments*: the ids the caller supplies (`arg_ids`) are never computed;
   * *constant folding*: every node that does not depend on an argument is
     evaluated **once** here and kept as a `Blob` (weights, literals, and any
     slicing or transposing on top of them);
   * *capture*: a kept step that reads something neither an argument nor
     produced by the trace (a constant built inside the function) is `retain()`ed
     so the plan can hold it.
3. **Replay** — `Plan.run` / `runAll` bind the arguments by position and execute
   the `Step` list through `ops.applyPrim`. The returned tensors are
   `retain()`ed: **the caller owns them**, the same rule as every other
   transform in the library.

## Implementation intent

* **Why tracing.** Zig has no AST hooks and `comptime` cannot look inside an
  arbitrary function body, so the plan is recorded by *running* the function once
  (JAX's approach). That is what lets one op implementation serve eager
  execution, `grad` and `jit` alike.
* **The recorder lives in the context.** The graph is installed in the
  thread-local `context`, and ops simply record "if one is installed". That is
  why a `grad` inside a `jit` (or the other way round) cannot lose a node.
* **Nodes carry ids, not pointers.** Keying on pointers would confuse a freed
  tensor with a new one that reused its address. `Tensor.id` is a monotonic
  counter, so ids are never reused.
* **The graph owns a reference.** There is no GC: `record` retains every input
  and output, and `Graph.deinit` releases one reference each, so a trace survives
  the `Scope` that produced it.
* **Folding runs with recording off.** `evalFolded` detaches the tape and the
  graph and evaluates in `.compiled` mode, so a folded node cannot record itself
  and grow the graph without end.
* **Replay allocates twice.** A `Scope` for the temporaries and one hash map from
  id to `Tensor`. The step array is fixed at build time and never grows.
* **A plan pins the shapes.** Since constants are baked in, a different shape
  needs a new plan; re-tracing explicitly (rather than silently) was the choice
  that makes the cost visible.

## Tests

* `jit folds constants and eliminates dead code` in `src/zpu/jit.zig` checks the
  folding, the elimination and that replaying does not add traces.
* `src/zpu/tests.zig` covers `jit` among the other transforms.
* `zig build bench` measures `jit(grad MLP) f32` — a whole gradient turned into a
  plan (`--filter jit` runs just that case).
