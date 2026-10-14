# Recording (`graph`)

`zpu.graph` is the first half of `jit`, on its own: it records the ops that run
into a flat list, and `zpu.plan` turns that list into an execution plan. Users
rarely touch it directly — it is the place to look when you want to know what a
trace actually contains, or when you want to write your own executor.

## Usage

```zig
var g = zpu.graph.Graph.init(alloc);
defer g.deinit();

_ = zpu.graph.setActiveGraph(&g); // recording starts here
defer _ = zpu.graph.setActiveGraph(null); // and stops here

const y = try zpu.tanh(x); // ops record whenever a graph is installed
_ = y;

try std.debug.print("{d} nodes\n", .{g.len()});
for (g.nodes.items) |n| {
    try std.debug.print("{s} ({d} inputs)\n", .{ @tagName(n.prim), n.n_inputs });
}

// id: consumed but never produced = the function's arguments (and outside
// constants)
const externals = try g.externalIds(alloc);
defer alloc.free(externals);
```

* A `Node` is a `prim`, up to three `inputs`, an `out` and `attrs`.
* `producedBy(id)` answers "does some node in this graph produce that id".
* When recording nests, restore the pointer `setActiveGraph` returns (a `grad`
  inside a `jit` has two recorders alive at once).

## Implementation intent

* **A flat list, not a tree.** A plan is built by walking back from the outputs,
  which needs neither recursion nor a depth limit — and recursion costs stack in
  Zig, so the flat shape is both safer and easier to read.
* **Recording is independent of the mode.** An op records whenever a graph is
  installed, even in plain `.eager` mode. That is what makes `jit(grad(f))` work
  without a special case: the backward pass goes through exactly the same path.
* **Ids, not pointers.** Keying on pointers would confuse a freed tensor with a
  new one that landed on the same address; `Tensor.id` is monotonic and never
  reused.
* **The graph owns a reference.** `record` retains every input and output and
  `Graph.deinit` gives one reference each back, so a trace outlives the `Scope`
  that produced it — which is exactly what lets the plan be built later.
* **Recording is best effort.** A failed append is swallowed (`catch {}`):
  recording changes no result, so there is no reason to fail a computation just
  because a diagnostic list could not grow.
* **`MAX_INPUTS = 3`.** Every primitive today takes one or two inputs, so three is
  enough; an op with four inputs would raise that constant and the copy in
  `plan.build` with it.

## Tests

* `graph records and releases` in `src/zpu/graph.zig` covers recording, release
  and the leak check (`std.testing.allocator`) in one test.
* The conversion into a plan is covered by the tests in `src/zpu/jit.zig`, and
  `jit` end to end by `src/zpu/tests.zig` and the `jit(...)` case in
  `zig build bench`.
