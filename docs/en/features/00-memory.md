# Memory and shapes (`dtype` / `shape` / `buffer` / `tensor`)

## Usage

```zig
const S = zpu.Shape;

// a compile-time shape: the compiler checks the indices
const x = try zpu.Tensor.zeros(S.initComptime(&.{ 2, 3 }), .f32);

// a run-time shape
const dims = [_]usize{ rows, cols };
const y = try zpu.Tensor.ones(try S.init(&dims), .f64);

x.shape.dims[0];        // 2
x.rank();               // 2
x.numel();              // 6
x.dtype();              // .f32
x.isContiguous();
x.strides[0];           // row stride

// reading and writing goes through element indices
x.readF64(0);
x.writeF64(0, 1.5);
try zpu.ops.copyInto(x, y);   // view to view
```

Lifetime is managed with scopes:

```zig
var scope = try zpu.Scope.init(alloc);
defer scope.deinit();                 // frees every reference created inside

const a = try zpu.Tensor.ones(S.initComptime(&.{4}), .f64);   // owned by the scope
const b = try zpu.exp(a);                                     // same

// values that must outlive the scope (model weights, optimiser state)
const state = try zpu.Tensor.zeros(S.initComptime(&.{4}), .f64);
zpu.untrack(state);                   // now the caller owns it
```

## Implementation intent

* **`Shape` is a fixed-rank value type.** It carries `dims` for up to `MAX_RANK`
  (8 by default); `initComptime` keeps comptime-known dimensions, `init` packs
  run-time ones. Broadcast and reduction code works on those fixed-size arrays,
  so there is no per-rank branching explosion, and `payloadRank()` (rank minus the
  `vmap` depth) gives the axes their meaning back.

* **`Buffer` is 64-byte aligned and reference counted.** The counter is an
  `std.atomic.Value(u32)`; `createWith` allocates the payload with
  `alignedAlloc(u8, .fromByteUnits(ALIGN), n)` and the descriptor separately. 64
  bytes suits both vectorised kernels and GPU buffer alignment, and the atomic
  count means a tensor can be handed to another thread.

* **A view is a new reference.** `viewOf` builds a `Tensor` with changed
  `(shape, strides, offset)` and *retains* the buffer; the new reference is
  adopted by the active scope (`track`). Without a scope the creator owns it — the
  reason a view created outside a scope must be released by hand.

* **`Scope`/`Tracker` is the only lifetime mechanism.** `Tracker.begin` installs
  the tracker into the per-thread context, `track` adopts every tensor created
  afterwards, and `end` releases them all. It is deliberately not thread local:
  the context is per thread, so two threads can each have their own scope.

* **`untrack` hands a creation reference to the caller.** Operations on tensors
  must never do this implicitly, because that is exactly how double frees happen:
  an object that the scope also plans to release. Owning a value across scope
  boundaries is always an explicit decision.

## Notes

* `readF64` / `writeF64` assert that the tensor is contiguous; they are the
  bridge to numbers, not the fast path.
* `Tensor.item()` on a multi-element tensor panics in debug builds; use
  `readF64(0)` once you have reduced to a scalar.

## Tests

`image basics`, `tensor basics` and `scopes own what they create` in
`src/zpu/tests.zig`, plus the module tests collected by `zig build test`, cover
strides, views, dtype round trips and scope ownership. Every test allocates
through `std.testing.allocator`, so a lifetime mistake fails the run.
