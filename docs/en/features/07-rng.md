# Random numbers (`rng`)

## Usage

```zig
var gen = zpu.rng.Rng.init(42);                     // the seed is explicit
const u = try gen.uniform(zpu.Shape.initComptime(&.{ 2, 3 }), .f32);
const n = try gen.normal(zpu.Shape.initComptime(&.{ 4 }), .f64);
_ = gen.nextU64();

// global helpers (implicit seed)
zpu.rng.seed(7);
const a = try zpu.rng.uniform(shape, .f64);
const b = try zpu.rng.normal(shape, .f32);
const c = try zpu.rng.randint(shape, 0, 10);
const p = try zpu.rng.permutation(10);              // a shuffle of 0..n-1
```

## Implementation intent

* **The state is passed around.** `Rng` is a value type: copying it gives an
  independent stream. Nothing depends on a global state, so tests are
  deterministic and adding threads does not change the sequence.

* **Uniform numbers use the high bits.** `nextU64` is mapped into `[0, 1)` from
  the *upper* mantissa bits, because the low bits of a linear congruential
  generator have a short period. The result is then scaled into the requested
  dtype.

* **Normal numbers come from Box-Muller.** Two at a time, discarding the spare
  when an odd number is requested, and written as `sqrt(-2 ln u)` with a single
  trig call instead of a `sin`/`cos` pair.

* **Sampling fills the tensor directly.** No intermediate array: one allocation
  for any tensor size, for `f32` and `f64` alike (`randint` covers integers).

* **Relation to `dropout`.** `nn.dropout` draws its mask with `gen.uniform` and
  scales by `1/(1-p)`. The generator is a parameter of the layer, so "fix the
  seed in the middle of training to reproduce" is written directly.

## Tests

`rng: determinism and distribution` checks that two generators with the same
seed produce identical tensors, and that 20000 normal samples have a mean near 0
and a variance near 1. The tolerances (0.03 and 0.05) are comfortably above the
standard error of that sample size (~0.007 for the mean, ~0.01 for the variance),
so the test is stable across platforms.
