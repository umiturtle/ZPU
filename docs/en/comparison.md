# How ZPU differs from other libraries

Read this if you already know NumPy, PyTorch or JAX and want to know what is
different here — not what is missing.

| | JAX | PyTorch | NumPy | **ZPU** |
| --- | --- | --- | --- | --- |
| Host language | Python + C++/XLA | Python + C++/CUDA | Python, C BLAS/LAPACK | **Zig and the standard library only** |
| Dependencies | CUDA/TPU toolchains | many | many | **none** — `std.DynLib` at run time, nothing at build time |
| API style | functional transforms | eager with autograd | array library | **both: eager ops plus transforms** |
| Memory | garbage collected | reference counted | garbage collected | **scopes with ref-counted buffers (explicit ownership)** |
| Differentiation | tracing transforms | tape per graph | — | **id-only tape, re-computation VJPs, transform based** |
| Backends | XLA (compiler) | many kernels | CPU (BLAS) | **one kernel interface over CPU / WebGPU / Vulkan / Metal / D3D12** |
| Build | pip / conda, code generation | pip | per project | **`zig build`, nothing generated** |

## The differences that actually matter

* **Ownership is part of the API.** Python relies on the garbage collector; here
  every tensor is a value with a reference-counted buffer, and *scopes* decide who
  frees what. That is the price of having no runtime, and the tests enforce it:
  leaks and double frees fail the test suite.

* **The tape stores ids, not values.** A `tape.Node` records the primitive, its
  attributes and the ids of its inputs. VJPs recompute whatever they need from the
  inputs. Memory stays flat and predictable, the JIT can fold whole constant
  subgraphs, and there is no "saved tensors" tuning knob to get wrong.

* **Transforms are ordinary Zig functions.** `grad`, `vmap`, `jvp` and `jit` open
  a scope, install a recorder, call your function, restore the previous state and
  `retain` only what escapes. There is no tracing DSL: if it compiles, it can be
  transformed.

* **Gradient coverage is checked at compile time.** Every primitive is classified
  as "has a VJP", "derivative is zero by definition" or "not implemented yet". An
  unclassified primitive is a build error, and differentiating through an
  unimplemented one is `Error.NotSupported` rather than a silent zero.

* **`comptime` instead of run-time detection.** Dtypes dispatch through `inline
  for` over the dtype enum; `Shape.initComptime` makes shapes compile-time values;
  the VJP coverage table is a `comptime` block. Nothing is discovered by probing
  at start-up beyond what the operating system genuinely has to answer (which GPU
  libraries exist).

* **The toolchain is the product.** `zig build test` / `check` / `bench` / `doc` /
  `examples` / `serve`. No Python, no Makefile, no shell script, no code
  generation step — the HTML documentation is produced by a Zig program in
  `docs/build_docs.zig`, and the benchmarks by `bench/bench.zig`.

* **Backends are a table, not a compile-time choice.** `zbackend` opens
  `libvulkan.so`, `wgpu_native.dll`, `Metal.framework` or `d3d12.dll` at run time,
  enumerates what is there, and dispatches with the same `Kernel` type on the CPU
  when nothing else is. CI does not need a GPU to prove the CPU path works.
