# Execution backends (`zbackend`)

## Usage

```zig
const zpu = @import("zpu");
const zb = zpu.zbackend;

// 1. ask what this machine has (the probe result is cached)
const report = zpu.backends();
std.debug.print("best compute backend: {s}\n", .{report.best().name()});
var buf: [16]u8 = undefined;
for (zb.ALL) |b| {
    const info = report.detail(b);
    std.debug.print("{s: <7} available={} devices={d} api={s}\n", .{
        b.name(), info.available, info.device_count, info.versionString(&buf),
    });
    for (0..info.device_count) |i| std.debug.print("  - {s}\n", .{info.deviceName(i)});
}

// 2. write the kernel once: a Zig body plus per-backend shading source
const Scale = struct {
    fn body(ctx: *zb.KernelCtx) void {
        const data = ctx.buffers[0].as(f32);
        data[ctx.id] = data[ctx.id] * 2.0 + 1.0;
    }
};
const kernel = zb.Kernel{
    .name = "scale_offset",
    .body = Scale.body,
    .source = .{
        .wgsl =
        \\@group(0) @binding(0) var<storage, read_write> data: array<f32>;
        \\@compute @workgroup_size({threads})
        \\fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {
        \\    let i = gid.x;
        \\    data[i] = data[i] * 2.0 + 1.0;
        \\}
        ,
        .msl = ..., .glsl = ..., .hlsl = ...,
    },
    // Vulkan needs SPIR-V. The shipped bytecode comes with the repository
    // (its provenance is recorded in `src/backend/spirv.zig`).
    .spirv = &zb.spirv.scale_offset,
};

// 3. run it: without a GPU the Zig body runs on the CPU
var buffers = [_]zb.Buffer{.{ .ptr = @ptrCast(data.ptr), .len = data.len * @sizeOf(f32) }};
var ctx = zb.KernelCtx{ .buffers = &buffers, .count = data.len, .threads = 64, .alloc = alloc };
try zb.dispatch(.cpu, kernel, &ctx);              // serial
try zb.dispatchThreaded(.cpu, kernel, &ctx, 8);   // split over 8 threads

// 4. let the environment pick the driver (optional): ZPU_VULKAN_LIB /
//    ZPU_WEBGPU_LIB name one library, VK_ICD_FILENAMES / VK_DRIVER_FILES are
//    read as further candidates.
zb.configureFromEnvironment(init.environ_map);

// 5. or hold a device yourself (opening one costs milliseconds)
if (try zb.openDevice(.vulkan)) |dev| {
    var device = dev;
    defer device.deinit();
    try zb.dispatchOn(&device, kernel, &ctx);
}

// 6. when the same data goes round and round, leave it on the device: no
//    upload, no download, no allocation, and no pipeline compile per dispatch.
if (try zb.device(.vulkan)) |dev| {
    var weights = try zb.createDeviceBuffer(dev, 4096, alloc);
    defer zb.destroyDeviceBuffer(dev, &weights);
    try zb.writeDeviceBuffer(dev, &weights, host_bytes);        // once
    for (0..1000) |_| {
        try zb.dispatchDeviceBuffers(dev, kernel, &.{&weights}, 1024, 64);
    }
    try zb.readDeviceBuffer(dev, &weights, into_host);          // when needed
}
```

`zig build run-gpu-probe` prints the table above, lowers the same kernel into five
shading languages and checks threaded CPU dispatch against the serial result. It
then **opens each backend's device for real** and, on a backend whose kernel has
code for it, runs the kernel and checks the numbers — a single wrong number fails
the run. On a machine without a GPU it prints "not available" and only the CPU
runs, which is not a failure but the correct answer for that machine. When at
least one device ran, it also reports the cost of the first dispatch (the
pipeline is built there) and the average of the ones after it.

## Implementation intent

* **Detection is real, and it is only detection.** `probe()` opens the platform
  loader with `std.DynLib` and asks it what exists. For Vulkan that means
  `vkEnumerateInstanceVersion`, a real `vkCreateInstance` and a
  `vkEnumeratePhysicalDevices` walk that reads `VkPhysicalDeviceProperties` for
  the device name and API version, then destroys the instance again. For WebGPU,
  Metal and D3D12 it resolves the single entry point each runtime needs
  (`wgpuCreateInstance`, `MTLCreateSystemDefaultDevice`, `D3D12CreateDevice`) and
  reports the library it came from. No SDK, no headers, no link time dependency:
  the ABI structs are declared locally and the entry points are loaded by name.

* **A missing runtime is a normal answer.** The report says `available=false`,
  `device_count=0`, and `open()` returns `null`; `best()` then falls back to the
  CPU backend. CI machines without a GPU therefore exercise the same code path as
  a laptop with one — just on the other branch.

* **One kernel, many languages.** A `Kernel` carries its source per shading
  language plus an optional pre-compiled SPIR-V blob. `lower()` substitutes only
  `{name}` and `{threads}` and leaves every other brace alone, because real shader
  source is full of them; `canRun(backend)` says which backends can execute a
  given kernel (the CPU backend can always run it, the GLSL path needs SPIR-V).

* **The CPU backend is the reference implementation.** `dispatch` runs the
  kernel body over the invocation space, and `dispatchThreaded` splits the same
  space into strided slices — exactly the partitioning a GPU dispatch performs —
  giving each worker its own context copy and collecting failures into one
  `Error.KernelFailed`. If the threaded result differs from the serial one, the
  kernel was not written to be parallel; the example checks that equivalence.

* **Verbose where it matters.** `Info` carries the loader path, the API version
  and up to eight device names in fixed buffers, so nothing has to be allocated to
  answer "what is on this machine?" — a question the documentation, the examples
  and the benchmarks all ask.

* **Devices come in three stages.** `probe()` (what is there) → `open()` (a
  description of the driver) → `openDevice()` (a live connection). `Device` is a
  tagged union over the backends: `.cpu`, `.vulkan` and `.webgpu` execute,
  `.metal` and `.d3d12` are placeholders. `dispatchOn(&dev, kernel, ctx)` runs on
  one device; `dispatch(backend, kernel, ctx)` decides for itself — a device when
  one opens, the CPU body otherwise. `device(backend)` opens once per process and
  keeps it (creating an instance and a device costs milliseconds, which would
  dwarf the arithmetic if it happened per dispatch).

* **The cache is lock-free.** Each slot walks
  `unopened → opening → opened | unavailable`; a `cmpxchg` lets exactly one
  thread open it. A thread that loses the race does not wait: it runs that
  dispatch on the CPU. A correct answer now beats a few milliseconds of waiting.

* **Vulkan follows the API's own stages.** No headers, no SDK, no link-time
  dependency: the ABI structs (`VkInstanceCreateInfo`, ...) are declared in this
  repository and entry points are resolved by name. The staging is not a
  formality: with a null instance `vkGetInstanceProcAddr` is only obliged to
  return the global commands (`vkCreateInstance` and friends) — measured against
  SwiftShader, everything else came back NULL. So (1) opening the library
  resolves the globals, (2) `createInstance` resolves the instance-level
  commands on the instance it just made (`Loader.resolveWith`), and (3)
  device-level commands come from `vkGetDeviceProcAddr`, the API's fast path
  where the loader hands back a direct pointer instead of a trampoline.
  `probe()` walks the same steps, reads the API version and the device names,
  and always destroys what it made. Once the device exists we look for a compute
  queue family and host-visible coherent memory, and buffers stay mapped for
  their lifetime (upload and readback are `memcpy`s). `count` and `threads`
  travel as a 16-byte push constant, `{ u32 count; u32 threads; u32 pad[2]; }`.

* **Pipelines are not rebuilt.** Creating a shader module, descriptor set
  layout, descriptor set, pipeline, command buffer and fence per dispatch used to
  include compiling the shader, and that dominated everything else. Now the
  device keeps them, keyed by the SPIR-V hash and the binding count (four fixed
  slots; a different kernel hashing to a slot recycles it), and one command
  buffer plus one fence are reset instead of reallocated. What changes per
  dispatch is the descriptor contents. Measured here (SwiftShader, 1024
  invocations, 200 dispatches): 1975–2160 µs per dispatch without the cache,
  60–85 µs with it.

* **Data that should stay put goes in a `DeviceBuffer`.** `dispatch` and
  `dispatchOn` send every `KernelCtx.buffer` to the device, run, and read it back
  — the right default for tensors that live in host memory, and a waste in a loop
  over the same data. `createDeviceBuffer` / `writeDeviceBuffer` /
  `dispatchDeviceBuffers` / `readDeviceBuffer` allocate and transfer once and keep
  the data on the device meanwhile. The one-shot path also draws its buffers from
  the device's pool and its descriptor arrays from the stack, so a warm device
  sees no allocation at all. Measured here (SwiftShader, 1024 invocations, five
  runs): 28.2 / 36.0 / 40.5 / 35.8 / 29.5 µs per dispatch on a resident buffer,
  against 30.9 / 36.1 / 31.2 / 29.1 / 33.2 µs when the host buffers are copied in
  and out — on a software implementation the two are the same, because "device
  memory" is host memory, the copy is a memcpy, and submit plus fence wait
  dominate. The gap appears on real hardware, where the question is whether every
  dispatch pays for a transfer over PCIe. Speed is not the only thing a
  `DeviceBuffer` buys: a loop that reuses one allocation never touches the
  driver's memory manager, so there is no churn and no OOM jitter.

* **The workgroup size comes from the module.** `@workgroup_size` is baked into
  the bytecode, so trusting the caller's `threads` would leave invocations
  unrun. `spirv.localSize()` reads `OpExecutionMode LocalSize`, and that value
  decides the workgroup count and what the push constant carries.

* **Drivers are found by name, and can be named.** Besides the per-platform
  names (`libvulkan.so.1`, `vulkan-1.dll`, `libMoltenVK.dylib`, ...), Linux also
  searches the multiarch directory and the software rasterizers (SwiftShader,
  lavapipe). `setDriverLibrary` / `ZPU_VULKAN_LIB` / `ZPU_WEBGPU_LIB` point at one
  library, `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` are read as further candidates,
  and `null` restores the defaults. An ICD works as well as a loader: a Vulkan
  driver exports `vkGetInstanceProcAddr` itself. In a Linux program that does not
  link libc only the loader's names are tried — Zig's built-in ELF loader does
  not run a library's constructors, so a driver written in C++ (SwiftShader, for
  instance) would never be initialised. Ordinary GPU programs link libc, and then
  the platform loader takes care of bringing the ICD up.

* **WebGPU follows the `webgpu.h` conventions.** To work with Dawn
  (`wgpuGetProcAddress`) and with implementations that export plain symbols
  (wgpu-native), the `Loader` field names and the `wgpu*` symbol names live in one
  table — and **a missing entry fails the build at comptime**. No runtime string
  juggling, and no instrumentation to keep in sync. Strings are `WGPUStringView`
  (length-prefixed, not NUL-terminated) and callbacks use the
  `AllowProcessEvents` mode driven by `wgpuInstanceProcessEvents`. Buffers are
  `MappedAtCreation`: write, unmap, dispatch, then `wgpuBufferMapAsync` and poll
  to read back.

* **There is exactly one kernel ABI.** `@group(0) @binding(i)` is
  `KernelCtx.buffers[i]`, and **every buffer is bound as a storage buffer**. The
  entry point is always `main` (`Kernel.lower` substitutes `{threads}`, so the
  shader owns its `@workgroup_size`). Those two rules plus Vulkan's push constant
  block are the whole contract a shader sees.

## Status

| Backend | Detection | Execution |
| --- | --- | --- |
| CPU | always | `dispatch` / `dispatchThreaded` / `dispatchOn` |
| Vulkan | loader + instance + physical device enumeration (API version and device names) | **done, verified on a real device**: device, compute queue, command pool, host-visible memory, cached SPIR-V pipelines, SSBOs, push constants, fence |
| WebGPU | loading the implementation (Dawn / wgpu-native) | **done**: adapter request, device request, WGSL module, storage buffers, bind group, compute pass, readback |
| Metal | `MTLCreateSystemDefaultDevice` | planned |
| Direct3D 12 | `D3D12CreateDevice` | planned |

Vulkan and WebGPU come first; the other two keep their detection code and their
seat in `Backend.ALL`.

**What is verified where.** This development machine has neither a GPU nor
`/dev/dri`, so the Vulkan path runs against a software driver:
`ZPU_VULKAN_LIB=/path/to/libvk_swiftshader.so zig build run-gpu-probe` opens the
device, runs the shipped SPIR-V, checks the numbers and measures the 60–85 µs per
dispatch quoted above. On a machine with a GPU, plain `zig build run-gpu-probe`
performs the same checks, because the loader and the ICD are already there. The
tests follow the same rule: where `probe()` finds a driver they **must** open the
device, run the kernel and match the numbers, and they only fall back to the CPU
equivalence checks where it does not. What can be checked without hardware:
(1) loader discovery and staged resolution, (2) ABI struct layouts (comptime
tests pin sizes and offsets), (3) completeness of the entry-point table
(comptime), (4) returning `null` and falling back to the CPU when a backend
cannot run, and (5) CPU execution equivalence.

## Notes

* `Backend.ALL` lists all five in a fixed order, so UIs and logs need no special
  cases.
* The probe result is cached per thread and dropped by `zpu.deinit()`, which is
  why calling `zpu.backends()` inside a loop is cheap.
* Kernel sources are plain strings: a host that knows nothing about WGSL can
  still hand them to a backend that does.
* **Vulkan requires SPIR-V** (`.spirv`). The GLSL source is kept for the record,
  but the translation into SPIR-V happens outside Zig (`glslangValidator` and
  friends) and the artifact is embedded — the build never calls an external tool.
  A kernel without `.spirv` is not runnable on Vulkan and falls back to the CPU
  (`canRun` says so honestly).
* **WebGPU's entry point is always `main`**, and `@workgroup_size` is baked into
  the source through `{threads}`; `ctx.threads` only affects how the id space is
  split.
* **Do not use uniforms.** Even a small constant goes into a storage buffer
  (`var<storage, read> scale: array<f32>;`), because the backends describe every
  buffer as a storage buffer.
* **Tensors in host memory travel on every dispatch.** `dispatch` sends the
  buffers, runs, and reads them back. The allocation comes from the pool, the
  descriptor arrays from the stack and the pipeline from the cache, so what
  remains is the transfer plus submit and fence. Data that goes round in a loop
  belongs in a `DeviceBuffer`, which removes the round trip as well (keeping the
  numeric core's tensors resident is the next step; today a loop calls the kernel
  directly).
* Close a device with `Device.deinit` (the value `openDevice` handed out) or
  `zpu.shutdownDevices()` (the process-wide cache) — never while a dispatch is
  running.
* **A `DeviceBuffer` dies when it is released.** Reading, writing or dispatching
  with a dead buffer, or with one from another device, fails with
  `Error.InvalidBuffer`; calling `destroyDeviceBuffer` twice is harmless. The
  memory goes back to the device's pool, so the next allocation of a similar size
  does not trouble the driver.
* `dispatchDeviceBuffers` binds at most 16 buffers per dispatch
  (`RunBuffers.max`); no kernel in the library needs more.
