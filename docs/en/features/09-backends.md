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
    .spirv = null, // Vulkan needs SPIR-V (see "Notes")
};

// 3. run it: without a GPU the Zig body runs on the CPU
var buffers = [_]zb.Buffer{.{ .ptr = @ptrCast(data.ptr), .len = data.len * @sizeOf(f32) }};
var ctx = zb.KernelCtx{ .buffers = &buffers, .count = data.len, .threads = 64, .alloc = alloc };
try zb.dispatch(.cpu, kernel, &ctx);              // serial
try zb.dispatchThreaded(.cpu, kernel, &ctx, 8);   // split over 8 threads

// 4. or hold a device yourself (opening one costs milliseconds)
if (try zb.openDevice(.vulkan)) |dev| {
    var device = dev;
    defer device.deinit();
    try zb.dispatchOn(&device, kernel, &ctx);
}
```

`zig build run-gpu-probe` prints the table above, lowers the same kernel into four
shading languages and checks threaded CPU dispatch against the serial result. It
then **opens each backend's device for real**, and on a backend whose kernel has
code for it, runs the kernel and checks the numbers. On a machine without a GPU
it prints "not available" for all four and only the CPU runs — that is not a
failure, it is the correct answer for that machine.

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

* **Vulkan talks to the loader only.** No headers, no SDK, no link-time
  dependency: the ABI structs (`VkInstanceCreateInfo`, ...) are declared in this
  repository and every entry point is resolved by name through
  `vkGetInstanceProcAddr`. Once the device exists we look for a compute queue
  family and host-visible coherent memory, and buffers stay mapped for their
  lifetime (upload and readback are `memcpy`s). Each dispatch creates and
  destroys a shader module, descriptor set layout, pipeline, descriptor set,
  command buffer and fence. `count` and `threads` travel as a 16-byte push
  constant, `{ u32 count; u32 threads; u32 pad[2]; }`.

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
| Vulkan | loader + instance + physical device enumeration | **done**: device, compute queue, command pool, host-visible memory, SPIR-V pipeline, SSBOs, push constants, fence |
| WebGPU | loading the implementation (Dawn / wgpu-native) | **done**: adapter request, device request, WGSL module, storage buffers, bind group, compute pass, readback |
| Metal | `MTLCreateSystemDefaultDevice` | planned |
| Direct3D 12 | `D3D12CreateDevice` | planned |

Vulkan and WebGPU come first; the other two keep their detection code and their
seat in `Backend.ALL`.

**What is verified where.** This development machine has no GPU (no `libvulkan`,
no Dawn, no `/dev/dri`), so the GPU paths themselves are checked on real hardware
by running `zig build run-gpu-probe`, which opens each backend's device, runs the
kernel on the backends that have code for it and checks the numbers. Without a
GPU the verifiable parts are: (1) loader probing and instance creation,
(2) ABI struct layouts (comptime tests pin sizes and offsets), (3) completeness of
the entry-point table (comptime), (4) returning `null` and falling back to the CPU
when a backend cannot run, and (5) CPU execution equivalence.

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
* **The GPU path copies buffers on every dispatch.** Each call allocates,
  uploads, dispatches and frees device memory, so a hot loop should hold its own
  device with `dispatchOn` and resident buffers (device-resident tensors are the
  next step).
* Close a device with `Device.deinit` (the value `openDevice` handed out) or
  `zpu.shutdownDevices()` (the process-wide cache) — never while a dispatch is
  running.

* `Backend.ALL` lists all five backends in a stable order, so a UI or a log line
  can iterate it without special cases.
* The probe result is cached in the per-thread context and dropped by
  `zpu.deinit()`, so `zpu.backends()` is free to call in a loop.
* Kernel sources are plain strings: a host that has no idea what WGSL is can
  still hand a kernel to a backend that does.
