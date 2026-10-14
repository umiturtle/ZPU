# Execution backends (`zbackend`)

## Usage

```zig
const zpu = @import("zpu");

const report = zpu.backends();                 // probed once, then cached
std.debug.print("compute backend: {s}\n", .{@tagName(report.best())});

for (zbackend.Backend.ALL) |b| {
    const info = report.detail(b);
    std.debug.print("{s: <7} available={} devices={d} api={s}\n", .{
        b.name(), info.available, info.device_count, info.versionString(&buf),
    });
    for (0..info.device_count) |i| std.debug.print("  - {s}\n", .{info.deviceName(i)});
}

// one kernel, described once
const scale_offset = zbackend.Kernel{
    .name = "scale_offset",
    .workgroup = .{ .x = 64 },
    .wgsl =
    \\@group(0) @binding(0) var<storage, read_write> data: array<f32>;
    \\@compute @workgroup_size({threads})
    \\fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\    let i = gid.x;
    \\    data[i] = data[i] * 2.0 + 1.0;
    \\}
    ,
    .msl = ...,
    .glsl = ...,
    .hlsl = ...,
};

var ctx = zbackend.KernelCtx{
    .id = 0,
    .buffers = &.{ .{ .handle = data.buf, .role = .storage, .bytes = data.numel() * 4 } },
};
try zpu.dispatchKernel(.cpu, scale_offset, &ctx);            // single threaded
try zpu.dispatchKernelThreaded(.cpu, scale_offset, &ctx, 8); // strided over 8 threads
```

`zig build run-gpu-probe` prints the table above, lowers one kernel into all four
shading languages at once, and verifies a threaded CPU dispatch against the
serial result.

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

## Status

| Backend | Detection | Execution |
| --- | --- | --- |
| CPU | always | `dispatch`, `dispatchThreaded` |
| Vulkan | loader + instance + physical devices | in progress (device, queue, pipeline, SSBO) |
| WebGPU | `wgpuCreateInstance` | in progress (buffer and compute pass) |
| Metal | `MTLCreateSystemDefaultDevice` | planned |
| Direct3D 12 | `D3D12CreateDevice` | planned |

Vulkan and WebGPU are being finished first; the other two keep their detection
code and their place in `Backend.ALL`.

## Notes

* `Backend.ALL` lists all five backends in a stable order, so a UI or a log line
  can iterate it without special cases.
* The probe result is cached in the per-thread context and dropped by
  `zpu.deinit()`, so `zpu.backends()` is free to call in a loop.
* Kernel sources are plain strings: a host that has no idea what WGSL is can
  still hand a kernel to a backend that does.
