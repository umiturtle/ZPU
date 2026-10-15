//! `zig build run-gpu-probe`
//!
//! Shows what the backend layer found on this machine and demonstrates the
//! "one kernel, several backends" pipeline:
//!
//! 1. `probe()` reports which drivers are present (CPU always is) and, when a
//!    Vulkan loader answers, enumerates the physical devices.
//! 2. The same kernel is written once in Zig and once per shading language. The
//!    example prints which backends could execute it.
//! 3. It then runs the kernel for real: on the CPU backend, threaded, and
//!    verifies the result.
const std = @import("std");
const zpu = @import("zpu");
const zb = @import("zbackend");

const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file = Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_file.interface;

    try zpu.init(alloc, .{});
    defer zpu.deinit();

    // A program owns its environment; ZPU takes paths, not variables. This reads
    // ZPU_VULKAN_LIB / ZPU_WEBGPU_LIB, then VK_DRIVER_FILES / VK_ICD_FILENAMES,
    // so pointing this example at a software rasterizer (or at a driver the
    // system loader cannot find) is one environment variable away.
    zb.configureFromEnvironment(init.environ_map);

    const report = zpu.backends();
    try out.print("zpu {s} — execution backends\n\n", .{"0.1.0"});
    try out.print("{s:<8} {s:<9} {s:<34} {s:>6} {s:>8}  {s}\n", .{ "backend", "present", "library", "api", "devices", "device names" });
    try out.print("{s:-<8} {s:-<9} {s:-<34} {s:->6} {s:->8}  {s:-<20}\n", .{ "", "", "", "", "", "" });
    for (zb.ALL) |b| {
        const info = report.detail(b);
        var version_buf: [16]u8 = undefined;
        const version = info.versionString(&version_buf);
        try out.print("{s:<8} {s:<9} {s:<34} {s:>6} {d:>8}  ", .{
            b.name(),
            if (info.available) "yes" else "no",
            info.library_path orelse "-",
            if (version.len == 0) "-" else version,
            info.device_count,
        });
        if (info.names_len == 0) {
            try out.print("{s}\n", .{"-"});
        } else {
            for (0..info.names_len) |i| {
                try out.print("{s}{s}", .{ if (i == 0) "" else ", ", info.deviceName(i) });
            }
            try out.print("\n", .{});
        }
    }
    try out.print("\nbest backend for compute: {s}\n", .{report.best().name()});

    // ---------------------------------------------------------------- kernel
    // One kernel, written once per backend. The Zig body is the reference the
    // CPU runs; the WGSL, GLSL and SPIR-V come from `zbackend.spirv`, the
    // artifacts this repository ships (that file records where they came from).
    const Scale = struct {
        fn body(ctx: *zb.KernelCtx) void {
            const data = ctx.buffers[0].as(f32);
            const scale = ctx.buffers[1].as(f32)[0];
            data[ctx.id] = data[ctx.id] * scale + @as(f32, @floatFromInt(ctx.id));
        }

        const kernel = zb.Kernel{
            .name = "scale_offset",
            .body = body,
            .source = .{
                .wgsl = zb.spirv.scale_offset_wgsl,
                .glsl = zb.spirv.scale_offset_glsl,
                .msl =
                \\#include <metal_stdlib>
                \\using namespace metal;
                \\kernel void {name}(device float *data [[buffer(0)]],
                \\                  device const float *scale [[buffer(1)]],
                \\                  uint i [[thread_position_in_grid]]) {
                \\    data[i] = data[i] * scale[0] + float(i);
                \\}
                ,
                .hlsl =
                \\RWStructuredBuffer<float> data : register(u0);
                \\StructuredBuffer<float> scale : register(t0);
                \\[numthreads({threads}, 1, 1)]
                \\void {name}(uint3 gid : SV_DispatchThreadID) {
                \\    uint i = gid.x;
                \\    data[i] = data[i] * scale[0] + float(i);
                \\}
                ,
            },
            // Real bytecode: the GLSL above, compiled by glslang. Vulkan needs
            // nothing else, which is why `canRun(.vulkan)` turns true here.
            .spirv = &zb.spirv.scale_offset,
        };
    };

    try out.print("\nkernel \"{s}\":\n", .{Scale.kernel.name});
    for (zb.ALL) |b| {
        var line_buf: [1024]u8 = undefined;
        var line = Io.Writer.fixed(&line_buf);
        if (try Scale.kernel.lower(b, alloc, 64)) |src| {
            defer alloc.free(src);
            var it = std.mem.splitScalar(u8, src, '\n');
            const first = it.next() orelse "";
            try line.print("{s}", .{first});
        } else {
            try line.print("{s}", .{"(no source — zig body only)"});
        }
        try out.print("  {s:<8} can run: {s:<3}  {s}\n", .{ b.name(), if (Scale.kernel.canRun(b)) "yes" else "no", line.buffered() });
    }

    // ------------------------------------------------------------ run it here
    const n = 1024;
    const data = try alloc.alloc(f32, n);
    defer alloc.free(data);
    for (data, 0..) |*v, i| v.* = @floatFromInt(i);
    var scale_value = [_]f32{2.0};

    var buffers = [_]zb.Buffer{
        .{ .ptr = @ptrCast(data.ptr), .len = @sizeOf(f32) * n },
        .{ .ptr = @ptrCast(&scale_value), .len = @sizeOf(f32), .role = .uniform },
    };
    var ctx = zb.KernelCtx{ .buffers = &buffers, .count = n, .threads = 64, .alloc = alloc };
    try zb.dispatchThreaded(.cpu, Scale.kernel, &ctx, 4);

    var ok = true;
    for (data, 0..) |v, i| {
        const want = @as(f32, @floatFromInt(i)) * 2.0 + @as(f32, @floatFromInt(i));
        if (@abs(v - want) > 1e-3) ok = false;
    }
    try out.print("\ncpu dispatch of {d} invocations on 4 threads: {s} (data[3] = {d})\n", .{ n, if (ok) "correct" else "WRONG", data[3] });

    // ------------------------------------------------------ devices, for real
    //
    // `openDevice` is the call `dispatch` makes for itself. When a backend
    // really opens here, the kernel runs on it and the numbers are checked —
    // this is the end-to-end verification, not a stub.
    try out.print("\ndevices:\n", .{});
    var all_ok = ok;
    var gpu_ran: ?zb.Backend = null;
    var device_ran: [zb.ALL.len]bool = @splat(false);
    for (zb.ALL) |b| {
        const maybe = try zb.openDevice(b);
        if (maybe == null) {
            try out.print("  {s:<8} not available on this machine\n", .{b.name()});
            continue;
        }
        var dev = maybe.?;
        defer dev.deinit();
        if (!Scale.kernel.canRun(b)) {
            try out.print("  {s:<8} open, but the kernel has no {s} code\n", .{ b.name(), b.name() });
            continue;
        }
        for (data, 0..) |*v, i| v.* = @floatFromInt(i);
        var device_ctx = zb.KernelCtx{ .buffers = &buffers, .count = n, .threads = 64, .alloc = alloc };
        try zb.dispatchOn(&dev, Scale.kernel, &device_ctx);
        var good = true;
        for (data, 0..) |v, i| {
            const want = @as(f32, @floatFromInt(i)) * 2.0 + @as(f32, @floatFromInt(i));
            if (@abs(v - want) > 1e-3) good = false;
        }
        all_ok = all_ok and good;
        device_ran[@backingInt(b)] = true;
        if (b != .cpu) gpu_ran = b;
        try out.print("  {s:<8} ran {d} invocations: {s} (data[3] = {d})\n", .{ b.name(), n, if (good) "correct" else "WRONG", data[3] });
    }
    // The same kernel with data that stays on the device: allocate once, write
    // once, dispatch as often as you like. This is the shape a hot loop wants —
    // no upload, no download, no allocation, no pipeline compile per call.
    if (gpu_ran) |backend| {
        const dev = (try zb.device(backend)) orelse {
            try out.print("\n{s}: no cached device to hold buffers\n", .{backend.name()});
            try out.flush();
            return error.DeviceRefused;
        };
        var resident = try zb.createBuffer(dev, @sizeOf(f32) * n, alloc);
        defer zb.destroyBuffer(dev, &resident);
        var factor = try zb.createBuffer(dev, @sizeOf(f32), alloc);
        defer zb.destroyBuffer(dev, &factor);

        for (data, 0..) |*v, i| v.* = @floatFromInt(i);
        try zb.writeBuffer(dev, &resident, std.mem.sliceAsBytes(data[0..]));
        const two = [1]f32{2.0};
        try zb.writeBuffer(dev, &factor, std.mem.sliceAsBytes(&two));

        const rounds: usize = 200;
        const first_start = std.Io.Clock.awake.now(init.io).nanoseconds;
        try zb.dispatchBuffers(dev, Scale.kernel, &.{ &resident, &factor }, n, 64);
        const first_ns = std.Io.Clock.awake.now(init.io).nanoseconds - first_start;
        const started = std.Io.Clock.awake.now(init.io).nanoseconds;
        var round: usize = 1;
        while (round < rounds) : (round += 1) {
            try zb.dispatchBuffers(dev, Scale.kernel, &.{ &resident, &factor }, n, 64);
        }
        const elapsed = std.Io.Clock.awake.now(init.io).nanoseconds - started;
        const per_dispatch_us = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(rounds - 1)) / 1000.0;

        // The timing loop ran the kernel 200 times, so the numbers there are
        // meaningless by now; put the seed back and check two known passes:
        // (i * 2 + i) * 2 + i.
        try zb.writeBuffer(dev, &resident, std.mem.sliceAsBytes(data[0..]));
        try zb.dispatchBuffers(dev, Scale.kernel, &.{ &resident, &factor }, n, 64);
        try zb.dispatchBuffers(dev, Scale.kernel, &.{ &resident, &factor }, n, 64);
        var check: [1024]f32 = undefined;
        try zb.readBuffer(dev, &resident, std.mem.sliceAsBytes(check[0..n]));
        var good = true;
        for (check[0..n], 0..) |v, i| {
            const index = @as(f32, @floatFromInt(i));
            const want = (index * 2.0 + index) * 2.0 + index;
            if (@abs(v - want) > 1e-2) good = false;
        }
        all_ok = all_ok and good;
        try out.print("\n{s}: resident buffers — first dispatch {d} us (shader compiled), then {d:.1} us each over {d} ({s})\n", .{
            backend.name(),
            @divTrunc(first_ns, std.time.ns_per_us),
            per_dispatch_us,
            rounds - 1,
            if (good) "correct" else "WRONG",
        });
    }

    // The same kernel the way a caller with host-memory tensors uses it: every
    // dispatch copies the buffers in and out. The device still reuses its
    // pipeline and pooled buffers, so this is the fair comparison with the
    // resident path above.
    if (gpu_ran) |backend| {
        const rounds: usize = 200;
        var timing = zb.KernelCtx{ .buffers = &buffers, .count = n, .threads = 64, .alloc = alloc };
        const loop_start = std.Io.Clock.awake.now(init.io).nanoseconds;
        var round: usize = 0;
        while (round < rounds) : (round += 1) try zb.dispatch(backend, Scale.kernel, &timing);
        const loop_ns = std.Io.Clock.awake.now(init.io).nanoseconds - loop_start;
        const per_dispatch_us = @as(f64, @floatFromInt(loop_ns)) / @as(f64, @floatFromInt(rounds)) / 1000.0;
        try out.print("{s}: host buffers copied in and out — {d:.1} us per dispatch over {d}\n", .{
            backend.name(), per_dispatch_us, rounds,
        });
    }

    // The cached devices `dispatch` uses are the same handles; closing them is
    // optional (the process exit does it), but saying so explicitly is tidier.
    zb.shutdownDevices();
    try out.flush();

    // A runner that installs a software driver wants to know the driver really
    // ran, not that the probe found nothing and the CPU took over:
    // `ZPU_REQUIRE_GPU=vulkan` fails the run unless that backend opened a device
    // *and* dispatched the kernel on it. Nothing is required by default, so a
    // machine without a GPU still passes.
    if (init.environ_map.get("ZPU_REQUIRE_GPU")) |want| {
        var missing: usize = 0;
        var names = std.mem.tokenizeAny(u8, want, ", \t");
        while (names.next()) |name| {
            const b = zb.parseBackend(name) orelse {
                try out.print("\nZPU_REQUIRE_GPU: unknown backend \"{s}\"\n", .{name});
                try out.flush();
                return error.UnknownBackend;
            };
            if (!device_ran[@backingInt(b)]) {
                try out.print("\nZPU_REQUIRE_GPU: {s} did not run a kernel on a device\n", .{name});
                missing += 1;
            }
        }
        if (missing != 0) {
            try out.flush();
            return error.BackendUnavailable;
        }
    }

    // Every device that ran the kernel had to produce the same numbers as the
    // CPU. This is what makes the example usable as a check: a wrong result, or
    // a device that answered but computed nonsense, fails the run.
    if (all_ok) return;
    return error.KernelResultMismatch;
}
