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
const zb = zpu.zbackend;

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
    for (zb.ALL) |b| {
        const maybe = try zb.openDevice(b);
        if (maybe == null) {
            try out.print("  {s:<8} not available on this machine\n", .{b.name()});
            continue;
        }
        var dev = maybe.?;
        defer dev.deinit();
        if (b != .cpu) gpu_ran = b;
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
        try out.print("  {s:<8} ran {d} invocations: {s} (data[3] = {d})\n", .{ b.name(), n, if (good) "correct" else "WRONG", data[3] });
    }
    // What one dispatch costs once the device is warm: the number covers the
    // whole round trip (write the buffers, run the kernel, read them back), and
    // the first dispatch is reported apart because it pays for the pipeline.
    if (gpu_ran) |backend| {
        const rounds: usize = 200;
        var timing = zb.KernelCtx{ .buffers = &buffers, .count = n, .threads = 64, .alloc = alloc };
        const warm_start = std.Io.Clock.awake.now(init.io).nanoseconds;
        try zb.dispatch(backend, Scale.kernel, &timing);
        const warm_ns = std.Io.Clock.awake.now(init.io).nanoseconds - warm_start;
        const loop_start = std.Io.Clock.awake.now(init.io).nanoseconds;
        var round: usize = 1;
        while (round < rounds) : (round += 1) try zb.dispatch(backend, Scale.kernel, &timing);
        const loop_ns = std.Io.Clock.awake.now(init.io).nanoseconds - loop_start;
        const per_dispatch_us = @as(f64, @floatFromInt(loop_ns)) / @as(f64, @floatFromInt(rounds - 1)) / 1000.0;
        try out.print("\n{s}: first dispatch {d} us (pipeline included), then {d:.1} us each over {d} more\n", .{
            backend.name(),
            @divTrunc(warm_ns, std.time.ns_per_us),
            per_dispatch_us,
            rounds - 1,
        });
    }

    // The cached devices `dispatch` uses are the same handles; closing them is
    // optional (the process exit does it), but saying so explicitly is tidier.
    zb.shutdownDevices();
    try out.flush();

    // Every device that ran the kernel had to produce the same numbers as the
    // CPU. This is what makes the example usable as a check: a wrong result, or
    // a device that answered but computed nonsense, fails the run.
    if (all_ok) return;
    return error.KernelResultMismatch;
}
