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
    const Scale = struct {
        /// Portable description: the same computation for every backend.
        fn body(ctx: *zb.KernelCtx) void {
            const data = ctx.buffers[0].as(f32);
            const scale = ctx.buffers[1].as(f32)[0];
            data[ctx.id] = data[ctx.id] * scale + @as(f32, @floatFromInt(ctx.id));
        }

        const kernel = zb.Kernel{
            .name = "scale_offset",
            .body = body,
            .source = .{
                .wgsl =
                \\@group(0) @binding(0) var<storage, read_write> data: array<f32>;
                \\@group(0) @binding(1) var<uniform> scale: f32;
                \\@compute @workgroup_size({threads})
                \\fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {
                \\    let i = gid.x;
                \\    data[i] = data[i] * scale + f32(i);
                \\}
                ,
                .msl =
                \\#include <metal_stdlib>
                \\using namespace metal;
                \\kernel void {name}(device float *data [[buffer(0)]],
                \\                  constant float &scale [[buffer(1)]],
                \\                  uint i [[thread_position_in_grid]]) {
                \\    data[i] = data[i] * scale + float(i);
                \\}
                ,
                .glsl =
                \\#version 450
                \\layout(local_size_x = {threads}) in;
                \\layout(std430, binding = 0) buffer Data { float data[]; };
                \\layout(binding = 1) uniform Scale { float scale; };
                \\void main() {
                \\    uint i = gl_GlobalInvocationID.x;
                \\    data[i] = data[i] * scale + float(i);
                \\}
                ,
                .hlsl =
                \\RWStructuredBuffer<float> data : register(u0);
                \\cbuffer Scale : register(b0) { float scale; };
                \\[numthreads({threads}, 1, 1)]
                \\void {name}(uint3 gid : SV_DispatchThreadID) {
                \\    uint i = gid.x;
                \\    data[i] = data[i] * scale + float(i);
                \\}
                ,
            },
            // no SPIR-V shipped: a Vulkan driver would need a compiler or
            // bytecode, so the layer falls back to the CPU there.
            .spirv = null,
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
    try out.flush();
}
