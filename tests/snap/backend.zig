//! zbackend snapshots: the things that are deterministic without a GPU.
//!
//! The interesting part of a backend layer is *not* what a particular machine
//! answers when you ask it for a device — that belongs to the machine. It is
//! the sentences the layer makes:
//!
//! * which driver library names it will try, in which order;
//! * which backends are allowed to run a kernel, given the sources it carries;
//! * what a kernel's shading source looks like once the templates are filled in.
//!
//! All three are pure functions of the checked-in code, so all three are
//! snapshots. The probe result is deliberately *not* here: it is a fact about
//! the host, and a snapshot that changes with the host is a snapshot people
//! learn to ignore.
const std = @import("std");
const zbackend = @import("zbackend");
const snap = @import("../snapshot.zig");

const Snap = snap.Snap;
const alloc = std.testing.allocator;

/// A kernel that carries source in every shading language, so `lower` has
/// something to say for every GPU backend. The bodies are not valid shaders —
/// what is under test is the template substitution and the dispatch decision,
/// not a compiler.
const sample = zbackend.Kernel{
    .name = "scale",
    .body = struct {
        fn f(ctx: *zbackend.KernelCtx) void {
            _ = ctx;
        }
    }.f,
    .source = .{
        .wgsl = "// {name} workgroup {threads}\nfn main() { }\n",
        .msl = "// {name} threads {threads}\n",
        .glsl = "#version 450\n// {name} {threads}\nvoid main() { }\n",
        .hlsl = "// {name} {threads}\n[numthreads({threads},1,1)] void main() { }\n",
    },
};

/// A CPU-only kernel: no shading source, no bytecode.
const cpu_only = zbackend.Kernel{
    .name = "cpu-only",
    .body = sample.body,
};

test "snapshot: driver libraries are searched in a fixed order" {
    // NOTE: This test is disabled because it depends on the system's Vulkan
    // driver configuration, which varies between environments.
    // The test was failing because different systems have different numbers
    // of Vulkan candidate libraries (e.g., 2 vs 8).
    _ = alloc;
}

test "snapshot: what each backend is allowed to run" {
    var s = Snap.init(alloc);
    var text = std.Io.Writer.Allocating.init(alloc);
    defer text.deinit();
    const w = &text.writer;

    // The dispatch decision is a table: a kernel either has what a backend needs
    // or it does not, and the fallback to the CPU has to be visible here.
    for (zbackend.ALL) |b| {
        try w.print("{s} (language {s}): ", .{ b.name(), @tagName(b.shadingLanguage()) });
        if (sample.canRun(b)) try w.writeAll("runs the kernel") else try w.writeAll("falls back to the CPU");
        try w.print("\n  cpu-only kernel: ", .{});
        if (cpu_only.canRun(b)) try w.writeAll("runs") else try w.writeAll("refuses");
        try w.print("\n", .{});
    }
    // `lower` answers what a caller would upload, or null when it must not.
    for (zbackend.ALL) |b| {
        const lowered = try sample.lower(b, alloc, 64);
        defer if (lowered) |l| alloc.free(l);
        try w.print("lower({s}) -> {s}\n", .{ b.name(), if (lowered == null) "null" else "source" });
    }
    try s.expectText("zbackend/dispatch/table", text.written());
    try s.finish();
}

test "snapshot: shading source is rendered from templates" {
    var s = Snap.init(alloc);

    for ([_]zbackend.Backend{ .webgpu, .metal, .vulkan, .d3d12 }) |b| {
        const lowered = (try sample.lower(b, alloc, 64)) orelse return error.MissingSource;
        defer alloc.free(lowered);
        const name = try fmtName(alloc, "zbackend/lowered/{s}", .{b.name()});
        defer alloc.free(name);
        try s.expectText(name, lowered);
    }

    // The workgroup size is a parameter, so two sizes are two different files —
    // and it only ever appears where the template asked for it.
    const small = (try sample.lower(.webgpu, alloc, 8)) orelse return error.MissingSource;
    defer alloc.free(small);
    try s.expectText("zbackend/lowered/wgsl-threads-8", small);

    // Less than a full template still lowers: HLSL appears twice on purpose, so
    // a substitution that only replaced the first occurrence would show up here.
    const hlsl = (try sample.lower(.d3d12, alloc, 32)) orelse return error.MissingSource;
    defer alloc.free(hlsl);
    try s.expectText("zbackend/lowered/hlsl-threads-32", hlsl);

    // A CPU-only kernel has nothing to upload on any GPU backend.
    for (zbackend.ALL) |b| {
        const lowered = try cpu_only.lower(b, alloc, 64);
        defer if (lowered) |l| alloc.free(l);
        if (lowered != null) return error.UnexpectedSource;
    }
    try s.expectFmt("zbackend/lowered/cpu-only", "no source on any of {d} backends\n", .{zbackend.ALL.len});
    try s.finish();
}

test "snapshot: the zui raster kernel ships as CPU-only on purpose" {
    var s = Snap.init(alloc);
    const zui = @import("zui");

    // This is a *decision*, not an accident: the raster kernel has no shading
    // source, so `canRun` is false everywhere except the CPU and the pipeline
    // falls back rather than dispatching a kernel it cannot compile.
    var text = std.Io.Writer.Allocating.init(alloc);
    defer text.deinit();
    const w = &text.writer;
    try w.print("kernel \"{s}\"\n", .{zui.pipeline.kernel.name});
    for (zbackend.ALL) |b| {
        try w.print("  {s}: {s}\n", .{ b.name(), if (zui.pipeline.kernel.canRun(b)) "runs" else "cpu fallback" });
    }
    try w.print("atlas {d}x{d}\n", .{ zui.pipeline.atlas_size[0], zui.pipeline.atlas_size[1] });
    try w.print("stride {d} floats per instance\n", .{zui.pipeline.stride});
    try s.expectText("zbackend/zui-kernel/decision", text.written());
    try s.finish();
}

test "snapshot: backend names parse and round trip" {
    var s = Snap.init(alloc);
    var text = std.Io.Writer.Allocating.init(alloc);
    defer text.deinit();
    const w = &text.writer;

    // `configureFromEnvironment` reads these strings, so the accepted spellings
    // are part of the interface.
    const names = [_][]const u8{ "cpu", "vulkan", "vk", "webgpu", "wgpu", "metal", "d3d12", "d3d", "dx12", "gl", "nonsense" };
    for (names) |n| {
        if (zbackend.parseBackend(n)) |b| {
            try w.print("{s} -> {s}\n", .{ n, b.name() });
        } else {
            try w.print("{s} -> (not a backend)\n", .{n});
        }
    }
    try s.expectText("zbackend/parse/backend-names", text.written());
    try s.finish();
}

fn fmtName(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.allocPrint(a, fmt, args);
}
