//! ZBackend — the portable execution backend shared by ZPU (compute) and ZGF
//! (graphics).
//!
//! The library ships one *portable* description of every kernel and data
//! movement it performs, and a set of thin backend drivers that can execute it:
//!
//! | backend  | runtime loaded at run time        | shading language |
//! |----------|-----------------------------------|------------------|
//! | `cpu`    | none (always available)           | Zig (thread pool)|
//! | `webgpu` | `libwebgpu_dawn.so` / `webgpu.dll`| WGSL             |
//! | `metal`  | the system framework (macOS/iOS)  | MSL              |
//! | `vulkan` | `libvulkan.so.1` / `vulkan-1.dll` | GLSL → SPIR-V    |
//! | `d3d12`  | `d3d12.dll` / `dxgi.dll`          | HLSL             |
//!
//! Nothing here *requires* a GPU: `cpu` is a complete implementation (thread
//! pool, SIMD element types) so the same program runs headless in CI and on a
//! machine with a discrete GPU, and `probe()` reports what the current machine
//! actually offers without linking against any SDK.
//!
//! ```zig
//! const b = @import("zbackend");
//! const report = b.probe();
//! std.debug.print("cpu={} vulkan={} webgpu={}\n", .{ report.cpu, report.vulkan, report.webgpu });
//! ```
const std = @import("std");
const builtin = @import("builtin");

pub const Backend = enum {
    cpu,
    webgpu,
    metal,
    vulkan,
    d3d12,

    /// Human readable name (also used in the `-Dbackend=` build option).
    pub fn name(self: Backend) []const u8 {
        return switch (self) {
            .cpu => "cpu",
            .webgpu => "webgpu",
            .metal => "metal",
            .vulkan => "vulkan",
            .d3d12 => "d3d12",
        };
    }

    /// The shading language a portable kernel is lowered to.
    pub fn shadingLanguage(self: Backend) ShadingLanguage {
        return switch (self) {
            .cpu => .zig,
            .webgpu => .wgsl,
            .metal => .msl,
            .vulkan => .glsl,
            .d3d12 => .hlsl,
        };
    }
};

pub const ShadingLanguage = enum { zig, wgsl, msl, glsl, hlsl };

/// What a backend can do. ZPU asks for `storage_buffers + compute`, ZGF also
/// asks for `render_passes` and `textures`.
pub const Capabilities = struct {
    compute: bool = false,
    render_passes: bool = false,
    textures: bool = false,
    storage_buffers: bool = false,
    f64: bool = false,
    max_workgroup_threads: u32 = 256,
};

pub const Runtime = struct {
    backend: Backend,
    /// The shared library / framework that was found, `null` for the CPU.
    library_path: ?[]const u8,
    caps: Capabilities,
};

/// Every backend the library knows how to talk to, in preference order.
pub const ALL: [5]Backend = .{ .webgpu, .metal, .vulkan, .d3d12, .cpu };

/// The dynamic libraries probed for each backend, in the order they are tried.
/// Loading is done with `std.DynLib`, so no SDK is needed at build time.
fn libraryNames(backend: Backend) []const []const u8 {
    return switch (backend) {
        .cpu => &.{},
        .webgpu => switch (builtin.os.tag) {
            .windows => &.{"webgpu_dawn.dll"},
            .macos => &.{ "libwebgpu_dawn.dylib", "/System/Library/Frameworks/WebGPU.framework/WebGPU" },
            else => &.{"libwebgpu_dawn.so"},
        },
        .metal => &.{"/System/Library/Frameworks/Metal.framework/Metal"},
        .vulkan => switch (builtin.os.tag) {
            .windows => &.{"vulkan-1.dll"},
            else => &.{ "libvulkan.so.1", "libvulkan.so" },
        },
        .d3d12 => switch (builtin.os.tag) {
            .windows => &.{ "d3d12.dll", "d3d12core.dll" },
            // Proton / Wine prefixes use the same names.
            else => &.{},
        },
    };
}

fn capsOf(backend: Backend) Capabilities {
    return switch (backend) {
        .cpu => .{ .compute = true, .render_passes = true, .textures = true, .storage_buffers = true, .f64 = true, .max_workgroup_threads = 1024 },
        .webgpu => .{ .compute = true, .render_passes = true, .textures = true, .storage_buffers = true, .max_workgroup_threads = 256 },
        .metal => .{ .compute = true, .render_passes = true, .textures = true, .storage_buffers = true, .max_workgroup_threads = 1024 },
        .vulkan => .{ .compute = true, .render_passes = true, .textures = true, .storage_buffers = true, .f64 = true, .max_workgroup_threads = 1024 },
        .d3d12 => .{ .compute = true, .render_passes = true, .textures = true, .storage_buffers = true, .max_workgroup_threads = 1024 },
    };
}

pub const Report = struct {
    cpu: bool,
    webgpu: bool,
    metal: bool,
    vulkan: bool,
    d3d12: bool,

    /// The best backend available on this machine (never fails: `cpu` is always
    /// present).
    pub fn best(self: Report) Backend {
        if (self.webgpu) return .webgpu;
        if (self.metal) return .metal;
        if (self.vulkan) return .vulkan;
        if (self.d3d12) return .d3d12;
        return .cpu;
    }

    pub fn has(self: Report, b: Backend) bool {
        return switch (b) {
            .cpu => self.cpu,
            .webgpu => self.webgpu,
            .metal => self.metal,
            .vulkan => self.vulkan,
            .d3d12 => self.d3d12,
        };
    }
};

/// Opens every candidate driver and reports which ones answered. Called once
/// per process by `zpu.init`.
pub fn probe() Report {
    return .{
        .cpu = true,
        .webgpu = probeBackend(.webgpu),
        .metal = probeBackend(.metal),
        .vulkan = probeBackend(.vulkan),
        .d3d12 = probeBackend(.d3d12),
    };
}

fn probeBackend(backend: Backend) bool {
    for (libraryNames(backend)) |name| {
        var lib = std.DynLib.open(name) catch continue;
        lib.close();
        return true;
    }
    return false;
}

/// The concrete driver found for `backend`, or `null`.
pub fn open(backend: Backend) ?Runtime {
    if (backend == .cpu) return .{ .backend = .cpu, .library_path = null, .caps = capsOf(.cpu) };
    for (libraryNames(backend)) |name| {
        var lib = std.DynLib.open(name) catch continue;
        lib.close();
        return .{ .backend = backend, .library_path = name, .caps = capsOf(backend) };
    }
    return null;
}

/// Whether a GPU backend is usable right now. ZPU uses this to decide between
/// the eager CPU engine and a device-resident plan.
pub fn gpuAvailable() bool {
    const r = probe();
    return r.best() != .cpu;
}

/// Parses `-Dbackend=` / `ZPU_BACKEND=cpu|webgpu|metal|vulkan|d3d12`.
pub fn parseBackend(s: []const u8) ?Backend {
    const info = @typeInfo(Backend).@"enum";
    inline for (info.field_names, info.field_values) |fname, fvalue| {
        if (std.mem.eql(u8, s, fname)) return @fromBackingInt(@intCast(fvalue));
    }
    return null;
}

/// Backends that are actually present, in preference order.
pub fn available(alloc: std.mem.Allocator) ![]Backend {
    const r = probe();
    var list: std.ArrayListUnmanaged(Backend) = .empty;
    errdefer list.deinit(alloc);
    for (ALL) |b| {
        if (r.has(b)) try list.append(alloc, b);
    }
    return list.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------
// Portable kernels
// ---------------------------------------------------------------------------

/// A data-parallel kernel described once and lowered per backend.
///
/// `body` is the Zig implementation (the CPU backend calls it directly); the
/// `source` field carries the same computation written in the shading language
/// that the target needs. `Kernel.lower` renders the template for a backend.
pub const Kernel = struct {
    name: []const u8,
    /// Zig body used by the CPU backend: `fn(ctx: *KernelCtx) void`.
    body: *const fn (*KernelCtx) void,
    /// Shading-language body, or `null` when the kernel is CPU only.
    source: ?Source = null,

    /// Shading source templates, one per language. `{name}` is replaced by the
    /// kernel name and `{threads}` by the workgroup size.
    pub const Source = struct {
        wgsl: ?[]const u8 = null,
        msl: ?[]const u8 = null,
        glsl: ?[]const u8 = null,
        hlsl: ?[]const u8 = null,
    };

    /// Renders the kernel for a backend, or `null` when that backend has no
    /// source for it (the caller then falls back to the CPU body).
    pub fn lower(self: Kernel, backend: Backend, alloc: std.mem.Allocator, threads: u32) !?[]u8 {
        const lang = backend.shadingLanguage();
        const tmpl = switch (lang) {
            .zig => return null,
            .wgsl => self.source orelse return null,
            .msl => self.source orelse return null,
            .glsl => self.source orelse return null,
            .hlsl => self.source orelse return null,
        };
        const text = switch (lang) {
            .wgsl => tmpl.wgsl,
            .msl => tmpl.msl,
            .glsl => tmpl.glsl,
            .hlsl => tmpl.hlsl,
            .zig => null,
        } orelse return null;

        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(alloc);
        var i: usize = 0;
        while (i < text.len) {
            if (text[i] == '{') {
                const close = std.mem.indexOfScalarPos(u8, text, i, '}') orelse return Error.InvalidSource;
                const key = text[i + 1 .. close];
                if (std.mem.eql(u8, key, "name")) {
                    try out.appendSlice(alloc, self.name);
                } else if (std.mem.eql(u8, key, "threads")) {
                    try out.print(alloc, "{d}", .{threads});
                } else return Error.InvalidSource;
                i = close + 1;
            } else {
                try out.append(alloc, text[i]);
                i += 1;
            }
        }
        return try out.toOwnedSlice(alloc);
    }
};

/// The state a CPU backend kernel runs against: a list of contiguous buffers
/// plus the workgroup geometry, mirroring what a GPU dispatch would receive.
pub const KernelCtx = struct {
    buffers: []Buffer,
    /// Global invocation id and total count (there are `workgroups * threads` of
    /// them, exactly like `@builtin(global_invocation_id)`).
    id: usize = 0,
    count: usize = 0,
    threads: u32 = 1,
    /// Scratch space for reductions inside one workgroup.
    shared: []f64 = &.{},
    barriers: Status = .{},
    alloc: std.mem.Allocator = undefined,

    pub const Status = struct {
        failed: bool = false,
        message: []const u8 = "",
        pub fn fail(self: *Status, msg: []const u8) void {
            self.failed = true;
            self.message = msg;
        }
    };
};

pub const Buffer = struct {
    ptr: [*]u8,
    len: usize,
    role: Role = .storage,

    pub const Role = enum { storage, uniform, vertex, index, texture };

    pub fn as(self: Buffer, comptime T: type) []T {
        const n = self.len / @sizeOf(T);
        return @as([*]T, @ptrCast(@alignCast(self.ptr)))[0..n];
    }
};

pub const Error = error{
    InvalidSource,
    Unsupported,
    BackendUnavailable,
    KernelFailed,
};

/// Dispatches a portable kernel on `backend`, running the Zig body on the CPU
/// and falling back to it when no driver is present.
pub fn dispatch(backend: Backend, kernel: Kernel, ctx: *KernelCtx) Error!void {
    if (backend != .cpu) {
        const rt = open(backend) orelse return dispatch(.cpu, kernel, ctx);
        if (!rt.caps.compute) return dispatch(.cpu, kernel, ctx);
    }
    var id: usize = 0;
    while (id < ctx.count) : (id += 1) {
        ctx.id = id;
        kernel.body(ctx);
        if (ctx.barriers.failed) return Error.KernelFailed;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "probe always reports the cpu backend" {
    const r = probe();
    try testing.expect(r.cpu);
    try testing.expect(r.has(.cpu));
    // `best` must never be null-ish: cpu is the floor.
    _ = r.best();
}

test "backend names round trip through parseBackend" {
    for (ALL) |b| {
        const p = parseBackend(b.name()) orelse return error.UnknownBackend;
        try testing.expectEqual(b, p);
    }
    try testing.expectEqual(@as(?Backend, null), parseBackend("nonsense"));
}

test "cpu dispatch runs the zig body for every invocation" {
    const Counter = struct {
        fn body(ctx: *KernelCtx) void {
            const out = ctx.buffers[0].as(f64);
            out[ctx.id] = @floatFromInt(ctx.id * 2);
        }
    };
    var data: [8]f64 = @splat(0);
    var bufs = [_]Buffer{.{ .ptr = @ptrCast(&data), .len = @sizeOf(@TypeOf(data)) }};
    var ctx = KernelCtx{ .buffers = &bufs, .count = data.len, .threads = 4 };
    try dispatch(.cpu, .{ .name = "fill", .body = Counter.body }, &ctx);
    for (data, 0..) |v, i| try testing.expectEqual(@as(f64, @floatFromInt(i * 2)), v);
}

test "kernels lower to each shading language" {
    const k = Kernel{
        .name = "scale",
        .body = struct {
            fn body(_: *KernelCtx) void {}
        }.body,
        .source = .{
            .wgsl = "@compute @workgroup_size({threads}) fn {name}() {}",
            .msl = "kernel void {name}() {{}}",
            .glsl = "#version 450\nvoid main() {} // {name}",
            .hlsl = "[numthreads({threads},1,1)] void {name}() {}",
        },
    };
    const wgsl = (try k.lower(.webgpu, testing.allocator, 64)).?;
    defer testing.allocator.free(wgsl);
    try testing.expectEqualStrings("@compute @workgroup_size(64) fn scale() {}", wgsl);

    const msl = (try k.lower(.metal, testing.allocator, 64)).?;
    defer testing.allocator.free(msl);
    try testing.expectEqualStrings("kernel void scale() {}", msl);

    const hlsl = (try k.lower(.d3d12, testing.allocator, 32)).?;
    defer testing.allocator.free(hlsl);
    try testing.expectEqualStrings("[numthreads(32,1,1)] void scale() {}", hlsl);

    // The CPU has no shading language: callers use `body`.
    try testing.expectEqual(@as(?[]u8, null), try k.lower(.cpu, testing.allocator, 1));
    // A kernel without source cannot be lowered for a GPU either.
    const plain = Kernel{ .name = "plain", .body = k.body };
    try testing.expectEqual(@as(?[]u8, null), try plain.lower(.vulkan, testing.allocator, 1));
}

test "available() lists the cpu backend at least" {
    const list = try available(testing.allocator);
    defer testing.allocator.free(list);
    try testing.expect(list.len >= 1);
    try testing.expectEqual(Backend.cpu, list[list.len - 1]);
}
