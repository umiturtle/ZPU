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
const lib_mod = @import("backend/lib.zig");

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
/// The names of the libraries for the backends that have no loader module of
/// their own: Vulkan and WebGPU keep theirs next to their ABI (see
/// `candidateList`).
fn libraryNames(backend: Backend) []const []const u8 {
    return switch (backend) {
        .cpu => &.{},
        .metal => &.{"/System/Library/Frameworks/Metal.framework/Metal"},
        .d3d12 => switch (builtin.os.tag) {
            .windows => &.{ "d3d12.dll", "d3d12core.dll" },
            // Proton / Wine prefixes use the same names.
            else => &.{},
        },
        .vulkan, .webgpu => &.{},
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

/// What a probe learned about one backend. The name buffers are fixed size so a
/// probe never allocates and can run before the allocator is configured.
pub const Info = struct {
    available: bool = false,
    /// The library that answered, `null` for the CPU.
    library_path: ?[]const u8 = null,
    /// Loader / driver API version, packed the way the API packs it.
    api_version: u32 = 0,
    /// Number of devices the loader reported (0 when it cannot be asked).
    device_count: u32 = 0,
    names: [8][256]u8 = @splat(@splat(0)),
    name_lens: [8]u16 = @splat(0),
    /// Set when a name was filled in.
    names_len: u8 = 0,

    pub fn deviceName(self: *const Info, i: usize) []const u8 {
        if (i >= self.names_len) return "";
        const len = self.name_lens[i];
        return self.names[i][0..len];
    }

    /// `major.minor` of an API version packed as Vulkan does.
    pub fn versionString(self: *const Info, buf: *[16]u8) []const u8 {
        const major = self.api_version >> 22;
        const minor = (self.api_version >> 12) & 0x3ff;
        if (self.api_version == 0) return "";
        return std.fmt.bufPrint(buf, "{d}.{d}", .{ major, minor }) catch "";
    }
};

pub const Report = struct {
    cpu: bool,
    webgpu: bool,
    metal: bool,
    vulkan: bool,
    d3d12: bool,
    info: [5]Info = @splat(.{}),

    /// The best backend available on this machine (never fails: `cpu` is always
    /// present).
    ///
    /// "Available" means *something to run on*, not just a library that loaded:
    /// a machine with the Vulkan loader installed and no ICD (a plain CI runner,
    /// for instance) enumerates zero devices, and picking vulkan there would send
    /// every dispatch down the "the device refused, use the CPU" path. The CPU is
    /// the honest answer, and `has` still tells the whole story.
    pub fn best(self: Report) Backend {
        if (self.usable(.webgpu)) return .webgpu;
        if (self.usable(.metal)) return .metal;
        if (self.usable(.vulkan)) return .vulkan;
        if (self.usable(.d3d12)) return .d3d12;
        return .cpu;
    }

    /// The backend is present *and* has a device to dispatch on.
    pub fn usable(self: Report, b: Backend) bool {
        if (b == .cpu) return true;
        return self.has(b) and self.detail(b).device_count > 0;
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

    pub fn detail(self: *const Report, b: Backend) *const Info {
        return &self.info[@backingInt(b)];
    }
};

/// Opens every candidate driver and reports what it found. Called once per
/// process by `zpu.init`; safe to call anywhere, allocates nothing.
pub fn probe() Report {
    var report = Report{ .cpu = true, .webgpu = false, .metal = false, .vulkan = false, .d3d12 = false };
    report.info[@backingInt(Backend.cpu)] = .{
        .available = true,
        .api_version = 0,
        .device_count = 1,
    };
    for (ALL) |b| {
        if (b == .cpu) continue;
        const info = probeBackend(b);
        report.info[@backingInt(b)] = info;
        switch (b) {
            .webgpu => report.webgpu = info.available,
            .metal => report.metal = info.available,
            .vulkan => report.vulkan = info.available,
            .d3d12 => report.d3d12 = info.available,
            .cpu => {},
        }
    }
    return report;
}

/// Driver libraries a program pointed at itself or through the environment, one
/// slot per backend: a name that comes first, plus extra candidates that come
/// after the platform's own names.
///
/// The strings are copied into fixed buffers, so the caller's environment (or
/// its temporary slice) can go away afterwards.
const DriverChoice = struct {
    named: [max_path]u8 = @splat(0),
    named_len: u8 = 0,
    extras: [max_extra][max_path]u8 = @splat(@splat(0)),
    extra_lens: [max_extra]u8 = @splat(0),
    extra_count: u8 = 0,
};

/// The longest driver path this layer remembers.
pub const max_path = 256;
/// How many extra candidates one backend can carry (the ICD variables).
pub const max_extra = 4;
/// The longest list `candidateList` can produce.
pub const max_candidates = max_extra + 8;

var choices: [ALL.len]DriverChoice = @splat(.{});

fn copyPath(dst: []u8, path: []const u8) u8 {
    const len = @min(path.len, dst.len);
    @memcpy(dst[0..len], path[0..len]);
    return @intCast(len);
}

/// Points a backend at one driver library, tried before the platform's own
/// names: a bare name searched on the platform's library path, or an absolute
/// path.
///
/// Use it for drivers the default list cannot know about — a software rasterizer
/// such as SwiftShader (`libvk_swiftshader.so`), an ICD named in
/// `VK_DRIVER_FILES`, MoltenVK, a browser's WebGPU implementation. `null`
/// restores the platform defaults.
///
/// Because the probe result is cached per thread, set this before the first
/// `probe()` (or before `zpu.init`); it only affects probes and `open` calls
/// that happen afterwards. Programs that have a `std.process.Init` can call
/// `configureFromEnvironment` instead of reading variables themselves.
pub fn setDriverLibrary(backend: Backend, library: ?[]const u8) void {
    const slot = &choices[@backingInt(backend)];
    if (library) |path| {
        slot.named_len = copyPath(&slot.named, path);
    } else {
        slot.named_len = 0;
    }
}

/// Adds a candidate that is tried *after* the platform's own names. This is how
/// a Vulkan ICD named in the environment gets used on a machine with no loader
/// installed.
pub fn addDriverLibrary(backend: Backend, library: []const u8) void {
    const slot = &choices[@backingInt(backend)];
    if (slot.extra_count >= max_extra) return;
    const index = slot.extra_count;
    slot.extra_lens[index] = copyPath(&slot.extras[index], library);
    slot.extra_count += 1;
}

/// The libraries `backend` is tried with, in order: the named one, then the
/// platform's names, then the extra candidates. Written into `buf`, which the
/// caller owns, so this allocates nothing.
pub fn candidateList(backend: Backend, buf: *[max_candidates][]const u8) []const []const u8 {
    const slot = &choices[@backingInt(backend)];
    var n: usize = 0;
    if (slot.named_len != 0) {
        buf[n] = slot.named[0..slot.named_len];
        n += 1;
    }
    const platform = switch (backend) {
        // The Vulkan and WebGPU lists live with their loader code, which is
        // where the per-platform names are.
        .vulkan => vulkan.libraryNames(),
        .webgpu => webgpu.libraryNames(),
        else => libraryNames(backend),
    };
    for (platform) |name| {
        if (n == buf.len) break;
        buf[n] = name;
        n += 1;
    }
    for (0..slot.extra_count) |i| {
        if (n == buf.len) break;
        buf[n] = slot.extras[i][0..slot.extra_lens[i]];
        n += 1;
    }
    return buf[0..n];
}

/// Reads the usual variables out of a program's environment and points the
/// backends at what they name:
///
/// * `ZPU_VULKAN_LIB` / `ZPU_WEBGPU_LIB` — a library name or an absolute path,
///   tried before the platform's own names,
/// * `VK_DRIVER_FILES` / `VK_ICD_FILENAMES` — the standard Vulkan variables,
///   tried after them. The standard use is to name ICD manifests (`.json`),
///   which only a loader can resolve; ZPU skips those and takes the entries that
///   name a library, because a Vulkan driver exports `vkGetInstanceProcAddr`
///   itself and can be driven directly on a machine with no loader.
///
/// Call this early — before the first `probe()` — because the probe result is
/// cached per thread.
pub fn configureFromEnvironment(environ_map: *std.process.Environ.Map) void {
    if (environ_map.get("ZPU_VULKAN_LIB")) |path| setDriverLibrary(.vulkan, path);
    if (environ_map.get("ZPU_WEBGPU_LIB")) |path| setDriverLibrary(.webgpu, path);
    const keys = [_][]const u8{ "VK_ICD_FILENAMES", "VK_DRIVER_FILES" };
    for (keys) |key| {
        const value = environ_map.get(key) orelse continue;
        var it = std.mem.tokenizeAny(u8, value, ":; \t");
        while (it.next()) |entry| {
            if (std.mem.endsWith(u8, entry, ".json")) continue;
            addDriverLibrary(.vulkan, entry);
        }
    }
}

/// The first library that opens for `backend`, and what the probe saw in it.
fn probeBackend(backend: Backend) Info {
    var buf: [max_candidates][]const u8 = undefined;
    for (candidateList(backend, &buf)) |name| {
        if (backend == .vulkan) {
            if (vulkanInfoFrom(name)) |info| return info;
            continue;
        }
        var lib = lib_mod.Library.open(name) catch continue;
        defer lib.close();
        var info = Info{ .available = true, .library_path = name };
        probeLib(backend, &lib, &info);
        if (info.available) return info;
    }
    return .{};
}

/// The Vulkan probe lives in the device module, with the ABI: creating an
/// instance and enumerating physical devices is exactly what `vulkan.probe`
/// does. This copies the fixed-size result into an `Info`.
fn vulkanInfoFrom(path: []const u8) ?Info {
    const probed = vulkan.probe(path) orelse return null;
    var info = Info{
        .available = true,
        .library_path = path,
        .api_version = probed.api_version,
        .device_count = probed.count,
    };
    const n = @min(@as(usize, probed.count), info.names.len);
    for (0..n) |i| {
        const device_name = probed.name(i);
        const len = @min(device_name.len, info.names[i].len);
        @memcpy(info.names[i][0..len], device_name[0..len]);
        info.name_lens[i] = @intCast(len);
    }
    info.names_len = @intCast(n);
    return info;
}

/// A driver is usable when the entry point the backend needs exists.
fn symbolProbe(lib: *lib_mod.Library, symbol: [:0]const u8, info: *Info) void {
    if (lib.lookup(*const anyopaque, symbol) == null) {
        info.available = false;
        return;
    }
    info.device_count = 1;
    const short = symbol[0..@min(symbol.len, info.names[0].len)];
    @memcpy(info.names[0][0..short.len], short);
    info.name_lens[0] = @intCast(short.len);
    info.names_len = 1;
}

fn probeLib(backend: Backend, lib: *lib_mod.Library, info: *Info) void {
    switch (backend) {
        .metal => symbolProbe(lib, "MTLCreateSystemDefaultDevice", info),
        .d3d12 => symbolProbe(lib, "D3D12CreateDevice", info),
        .webgpu => symbolProbe(lib, "wgpuCreateInstance", info),
        // Vulkan goes through `vulkanInfo`: its probe needs the ABI.
        .vulkan, .cpu => {},
    }
}

/// The concrete driver found for `backend`, or `null`.
pub fn open(backend: Backend) ?Runtime {
    if (backend == .cpu) return .{ .backend = .cpu, .library_path = null, .caps = capsOf(.cpu) };
    const info = probeBackend(backend);
    if (!info.available) return null;
    var caps = capsOf(backend);
    if (info.device_count == 0) caps.compute = false;
    return .{ .backend = backend, .library_path = info.library_path, .caps = caps };
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
    /// Precompiled bytecode (SPIR-V) for backends that cannot compile source
    /// themselves. Shipping this is optional: without it the kernel falls back
    /// to the CPU.
    spirv: ?[]const u32 = null,

    /// Shading source templates, one per language. `{name}` is replaced by the
    /// kernel name and `{threads}` by the workgroup size.
    pub const Source = struct {
        wgsl: ?[]const u8 = null,
        msl: ?[]const u8 = null,
        glsl: ?[]const u8 = null,
        hlsl: ?[]const u8 = null,
    };

    /// Whether `backend` can actually execute this kernel: the CPU always can,
    /// other backends need either source in their shading language or
    /// precompiled bytecode.
    pub fn canRun(self: Kernel, backend: Backend) bool {
        return switch (backend.shadingLanguage()) {
            .zig => true,
            .glsl => self.spirv != null,
            .wgsl => if (self.source) |src| src.wgsl != null else false,
            .msl => if (self.source) |src| src.msl != null else false,
            .hlsl => if (self.source) |src| src.hlsl != null else false,
        };
    }

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
            // Only `{name}` and `{threads}` are placeholders. Any other brace is
            // literal shader code (`void main() { ... }`), so it is copied as is.
            if (text[i] == '{') {
                if (std.mem.indexOfScalarPos(u8, text, i, '}')) |close| {
                    const key = text[i + 1 .. close];
                    if (std.mem.eql(u8, key, "name")) {
                        try out.appendSlice(alloc, self.name);
                        i = close + 1;
                        continue;
                    }
                    if (std.mem.eql(u8, key, "threads")) {
                        try out.print(alloc, "{d}", .{threads});
                        i = close + 1;
                        continue;
                    }
                }
            }
            try out.append(alloc, text[i]);
            i += 1;
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
    /// A device buffer was already released, or comes from another device.
    InvalidBuffer,
    DeviceFailed,
    OutOfMemory,
};

/// Dispatches a portable kernel on `backend`.
///
/// The kernel runs on the device when there is one and the kernel carries code
/// for it (`Kernel.canRun`); otherwise the Zig body runs on the CPU. That
/// fallback is silent on purpose — a program written once runs everywhere, and
/// `probe()` tells the caller what actually happened.
pub fn dispatch(backend: Backend, kernel: Kernel, ctx: *KernelCtx) Error!void {
    if (backend != .cpu and kernel.canRun(backend)) {
        if (try device(backend)) |dev| return dispatchOn(dev, kernel, ctx);
    }
    return runCpu(kernel, ctx);
}

/// Runs the Zig body once per invocation, in order — the CPU backend, and the
/// fallback of every other backend.
pub fn runCpu(kernel: Kernel, ctx: *KernelCtx) Error!void {
    var id: usize = 0;
    while (id < ctx.count) : (id += 1) {
        ctx.id = id;
        kernel.body(ctx);
        if (ctx.barriers.failed) return Error.KernelFailed;
    }
}

/// Runs `kernel` over `ctx.count` invocations using `thread_count` OS threads.
///
/// The CPU backend is a complete implementation, not a debug fallback, so it
/// has the same execution model as a GPU dispatch: invocations are independent,
/// and each worker walks a strided slice of the global id space. A kernel that
/// needs cross-invocation ordering (a scan, for example) must not use this.
pub fn dispatchThreaded(backend: Backend, kernel: Kernel, ctx: *KernelCtx, thread_count: usize) Error!void {
    const workers = @max(@min(thread_count, ctx.count), 1);
    if (backend != .cpu or workers == 1) return dispatch(backend, kernel, ctx);

    var failures = std.atomic.Value(usize).init(0);
    var handles: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;
    const self_index = workers - 1;
    for (0..self_index) |i| {
        handles[spawned] = std.Thread.spawn(.{}, workerBody, .{ kernel, ctx, i, workers, &failures }) catch break;
        spawned += 1;
    }
    workerBody(kernel, ctx, self_index, workers, &failures);
    // Slices whose thread could not be spawned run here instead.
    for (spawned..self_index) |i| workerBody(kernel, ctx, i, workers, &failures);
    for (handles[0..spawned]) |h| h.join();
    if (failures.load(.monotonic) != 0) return Error.KernelFailed;
}

pub const max_threads = 64;

fn workerBody(kernel: Kernel, ctx: *KernelCtx, slice: usize, workers: usize, failures: *std.atomic.Value(usize)) void {
    // Each worker gets its own copy of the context, so `id` is private while the
    // buffers stay shared — the same split a GPU makes between shared memory and
    // per-invocation registers.
    var local = ctx.*;
    local.id = 0;
    var id = slice;
    while (id < ctx.count) : (id += workers) {
        local.id = id;
        kernel.body(&local);
        if (local.barriers.failed) {
            _ = failures.fetchAdd(1, .monotonic);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Devices: opening a driver and dispatching on it
// ---------------------------------------------------------------------------

pub const vulkan = @import("backend/vulkan.zig");
pub const webgpu = @import("backend/webgpu.zig");

/// SPIR-V bytecode (and the sources it came from) for the kernels this
/// repository ships as verified artifacts.
pub const spirv = @import("backend/spirv.zig");

/// A live connection to a backend.
///
/// `open` describes a driver; a `Device` is an open one. Only `.cpu`,
/// `.vulkan` and `.webgpu` have an implementation today; the other two exist so
/// the tag set matches `Backend` (and so `Device.deinit` needs no `else`).
pub const Device = union(Backend) {
    // Field order must match `Backend`'s; Zig checks it at comptime.
    cpu: void,
    webgpu: webgpu.Device,
    metal: void,
    vulkan: vulkan.Device,
    d3d12: void,

    /// The backend this device drives.
    pub fn backend(self: Device) Backend {
        return self;
    }

    pub fn deinit(self: *Device) void {
        switch (self.*) {
            .vulkan => |*d| d.deinit(),
            .webgpu => |*d| d.deinit(),
            .cpu, .metal, .d3d12 => {},
        }
    }
};

/// Opens a driver for `backend`: `null` when this machine cannot run it (no
/// library, no adapter, no device), an error only when something went wrong
/// while opening.
pub fn openDevice(backend: Backend) Error!?Device {
    return openDeviceNamed(backend, null);
}

/// Opens `backend` through one specific library, ignoring both the platform
/// defaults and `setDriverLibrary`.
pub fn openDeviceAt(backend: Backend, library: []const u8) Error!?Device {
    return openDeviceNamed(backend, library);
}

fn openDeviceNamed(backend: Backend, explicit: ?[]const u8) Error!?Device {
    if (backend == .cpu) return Device{ .cpu = {} };
    if (explicit) |name| return openDeviceVia(backend, name);
    var buf: [max_candidates][]const u8 = undefined;
    for (candidateList(backend, &buf)) |name| {
        // A library that does not open, or opens without a device, is not an
        // error — the next candidate gets its turn, and the CPU is the floor.
        if (openDeviceVia(backend, name) catch null) |opened| return opened;
    }
    return null;
}

fn openDeviceVia(backend: Backend, library: []const u8) Error!?Device {
    return switch (backend) {
        .cpu => Device{ .cpu = {} },
        .vulkan => if (vulkan.Device.openAt(library, null) catch |e| return mapDeviceError(e)) |d| Device{ .vulkan = d } else null,
        .webgpu => if (webgpu.Device.openAt(library) catch |e| return mapDeviceError(e)) |d| Device{ .webgpu = d } else null,
        .metal, .d3d12 => null,
    };
}

/// Devices opened on demand and kept for the process lifetime.
///
/// Creating an instance and a device costs milliseconds; a numeric library
/// dispatches thousands of kernels per second, so `dispatch` opens each backend
/// once instead of once per call. The state machine is lock-free: a thread that
/// finds another thread mid-open does not wait, it just falls back to the CPU
/// for that dispatch (`docs/features/09-backends.md` explains the choice).
const cache = struct {
    var slots: [ALL.len]Slot = @splat(.{});

    const Slot = struct {
        state: std.atomic.Value(State) = .init(.unopened),
        device: ?Device = null,
    };

    const State = enum(u8) { unopened, opening, opened, unavailable };
};

/// The process-wide device for `backend`, opening it on first use.
///
/// `null` means the backend cannot run here — no library, no device, or another
/// thread is opening it right now. The returned pointer stays valid until
/// `shutdownDevices`.
pub fn device(backend: Backend) Error!?*Device {
    const slot = &cache.slots[@backingInt(backend)];
    switch (slot.state.load(.acquire)) {
        .opened => return if (slot.device) |*d| d else null,
        .unavailable, .opening => return null,
        .unopened => {},
    }
    if (slot.state.cmpxchgStrong(.unopened, .opening, .acq_rel, .acquire) != null) {
        // Another thread owns the open; this call stays on the CPU.
        return if (slot.device) |*d| d else null;
    }
    errdefer slot.state.store(.unavailable, .release);
    if (try openDevice(backend)) |d| {
        slot.device = d;
        slot.state.store(.opened, .release);
    } else {
        slot.state.store(.unavailable, .release);
    }
    return if (slot.device) |*d| d else null;
}

/// Closes every cached device; the next `device` call reopens what it needs.
/// Call it at shutdown, or between tests — never while a dispatch is running.
pub fn shutdownDevices() void {
    for (&cache.slots) |*slot| {
        if (slot.device) |*d| d.deinit();
        slot.device = null;
        slot.state.store(.unopened, .release);
    }
}

/// Runs `kernel` on an already open device.
///
/// The CPU device runs the Zig body. The GPU devices run the kernel's shader
/// code, which means copying each `KernelCtx.buffer` to the device and back —
/// one-shot buffers, so a dispatch is not free. Device-resident buffers are the
/// next step for these backends.
pub fn dispatchOn(dev: *Device, kernel: Kernel, ctx: *KernelCtx) Error!void {
    switch (dev.*) {
        .cpu, .metal, .d3d12 => return runCpu(kernel, ctx),
        .vulkan => |*d| return dispatchVulkan(d, kernel, ctx),
        .webgpu => |*d| return dispatchWebgpu(d, kernel, ctx),
    }
}

/// The name of the shader entry point every GPU backend expects: WGSL must
/// declare `@compute fn main(...)`, SPIR-V must have an entry point `main`.
pub const entry_point = "main";

fn dispatchVulkan(d: *vulkan.Device, kernel: Kernel, ctx: *KernelCtx) Error!void {
    const code = kernel.spirv orelse return Error.Unsupported;
    if (ctx.buffers.len > RunBuffers.max) return Error.Unsupported;

    // Both arrays live on the stack: a dispatch to a warm device should not call
    // an allocator, and it should get its buffers back from the device's pool
    // rather than from the driver.
    var local: [RunBuffers.max]vulkan.Device.Allocation = undefined;
    const allocs = local[0..ctx.buffers.len];
    var created: usize = 0;
    defer for (allocs[0..created]) |a| d.release(a);
    for (ctx.buffers, 0..) |b, i| {
        allocs[i] = d.acquire(b.len) catch |e| return mapDeviceError(e);
        created += 1;
        d.upload(allocs[i], b.ptr[0..b.len]);
    }

    var stack: [RunBuffers.max]vulkan.Device.BufferBinding = undefined;
    const bindings = stack[0..allocs.len];
    for (allocs, 0..) |a, i| bindings[i] = .{ .buffer = a.buffer, .size = a.size };

    d.dispatchCompute(code, bindings, ctx.count, ctx.threads) catch |e| return mapDeviceError(e);
    // Host-coherent memory: the device's writes are already visible.
    for (ctx.buffers, 0..) |b, i| d.download(allocs[i], b.ptr[0..b.len]);
}

fn dispatchWebgpu(d: *webgpu.Device, kernel: Kernel, ctx: *KernelCtx) Error!void {
    const alloc = ctx.alloc;
    const wgsl = (try kernel.lower(.webgpu, alloc, ctx.threads)) orelse return Error.Unsupported;
    defer alloc.free(wgsl);
    const allocs = try alloc.alloc(webgpu.Device.Allocation, ctx.buffers.len);
    defer alloc.free(allocs);
    var created: usize = 0;
    defer for (allocs[0..created]) |a| d.release(a);
    for (ctx.buffers, 0..) |b, i| {
        allocs[i] = d.acquire(b.len) catch |e| return mapDeviceError(e);
        created += 1;
        // Buffers are created mapped: write, then unmaps so the GPU may read.
        d.upload(allocs[i], b.ptr[0..b.len]);
        d.commit(allocs[i]);
    }
    d.run(alloc, wgsl, entry_point, allocs, ctx.count, ctx.threads) catch |e| return mapDeviceError(e);
    for (ctx.buffers, 0..) |b, i| d.download(allocs[i], b.ptr[0..b.len]) catch |e| return mapDeviceError(e);
}

/// Device failures collapse into one backend error; allocation failures keep
/// their own name because callers handle them differently.
fn mapDeviceError(e: anyerror) Error {
    return switch (e) {
        error.OutOfMemory => Error.OutOfMemory,
        else => Error.DeviceFailed,
    };
}

// ---------------------------------------------------------------------------
// Device buffers: memory that stays on the device
// ---------------------------------------------------------------------------

/// A buffer that lives on the device between dispatches.
///
/// `dispatch` and `dispatchOn` copy every `KernelCtx.buffer` in, run, and copy it
/// back. That is the right default for a library whose tensors live in host
/// memory, and a waste in a loop. A `DeviceBuffer` is the other option:
/// allocate once, write once, dispatch as often as you like, read back when you
/// want the numbers.
///
/// ```zig
/// var weights = try zb.createBuffer(&device, 4096, alloc);
/// defer zb.destroyBuffer(&device, &weights);
/// try zb.writeBuffer(&device, &weights, host_bytes);
/// try zb.dispatchBuffers(&device, kernel, &.{&weights}, 1024, 64);
/// try zb.readBuffer(&device, &weights, into_host);
/// ```
///
/// The CPU device keeps the bytes in host memory, so the same code runs (and the
/// same tests pass) on a machine with no GPU.
pub const DeviceBuffer = struct {
    backend: Backend,
    size: usize,
    /// Cleared by `destroyBuffer`; every other call refuses a dead buffer
    /// instead of touching a handle the driver has already taken back.
    live: bool = true,
    memory: Memory,

    const Memory = union(Backend) {
        // Field order must match `Backend`'s; Zig checks it at comptime.
        cpu: Cpu,
        webgpu: webgpu.Device.Allocation,
        metal: void,
        vulkan: vulkan.Device.Allocation,
        d3d12: void,

        const Cpu = struct {
            bytes: []u8,
            alloc: std.mem.Allocator,
        };
    };

    /// How many bytes the buffer holds.
    pub fn len(self: *const DeviceBuffer) usize {
        return self.size;
    }
};

/// Allocates device memory that outlives a dispatch. `alloc` is used for the
/// CPU's host memory and ignored by the GPU backends.
pub fn createBuffer(dev: *Device, size: usize, alloc: std.mem.Allocator) Error!DeviceBuffer {
    const wanted = @max(size, 4);
    return switch (dev.*) {
        .cpu => .{
            .backend = .cpu,
            .size = wanted,
            .memory = .{ .cpu = .{
                .bytes = try alloc.alloc(u8, wanted),
                .alloc = alloc,
            } },
        },
        .vulkan => |*d| blk: {
            const allocation = d.acquire(wanted) catch |e| return mapDeviceError(e);
            break :blk .{ .backend = .vulkan, .size = wanted, .memory = .{ .vulkan = allocation } };
        },
        .webgpu => |*d| blk: {
            const allocation = d.acquire(wanted) catch |e| return mapDeviceError(e);
            break :blk .{ .backend = .webgpu, .size = wanted, .memory = .{ .webgpu = allocation } };
        },
        .metal, .d3d12 => Error.BackendUnavailable,
    };
}

/// Frees a buffer. Safe to call twice; the second call does nothing.
///
/// A buffer made by `createBuffer` on the GPU goes back into the device's pool,
/// so the next `createBuffer` of a similar size does not ask the driver.
pub fn destroyBuffer(dev: *Device, buf: *DeviceBuffer) void {
    if (!buf.live) return;
    switch (buf.memory) {
        .cpu => |c| c.alloc.free(c.bytes),
        .vulkan => |a| if (dev.* == .vulkan) dev.vulkan.release(a),
        .webgpu => |a| if (dev.* == .webgpu) dev.webgpu.release(a),
        .metal, .d3d12 => {},
    }
    buf.live = false;
}

/// Writes host bytes into a device buffer — what a one-shot dispatch does for
/// the caller, made explicit so it can happen once instead of every time.
pub fn writeBuffer(dev: *Device, buf: *DeviceBuffer, bytes: []const u8) Error!void {
    if (!buf.live) return Error.InvalidBuffer;
    const slice = bytes[0..@min(bytes.len, buf.size)];
    switch (buf.memory) {
        .cpu => |c| @memcpy(c.bytes[0..slice.len], slice),
        .vulkan => |a| if (dev.* == .vulkan) dev.vulkan.upload(a, slice),
        .webgpu => |a| if (dev.* == .webgpu) {
            dev.webgpu.upload(a, slice);
            dev.webgpu.commit(a);
        },
        .metal, .d3d12 => return Error.BackendUnavailable,
    }
}

/// Reads a device buffer back into host memory.
pub fn readBuffer(dev: *Device, buf: *const DeviceBuffer, out: []u8) Error!void {
    if (!buf.live) return Error.InvalidBuffer;
    const slice = out[0..@min(out.len, buf.size)];
    switch (buf.memory) {
        .cpu => |c| @memcpy(slice, c.bytes[0..slice.len]),
        .vulkan => |a| if (dev.* == .vulkan) dev.vulkan.download(a, slice),
        .webgpu => |a| if (dev.* == .webgpu) dev.webgpu.download(a, slice) catch |e| return mapDeviceError(e),
        .metal, .d3d12 => return Error.BackendUnavailable,
    }
}

/// Runs `kernel` on buffers that already live on the device: no upload, no
/// download, no allocation, no descriptor setup the device has not already done.
///
/// The buffers must have been created on this device. The CPU device accepts up
/// to `RunBuffers.max` of them and runs the kernel's Zig body.
pub fn dispatchBuffers(
    dev: *Device,
    kernel: Kernel,
    buffers: []const *const DeviceBuffer,
    count: usize,
    threads: u32,
) Error!void {
    for (buffers) |b| {
        if (!b.live or b.backend != dev.backend()) return Error.InvalidBuffer;
    }
    switch (dev.*) {
        .cpu => {
            var host: [RunBuffers.max]Buffer = undefined;
            if (buffers.len > host.len) return Error.Unsupported;
            for (buffers, 0..) |b, i| host[i] = .{ .ptr = b.memory.cpu.bytes.ptr, .len = b.size };
            var ctx = KernelCtx{
                .buffers = host[0..buffers.len],
                .count = count,
                .threads = threads,
                .alloc = std.heap.page_allocator,
            };
            return runCpu(kernel, &ctx);
        },
        .vulkan => |*d| {
            const code = kernel.spirv orelse return Error.Unsupported;
            // The binding descriptions live on the stack: a dispatch should not
            // call an allocator at all.
            var stack: [RunBuffers.max]vulkan.Device.BufferBinding = undefined;
            if (buffers.len > stack.len) return Error.Unsupported;
            for (buffers, 0..) |b, i| {
                stack[i] = .{ .buffer = b.memory.vulkan.buffer, .size = b.memory.vulkan.size };
            }
            d.dispatchCompute(code, stack[0..buffers.len], count, threads) catch |e| return mapDeviceError(e);
        },
        .webgpu => |*d| {
            const alloc = std.heap.page_allocator;
            const wgsl = (try kernel.lower(.webgpu, alloc, threads)) orelse return Error.Unsupported;
            defer alloc.free(wgsl);
            const allocs = alloc.alloc(webgpu.Device.Allocation, buffers.len) catch return Error.OutOfMemory;
            defer alloc.free(allocs);
            for (buffers, 0..) |b, i| allocs[i] = b.memory.webgpu;
            d.run(alloc, wgsl, entry_point, allocs, count, threads) catch |e| return mapDeviceError(e);
        },
        .metal, .d3d12 => return Error.BackendUnavailable,
    }
}

/// The bound a device dispatch onto resident buffers has: sixteen buffers is far
/// more than the kernel ABI needs, and it keeps the host-side array on the stack.
pub const RunBuffers = struct {
    pub const max = 16;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "best asks for a device, not just a loader" {
    // A machine with the loader installed and no ICD — a plain CI runner — sees
    // `vkGetInstanceProcAddr`, opens an instance and enumerates *zero* devices.
    // Reporting that as "the best backend for compute" would make every dispatch
    // take the "device refused, fall back to the CPU" path, so `best` asks for a
    // device and `has` keeps reporting the whole truth.
    var report = Report{
        .cpu = true,
        .webgpu = false,
        .metal = false,
        .vulkan = true,
        .d3d12 = false,
    };
    report.info[@backingInt(Backend.vulkan)] = .{
        .available = true,
        .library_path = "libvulkan.so.1",
        .api_version = 0x00403000,
        .device_count = 0,
    };
    try testing.expect(report.has(.vulkan));
    try testing.expect(!report.usable(.vulkan));
    try testing.expectEqual(Backend.cpu, report.best());

    // Once an ICD answers with one device, vulkan is the right answer again.
    report.info[@backingInt(Backend.vulkan)].device_count = 1;
    try testing.expect(report.usable(.vulkan));
    try testing.expectEqual(Backend.vulkan, report.best());

    // The CPU needs no device list to be usable.
    try testing.expect(report.usable(.cpu));
}

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
            .msl = "kernel void {name}() {}",
            .glsl = "#version 450\nvoid main() { imageStore(0, ivec2(0), vec4(0.0)); } // {name}",
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

test "probe detail describes the cpu backend and reports versions" {
    const r = probe();
    const cpu = r.detail(.cpu);
    try testing.expect(cpu.available);
    try testing.expectEqual(@as(u32, 1), cpu.device_count);
    try testing.expectEqualStrings("", cpu.deviceName(0));

    var buf: [16]u8 = undefined;
    var fake = Info{ .api_version = (1 << 22) | (3 << 12) };
    try testing.expectEqualStrings("1.3", fake.versionString(&buf));
    fake.api_version = 0;
    try testing.expectEqualStrings("", fake.versionString(&buf));

    // A backend that is not present must not claim devices.
    if (!r.vulkan) try testing.expectEqual(@as(u32, 0), r.detail(.vulkan).device_count);
    if (!r.metal) try testing.expect(!r.detail(.metal).available);
}

test "threaded dispatch matches serial dispatch" {
    const Scale = struct {
        fn body(ctx: *KernelCtx) void {
            const out = ctx.buffers[0].as(f64);
            out[ctx.id] = out[ctx.id] * 2 + 1;
        }
    };
    const kernel = Kernel{ .name = "scale", .body = Scale.body };
    const n = 4096;
    var serial: [n]f64 = undefined;
    var threaded: [n]f64 = undefined;
    for (&serial, 0..) |*v, i| v.* = @floatFromInt(i);
    @memcpy(&threaded, &serial);

    var s_bufs = [_]Buffer{.{ .ptr = @ptrCast(&serial), .len = @sizeOf(@TypeOf(serial)) }};
    var s_ctx = KernelCtx{ .buffers = &s_bufs, .count = n };
    try dispatch(.cpu, kernel, &s_ctx);

    var t_bufs = [_]Buffer{.{ .ptr = @ptrCast(&threaded), .len = @sizeOf(@TypeOf(threaded)) }};
    var t_ctx = KernelCtx{ .buffers = &t_bufs, .count = n };
    try dispatchThreaded(.cpu, kernel, &t_ctx, 4);

    try testing.expectEqualSlices(f64, &serial, &threaded);
}

test "threaded dispatch reports a kernel failure" {
    const Boom = struct {
        fn body(ctx: *KernelCtx) void {
            if (ctx.id == 3) ctx.barriers.fail("deliberate");
        }
    };
    var data: [16]f64 = @splat(0);
    var bufs = [_]Buffer{.{ .ptr = @ptrCast(&data), .len = @sizeOf(@TypeOf(data)) }};
    var ctx = KernelCtx{ .buffers = &bufs, .count = data.len };
    try testing.expectError(Error.KernelFailed, dispatchThreaded(.cpu, .{ .name = "boom", .body = Boom.body }, &ctx, 2));
}

test "canRun decides per backend" {
    const base = Kernel{ .name = "k", .body = struct {
        fn body(_: *KernelCtx) void {}
    }.body };
    try testing.expect(base.canRun(.cpu));
    try testing.expect(!base.canRun(.metal));
    try testing.expect(!base.canRun(.vulkan));

    const wgsl_only = Kernel{ .name = "k", .body = base.body, .source = .{ .wgsl = "x" } };
    try testing.expect(wgsl_only.canRun(.webgpu));
    try testing.expect(!wgsl_only.canRun(.metal));
    try testing.expect(!wgsl_only.canRun(.vulkan));

    const with_spirv = Kernel{ .name = "k", .body = base.body, .spirv = &.{ 0x07230203, 0, 1, 0 } };
    try testing.expect(with_spirv.canRun(.vulkan));
    try testing.expect(!with_spirv.canRun(.webgpu));
}

test "the cpu device is always openable and runs the zig body" {
    var dev = (try openDevice(.cpu)).?;
    defer dev.deinit();
    try testing.expectEqual(Backend.cpu, dev.backend());

    const Fill = struct {
        fn body(ctx: *KernelCtx) void {
            ctx.buffers[0].as(f64)[ctx.id] = @floatFromInt(ctx.id);
        }
    };
    var data: [16]f64 = @splat(0);
    var bufs = [_]Buffer{.{ .ptr = @ptrCast(&data), .len = @sizeOf(@TypeOf(data)) }};
    var ctx = KernelCtx{ .buffers = &bufs, .count = data.len, .alloc = testing.allocator };
    try dispatchOn(&dev, .{ .name = "fill", .body = Fill.body }, &ctx);
    for (data, 0..) |v, i| try testing.expectEqual(@as(f64, @floatFromInt(i)), v);
}

test "a backend that cannot be opened falls back to the cpu" {
    const Fill = struct {
        fn body(ctx: *KernelCtx) void {
            ctx.buffers[0].as(f64)[ctx.id] = @floatFromInt(ctx.id);
        }
    };
    // Real SPIR-V would run on a real driver; this is a placeholder, so the
    // machine either has no Vulkan (the usual case, and the one under test) or
    // the kernel still must not corrupt the host buffer.
    const kernel = Kernel{ .name = "fill", .body = Fill.body, .spirv = &.{ 0x07230203, 0, 1, 0 } };
    for ([_]Backend{ .vulkan, .webgpu, .metal, .d3d12 }) |b| {
        if (try device(b) != null) continue;
        var data: [16]f64 = @splat(0);
        var bufs = [_]Buffer{.{ .ptr = @ptrCast(&data), .len = @sizeOf(@TypeOf(data)) }};
        var ctx = KernelCtx{ .buffers = &bufs, .count = data.len, .alloc = testing.allocator };
        try dispatch(b, kernel, &ctx);
        for (data, 0..) |v, i| try testing.expectEqual(@as(f64, @floatFromInt(i)), v);
    }
}

test "devices are opened once and kept" {
    const first = try device(.cpu);
    const second = try device(.cpu);
    try testing.expect(first != null);
    try testing.expect(first == second);
}

test "available() lists the cpu backend at least" {
    const list = try available(testing.allocator);
    defer testing.allocator.free(list);
    try testing.expect(list.len >= 1);
    try testing.expectEqual(Backend.cpu, list[list.len - 1]);
}

test "when a driver is installed the kernel really runs on the device" {
    // On a machine with no driver the CPU fallback (tested above) is all there
    // is, and `probe` says so. When the probe *does* see a driver — a real GPU,
    // or a software rasterizer such as SwiftShader or lavapipe — the device has
    // to open and the shipped SPIR-V has to produce the right numbers.
    if (!probe().has(.vulkan)) return;
    const dev = (try device(.vulkan)) orelse return error.DeviceRefused;
    defer shutdownDevices();

    var data: [256]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i);
    var scale = [1]f32{2.0};
    var bufs = [_]Buffer{
        .{ .ptr = @ptrCast(&data), .len = @sizeOf(@TypeOf(data)) },
        .{ .ptr = @ptrCast(&scale), .len = @sizeOf(@TypeOf(scale)) },
    };
    var ctx = KernelCtx{
        .buffers = &bufs,
        .count = data.len,
        .threads = 64,
        .alloc = testing.allocator,
    };
    try dispatchOn(dev, .{
        .name = "scale_offset",
        .body = struct {
            fn body(_: *KernelCtx) void {}
        }.body,
        .source = .{ .wgsl = spirv.scale_offset_wgsl, .glsl = spirv.scale_offset_glsl },
        .spirv = &spirv.scale_offset,
    }, &ctx);
    for (data, 0..) |v, i| {
        const want = @as(f32, @floatFromInt(i)) * 2.0 + @as(f32, @floatFromInt(i));
        try testing.expectApproxEqAbs(want, v, 1e-3);
    }
}

test "a named driver is tried first and environment candidates last" {
    setDriverLibrary(.vulkan, "/tmp/zpu-test-vulkan.so");
    addDriverLibrary(.vulkan, "/tmp/zpu-test-icd.so");
    defer {
        setDriverLibrary(.vulkan, null);
        choices[@backingInt(Backend.vulkan)].extra_count = 0;
    }

    var buf: [max_candidates][]const u8 = undefined;
    const list = candidateList(.vulkan, &buf);
    try testing.expectEqualStrings("/tmp/zpu-test-vulkan.so", list[0]);
    try testing.expectEqualStrings("/tmp/zpu-test-icd.so", list[list.len - 1]);
    // The platform's own names stay in between: naming a driver must not turn
    // the list into an all-or-nothing switch.
    try testing.expect(list.len >= 3);

    // With nothing named, every backend starts from its platform list.
    var plain: [max_candidates][]const u8 = undefined;
    const webgpu_list = candidateList(.webgpu, &plain);
    try testing.expectEqualStrings(webgpu.libraryNames()[0], webgpu_list[0]);
}

test "device buffers hold a kernel's data between dispatches" {
    const Scale = struct {
        fn body(ctx: *KernelCtx) void {
            const data = ctx.buffers[0].as(f32);
            const scale = ctx.buffers[1].as(f32)[0];
            data[ctx.id] = data[ctx.id] * scale + @as(f32, @floatFromInt(ctx.id));
        }
    };
    const kernel = Kernel{
        .name = "scale_offset",
        .body = Scale.body,
        .source = .{ .wgsl = spirv.scale_offset_wgsl, .glsl = spirv.scale_offset_glsl },
        .spirv = &spirv.scale_offset,
    };

    // The CPU device keeps the bytes in host memory, so this checks the API
    // itself; the test below runs the same kernel on a real device when there is
    // one.
    const dev = (try device(.cpu)).?;
    var data = try createBuffer(dev, 256 * @sizeOf(f32), testing.allocator);
    defer destroyBuffer(dev, &data);
    var scale = try createBuffer(dev, @sizeOf(f32), testing.allocator);
    defer destroyBuffer(dev, &scale);

    var seed: [256]f32 = undefined;
    for (&seed, 0..) |*v, i| v.* = @floatFromInt(i);
    try writeBuffer(dev, &data, std.mem.sliceAsBytes(&seed));
    const two = [1]f32{2.0};
    try writeBuffer(dev, &scale, std.mem.sliceAsBytes(&two));

    try dispatchBuffers(dev, kernel, &.{ &data, &scale }, seed.len, 8);

    var out: [256]f32 = undefined;
    try readBuffer(dev, &data, std.mem.sliceAsBytes(&out));
    for (out, 0..) |v, i| {
        const want = @as(f32, @floatFromInt(i)) * 2.0 + @as(f32, @floatFromInt(i));
        try testing.expectApproxEqAbs(want, v, 1e-4);
    }

    // A released buffer is refused, not dereferenced.
    destroyBuffer(dev, &data);
    try testing.expectError(Error.InvalidBuffer, writeBuffer(dev, &data, &.{ 0, 0, 0, 0 }));
    try testing.expectError(Error.InvalidBuffer, readBuffer(dev, &data, std.mem.sliceAsBytes(&out)));
    // ...and destroying it twice is harmless.
    destroyBuffer(dev, &data);
}

test "resident buffers keep their data on a real device" {
    if (!probe().has(.vulkan)) return;
    const dev = (try device(.vulkan)) orelse return error.DeviceRefused;
    defer shutdownDevices();

    const Scale = struct {
        fn body(ctx: *KernelCtx) void {
            const data = ctx.buffers[0].as(f32);
            const scale = ctx.buffers[1].as(f32)[0];
            data[ctx.id] = data[ctx.id] * scale + @as(f32, @floatFromInt(ctx.id));
        }
    };
    const kernel = Kernel{
        .name = "scale_offset",
        .body = Scale.body,
        .source = .{ .wgsl = spirv.scale_offset_wgsl, .glsl = spirv.scale_offset_glsl },
        .spirv = &spirv.scale_offset,
    };

    var data = try createBuffer(dev, 256 * @sizeOf(f32), testing.allocator);
    defer destroyBuffer(dev, &data);
    var scale = try createBuffer(dev, @sizeOf(f32), testing.allocator);
    defer destroyBuffer(dev, &scale);

    var seed: [256]f32 = undefined;
    for (&seed, 0..) |*v, i| v.* = @floatFromInt(i);
    try writeBuffer(dev, &data, std.mem.sliceAsBytes(&seed));
    const two = [1]f32{2.0};
    try writeBuffer(dev, &scale, std.mem.sliceAsBytes(&two));

    // Twice on the same resident data: the second pass has to build on the
    // first, and it reuses the pipeline the first one compiled.
    try dispatchBuffers(dev, kernel, &.{ &data, &scale }, seed.len, 64);
    try dispatchBuffers(dev, kernel, &.{ &data, &scale }, seed.len, 64);

    var out: [256]f32 = undefined;
    try readBuffer(dev, &data, std.mem.sliceAsBytes(&out));
    for (out, 0..) |v, i| {
        const index = @as(f32, @floatFromInt(i));
        const want = (index * 2.0 + index) * 2.0 + index;
        try testing.expectApproxEqAbs(want, v, 1e-2);
    }
}
