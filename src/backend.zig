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

fn probeBackend(backend: Backend) Info {
    for (libraryNames(backend)) |name| {
        var lib = std.DynLib.open(name) catch continue;
        defer lib.close();
        var info = Info{ .available = true, .library_path = name };
        switch (backend) {
            .vulkan => vulkanProbe(&lib, &info),
            .metal => symbolProbe(&lib, "MTLCreateSystemDefaultDevice", &info),
            .d3d12 => symbolProbe(&lib, "D3D12CreateDevice", &info),
            .webgpu => symbolProbe(&lib, "wgpuCreateInstance", &info),
            .cpu => {},
        }
        return info;
    }
    return .{};
}

/// A driver is usable when the entry point the backend needs exists.
fn symbolProbe(lib: *std.DynLib, symbol: [:0]const u8, info: *Info) void {
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

// ---------------------------------------------------------------------------
// Vulkan loader probe
//
// The Vulkan loader is a plain C ABI, so a probe can create an instance,
// enumerate the physical devices and read their names without any SDK headers
// being present at build time. The struct layouts below are the first fields of
// the real ones; `VkPhysicalDeviceProperties` is read into an oversized buffer
// because the driver writes the whole struct.
// ---------------------------------------------------------------------------
const vk = struct {
    const Success: i32 = 0;
    const ApplicationInfoStype: u32 = 0;
    const InstanceCreateInfoStype: u32 = 1;
    const physical_device_type_discrete_gpu: u32 = 2;

    const ApplicationInfo = extern struct {
        sType: u32 = ApplicationInfoStype,
        pNext: ?*const anyopaque = null,
        pApplicationName: ?[*:0]const u8 = "zpu",
        applicationVersion: u32 = 1,
        pEngineName: ?[*:0]const u8 = "zpu",
        engineVersion: u32 = 1,
        apiVersion: u32 = (1 << 22),
    };

    const InstanceCreateInfo = extern struct {
        sType: u32 = InstanceCreateInfoStype,
        pNext: ?*const anyopaque = null,
        flags: u32 = 0,
        pApplicationInfo: ?*const ApplicationInfo = null,
        enabledLayerCount: u32 = 0,
        ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
        enabledExtensionCount: u32 = 0,
        ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    };

    const CreateInstanceFn = *const fn (*const InstanceCreateInfo, ?*const anyopaque, *?*anyopaque) callconv(.c) i32;
    const EnumerateInstanceVersionFn = *const fn (*u32) callconv(.c) i32;
    const GetInstanceProcAddrFn = *const fn (?*anyopaque, [*:0]const u8) callconv(.c) ?*const anyopaque;
    const EnumeratePhysicalDevicesFn = *const fn (?*anyopaque, *u32, ?[*]?*anyopaque) callconv(.c) i32;
    const GetPhysicalDevicePropertiesFn = *const fn (?*anyopaque, *anyopaque) callconv(.c) void;
    const DestroyInstanceFn = *const fn (?*anyopaque, ?*const anyopaque) callconv(.c) void;
};

fn vulkanProbe(lib: *std.DynLib, info: *Info) void {
    const gpa = lib.lookup(vk.GetInstanceProcAddrFn, "vkGetInstanceProcAddr") orelse {
        info.available = false;
        return;
    };

    // loader version (optional: 1.0 loaders do not export it)
    if (gpa(null, "vkEnumerateInstanceVersion")) |ptr| {
        const f: vk.EnumerateInstanceVersionFn = @ptrCast(ptr);
        var version: u32 = 0;
        if (f(&version) == vk.Success) info.api_version = version;
    }

    const create_ptr = gpa(null, "vkCreateInstance") orelse {
        info.available = false;
        return;
    };
    const create: vk.CreateInstanceFn = @ptrCast(create_ptr);

    var app = vk.ApplicationInfo{};
    var ci = vk.InstanceCreateInfo{ .pApplicationInfo = &app };
    if (info.api_version >= (1 << 22)) {
        app.apiVersion = @min(info.api_version, @as(u32, 3 << 22));
        ci.pApplicationInfo = &app;
    }
    var instance: ?*anyopaque = null;
    if (create(&ci, null, &instance) != vk.Success) {
        // The loader is present but no driver answered: still usable in
        // principle, just not on this machine.
        info.available = true;
        info.device_count = 0;
        return;
    }
    const inst = instance orelse {
        info.device_count = 0;
        return;
    };
    defer {
        if (gpa(inst, "vkDestroyInstance")) |ptr| {
            const destroy: vk.DestroyInstanceFn = @ptrCast(ptr);
            destroy(inst, null);
        }
    }

    const enum_ptr = gpa(inst, "vkEnumeratePhysicalDevices") orelse return;
    const enumerate: vk.EnumeratePhysicalDevicesFn = @ptrCast(enum_ptr);
    const props_ptr = gpa(inst, "vkGetPhysicalDeviceProperties") orelse return;
    const get_props: vk.GetPhysicalDevicePropertiesFn = @ptrCast(props_ptr);

    var count: u32 = 0;
    if (enumerate(inst, &count, null) != vk.Success) return;
    info.device_count = count;
    if (count == 0) return;

    var devices: [8]?*anyopaque = @splat(null);
    var want: u32 = @min(count, 8);
    if (enumerate(inst, &want, &devices) != vk.Success) return;

    for (devices[0..want], 0..) |dev, i| {
        // The driver fills the whole VkPhysicalDeviceProperties; the fields we
        // read are the first ones, so an oversized zeroed buffer is safe.
        var props: [2048]u8 = @splat(0);
        get_props(dev, &props);
        const api_version = std.mem.readInt(u32, props[0..4], .little);
        if (i == 0) info.api_version = api_version;
        const dev_name = props[20..276]; // VkPhysicalDeviceProperties.deviceName
        const len = std.mem.indexOfScalar(u8, dev_name, 0) orelse dev_name.len;
        const n = @min(len, info.names[i].len);
        @memcpy(info.names[i][0..n], dev_name[0..n]);
        info.name_lens[i] = @intCast(n);
    }
    info.names_len = @intCast(want);
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

const vulkan = @import("backend/vulkan.zig");
const webgpu = @import("backend/webgpu.zig");

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
    return switch (backend) {
        .cpu => Device{ .cpu = {} },
        .vulkan => if (vulkan.Device.open(null) catch |e| return mapDeviceError(e)) |d| Device{ .vulkan = d } else null,
        .webgpu => if (webgpu.Device.open() catch |e| return mapDeviceError(e)) |d| Device{ .webgpu = d } else null,
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
    const spirv = kernel.spirv orelse return Error.Unsupported;
    const alloc = ctx.alloc;
    const allocs = try alloc.alloc(vulkan.Device.Allocation, ctx.buffers.len);
    defer alloc.free(allocs);
    var created: usize = 0;
    defer for (allocs[0..created]) |a| d.free(a);
    for (ctx.buffers, 0..) |b, i| {
        allocs[i] = d.allocBuffer(b.len) catch |e| return mapDeviceError(e);
        created += 1;
        d.upload(allocs[i], b.ptr[0..b.len]);
    }
    const bindings = try alloc.alloc(vulkan.Device.BufferBinding, ctx.buffers.len);
    defer alloc.free(bindings);
    for (allocs, 0..) |a, i| bindings[i] = .{ .buffer = a.buffer, .size = a.size };
    d.dispatchCompute(spirv, bindings, ctx.count, ctx.threads) catch |e| return mapDeviceError(e);
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
    defer for (allocs[0..created]) |a| d.destroyBuffer(a);
    for (ctx.buffers, 0..) |b, i| {
        allocs[i] = d.createBuffer(b.len) catch |e| return mapDeviceError(e);
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

    const spirv = Kernel{ .name = "k", .body = base.body, .spirv = &.{ 0x07230203, 0, 1, 0 } };
    try testing.expect(spirv.canRun(.vulkan));
    try testing.expect(!spirv.canRun(.webgpu));
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
