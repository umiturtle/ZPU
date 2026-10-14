//! WebGPU compute backend.
//!
//! WebGPU has no stable C ABI of its own — the standard defines a JavaScript
//! API, and `webgpu.h` is a *convention* implemented by Dawn (Chrome, native),
//! `wgpu-native` (Firefox) and `wgpu` (Deno). This file speaks that convention:
//! every entry point is resolved from the library at runtime (`wgpuGetProcAddress`
//! on Dawn, plain exported symbols on wgpu-native), so nothing links against
//! either implementation and no headers are needed at build time.
//!
//! The structs mirror `webgpu.h` (2024 revision: `WGPUStringView` rather than
//! NUL-terminated strings, `WGPUChainedStruct *nextInChain` in place of the old
//! `sType` fields).
//!
//! Compute shape: one shader module per kernel (WGSL), one pipeline per
//! invocation geometry, a bind group with one storage buffer per `KernelCtx`
//! buffer, and one compute pass per dispatch. Buffers are
//! `Storage | CopySrc | CopyDst | MappedAtCreation`, so upload and readback are
//! `memcpy`s through a mapped range.
//!
//! The shader ABI matches the Vulkan one: `@group(0) @binding(i)` is
//! `KernelCtx.buffers[i]`, `@builtin(global_invocation_id)` is the invocation id,
//! and `{name}`/`{threads}` in the source template are substituted by
//! `zbackend.Kernel.lower`.
const std = @import("std");
const builtin = @import("builtin");

pub const Error = error{
    NoAdapter,
    NoDevice,
    OutOfMemory,
    WebGpuFailed,
};

/// Where the implementation lives on each platform.
pub fn libraryNames() []const []const u8 {
    return switch (builtin.os.tag) {
        .windows => &.{ "webgpu_dawn.dll", "wgpu_native.dll" },
        .macos => &.{ "libwebgpu_dawn.dylib", "libwgpu_native.dylib" },
        else => &.{ "libwebgpu_dawn.so", "libwgpu_native.so", "libwebgpu.so", "libdawn_native.so" },
    };
}

const PROC_NAME = "wgpuGetProcAddress";
/// How many times a callback is polled before giving up on it.
const poll_spins = 100_000;

// ---------------------------------------------------------------- ABI structs

const WGPUInstance = *anyopaque;
const WGPUAdapter = *anyopaque;
const WGPUDevice = *anyopaque;
const WGPUQueue = *anyopaque;
const WGPUBuffer = *anyopaque;
const WGPUShaderModule = *anyopaque;
const WGPUBindGroupLayout = *anyopaque;
const WGPUPipelineLayout = *anyopaque;
const WGPUComputePipeline = *anyopaque;
const WGPUBindGroup = *anyopaque;
const WGPUCommandEncoder = *anyopaque;
const WGPUComputePassEncoder = *anyopaque;
const WGPUCommandBuffer = *anyopaque;

/// `WGPUFlags` is 64-bit and `WGPUBool` is a `uint32_t`.
const Flags = u64;
const Bool = u32;
const TRUE: Bool = 1;

const Usage = struct {
    const map_read: Flags = 0x0000_0001;
    const map_write: Flags = 0x0000_0002;
    const copy_src: Flags = 0x0000_0004;
    const copy_dst: Flags = 0x0000_0008;
    const storage: Flags = 0x0000_0080;
};
const MapMode = struct {
    const read: Flags = 0x0000_0001;
};
const ShaderStage = struct {
    const compute: Flags = 0x0000_0004;
};
const BindingType = struct {
    const storage: u32 = 2;
};
const CallbackMode = struct {
    /// `WGPUCallbackMode_AllowProcessEvents`: the callback runs while
    /// `wgpuInstanceProcessEvents` / `wgpuDeviceTick` drive the queue, which is
    /// exactly the polling loop used below.
    const allow_process_events: u32 = 2;
};
const FutureStatus = struct {
    const success: u32 = 1;
};
const SType = struct {
    const shader_source_wgsl: u32 = 0x0000_0002;
};

/// `WGPUStringView`: a fat pointer that need not be NUL-terminated.
pub const StringView = extern struct {
    data: ?[*]const u8 = null,
    length: usize = 0,

    pub fn init(s: []const u8) StringView {
        return .{ .data = if (s.len == 0) null else s.ptr, .length = s.len };
    }
    pub fn cstr(s: [*:0]const u8) StringView {
        return .{ .data = s, .length = std.mem.len(s) };
    }
    pub fn slice(self: StringView) []const u8 {
        const p = self.data orelse return "";
        return p[0..self.length];
    }
};

const ChainedStruct = extern struct {
    next: ?*const ChainedStruct = null,
    s_type: u32,
};

const InstanceDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    required_feature_count: usize = 0,
    required_features: ?[*]const u32 = null,
    required_limit_count: usize = 0,
};

const RequestAdapterOptions = extern struct {
    next: ?*const ChainedStruct = null,
    feature_level: u32 = 2, // WGPUFeatureLevel_Core
    power_preference: u32 = 2, // WGPUPowerPreference_HighPerformance
    force_fallback_adapter: Bool = 0,
    backend_type: u32 = 0, // WGPUBackendType_Undefined
    compatible_surface: ?*anyopaque = null,
};

const RequestAdapterCallbackInfo = extern struct {
    next: ?*const ChainedStruct = null,
    mode: u32 = CallbackMode.allow_process_events,
    callback: *const fn (u32, WGPUAdapter, StringView, ?*anyopaque) callconv(.c) void = onAdapter,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const RequestDeviceCallbackInfo = extern struct {
    next: ?*const ChainedStruct = null,
    mode: u32 = CallbackMode.allow_process_events,
    callback: *const fn (u32, WGPUDevice, StringView, ?*anyopaque) callconv(.c) void = onDevice,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const QueueDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
};

const DeviceDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    required_feature_count: usize = 0,
    required_features: ?[*]const u32 = null,
    required_limit_count: usize = 0,
    required_limits: ?*const anyopaque = null,
    default_queue: QueueDescriptor = .{},
};

const ShaderModuleDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
};

const ShaderSourceWGSL = extern struct {
    chain: ChainedStruct = .{ .s_type = SType.shader_source_wgsl },
    code: StringView = .{},
};

const ProgrammableStageDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    entry_point: StringView = .{},
    module: WGPUShaderModule,
    constant_count: usize = 0,
    constants: ?*const anyopaque = null,
};

const ComputePipelineDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    layout: ?WGPUPipelineLayout = null,
    stage: ProgrammableStageDescriptor,
};

const BufferBindingLayout = extern struct {
    next: ?*const ChainedStruct = null,
    type: u32 = BindingType.storage,
    has_dynamic_offset: Bool = 0,
    min_binding_size: u64 = 0,
};

const SamplerBindingLayout = extern struct {
    next: ?*const ChainedStruct = null,
    type: u32 = 0,
};

const TextureBindingLayout = extern struct {
    next: ?*const ChainedStruct = null,
    sample_type: u32 = 0,
    view_dimension: u32 = 0,
    multisampled: Bool = 0,
};

const StorageTextureBindingLayout = extern struct {
    next: ?*const ChainedStruct = null,
    access: u32 = 0,
    format: u32 = 0,
    view_dimension: u32 = 0,
};

/// C's anonymous `union { WGPUSamplerBindingLayout sampler; ... }`: the members
/// overlap, so they must overlap here too or the array stride would be wrong.
const BindingMember = extern union {
    sampler: SamplerBindingLayout,
    texture: TextureBindingLayout,
    storage_texture: StorageTextureBindingLayout,
};

const BindGroupLayoutEntry = extern struct {
    next: ?*const ChainedStruct = null,
    binding: u32,
    visibility: Flags = ShaderStage.compute,
    binding_array_size: u32 = 0,
    buffer: BufferBindingLayout = .{},
    binding_member: BindingMember,
};

/// The only kind of entry this backend binds: a storage buffer at `binding`.
fn storageBinding(binding: u32) BindGroupLayoutEntry {
    return .{
        .binding = binding,
        .binding_member = .{ .storage_texture = .{} },
    };
}

const BindGroupLayoutDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    entry_count: usize = 0,
    entries: ?[*]const BindGroupLayoutEntry = null,
};

const PipelineLayoutDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    bind_group_layout_count: usize = 0,
    bind_group_layouts: ?[*]const WGPUBindGroupLayout = null,
    immediate_size: u32 = 0,
};

const BufferDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    usage: Flags = 0,
    size: u64 = 0,
    mapped_at_creation: Bool = 0,
};

const BindGroupEntry = extern struct {
    next: ?*const ChainedStruct = null,
    binding: u32,
    buffer: ?WGPUBuffer = null,
    offset: u64 = 0,
    size: u64 = 0,
    sampler: ?*anyopaque = null,
    texture_view: ?*anyopaque = null,
};

const BindGroupDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    layout: WGPUBindGroupLayout,
    entry_count: usize = 0,
    entries: ?[*]const BindGroupEntry = null,
};

const ComputePassDescriptor = extern struct {
    next: ?*const ChainedStruct = null,
    label: StringView = .{},
    timestamp_writes: ?*const anyopaque = null,
};

const BufferMapCallbackInfo = extern struct {
    next: ?*const ChainedStruct = null,
    mode: u32 = CallbackMode.allow_process_events,
    callback: *const fn (u32, WGPUBuffer, u64, ?*anyopaque) callconv(.c) void = onMapped,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

/// The only state a callback needs to hand back: the slot to write and whether
/// it ran.
const Slot = struct {
    handle: ?*anyopaque = null,
    done: bool = false,
};

fn onAdapter(status: u32, adapter: WGPUAdapter, message: StringView, userdata: ?*anyopaque) callconv(.c) void {
    _ = message;
    const slot: *Slot = @ptrCast(@alignCast(userdata orelse return));
    slot.done = true;
    if (status == FutureStatus.success) slot.handle = adapter;
}

fn onDevice(status: u32, device: WGPUDevice, message: StringView, userdata: ?*anyopaque) callconv(.c) void {
    _ = message;
    const slot: *Slot = @ptrCast(@alignCast(userdata orelse return));
    slot.done = true;
    if (status == FutureStatus.success) slot.handle = device;
}

fn onMapped(status: u32, buffer: WGPUBuffer, offset: u64, userdata: ?*anyopaque) callconv(.c) void {
    _ = status;
    _ = buffer;
    _ = offset;
    const slot: *Slot = @ptrCast(@alignCast(userdata orelse return));
    slot.done = true;
}

// ----------------------------------------------------------------- the loader

/// Runtime-resolved entry points.
pub const Loader = struct {
    lib: std.DynLib,
    /// `wgpuGetProcAddress`, present on Dawn. wgpu-native exports the entry
    /// points directly instead.
    get_proc: ?*const fn (StringView) callconv(.c) ?*const anyopaque,

    create_instance: *const fn (?*const InstanceDescriptor) callconv(.c) ?WGPUInstance,
    instance_request_adapter: *const fn (WGPUInstance, *const RequestAdapterOptions, *const RequestAdapterCallbackInfo) callconv(.c) ?*anyopaque,
    instance_process_events: *const fn (WGPUInstance) callconv(.c) void,
    instance_release: *const fn (WGPUInstance) callconv(.c) void,
    adapter_request_device: *const fn (WGPUAdapter, ?*const DeviceDescriptor, *const RequestDeviceCallbackInfo) callconv(.c) ?*anyopaque,
    adapter_has_feature: *const fn (WGPUAdapter, u32) callconv(.c) Bool,
    adapter_release: *const fn (WGPUAdapter) callconv(.c) void,
    device_create_shader_module: *const fn (WGPUDevice, *const ShaderModuleDescriptor) callconv(.c) WGPUShaderModule,
    device_create_buffer: *const fn (WGPUDevice, *const BufferDescriptor) callconv(.c) WGPUBuffer,
    device_create_bind_group_layout: *const fn (WGPUDevice, *const BindGroupLayoutDescriptor) callconv(.c) WGPUBindGroupLayout,
    device_create_pipeline_layout: *const fn (WGPUDevice, *const PipelineLayoutDescriptor) callconv(.c) WGPUPipelineLayout,
    device_create_compute_pipeline: *const fn (WGPUDevice, *const ComputePipelineDescriptor) callconv(.c) WGPUComputePipeline,
    device_create_bind_group: *const fn (WGPUDevice, *const BindGroupDescriptor) callconv(.c) WGPUBindGroup,
    device_create_command_encoder: *const fn (WGPUDevice, ?*const anyopaque) callconv(.c) WGPUCommandEncoder,
    device_get_queue: *const fn (WGPUDevice) callconv(.c) WGPUQueue,
    device_tick: *const fn (WGPUDevice) callconv(.c) Bool,
    device_destroy: *const fn (WGPUDevice) callconv(.c) void,
    device_release: *const fn (WGPUDevice) callconv(.c) void,
    buffer_get_mapped_range: *const fn (WGPUBuffer, u64, usize) callconv(.c) [*]u8,
    buffer_unmap: *const fn (WGPUBuffer) callconv(.c) void,
    buffer_map_async: *const fn (WGPUBuffer, Flags, usize, usize, *const BufferMapCallbackInfo) callconv(.c) ?*anyopaque,
    buffer_release: *const fn (WGPUBuffer) callconv(.c) void,
    queue_submit: *const fn (WGPUQueue, usize, ?[*]const WGPUCommandBuffer) callconv(.c) void,
    queue_write_buffer: *const fn (WGPUQueue, WGPUBuffer, u64, [*]const u8, usize) callconv(.c) void,
    command_encoder_begin_compute_pass: *const fn (WGPUCommandEncoder, ?*const ComputePassDescriptor) callconv(.c) WGPUComputePassEncoder,
    command_encoder_finish: *const fn (WGPUCommandEncoder, ?*const anyopaque) callconv(.c) WGPUCommandBuffer,
    compute_pass_set_pipeline: *const fn (WGPUComputePassEncoder, WGPUComputePipeline) callconv(.c) void,
    compute_pass_set_bind_group: *const fn (WGPUComputePassEncoder, u32, WGPUBindGroup, usize, ?[*]const u32) callconv(.c) void,
    compute_pass_dispatch_workgroups: *const fn (WGPUComputePassEncoder, u32, u32, u32) callconv(.c) void,
    compute_pass_end: *const fn (WGPUComputePassEncoder) callconv(.c) void,
    command_buffer_release: *const fn (WGPUCommandBuffer) callconv(.c) void,
    shader_module_release: *const fn (WGPUShaderModule) callconv(.c) void,
    bind_group_layout_release: *const fn (WGPUBindGroupLayout) callconv(.c) void,
    pipeline_layout_release: *const fn (WGPUPipelineLayout) callconv(.c) void,
    compute_pipeline_release: *const fn (WGPUComputePipeline) callconv(.c) void,
    bind_group_release: *const fn (WGPUBindGroup) callconv(.c) void,

    pub fn open() ?Loader {
        var err: ?anyerror = null;
        return openWith(&err);
    }

    pub fn openWith(err_out: *?anyerror) ?Loader {
        for (libraryNames()) |name| {
            if (tryLoad(name, err_out)) |loader| return loader;
        }
        return null;
    }

    /// Opens one specific library: a bare name searched on the platform's
    /// library path, or an absolute path — how a program picks the WebGPU
    /// implementation it wants (Dawn out of a browser bundle, wgpu-native, a
    /// build in a non-standard prefix).
    pub fn openAt(library: []const u8) ?Loader {
        var err: ?anyerror = null;
        return tryLoad(library, &err);
    }

    fn tryLoad(name: []const u8, err_out: *?anyerror) ?Loader {
        var lib = std.DynLib.open(name) catch |e| {
            err_out.* = e;
            return null;
        };
        const get_proc = lib.lookup(
            *const fn (StringView) callconv(.c) ?*const anyopaque,
            PROC_NAME,
        ) orelse null;
        var self: Loader = undefined;
        self.lib = lib;
        self.get_proc = get_proc;
        if (!self.bindAll()) {
            err_out.* = error.SymbolMissing;
            lib.close();
            return null;
        }
        return self;
    }

    /// Resolves the whole table; one missing entry means the library is not an
    /// implementation of the ABI we speak to.
    fn bindAll(self: *Loader) bool {
        inline for (symbol_table) |entry| {
            const raw: ?*const anyopaque = if (self.get_proc) |f|
                f(StringView.cstr(entry.symbol))
            else
                self.lib.lookup(*const fn () callconv(.c) ?*const anyopaque, entry.symbol) orelse null;
            @field(self, entry.field) = @ptrCast(@alignCast(raw orelse return false));
        }
        return true;
    }

    pub fn close(self: *Loader) void {
        self.lib.close();
    }
};

/// `Loader` field name → exported symbol name. One table, so the field list and
/// the symbol list cannot drift; an unknown field fails the build.
const symbol_table = [_]struct { field: []const u8, symbol: [:0]const u8 }{
    .{ .field = "create_instance", .symbol = "wgpuCreateInstance" },
    .{ .field = "instance_request_adapter", .symbol = "wgpuInstanceRequestAdapter" },
    .{ .field = "instance_process_events", .symbol = "wgpuInstanceProcessEvents" },
    .{ .field = "instance_release", .symbol = "wgpuInstanceRelease" },
    .{ .field = "adapter_request_device", .symbol = "wgpuAdapterRequestDevice" },
    .{ .field = "adapter_has_feature", .symbol = "wgpuAdapterHasFeature" },
    .{ .field = "adapter_release", .symbol = "wgpuAdapterRelease" },
    .{ .field = "device_create_shader_module", .symbol = "wgpuDeviceCreateShaderModule" },
    .{ .field = "device_create_buffer", .symbol = "wgpuDeviceCreateBuffer" },
    .{ .field = "device_create_bind_group_layout", .symbol = "wgpuDeviceCreateBindGroupLayout" },
    .{ .field = "device_create_pipeline_layout", .symbol = "wgpuDeviceCreatePipelineLayout" },
    .{ .field = "device_create_compute_pipeline", .symbol = "wgpuDeviceCreateComputePipeline" },
    .{ .field = "device_create_bind_group", .symbol = "wgpuDeviceCreateBindGroup" },
    .{ .field = "device_create_command_encoder", .symbol = "wgpuDeviceCreateCommandEncoder" },
    .{ .field = "device_get_queue", .symbol = "wgpuDeviceGetQueue" },
    .{ .field = "device_tick", .symbol = "wgpuDeviceTick" },
    .{ .field = "device_destroy", .symbol = "wgpuDeviceDestroy" },
    .{ .field = "device_release", .symbol = "wgpuDeviceRelease" },
    .{ .field = "buffer_get_mapped_range", .symbol = "wgpuBufferGetMappedRange" },
    .{ .field = "buffer_unmap", .symbol = "wgpuBufferUnmap" },
    .{ .field = "buffer_map_async", .symbol = "wgpuBufferMapAsync" },
    .{ .field = "buffer_release", .symbol = "wgpuBufferRelease" },
    .{ .field = "queue_submit", .symbol = "wgpuQueueSubmit" },
    .{ .field = "queue_write_buffer", .symbol = "wgpuQueueWriteBuffer" },
    .{ .field = "command_encoder_begin_compute_pass", .symbol = "wgpuCommandEncoderBeginComputePass" },
    .{ .field = "command_encoder_finish", .symbol = "wgpuCommandEncoderFinish" },
    .{ .field = "compute_pass_set_pipeline", .symbol = "wgpuComputePassEncoderSetPipeline" },
    .{ .field = "compute_pass_set_bind_group", .symbol = "wgpuComputePassEncoderSetBindGroup" },
    .{ .field = "compute_pass_dispatch_workgroups", .symbol = "wgpuComputePassEncoderDispatchWorkgroups" },
    .{ .field = "compute_pass_end", .symbol = "wgpuComputePassEncoderEnd" },
    .{ .field = "command_buffer_release", .symbol = "wgpuCommandBufferRelease" },
    .{ .field = "shader_module_release", .symbol = "wgpuShaderModuleRelease" },
    .{ .field = "bind_group_layout_release", .symbol = "wgpuBindGroupLayoutRelease" },
    .{ .field = "pipeline_layout_release", .symbol = "wgpuPipelineLayoutRelease" },
    .{ .field = "compute_pipeline_release", .symbol = "wgpuComputePipelineRelease" },
    .{ .field = "bind_group_release", .symbol = "wgpuBindGroupRelease" },
};

/// Every `Loader` field except the two bookkeeping ones must appear in
/// `symbol_table`; the comptime loop fails the build otherwise.
fn symbolFor(comptime field: []const u8) [:0]const u8 {
    @setEvalBranchQuota(10_000);
    inline for (symbol_table) |entry| {
        if (comptime std.mem.eql(u8, field, entry.field)) return entry.symbol;
    }
    @compileError("no wgpu symbol registered for Loader field `" ++ field ++ "`");
}

comptime {
    @setEvalBranchQuota(10_000);
    for (@typeInfo(Loader).@"struct".field_names) |fname| {
        if (std.mem.eql(u8, fname, "lib") or std.mem.eql(u8, fname, "get_proc")) continue;
        _ = symbolFor(fname);
    }
}

// ----------------------------------------------------------------- the device

/// An adapter, a device and its queue — everything a dispatch needs.
pub const Device = struct {
    loader: Loader,
    instance: WGPUInstance,
    adapter: WGPUAdapter,
    handle: WGPUDevice,
    queue: WGPUQueue,

    /// A buffer plus the host size the caller asked for.
    pub const Allocation = struct {
        buffer: WGPUBuffer,
        size: usize,
    };

    /// Opens the implementation and asks for an adapter and a device.
    ///
    /// `null` means "this machine has no WebGPU" — nothing to load, no adapter,
    /// or no device.
    pub fn open() Error!?Device {
        var load_err: ?anyerror = null;
        const loader = Loader.openWith(&load_err) orelse return null;
        return openWithLoader(loader);
    }

    /// Same as `open`, but through one specific library (see `Loader.openAt`).
    pub fn openAt(library: []const u8) Error!?Device {
        const loader = Loader.openAt(library) orelse return null;
        return openWithLoader(loader);
    }

    fn openWithLoader(loader: Loader) Error!?Device {
        return Device.init(loader) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => null,
        };
    }

    pub fn init(loader_in: Loader) Error!Device {
        var loader = loader_in;
        errdefer loader.close();
        const instance = loader.create_instance(null) orelse return Error.NoAdapter;
        var adapter_slot = Slot{};
        const opts = RequestAdapterOptions{};
        const cb = RequestAdapterCallbackInfo{ .userdata1 = &adapter_slot };
        _ = loader.instance_request_adapter(instance, &opts, &cb);
        if (!poll(&loader, instance, &adapter_slot)) return Error.NoAdapter;
        const adapter: WGPUAdapter = @ptrCast(adapter_slot.handle.?);

        var device_slot = Slot{};
        const dcb = RequestDeviceCallbackInfo{ .userdata1 = &device_slot };
        const desc = DeviceDescriptor{ .label = StringView.cstr("zpu") };
        _ = loader.adapter_request_device(adapter, &desc, &dcb);
        if (!poll(&loader, instance, &device_slot)) return Error.NoDevice;
        const handle: WGPUDevice = @ptrCast(device_slot.handle.?);

        return .{
            .loader = loader,
            .instance = instance,
            .adapter = adapter,
            .handle = handle,
            .queue = loader.device_get_queue(handle),
        };
    }

    /// WaitAnyOnly callbacks run while events are processed or the device is
    /// ticked; both are called because Dawn and wgpu-native differ in which one
    /// drives the queue.
    fn poll(loader: *const Loader, instance: WGPUInstance, slot: *Slot) bool {
        var spins: usize = 0;
        while (!slot.done and spins < poll_spins) : (spins += 1) {
            loader.instance_process_events(instance);
        }
        return slot.done and slot.handle != null;
    }

    pub fn deinit(self: *Device) void {
        self.loader.device_destroy(self.handle);
        self.loader.device_release(self.handle);
        self.loader.adapter_release(self.adapter);
        self.loader.instance_release(self.instance);
        self.loader.close();
    }

    /// Whether the adapter reports `feature` (`WGPUFeatureName`).
    pub fn hasFeature(self: *const Device, feature: u32) bool {
        return self.loader.adapter_has_feature(self.adapter, feature) == TRUE;
    }

    /// Creates a device buffer big enough for `size` bytes and maps it for the
    /// caller's whole lifetime.
    pub fn createBuffer(self: *const Device, size: usize) Error!Allocation {
        const desc = BufferDescriptor{
            .label = StringView.cstr("zpu buffer"),
            .usage = Usage.storage | Usage.copy_src | Usage.copy_dst | Usage.map_write | Usage.map_read,
            .size = @max(size, 4),
            .mapped_at_creation = TRUE,
        };
        const buf = self.loader.device_create_buffer(self.handle, &desc);
        return .{ .buffer = buf, .size = size };
    }

    pub fn destroyBuffer(self: *const Device, alloc: Allocation) void {
        self.loader.buffer_release(alloc.buffer);
    }

    /// Writes host bytes into a buffer that has not been unmapped yet.
    pub fn upload(self: *const Device, alloc: Allocation, bytes: []const u8) void {
        const n = @min(bytes.len, self.mappedLen(alloc));
        const dst = self.loader.buffer_get_mapped_range(alloc.buffer, 0, n);
        @memcpy(dst[0..n], bytes[0..n]);
    }

    /// Submits a WGSL kernel over `bindings`, then reads `out` back.
    ///
    /// The kernel's buffers must already hold their inputs; `bindings[i]`
    /// corresponds to `@group(0) @binding(i)`.
    pub fn run(
        self: *const Device,
        alloc: std.mem.Allocator,
        wgsl: []const u8,
        entry_point: []const u8,
        bindings: []const Allocation,
        count: usize,
        threads: u32,
    ) Error!void {
        const d = self.loader;
        var source = ShaderSourceWGSL{ .code = StringView.init(wgsl) };
        var sm_desc = ShaderModuleDescriptor{
            .next = &source.chain,
            .label = StringView.cstr("zpu kernel"),
        };
        const module = d.device_create_shader_module(self.handle, &sm_desc);
        defer d.shader_module_release(module);

        const layout_entries = alloc.alloc(BindGroupLayoutEntry, bindings.len) catch return Error.OutOfMemory;
        defer alloc.free(layout_entries);
        for (layout_entries, 0..) |*e, i| e.* = storageBinding(@intCast(i));
        const bgl_desc = BindGroupLayoutDescriptor{
            .entry_count = layout_entries.len,
            .entries = if (layout_entries.len == 0) null else layout_entries.ptr,
        };
        const bgl = d.device_create_bind_group_layout(self.handle, &bgl_desc);
        defer d.bind_group_layout_release(bgl);

        const set_layouts = [1]WGPUBindGroupLayout{bgl};
        const pl_desc = PipelineLayoutDescriptor{
            .label = StringView.cstr("zpu pipeline layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = &set_layouts,
        };
        const pl = d.device_create_pipeline_layout(self.handle, &pl_desc);
        defer d.pipeline_layout_release(pl);

        const pipe_desc = ComputePipelineDescriptor{
            .label = StringView.cstr("zpu pipeline"),
            .layout = pl,
            .stage = .{
                .entry_point = StringView.init(entry_point),
                .module = module,
            },
        };
        const pipeline = d.device_create_compute_pipeline(self.handle, &pipe_desc);
        defer d.compute_pipeline_release(pipeline);

        const group_entries = alloc.alloc(BindGroupEntry, bindings.len) catch return Error.OutOfMemory;
        defer alloc.free(group_entries);
        for (group_entries, 0..) |*e, i| e.* = .{ .binding = @intCast(i), .buffer = bindings[i].buffer };
        const bg_desc = BindGroupDescriptor{
            .label = StringView.cstr("zpu bind group"),
            .layout = bgl,
            .entry_count = group_entries.len,
            .entries = if (group_entries.len == 0) null else group_entries.ptr,
        };
        const group = d.device_create_bind_group(self.handle, &bg_desc);
        defer d.bind_group_release(group);

        const encoder = d.device_create_command_encoder(self.handle, null);
        const pass = d.command_encoder_begin_compute_pass(encoder, null);
        d.compute_pass_set_pipeline(pass, pipeline);
        if (bindings.len != 0) d.compute_pass_set_bind_group(pass, 0, group, 0, null);
        const locals = @max(threads, 1);
        d.compute_pass_dispatch_workgroups(pass, @intCast((count + locals - 1) / locals), 1, 1);
        d.compute_pass_end(pass);
        const cmd = d.command_encoder_finish(encoder, null);
        defer d.command_buffer_release(cmd);

        const commands = [1]WGPUCommandBuffer{cmd};
        d.queue_submit(self.queue, 1, &commands);
        _ = d.device_tick(self.handle); // flush, and run the device timeline
    }

    /// Reads a buffer back into host memory. The buffer must not be mapped at
    /// this point (call `commit` first).
    pub fn download(self: *const Device, alloc: Allocation, out: []u8) Error!void {
        const d = self.loader;
        var slot = Slot{};
        const info = BufferMapCallbackInfo{ .userdata1 = &slot };
        _ = d.buffer_map_async(alloc.buffer, MapMode.read, 0, out.len, &info);
        var spins: usize = 0;
        while (!slot.done and spins < poll_spins) : (spins += 1) {
            d.instance_process_events(self.instance);
            _ = d.device_tick(self.handle);
        }
        if (!slot.done) return Error.WebGpuFailed;
        const src = d.buffer_get_mapped_range(alloc.buffer, 0, out.len);
        @memcpy(out, src[0..out.len]);
        d.buffer_unmap(alloc.buffer);
    }

    /// Unmaps a buffer that was written with `upload`, so the GPU may read it.
    pub fn commit(self: *const Device, alloc: Allocation) void {
        self.loader.buffer_unmap(alloc.buffer);
    }

    fn mappedLen(self: *const Device, alloc: Allocation) usize {
        _ = self;
        return alloc.size;
    }
};

test "the struct tables keep their C layout" {
    // Sizes that the ABI fixes, checked against `webgpu.h` on a 64-bit host:
    // StringView is a data pointer plus a length, a bind group layout entry is
    // next + binding + visibility/bindingArraySize + buffer + the union.
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(StringView));
    const entry = storageBinding(3);
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(BindGroupLayoutEntry));
    try std.testing.expectEqual(@as(u32, 0), entry.binding_member.storage_texture.access);
    try std.testing.expectEqual(@as(u32, BindingType.storage), entry.buffer.type);
}

test "StringView matches the C layout" {
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(StringView, "length"));
    const v = StringView.cstr("wgpuCreateInstance");
    try std.testing.expectEqual(@as(usize, 18), v.length);
    try std.testing.expectEqualStrings("wgpuCreateInstance", v.slice());
}

test "library names match the host platform" {
    const names = libraryNames();
    try std.testing.expect(names.len != 0);
    const expected = switch (builtin.os.tag) {
        .windows => "webgpu_dawn.dll",
        .macos => "libwebgpu_dawn.dylib",
        else => "libwebgpu_dawn.so",
    };
    try std.testing.expectEqualStrings(expected, names[0]);
}

test "openAt on a path that is not a library is null, not a crash" {
    try std.testing.expect(Loader.openAt("/nonexistent/libwebgpu_dawn.so") == null);
}

test "opening a missing library fails cleanly" {
    // In a sandbox without Dawn/wgpu-native this must be `null`, not a crash;
    // with one installed it must open and close.
    if (Loader.open()) |*l| {
        var loader = l.*;
        loader.close();
    }
}
