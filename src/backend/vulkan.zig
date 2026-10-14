//! Vulkan compute backend.
//!
//! Everything here talks to the *loader* (`libvulkan.so.1`, `vulkan-1.dll`,
//! `libvulkan.dylib`) through `std.DynLib`: no SDK, no headers, no link-time
//! dependency. The ABI structs below are declared locally with the exact layout
//! the API requires, and every entry point is looked up by name.
//!
//! The pipeline is the classic compute one:
//!
//! 1. `vkCreateInstance` (no layers, no extensions — compute needs none),
//! 2. enumerate physical devices and pick one,
//! 3. a queue family with `QUEUE_COMPUTE_BIT`, a logical device and one queue,
//! 4. host-visible, coherent memory for every buffer (upload and readback are
//!    `memcpy`s into the persistently mapped allocation),
//! 5. per dispatch: shader module from the kernel's SPIR-V, descriptor set
//!    layout with one storage-buffer binding per `KernelCtx` buffer, pipeline,
//!    descriptor set, push constants with the invocation count,
//!    `vkCmdDispatch`, a fence, and `vkQueueSubmit` + `vkWaitForFences`.
//!
//! The kernel ABI (what a shader may expect) is documented in
//! `docs/features/09-backends.md`: set 0, binding *i* = `KernelCtx.buffers[i]` as
//! a storage buffer, and a push constant block `{ u32 count; u32 threads; }`.
const std = @import("std");
const builtin = @import("builtin");

/// Vulkan's ABI: `__stdcall` on Windows, the C convention elsewhere.
const VKAPI = if (builtin.os.tag == .windows) std.builtin.CallingConvention.winapi else std.builtin.CallingConvention.c;

pub const Error = error{
    LoaderUnavailable,
    NoDevice,
    NoComputeQueue,
    NoHostVisibleMemory,
    OutOfMemory,
    VulkanFailed,
};

/// The libraries the loader is known by, per platform.
pub fn libraryNames() []const []const u8 {
    return switch (builtin.os.tag) {
        .windows => &.{"vulkan-1.dll"},
        .macos => &.{ "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib" },
        else => &.{ "libvulkan.so.1", "libvulkan.so" },
    };
}

// ---------------------------------------------------------------- ABI structs

const VkInstance = *anyopaque;
const VkPhysicalDevice = *anyopaque;
const VkDevice = *anyopaque;
const VkQueue = *anyopaque;
const VkBuffer = *anyopaque;
const VkDeviceMemory = *anyopaque;
const VkShaderModule = *anyopaque;
const VkDescriptorSetLayout = *anyopaque;
const VkPipelineLayout = *anyopaque;
const VkPipeline = *anyopaque;
const VkDescriptorPool = *anyopaque;
const VkDescriptorSet = *anyopaque;
const VkCommandPool = *anyopaque;
const VkCommandBuffer = *anyopaque;
const VkFence = *anyopaque;

const VK_SUCCESS: i32 = 0;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;

const VK_QUEUE_COMPUTE_BIT: u32 = 0x0000_0002;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0000_0020;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0000_0020;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x0000_0002;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x1;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x2;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x4;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x1;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

const ApplicationInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    p_next: ?*const anyopaque = null,
    application_name: ?[*:0]const u8 = "zpu",
    application_version: u32 = 1,
    engine_name: ?[*:0]const u8 = "zpu",
    engine_version: u32 = 1,
    api_version: u32,
};

const InstanceCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    application_info: ?*const ApplicationInfo = null,
    layer_count: u32 = 0,
    layer_names: ?[*]const [*:0]const u8 = null,
    extension_count: u32 = 0,
    extension_names: ?[*]const [*:0]const u8 = null,
};

/// The first 2048 bytes of `VkPhysicalDeviceProperties`. Only the name and the
/// API version are read; the limits are left untouched.
pub const PhysicalDeviceProperties = extern struct {
    api_version: u32 = 0,
    driver_version: u32 = 0,
    vendor_id: u32 = 0,
    device_id: u32 = 0,
    device_type: u32 = 0,
    device_name: [256]u8 = @splat(0),
    pipeline_cache_uuid: [16]u8 = @splat(0),
    reserved: [2020]u8 = @splat(0),

    pub fn name(self: *const PhysicalDeviceProperties) []const u8 {
        return std.mem.sliceTo(&self.device_name, 0);
    }
};

const MemoryType = extern struct { property_flags: u32, heap_index: u32 };
const MemoryHeap = extern struct { size: u64, flags: u32 };

const PhysicalDeviceMemoryProperties = extern struct {
    memory_type_count: u32 = 0,
    memory_types: [32]MemoryType = @splat(.{ .property_flags = 0, .heap_index = 0 }),
    memory_heap_count: u32 = 0,
    memory_heaps: [16]MemoryHeap = @splat(.{ .size = 0, .flags = 0 }),
};

const QueueFamilyProperties = extern struct {
    queue_flags: u32 = 0,
    queue_count: u32 = 0,
    timestamp_valid_bits: u32 = 0,
    min_image_transfer_granularity: [3]u32 = @splat(0),
};

const DeviceQueueCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    queue_family_index: u32,
    queue_count: u32 = 1,
    queue_priorities: [*]const f32,
};

const DeviceCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    queue_create_info_count: u32 = 1,
    queue_create_infos: [*]const DeviceQueueCreateInfo,
    layer_count: u32 = 0,
    layer_names: ?[*]const [*:0]const u8 = null,
    extension_count: u32 = 0,
    extension_names: ?[*]const [*:0]const u8 = null,
    enabled_features: ?*const anyopaque = null,
};

const MemoryAllocateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    p_next: ?*const anyopaque = null,
    allocation_size: u64,
    memory_type_index: u32,
};

const BufferCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    size: u64,
    usage: u32,
    sharing_mode: u32 = VK_SHARING_MODE_EXCLUSIVE,
    queue_family_index_count: u32 = 0,
    queue_family_indices: ?[*]const u32 = null,
};

const MemoryRequirements = extern struct {
    size: u64 = 0,
    alignment: u64 = 0,
    memory_type_bits: u32 = 0,
};

const ShaderModuleCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    code_size: usize,
    code: [*]const u32,
};

const DescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptor_type: u32 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    descriptor_count: u32 = 1,
    stage_flags: u32 = VK_SHADER_STAGE_COMPUTE_BIT,
    immutable_samplers: ?*const anyopaque = null,
};

const DescriptorSetLayoutCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    binding_count: u32 = 0,
    bindings: ?[*]const DescriptorSetLayoutBinding = null,
};

const PushConstantRange = extern struct {
    stage_flags: u32 = VK_SHADER_STAGE_COMPUTE_BIT,
    offset: u32 = 0,
    size: u32 = 0,
};

const PipelineLayoutCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    set_layout_count: u32 = 0,
    set_layouts: ?[*]const VkDescriptorSetLayout = null,
    push_constant_range_count: u32 = 0,
    push_constant_ranges: ?[*]const PushConstantRange = null,
};

const PipelineShaderStageCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32 = VK_SHADER_STAGE_COMPUTE_BIT,
    module: VkShaderModule,
    name: [*:0]const u8 = "main",
    specialization_info: ?*const anyopaque = null,
};

const ComputePipelineCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: PipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    base_pipeline_handle: ?VkPipeline = null,
    base_pipeline_index: i32 = -1,
};

const DescriptorPoolSize = extern struct {
    descriptor_type: u32 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    descriptor_count: u32 = 1,
};

const DescriptorPoolCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    max_sets: u32 = 1,
    pool_size_count: u32 = 1,
    pool_sizes: [*]const DescriptorPoolSize,
};

const DescriptorSetAllocateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    p_next: ?*const anyopaque = null,
    descriptor_pool: VkDescriptorPool,
    descriptor_set_count: u32 = 1,
    set_layouts: [*]const VkDescriptorSetLayout,
};

const DescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: u64 = 0,
    range: u64,
};

const WriteDescriptorSet = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    p_next: ?*const anyopaque = null,
    dst_set: VkDescriptorSet,
    dst_binding: u32,
    dst_array_element: u32 = 0,
    descriptor_count: u32 = 1,
    descriptor_type: u32 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    image_info: ?*const anyopaque = null,
    buffer_info: ?[*]const DescriptorBufferInfo = null,
    texel_buffer_view: ?*const anyopaque = null,
};

const CommandPoolCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    queue_family_index: u32,
};

const CommandBufferAllocateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    p_next: ?*const anyopaque = null,
    command_pool: VkCommandPool,
    level: u32 = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    command_buffer_count: u32 = 1,
};

const CommandBufferBeginInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    inheritance_info: ?*const anyopaque = null,
};

const SubmitInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    p_next: ?*const anyopaque = null,
    wait_semaphore_count: u32 = 0,
    wait_semaphores: ?[*]const *anyopaque = null,
    wait_stage_flags: ?[*]const u32 = null,
    command_buffer_count: u32 = 1,
    command_buffers: [*]const VkCommandBuffer,
    signal_semaphore_count: u32 = 0,
    signal_semaphores: ?[*]const *anyopaque = null,
};

const FenceCreateInfo = extern struct {
    s_type: u32 = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
};

/// The push constant block every kernel may rely on.
pub const PushParams = extern struct {
    count: u32,
    threads: u32,
    pad: [2]u32 = @splat(0),
};

// ----------------------------------------------------------------- the loader

/// Function pointers resolved by name. Vulkan exposes instance level entry
/// points through `vkGetInstanceProcAddr` and device level ones through
/// `vkGetDeviceProcAddr`; the CPU-side fallbacks (here: core 1.0 only) are
/// resolved from the loader itself.
pub const Loader = struct {
    lib: std.DynLib,
    /// Held so the device entry points can be looked up later.
    get_instance_proc: *const fn (?VkInstance, [*:0]const u8, ?*const anyopaque) callconv(VKAPI) ?*const anyopaque,

    create_instance: *const fn (*const InstanceCreateInfo, ?*const anyopaque, *VkInstance) callconv(VKAPI) i32,
    destroy_instance: *const fn (VkInstance, ?*const anyopaque) callconv(VKAPI) void,
    enumerate_physical_devices: *const fn (VkPhysicalDevice, ?*u32, ?[*]VkPhysicalDevice) callconv(VKAPI) i32,
    get_physical_device_properties: *const fn (VkPhysicalDevice, *PhysicalDeviceProperties) callconv(VKAPI) void,
    get_physical_device_memory_properties: *const fn (VkPhysicalDevice, *PhysicalDeviceMemoryProperties) callconv(VKAPI) void,
    get_physical_device_queue_family_properties: *const fn (VkPhysicalDevice, *u32, ?[*]QueueFamilyProperties) callconv(VKAPI) void,
    create_device: *const fn (VkPhysicalDevice, *const DeviceCreateInfo, ?*const anyopaque, *VkDevice) callconv(VKAPI) i32,

    /// Opens the first library that provides `vkGetInstanceProcAddr`.
    pub fn open() ?Loader {
        var err: ?anyerror = null;
        return openWith(&err);
    }

    pub fn openWith(err_out: *?anyerror) ?Loader {
        for (libraryNames()) |name| {
            var lib = std.DynLib.open(name) catch |e| {
                err_out.* = e;
                continue;
            };
            const get_instance_proc = lib.lookup(
                *const fn (?VkInstance, [*:0]const u8, ?*const anyopaque) callconv(VKAPI) ?*const anyopaque,
                "vkGetInstanceProcAddr",
            ) orelse continue;
            const loader = Loader.fromHandle(lib, get_instance_proc) orelse {
                lib.close();
                continue;
            };
            return loader;
        }
        return null;
    }

    fn fromHandle(lib: std.DynLib, get_instance_proc: anytype) ?Loader {
        var self = Loader{
            .lib = lib,
            .get_instance_proc = get_instance_proc,
            .create_instance = undefined,
            .destroy_instance = undefined,
            .enumerate_physical_devices = undefined,
            .get_physical_device_properties = undefined,
            .get_physical_device_memory_properties = undefined,
            .get_physical_device_queue_family_properties = undefined,
            .create_device = undefined,
        };
        self.create_instance = self.load(@TypeOf(self.create_instance), "vkCreateInstance") orelse return null;
        self.destroy_instance = self.load(@TypeOf(self.destroy_instance), "vkDestroyInstance") orelse return null;
        self.enumerate_physical_devices = self.load(@TypeOf(self.enumerate_physical_devices), "vkEnumeratePhysicalDevices") orelse return null;
        self.get_physical_device_properties = self.load(@TypeOf(self.get_physical_device_properties), "vkGetPhysicalDeviceProperties") orelse return null;
        self.get_physical_device_memory_properties = self.load(@TypeOf(self.get_physical_device_memory_properties), "vkGetPhysicalDeviceMemoryProperties") orelse return null;
        self.get_physical_device_queue_family_properties = self.load(@TypeOf(self.get_physical_device_queue_family_properties), "vkGetPhysicalDeviceQueueFamilyProperties") orelse return null;
        self.create_device = self.load(@TypeOf(self.create_device), "vkCreateDevice") orelse return null;
        return self;
    }

    fn load(self: *const Loader, comptime T: type, symbol: [*:0]const u8) ?T {
        const raw = self.get_instance_proc(null, symbol, null) orelse return null;
        return @ptrCast(@alignCast(raw));
    }

    pub fn close(self: *Loader) void {
        self.lib.close();
    }

    /// The API version the loader itself was built against, `0` when it does not
    /// expose `vkEnumerateInstanceVersion` (Vulkan 1.0).
    pub fn instanceVersion(self: *const Loader) u32 {
        const PFn = *const fn (*u32) callconv(VKAPI) i32;
        const f = self.load(PFn, "vkEnumerateInstanceVersion") orelse return 0;
        var v: u32 = 0;
        if (f(&v) != VK_SUCCESS) return 0;
        return v;
    }

    pub fn createInstance(self: *const Loader, api_version: u32) Error!VkInstance {
        const app = ApplicationInfo{ .api_version = api_version };
        const info = InstanceCreateInfo{ .application_info = &app };
        var instance: VkInstance = undefined;
        if (self.create_instance(&info, null, &instance) != VK_SUCCESS) return Error.VulkanFailed;
        return instance;
    }
};

// ----------------------------------------------------------------- the device

/// A ready-to-dispatch Vulkan device: one compute queue and one command pool.
pub const Device = struct {
    loader: Loader,
    instance: VkInstance,
    physical: VkPhysicalDevice,
    handle: VkDevice,
    queue: VkQueue,
    command_pool: VkCommandPool,
    queue_family: u32,
    memory_type: u32,
    props: PhysicalDeviceProperties,

    // device level entry points
    destroy_device: *const fn (VkDevice, ?*const anyopaque) callconv(VKAPI) void,
    get_device_queue: *const fn (VkDevice, u32, u32, *VkQueue) callconv(VKAPI) void,
    device_wait_idle: *const fn (VkDevice) callconv(VKAPI) i32,
    create_buffer: *const fn (VkDevice, *const BufferCreateInfo, ?*const anyopaque, *VkBuffer) callconv(VKAPI) i32,
    destroy_buffer: *const fn (VkDevice, VkBuffer, ?*const anyopaque) callconv(VKAPI) void,
    get_buffer_requirements: *const fn (VkDevice, VkBuffer, *MemoryRequirements) callconv(VKAPI) void,
    bind_buffer_memory: *const fn (VkDevice, VkBuffer, VkDeviceMemory, u64) callconv(VKAPI) i32,
    allocate_memory: *const fn (VkDevice, *const MemoryAllocateInfo, ?*const anyopaque, *VkDeviceMemory) callconv(VKAPI) i32,
    free_memory: *const fn (VkDevice, VkDeviceMemory, ?*const anyopaque) callconv(VKAPI) void,
    map_memory: *const fn (VkDevice, VkDeviceMemory, u64, u64, u32, *?*anyopaque) callconv(VKAPI) i32,
    unmap_memory: *const fn (VkDevice, VkDeviceMemory) callconv(VKAPI) void,
    create_shader_module: *const fn (VkDevice, *const ShaderModuleCreateInfo, ?*const anyopaque, *VkShaderModule) callconv(VKAPI) i32,
    destroy_shader_module: *const fn (VkDevice, VkShaderModule, ?*const anyopaque) callconv(VKAPI) void,
    create_descriptor_set_layout: *const fn (VkDevice, *const DescriptorSetLayoutCreateInfo, ?*const anyopaque, *VkDescriptorSetLayout) callconv(VKAPI) i32,
    destroy_descriptor_set_layout: *const fn (VkDevice, VkDescriptorSetLayout, ?*const anyopaque) callconv(VKAPI) void,
    create_pipeline_layout: *const fn (VkDevice, *const PipelineLayoutCreateInfo, ?*const anyopaque, *VkPipelineLayout) callconv(VKAPI) i32,
    destroy_pipeline_layout: *const fn (VkDevice, VkPipelineLayout, ?*const anyopaque) callconv(VKAPI) void,
    create_compute_pipelines: *const fn (VkDevice, ?VkPipeline, u32, [*]const ComputePipelineCreateInfo, ?*const anyopaque, [*]VkPipeline) callconv(VKAPI) i32,
    destroy_pipeline: *const fn (VkDevice, VkPipeline, ?*const anyopaque) callconv(VKAPI) void,
    create_descriptor_pool: *const fn (VkDevice, *const DescriptorPoolCreateInfo, ?*const anyopaque, *VkDescriptorPool) callconv(VKAPI) i32,
    destroy_descriptor_pool: *const fn (VkDevice, VkDescriptorPool, ?*const anyopaque) callconv(VKAPI) void,
    allocate_descriptor_sets: *const fn (VkDevice, *const DescriptorSetAllocateInfo, [*]VkDescriptorSet) callconv(VKAPI) i32,
    update_descriptor_sets: *const fn (VkDevice, u32, [*]const WriteDescriptorSet, u32, ?*const anyopaque) callconv(VKAPI) void,
    create_command_pool: *const fn (VkDevice, *const CommandPoolCreateInfo, ?*const anyopaque, *VkCommandPool) callconv(VKAPI) i32,
    destroy_command_pool: *const fn (VkDevice, VkCommandPool, ?*const anyopaque) callconv(VKAPI) void,
    allocate_command_buffers: *const fn (VkDevice, *const CommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(VKAPI) i32,
    free_command_buffers: *const fn (VkDevice, VkCommandPool, u32, [*]const VkCommandBuffer) callconv(VKAPI) void,
    begin_command_buffer: *const fn (VkCommandBuffer, *const CommandBufferBeginInfo) callconv(VKAPI) i32,
    end_command_buffer: *const fn (VkCommandBuffer) callconv(VKAPI) i32,
    reset_command_buffer: *const fn (VkCommandBuffer, u32) callconv(VKAPI) i32,
    cmd_bind_pipeline: *const fn (VkCommandBuffer, u32, VkPipeline) callconv(VKAPI) void,
    cmd_bind_descriptor_sets: *const fn (VkCommandBuffer, u32, VkPipelineLayout, u32, u32, [*]const VkDescriptorSet, u32, ?[*]const u32) callconv(VKAPI) void,
    cmd_push_constants: *const fn (VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const anyopaque) callconv(VKAPI) void,
    cmd_dispatch: *const fn (VkCommandBuffer, u32, u32, u32) callconv(VKAPI) void,
    create_fence: *const fn (VkDevice, *const FenceCreateInfo, ?*const anyopaque, *VkFence) callconv(VKAPI) i32,
    destroy_fence: *const fn (VkDevice, VkFence, ?*const anyopaque) callconv(VKAPI) void,
    wait_for_fences: *const fn (VkDevice, u32, [*]const VkFence, u32, u64) callconv(VKAPI) i32,
    queue_submit: *const fn (VkQueue, u32, [*]const SubmitInfo, VkFence) callconv(VKAPI) i32,

    /// Opens the first physical device with a compute queue.
    pub fn init(loader_in: Loader, device_index: ?usize) Error!Device {
        var loader = loader_in;
        errdefer loader.close();
        const api_version = @max(loader.instanceVersion(), 1 << 22); // at least 1.0
        const instance = try loader.createInstance(api_version);
        errdefer loader.destroy_instance(instance, null);

        var count: u32 = 0;
        if (loader.enumerate_physical_devices(instance, &count, null) != VK_SUCCESS or count == 0) {
            return Error.NoDevice;
        }
        var devices: [16]VkPhysicalDevice = undefined;
        var want: u32 = @min(count, devices.len);
        if (loader.enumerate_physical_devices(instance, &want, &devices) != VK_SUCCESS) return Error.NoDevice;

        const pick = device_index orelse 0;
        if (pick >= want) return Error.NoDevice;
        const physical = devices[pick];

        var props = PhysicalDeviceProperties{};
        loader.get_physical_device_properties(physical, &props);

        // a queue family that can compute
        var family_count: u32 = 0;
        loader.get_physical_device_queue_family_properties(physical, &family_count, null);
        if (family_count == 0) return Error.NoComputeQueue;
        var families: [32]QueueFamilyProperties = undefined;
        var fam_want: u32 = @min(family_count, families.len);
        loader.get_physical_device_queue_family_properties(physical, &fam_want, &families);
        var family: ?u32 = null;
        for (families[0..fam_want], 0..) |f, i| {
            if (f.queue_flags & VK_QUEUE_COMPUTE_BIT != 0) {
                family = @intCast(i);
                break;
            }
        }
        const queue_family = family orelse return Error.NoComputeQueue;

        var memory_props = PhysicalDeviceMemoryProperties{};
        loader.get_physical_device_memory_properties(physical, &memory_props);
        var host_type: ?u32 = null;
        for (0..memory_props.memory_type_count) |i| {
            const flags = memory_props.memory_types[i].property_flags;
            if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0) continue;
            if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT == 0) continue;
            if (host_type == null or flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0) {
                host_type = @intCast(i);
            }
        }
        const memory_type = host_type orelse return Error.NoHostVisibleMemory;

        const priority: f32 = 1.0;
        const queue_info = DeviceQueueCreateInfo{
            .queue_family_index = queue_family,
            .queue_priorities = @ptrCast(&priority),
        };
        const dev_info = DeviceCreateInfo{ .queue_create_infos = @ptrCast(&queue_info) };
        var device: VkDevice = undefined;
        if (loader.create_device(physical, &dev_info, null, &device) != VK_SUCCESS) return Error.VulkanFailed;
        errdefer self_destroy(&loader, device);

        var self = Device{
            .loader = loader,
            .instance = instance,
            .physical = physical,
            .handle = device,
            .queue = undefined,
            .command_pool = undefined,
            .queue_family = queue_family,
            .memory_type = memory_type,
            .props = props,

            .destroy_device = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_device), "vkDestroyDevice"),
            .get_device_queue = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).get_device_queue), "vkGetDeviceQueue"),
            .device_wait_idle = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).device_wait_idle), "vkDeviceWaitIdle"),
            .create_buffer = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_buffer), "vkCreateBuffer"),
            .destroy_buffer = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_buffer), "vkDestroyBuffer"),
            .get_buffer_requirements = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).get_buffer_requirements), "vkGetBufferMemoryRequirements"),
            .bind_buffer_memory = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).bind_buffer_memory), "vkBindBufferMemory"),
            .allocate_memory = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).allocate_memory), "vkAllocateMemory"),
            .free_memory = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).free_memory), "vkFreeMemory"),
            .map_memory = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).map_memory), "vkMapMemory"),
            .unmap_memory = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).unmap_memory), "vkUnmapMemory"),
            .create_shader_module = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_shader_module), "vkCreateShaderModule"),
            .destroy_shader_module = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_shader_module), "vkDestroyShaderModule"),
            .create_descriptor_set_layout = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_descriptor_set_layout), "vkCreateDescriptorSetLayout"),
            .destroy_descriptor_set_layout = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_descriptor_set_layout), "vkDestroyDescriptorSetLayout"),
            .create_pipeline_layout = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_pipeline_layout), "vkCreatePipelineLayout"),
            .destroy_pipeline_layout = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_pipeline_layout), "vkDestroyPipelineLayout"),
            .create_compute_pipelines = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_compute_pipelines), "vkCreateComputePipelines"),
            .destroy_pipeline = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_pipeline), "vkDestroyPipeline"),
            .create_descriptor_pool = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_descriptor_pool), "vkCreateDescriptorPool"),
            .destroy_descriptor_pool = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_descriptor_pool), "vkDestroyDescriptorPool"),
            .allocate_descriptor_sets = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).allocate_descriptor_sets), "vkAllocateDescriptorSets"),
            .update_descriptor_sets = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).update_descriptor_sets), "vkUpdateDescriptorSets"),
            .create_command_pool = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_command_pool), "vkCreateCommandPool"),
            .destroy_command_pool = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_command_pool), "vkDestroyCommandPool"),
            .allocate_command_buffers = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).allocate_command_buffers), "vkAllocateCommandBuffers"),
            .free_command_buffers = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).free_command_buffers), "vkFreeCommandBuffers"),
            .begin_command_buffer = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).begin_command_buffer), "vkBeginCommandBuffer"),
            .end_command_buffer = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).end_command_buffer), "vkEndCommandBuffer"),
            .reset_command_buffer = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).reset_command_buffer), "vkResetCommandBuffer"),
            .cmd_bind_pipeline = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).cmd_bind_pipeline), "vkCmdBindPipeline"),
            .cmd_bind_descriptor_sets = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).cmd_bind_descriptor_sets), "vkCmdBindDescriptorSets"),
            .cmd_push_constants = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).cmd_push_constants), "vkCmdPushConstants"),
            .cmd_dispatch = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).cmd_dispatch), "vkCmdDispatch"),
            .create_fence = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).create_fence), "vkCreateFence"),
            .destroy_fence = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).destroy_fence), "vkDestroyFence"),
            .wait_for_fences = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).wait_for_fences), "vkWaitForFences"),
            .queue_submit = try loadDeviceFn(&loader, device, @TypeOf(@as(Device, undefined).queue_submit), "vkQueueSubmit"),
        };
        self.get_device_queue(device, queue_family, 0, &self.queue);

        const pool_info = CommandPoolCreateInfo{ .queue_family_index = queue_family };
        if (self.create_command_pool(device, &pool_info, null, &self.command_pool) != VK_SUCCESS) {
            return Error.VulkanFailed;
        }
        return self;
    }

    /// Opens the loader and the `device_index`-th compute device.
    ///
    /// `null` means "this machine has no usable Vulkan" — no loader, no driver,
    /// or a driver that failed to hand out a device. Only allocation failures
    /// and the impossible cases come back as errors.
    pub fn open(device_index: ?usize) Error!?Device {
        var load_err: ?anyerror = null;
        const loader = Loader.openWith(&load_err) orelse return null;
        return Device.init(loader, device_index) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => null,
        };
    }

    fn self_destroy(loader: *const Loader, device: VkDevice) void {
        const PFn = *const fn (VkDevice, ?*const anyopaque) callconv(VKAPI) void;
        const raw = loader.get_instance_proc(null, "vkDestroyDevice", null) orelse return;
        const f: PFn = @ptrCast(@alignCast(raw));
        f(device, null);
    }

    fn loadDeviceFn(loader: *const Loader, device: VkDevice, comptime T: type, symbol: [*:0]const u8) Error!T {
        const raw = loader.get_instance_proc(device, symbol, null) orelse return Error.VulkanFailed;
        return @ptrCast(@alignCast(raw));
    }

    pub fn name(self: *const Device) []const u8 {
        return self.props.name();
    }

    pub fn deinit(self: *Device) void {
        _ = self.device_wait_idle(self.handle);
        self.destroy_command_pool(self.handle, self.command_pool, null);
        self.destroy_device(self.handle, null);
        self.loader.destroy_instance(self.instance, null);
        self.loader.close();
    }

    /// A device buffer that stays mapped: `data` is the host view of it.
    pub const Allocation = struct {
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        data: []u8,
        size: u64,

        pub fn bytes(self: Allocation) []u8 {
            return self.data;
        }
    };

    /// Allocates `size` bytes in host-visible memory and maps them for the whole
    /// lifetime of the allocation (upload and readback are plain `memcpy`s).
    pub fn allocBuffer(self: *Device, size: usize) Error!Allocation {
        const info = BufferCreateInfo{
            .size = @max(size, 4),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        var buffer: VkBuffer = undefined;
        if (self.create_buffer(self.handle, &info, null, &buffer) != VK_SUCCESS) return Error.VulkanFailed;
        errdefer self.destroy_buffer(self.handle, buffer, null);

        var req = MemoryRequirements{};
        self.get_buffer_requirements(self.handle, buffer, &req);
        const mem_info = MemoryAllocateInfo{
            .allocation_size = req.size,
            .memory_type_index = self.memory_type,
        };
        var memory: VkDeviceMemory = undefined;
        if (self.allocate_memory(self.handle, &mem_info, null, &memory) != VK_SUCCESS) return Error.VulkanFailed;
        errdefer self.free_memory(self.handle, memory, null);

        if (self.bind_buffer_memory(self.handle, buffer, memory, 0) != VK_SUCCESS) return Error.VulkanFailed;

        var mapped: ?*anyopaque = null;
        if (self.map_memory(self.handle, memory, 0, req.size, 0, &mapped) != VK_SUCCESS) return Error.VulkanFailed;
        const ptr: [*]u8 = @ptrCast(mapped orelse return Error.VulkanFailed);
        return .{
            .buffer = buffer,
            .memory = memory,
            .data = ptr[0..size],
            .size = req.size,
        };
    }

    pub fn free(self: *Device, alloc: Allocation) void {
        self.unmap_memory(self.handle, alloc.memory);
        self.free_memory(self.handle, alloc.memory, null);
        self.destroy_buffer(self.handle, alloc.buffer, null);
    }

    /// Uploads host bytes into a device allocation.
    pub fn upload(self: *Device, alloc: Allocation, bytes: []const u8) void {
        _ = self;
        @memcpy(alloc.data[0..@min(bytes.len, alloc.data.len)], bytes[0..@min(bytes.len, alloc.data.len)]);
    }

    /// Copies the device allocation back into host memory.
    pub fn download(self: *Device, alloc: Allocation, out: []u8) void {
        _ = self;
        const n = @min(out.len, alloc.data.len);
        @memcpy(out[0..n], alloc.data[0..n]);
    }

    /// Glue for `zbackend.KernelCtx`: runs `spirv` over the contexts' buffers.
    ///
    /// `bindings[i]` must be a persistent allocation created by `allocBuffer`.
    pub fn dispatchCompute(
        self: *Device,
        spirv: []const u32,
        buffers: []const BufferBinding,
        count: usize,
        threads: u32,
    ) Error!void {
        if (spirv.len < 5) return Error.VulkanFailed;
        const locals = @max(threads, 1);
        const workgroups: u32 = @intCast((count + locals - 1) / locals);
        const binding_count: u32 = @intCast(buffers.len);

        // shader module
        const module_info = ShaderModuleCreateInfo{ .code_size = spirv.len * 4, .code = spirv.ptr };
        var module: VkShaderModule = undefined;
        if (self.create_shader_module(self.handle, &module_info, null, &module) != VK_SUCCESS) {
            return Error.VulkanFailed;
        }
        defer self.destroy_shader_module(self.handle, module, null);

        // descriptor set layout: binding i = buffers[i]
        const layouts = std.heap.page_allocator.alloc(DescriptorSetLayoutBinding, buffers.len) catch return Error.OutOfMemory;
        defer std.heap.page_allocator.free(layouts);
        for (layouts, 0..) |*b, i| b.* = .{ .binding = @intCast(i) };
        const layout_info = DescriptorSetLayoutCreateInfo{ .binding_count = binding_count, .bindings = layouts.ptr };
        var set_layout: VkDescriptorSetLayout = undefined;
        if (self.create_descriptor_set_layout(self.handle, &layout_info, null, &set_layout) != VK_SUCCESS) {
            return Error.VulkanFailed;
        }
        defer self.destroy_descriptor_set_layout(self.handle, set_layout, null);

        const range = PushConstantRange{ .offset = 0, .size = @sizeOf(PushParams) };
        const pipeline_layout_info = PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .set_layouts = @ptrCast(&set_layout),
            .push_constant_range_count = 1,
            .push_constant_ranges = @ptrCast(&range),
        };
        var pipeline_layout: VkPipelineLayout = undefined;
        if (self.create_pipeline_layout(self.handle, &pipeline_layout_info, null, &pipeline_layout) != VK_SUCCESS) {
            return Error.VulkanFailed;
        }
        defer self.destroy_pipeline_layout(self.handle, pipeline_layout, null);

        const stage = PipelineShaderStageCreateInfo{ .module = module };
        const pipe_info = ComputePipelineCreateInfo{ .stage = stage, .layout = pipeline_layout };
        var pipeline: VkPipeline = undefined;
        if (self.create_compute_pipelines(self.handle, null, 1, @ptrCast(&pipe_info), null, @ptrCast(&pipeline)) != VK_SUCCESS) {
            return Error.VulkanFailed;
        }
        defer self.destroy_pipeline(self.handle, pipeline, null);

        const pool_size = DescriptorPoolSize{ .descriptor_count = @max(binding_count, 1) };
        const pool_info = DescriptorPoolCreateInfo{ .pool_sizes = @ptrCast(&pool_size) };
        var pool: VkDescriptorPool = undefined;
        if (self.create_descriptor_pool(self.handle, &pool_info, null, &pool) != VK_SUCCESS) return Error.VulkanFailed;
        defer self.destroy_descriptor_pool(self.handle, pool, null);

        const alloc_info = DescriptorSetAllocateInfo{ .descriptor_pool = pool, .set_layouts = @ptrCast(&set_layout) };
        var set: VkDescriptorSet = undefined;
        if (self.allocate_descriptor_sets(self.handle, &alloc_info, @ptrCast(&set)) != VK_SUCCESS) return Error.VulkanFailed;

        const infos = std.heap.page_allocator.alloc(DescriptorBufferInfo, buffers.len) catch return Error.OutOfMemory;
        defer std.heap.page_allocator.free(infos);
        const writes = std.heap.page_allocator.alloc(WriteDescriptorSet, buffers.len) catch return Error.OutOfMemory;
        defer std.heap.page_allocator.free(writes);
        for (buffers, 0..) |b, i| {
            infos[i] = .{ .buffer = b.buffer, .offset = 0, .range = b.size };
            writes[i] = .{
                .dst_set = set,
                .dst_binding = @intCast(i),
                .buffer_info = @ptrCast(&infos[i]),
            };
        }
        if (buffers.len != 0) self.update_descriptor_sets(self.handle, binding_count, writes.ptr, 0, null);

        // command buffer
        const cb_info = CommandBufferAllocateInfo{ .command_pool = self.command_pool };
        var cmd: VkCommandBuffer = undefined;
        if (self.allocate_command_buffers(self.handle, &cb_info, @ptrCast(&cmd)) != VK_SUCCESS) return Error.VulkanFailed;
        defer self.free_command_buffers(self.handle, self.command_pool, 1, @ptrCast(&cmd));

        const begin = CommandBufferBeginInfo{};
        if (self.begin_command_buffer(cmd, &begin) != VK_SUCCESS) return Error.VulkanFailed;
        self.cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        if (buffers.len != 0) {
            self.cmd_bind_descriptor_sets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
        }
        const params = PushParams{ .count = @intCast(count), .threads = locals };
        self.cmd_push_constants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushParams), @ptrCast(&params));
        self.cmd_dispatch(cmd, workgroups, 1, 1);
        if (self.end_command_buffer(cmd) != VK_SUCCESS) return Error.VulkanFailed;

        const fence_info = FenceCreateInfo{};
        var fence: VkFence = undefined;
        if (self.create_fence(self.handle, &fence_info, null, &fence) != VK_SUCCESS) return Error.VulkanFailed;
        defer self.destroy_fence(self.handle, fence, null);

        const submit = SubmitInfo{ .command_buffers = @ptrCast(&cmd) };
        if (self.queue_submit(self.queue, 1, @ptrCast(&submit), fence) != VK_SUCCESS) return Error.VulkanFailed;
        if (self.wait_for_fences(self.handle, 1, @ptrCast(&fence), 1, std.time.ns_per_s) != VK_SUCCESS) {
            return Error.VulkanFailed;
        }
    }

    /// The description one dispatch needs per bound buffer.
    pub const BufferBinding = struct {
        buffer: VkBuffer,
        size: u64,
    };
};

test "loader names match the host platform" {
    const names = libraryNames();
    try std.testing.expect(names.len != 0);
    const expected = switch (builtin.os.tag) {
        .windows => "vulkan-1.dll",
        .macos => "libvulkan.1.dylib",
        else => "libvulkan.so.1",
    };
    try std.testing.expectEqualStrings(expected, names[0]);
}

test "opening a missing loader fails cleanly" {
    // The sandbox (and CI) usually has no Vulkan loader; that must be a `null`,
    // never a crash.
    const maybe = Loader.open();
    if (maybe) |*l| {
        var loader = l.*;
        _ = loader.instanceVersion();
        loader.close();
    }
}

test "PushParams matches the shader ABI" {
    // std430-compatible: two u32s, then padding to 16 so a WGSL/GLSL
    // `vec4<u32>`-shaped block lines up with it too.
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(PushParams));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(PushParams, "pad"));
}
