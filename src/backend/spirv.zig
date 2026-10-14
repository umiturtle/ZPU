//! SPIR-V bytecode for the kernels this repository ships as *verified*
//! artifacts.
//!
//! Vulkan executes SPIR-V, and ZPU has no shader compiler inside it (that is a
//! boundary the project keeps on purpose: the build never runs an external
//! tool). So the bytecode below is checked in, with the exact source it came
//! from and the exact command that produced it:
//!
//! ```sh
//! # glslang 0.0.15 (WASM build), from the GLSL in `scale_offset_glsl`
//! node -e "require('@webgpu/glslang')() \
//!   .then(m => console.log(m.compileGLSL(src, 'compute', false, '1.1')))"
//! ```
//!
//! `comptime` checks the header (magic number, version, length) so a corrupted
//! or truncated blob fails the build instead of a driver.
const std = @import("std");

/// The GLSL the bytecode below was compiled from — the same computation as the
/// WGSL and as `Kernel.body` in the examples and tests.
pub const scale_offset_glsl =
    \\#version 450
    \\layout(local_size_x = 64) in;
    \\layout(std430, binding = 0) buffer Data { float data[]; };
    \\layout(std430, binding = 1) readonly buffer Scale { float scale[]; };
    \\void main() {
    \\    uint i = gl_GlobalInvocationID.x;
    \\    data[i] = data[i] * scale[0] + float(i);
    \\}
;

/// The workgroup size the module declares itself (`OpExecutionMode LocalSize`),
/// or `null` when it does not say.
///
/// The size is baked into the code, so a caller that passes a different
/// `threads` value would silently leave invocations unrun. Reading it back lets
/// the dispatcher use what the shader actually has.
pub fn localSize(code: []const u32) ?[3]u32 {
    if (code.len < 6) return null;
    var i: usize = 5; // past magic, version, generator, bound, schema
    while (i < code.len) {
        const word_count: usize = code[i] >> 16;
        const opcode = code[i] & 0xffff;
        if (word_count == 0) return null;
        // 16 = OpExecutionMode, 17 = LocalSize, then three literal sizes.
        if (opcode == 16 and word_count >= 6 and code[i + 2] == 17) {
            return .{ code[i + 3], code[i + 4], code[i + 5] };
        }
        i += word_count;
    }
    return null;
}

/// The same kernel in WGSL, for the WebGPU backend. The entry point is `main`
/// on every backend (that is the name `zbackend.entry_point` passes to the
/// driver), and only `{threads}` is substituted.
pub const scale_offset_wgsl =
    \\@group(0) @binding(0) var<storage, read_write> data: array<f32>;
    \\@group(0) @binding(1) var<storage, read> scale: array<f32>;
    \\@compute @workgroup_size({threads})
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\    let i = gid.x;
    \\    data[i] = data[i] * scale[0] + f32(i);
    \\}
;

/// `data[i] = data[i] * scale[0] + float(i)` over 64-invocation workgroups.
pub const scale_offset = [_]u32{
    0x07230203, 0x00010100, 0x00080008, 0x0000002a, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000012,
    0x61746144, 0x00000000, 0x00050006, 0x00000012, 0x00000000, 0x61746164, 0x00000000, 0x00030005,
    0x00000014, 0x00000000, 0x00040005, 0x0000001d, 0x6c616353, 0x00000065, 0x00050006, 0x0000001d,
    0x00000000, 0x6c616373, 0x00000065, 0x00030005, 0x0000001f, 0x00000000, 0x00040047, 0x0000000b,
    0x0000000b, 0x0000001c, 0x00040047, 0x00000011, 0x00000006, 0x00000004, 0x00050048, 0x00000012,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000012, 0x00000003, 0x00040047, 0x00000014,
    0x00000022, 0x00000000, 0x00040047, 0x00000014, 0x00000021, 0x00000000, 0x00040047, 0x0000001c,
    0x00000006, 0x00000004, 0x00040048, 0x0000001d, 0x00000000, 0x00000018, 0x00050048, 0x0000001d,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000001d, 0x00000003, 0x00040047, 0x0000001f,
    0x00000022, 0x00000000, 0x00040047, 0x0000001f, 0x00000021, 0x00000001, 0x00040047, 0x00000029,
    0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
    0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017,
    0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b,
    0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020,
    0x0000000d, 0x00000001, 0x00000006, 0x00030016, 0x00000010, 0x00000020, 0x0003001d, 0x00000011,
    0x00000010, 0x0003001e, 0x00000012, 0x00000011, 0x00040020, 0x00000013, 0x00000002, 0x00000012,
    0x0004003b, 0x00000013, 0x00000014, 0x00000002, 0x00040015, 0x00000015, 0x00000020, 0x00000001,
    0x0004002b, 0x00000015, 0x00000016, 0x00000000, 0x00040020, 0x00000019, 0x00000002, 0x00000010,
    0x0003001d, 0x0000001c, 0x00000010, 0x0003001e, 0x0000001d, 0x0000001c, 0x00040020, 0x0000001e,
    0x00000002, 0x0000001d, 0x0004003b, 0x0000001e, 0x0000001f, 0x00000002, 0x0004002b, 0x00000006,
    0x00000027, 0x00000040, 0x0004002b, 0x00000006, 0x00000028, 0x00000001, 0x0006002c, 0x00000009,
    0x00000029, 0x00000027, 0x00000028, 0x00000028, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041,
    0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e,
    0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000017, 0x00000008, 0x0004003d,
    0x00000006, 0x00000018, 0x00000008, 0x00060041, 0x00000019, 0x0000001a, 0x00000014, 0x00000016,
    0x00000018, 0x0004003d, 0x00000010, 0x0000001b, 0x0000001a, 0x00060041, 0x00000019, 0x00000020,
    0x0000001f, 0x00000016, 0x00000016, 0x0004003d, 0x00000010, 0x00000021, 0x00000020, 0x00050085,
    0x00000010, 0x00000022, 0x0000001b, 0x00000021, 0x0004003d, 0x00000006, 0x00000023, 0x00000008,
    0x00040070, 0x00000010, 0x00000024, 0x00000023, 0x00050081, 0x00000010, 0x00000025, 0x00000022,
    0x00000024, 0x00060041, 0x00000019, 0x00000026, 0x00000014, 0x00000016, 0x00000017, 0x0003003e,
    0x00000026, 0x00000025, 0x000100fd, 0x00010038,
};

comptime {
    // The SPIR-V header: magic, version, generator, bound (the id space), and a
    // reserved word that must be zero. A truncated or corrupted paste fails the
    // build here instead of confusing a driver.
    if (scale_offset[0] != 0x0723_0203) @compileError("scale_offset.spirv: bad magic number");
    // SPIR-V 1.1: major in bits 16..23, minor in bits 8..15.
    if (scale_offset[1] != 0x0001_0100) @compileError("scale_offset.spirv: expected SPIR-V 1.1");
    if (scale_offset[2] >> 16 == 0) @compileError("scale_offset.spirv: no generator recorded");
    if (scale_offset[3] == 0) @compileError("scale_offset.spirv: bound must be positive");
    if (scale_offset[4] != 0) @compileError("scale_offset.spirv: reserved schema word must be zero");
    if (scale_offset.len % 4 != 0) @compileError("scale_offset.spirv: a module is a multiple of four words");
}

test "the shipped bytecode has a valid SPIR-V header" {
    try std.testing.expectEqual(@as(u32, 0x0723_0203), scale_offset[0]);
    try std.testing.expectEqual(@as(u32, 0x0001_0100), scale_offset[1]);
    // glslang's tool id (8) in the high half of the generator word.
    try std.testing.expectEqual(@as(u32, 8), scale_offset[2] >> 16);
    try std.testing.expect(scale_offset[3] > 1); // id bound
    try std.testing.expectEqual(@as(u32, 0), scale_offset[4]);
    try std.testing.expect(scale_offset.len > 64);
}

test "the sources describe the same computation" {
    // Cheap consistency guard between the three spellings of the kernel.
    try std.testing.expect(std.mem.indexOf(u8, scale_offset_glsl, "local_size_x = 64") != null);
    try std.testing.expect(std.mem.indexOf(u8, scale_offset_wgsl, "workgroup_size({threads})") != null);
    try std.testing.expect(std.mem.indexOf(u8, scale_offset_wgsl, "fn main(") != null);
    try std.testing.expect(std.mem.indexOf(u8, scale_offset_glsl, "void main()") != null);

    // The dispatcher asks the module what workgroup size it was built with
    // instead of trusting the number in the kernel struct.
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, localSize(&scale_offset).?);
    try std.testing.expectEqual(@as(?[3]u32, null), localSize(&.{ 0x07230203, 0x00010000, 0, 1, 0 }));
    // A malformed word count must not walk off the end.
    try std.testing.expectEqual(@as(?[3]u32, null), localSize(&.{ 0x07230203, 0x00010000, 0, 1, 0, 0xffff0000 }));
    try std.testing.expect(std.mem.indexOf(u8, scale_offset_glsl, "scale[0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, scale_offset_wgsl, "scale[0]") != null);
}
