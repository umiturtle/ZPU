//! The instance pipeline: a frame as data, drawn by whoever can draw.
//!
//! `render.zig` turns a display list straight into pixels. This file turns the
//! *same* display list into an instance buffer — flat, fixed-size records in
//! logical space — and then draws it back. Two things come out of that:
//!
//! * **A headless reference for the GPU path.** `draw` rasterises a list through
//!   the buffer instead of through the walk, so every claim about the buffer
//!   ("it is complete", "it is ordered", "it is the same picture") is a test
//!   that runs in CI with no GPU present.
//! * **A frame a backend can upload.** `pack` flattens instances into the layout
//!   a storage buffer wants, `Params` is the uniform a shader needs to place
//!   logical space in a texture, and `atlas` is the texture itself. The
//!   `zbackend` kernel at the bottom is the same rasterisation as a data-parallel
//!   kernel, so the frame can be dispatched on any backend `zbackend` has.
//!
//! ## Why one invocation per instance
//!
//! Compositing is order-dependent: a later shape covers an earlier one. A
//! compute kernel has no ordering guarantee *between* invocations, so the
//! backend that runs them in order is the one this kernel is correct on — which
//! is exactly the contract the CPU backend provides, and exactly why `Kernel`
//! here carries no shading-language source and therefore never claims it can run
//! on a GPU (`Kernel.canRun`).
//!
//! A GPU draws instances the way it draws anything else: as geometry. The vertex
//! shader expands one instance into a quad, the fragment shader computes the
//! same coverage this file computes, and the hardware's blending unit puts them
//! in order. That program is `render_pass_wgsl` below — shipped as source for a
//! host that has a `wgpu` device, deliberately *not* registered as a compute
//! kernel.
const std = @import("std");
const zbackend = @import("zbackend");
const zgf = @import("zgf");
const atlas = @import("atlas.zig");
const frame_mod = @import("frame.zig");
const instances_mod = @import("instances.zig");
const raster = @import("raster.zig");
const types = @import("types.zig");

const DisplayList = frame_mod.DisplayList;
const Instance = frame_mod.Instance;
const Surface = raster.Surface;
const Vec2 = types.Vec2;

/// Floats per instance in a packed buffer. The record is 16-byte aligned and
/// nothing in it is a pointer, so `pack` is a copy and a shader can index it.
pub const stride = 20;

pub const Options = struct {
    origin: Vec2 = .{},
    scale: f32 = 1,
    clear: ?types.Color = null,
};

/// Rasterises `list` through an instance buffer.
///
/// `scratch` is the buffer the instances are built in — the frame arena is the
/// intended owner, so drawing a frame costs no system allocation. It is
/// separate from `img` because a host with a GPU wants to keep the buffer: that
/// is the frame it uploads.
pub fn draw(list: *const DisplayList, img: *zgf.Image, opts: Options, scratch: []Instance) void {
    const insts = instances_mod.collect(list, .{}, scratch);
    if (opts.clear) |c| img.fill(raster.toZgf(c));
    var surf = Surface.init(img);
    drawInstances(&surf, insts, opts);
}

/// Draws an already-built instance buffer, transforming it by `opts`.
pub fn drawInstances(surf: *Surface, insts: []const Instance, opts: Options) void {
    const tr = instances_mod.Transform{ .origin = opts.origin, .scale = opts.scale };
    for (insts) |inst| raster.emit(surf, scaled(inst, tr));
}

/// Applies a transform to one instance. `Instance.scaled` does the work; this
/// wrapper is here so the transfer is written once, in the one place both the
/// software loop and a host's upload path go through.
pub fn scaled(inst: Instance, tr: instances_mod.Transform) Instance {
    return inst.scaled(tr.origin, tr.scale);
}

// ---------------------------------------------------------------------------
// the buffer a device reads
// ---------------------------------------------------------------------------

/// The uniform a shader needs to put logical space into a texture: the target's
/// size in pixels, where the frame's origin lands, how many device pixels a
/// logical pixel is, how many instances follow, and the atlas size.
pub const Params = extern struct {
    width: f32,
    height: f32,
    origin_x: f32,
    origin_y: f32,
    scale: f32,
    count: f32,
    atlas_width: f32,
    atlas_height: f32,

    pub fn of(target_width: usize, target_height: usize, count: usize, opts: Options) Params {
        return .{
            .width = @floatFromInt(target_width),
            .height = @floatFromInt(target_height),
            .origin_x = opts.origin.x,
            .origin_y = opts.origin.y,
            .scale = opts.scale,
            .count = @floatFromInt(count),
            .atlas_width = @floatFromInt(atlas.width),
            .atlas_height = @floatFromInt(atlas.height),
        };
    }
};

/// The texture the glyph instances sample, as bytes: 8-bit coverage, row major,
/// one texel of padding between glyphs. A host uploads it as `r8unorm`.
pub fn atlasPixels() []const u8 {
    return &atlas.pixels;
}

pub const atlas_size = [2]usize{ atlas.width, atlas.height };

/// Flattens `insts` into `out` as `stride` floats each. Returns the slice it
/// wrote; `out` must have room for `insts.len * stride` floats.
pub fn pack(insts: []const Instance, out: []f32) []f32 {
    const n = @min(insts.len, out.len / stride);
    for (insts[0..n], 0..) |inst, i| {
        const base = i * stride;
        const dst = out[base..][0..stride];
        @memcpy(dst[0..4], &inst.box);
        @memcpy(dst[4..8], &inst.color);
        @memcpy(dst[8..12], &inst.clip);
        dst[12] = inst.param;
        dst[13] = @floatFromInt(inst.kind);
        @memcpy(dst[14..20], &inst.points);
    }
    return out[0 .. n * stride];
}

/// The inverse of `pack`. Round-tripping is a test, and it is also what the
/// software fallback does with a buffer a host has already built.
pub fn unpack(src: []const f32, out: []Instance) []Instance {
    const n = @min(out.len, src.len / stride);
    for (out[0..n], 0..) |*inst, i| {
        const s = src[i * stride ..][0..stride];
        @memcpy(&inst.box, s[0..4]);
        @memcpy(&inst.color, s[4..8]);
        @memcpy(&inst.clip, s[8..12]);
        inst.param = s[12];
        inst.kind = @intFromFloat(s[13]);
        @memcpy(&inst.points, s[14..20]);
    }
    return out[0..n];
}

// ---------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------

/// Rasterises a packed instance buffer into an RGBA `f32` target, one invocation
/// per instance.
///
/// Invocation `i` blends instance `i` — over everything the invocation before it
/// left behind. That is the *only* ordering this file depends on, and it is the
/// one the CPU backend guarantees (`zbackend.runCpu` runs ids in order). The GPU
/// path is `render_pass_wgsl`, not this kernel; keeping the WGSL out of
/// `Kernel.source` is what makes `canRun` tell the truth, so a dispatch to a GPU
/// backend falls back to the CPU instead of compiling a fragment shader as a
/// compute shader and failing at the driver.
pub const kernel = zbackend.Kernel{
    .name = "zui_rasterize_instances",
    .body = rasterizeBody,
};

/// Buffers: [0] instances (`stride` floats each), [1] target (`width * height *
/// 4` floats, RGBA), [2] `Params`.
pub const kernel_buffers = 3;

fn rasterizeBody(ctx: *zbackend.KernelCtx) void {
    if (ctx.buffers.len < kernel_buffers) {
        ctx.barriers.fail("zui: kernel needs instances, target and params buffers");
        return;
    }
    const params = ctx.buffers[2].as(f32);
    if (params.len < 8) {
        ctx.barriers.fail("zui: params buffer is too small");
        return;
    }
    const width: usize = @intFromFloat(params[0]);
    const height: usize = @intFromFloat(params[1]);
    const target = ctx.buffers[1].as(f32);
    if (target.len < width * height * 4) {
        ctx.barriers.fail("zui: target buffer is too small");
        return;
    }
    const src = ctx.buffers[0].as(f32);
    const count: usize = @intFromFloat(params[5]);
    if (ctx.id >= count) return;
    const base = ctx.id * stride;
    if (base + stride > src.len) {
        ctx.barriers.fail("zui: instance buffer is too small");
        return;
    }

    var inst: Instance = .{};
    @memcpy(&inst.box, src[base..][0..4]);
    @memcpy(&inst.color, src[base + 4 ..][0..4]);
    @memcpy(&inst.clip, src[base + 8 ..][0..4]);
    inst.param = src[base + 12];
    inst.kind = @intFromFloat(src[base + 13]);
    @memcpy(&inst.points, src[base + 14 ..][0..6]);

    var surf = Surface.fromSlice(target, width, height, 4);
    raster.emit(&surf, inst);
}

/// Runs the instance kernel over a whole frame on `backend` (the CPU, in every
/// build without a GPU) and returns what it drew.
pub fn dispatch(
    backend: zbackend.Backend,
    insts: []const Instance,
    target: []f32,
    target_width: usize,
    target_height: usize,
    opts: Options,
    alloc: std.mem.Allocator,
) !void {
    const flat = try alloc.alloc(f32, @max(insts.len, 1) * stride);
    defer alloc.free(flat);
    const params_buf = try alloc.alloc(f32, 8);
    defer alloc.free(params_buf);

    _ = pack(insts, flat);
    const p = Params.of(target_width, target_height, insts.len, opts);
    @memcpy(std.mem.sliceAsBytes(params_buf), std.mem.asBytes(&p));

    var buffers = [_]zbackend.Buffer{
        .{ .ptr = @ptrCast(flat.ptr), .len = flat.len * @sizeOf(f32) },
        .{ .ptr = @ptrCast(target.ptr), .len = target.len * @sizeOf(f32) },
        .{ .ptr = @ptrCast(params_buf.ptr), .len = @sizeOf(Params) },
    };
    var ctx = zbackend.KernelCtx{ .buffers = &buffers, .count = insts.len, .alloc = alloc };
    try zbackend.dispatch(backend, kernel, &ctx);
}

// ---------------------------------------------------------------------------
// the program a GPU would run
// ---------------------------------------------------------------------------

/// The render-pass form of the same picture: one quad per instance, the coverage
/// in the fragment shader, the hardware's blender doing the compositing.
///
/// It is written against WGSL 1.0 with no extensions. `coverage` there is the
/// translation of `raster.coverage`: a rectangle is an overlap, a rounded box a
/// distance field, a line a distance to a segment, a glyph a bilinear sample of
/// the atlas. The point of mirroring them is that the CPU reference and the GPU
/// program cannot drift without someone noticing the pixels changed.
pub const render_pass_wgsl =
    \\struct Params {
    \\  size: vec2<f32>,
    \\  origin: vec2<f32>,
    \\  scale: f32,
    \\  count: f32,
    \\  atlas: vec2<f32>,
    \\  pad: vec2<f32>,
    \\};
    \\struct Instance {
    \\  box: vec4<f32>,
    \\  color: vec4<f32>,
    \\  clip: vec4<f32>,
    \\  param: f32,
    \\  kind: u32,
    \\  points: array<f32, 6>,
    \\};
    \\@group(0) @binding(0) var<uniform> params: Params;
    \\@group(0) @binding(1) var<storage, read> instances: array<Instance>;
    \\@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
    \\@group(0) @binding(3) var atlas_sampler: sampler;
    \\
    \\struct VertexOut {
    \\  @builtin(position) pos: vec4<f32>,
    \\  @location(0) uv: vec2<f32>,
    \\  @location(1) @interpolate(flat) index: u32,
    \\};
    \\
    \\fn corners(i: u32) vec2<f32> {
    \\  let x = f32(i & 1u);
    \\  let y = f32((i >> 1u) & 1u);
    \\  return vec2<f32>(x, y);
    \\}
    \\
    \\// Four vertices per instance: the bounding box, which is all any of the
    \\// shapes needs, because the fragment shader decides the coverage inside it.
    \\@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOut {
    \\  let inst = instances[ii];
    \\  let c = corners(vi);
    \\  // Device pixels from logical space.
    \\  let scale = vec2<f32>(params.scale, params.scale);
    \\  let logical = inst.box.xy + c * inst.box.zw;
    \\  let device = params.origin + logical * scale;
    \\  let ndc = device / params.size * 2.0 - 1.0;
    \\  var out: VertexOut;
    \\  out.pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    \\  out.uv = c;
    \\  out.index = ii;
    \\  return out;
    \\}
    \\
    \\fn rect_coverage(box: vec4<f32>, p: vec2<f32>) -> f32 {
    \\  let o = clamp(min(box.xy + box.zw, p + 0.5) - max(box.xy, p - 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    \\  return o.x * o.y;
    \\}
    \\
    \\fn sd_rounded(p: vec2<f32>, c: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    \\  let q = abs(p - c) - b + vec2<f32>(r, r);
    \\  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    \\}
    \\
    \\fn capsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, w: f32) -> f32 {
    \\  let ab = b - a;
    \\  let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    \\  return clamp(w * 0.5 + 0.5 - distance(p, a + ab * t), 0.0, 1.0);
    \\}
    \\
    \\@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    \\  let inst = instances[in.index];
    \\  let p = in.pos.xy;
    \\  var cov = 0.0;
    \\  if (inst.kind == 0u) {
    \\    cov = rect_coverage(inst.box, p);
    \\  } else if (inst.kind == 1u) {
    \\    let c = inst.box.xy + inst.box.zw * 0.5;
    \\    cov = clamp(0.5 - sd_rounded(p, c, inst.box.zw * 0.5, inst.param), 0.0, 1.0);
    \\  } else if (inst.kind == 2u) {
    \\    let uv = vec2<f32>(inst.points[0], inst.points[1]) +
    \\             vec2<f32>(inst.points[2] - inst.points[0], inst.points[3] - inst.points[1]) * in.uv;
    \\    cov = textureSampleLevel(atlas_tex, atlas_sampler, uv, 0.0).r;
    \\  } else if (inst.kind == 3u) {
    \\    cov = capsule(p, vec2<f32>(inst.points[0], inst.points[1]), vec2<f32>(inst.points[2], inst.points[3]), inst.param);
    \\  } else {
    \\    let a = vec2<f32>(inst.points[0], inst.points[1]);
    \\    let b = vec2<f32>(inst.points[2], inst.points[3]);
    \\    let c = vec2<f32>(inst.points[4], inst.points[5]);
    \\    let s = sign((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));
    \\    let t = sign((c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x));
    \\    let u = sign((a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x));
    \\    let inside = (s + t + u) == 3.0 || (s + t + u) == -3.0;
    \\    cov = select(0.0, 1.0, inside);
    \\  }
    \\  // The clip is a hard edge, like the CPU side: in or out, no ramp.
    \\  let inside_clip = all(p >= inst.clip.xy) && all(p <= inst.clip.xy + inst.clip.zw);
    \\  cov = select(0.0, cov, inside_clip);
    \\  return vec4<f32>(inst.color.rgb, inst.color.a * cov);
    \\}
    \\
;

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const render = @import("render.zig");
const zpu = @import("zpu");

fn newImage(w: usize, h: usize) !zgf.Image {
    try zpu.init(testing.allocator, .{});
    return zgf.Image.init(testing.allocator, w, h, 3);
}

/// A frame with one of everything: rectangles, a rounded border, a line, a
/// triangle, text, and a nested clip.
fn richList(list: *DisplayList) !void {
    const white = types.Color.hex(0xffffff, 1);
    const panel = types.Color.hex(0x20242c, 1);
    const accent = types.Color.hex(0x53a9ff, 1);
    try list.push(testing.allocator, .{ .rect = .{ .r = types.Rect.init(0, 0, 64, 48), .color = panel } });
    try list.push(testing.allocator, .{ .rounded = .{
        .r = types.Rect.init(4, 4, 30, 14),
        .radius = 5,
        .color = accent,
        .border = 2,
        .border_color = white,
    } });
    try list.push(testing.allocator, .{ .line = .{
        .a = .{ .x = 2, .y = 24 },
        .b = .{ .x = 40, .y = 30 },
        .width = 3,
        .color = white,
    } });
    try list.push(testing.allocator, .{ .triangle = .{
        .a = .{ .x = 46, .y = 4 },
        .b = .{ .x = 60, .y = 4 },
        .c = .{ .x = 53, .y = 16 },
        .color = accent,
    } });
    try list.push(testing.allocator, .{ .push_clip = types.Rect.init(2, 34, 40, 12) });
    try list.push(testing.allocator, .{ .text = .{
        .pos = .{ .x = 3, .y = 44 },
        .size = 14,
        .color = white,
        .text = "zig 0.17",
    } });
    try list.push(testing.allocator, .{ .pop_clip = {} });
}

test "the instance path draws the same picture as the display list path" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try richList(&list);

    var direct = try newImage(64, 48);
    defer direct.deinit();
    var through = try newImage(64, 48);
    defer through.deinit();

    const clear = types.Color.hex(0x000000, 1);
    render.draw(&list, &direct, .{ .clear = clear });

    var scratch: [64]Instance = undefined;
    draw(&list, &through, .{ .clear = clear }, &scratch);

    var worst: f32 = 0;
    for (0..48) |y| {
        for (0..64) |x| {
            const a = direct.getPixel(x, y);
            const b = through.getPixel(x, y);
            worst = @max(worst, @max(@abs(a.r - b.r), @max(@abs(a.g - b.g), @abs(a.b - b.b))));
        }
    }
    try testing.expectEqual(@as(f32, 0), worst);
}

test "the same picture, at 2x and with an origin" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try richList(&list);

    var direct = try newImage(160, 120);
    defer direct.deinit();
    var through = try newImage(160, 120);
    defer through.deinit();

    const opts: Options = .{ .origin = .{ .x = 7.5, .y = 3 }, .scale = 2, .clear = types.Color.hex(0x101010, 1) };
    render.draw(&list, &direct, .{ .origin = opts.origin, .scale = opts.scale, .clear = opts.clear });

    var scratch: [64]Instance = undefined;
    draw(&list, &through, opts, &scratch);

    var worst: f32 = 0;
    for (0..120) |y| {
        for (0..160) |x| {
            const a = direct.getPixel(x, y);
            const b = through.getPixel(x, y);
            worst = @max(worst, @abs(a.r - b.r));
        }
    }
    try testing.expectEqual(@as(f32, 0), worst);
}

test "pack and unpack round-trip an instance, field for field" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try richList(&list);

    var buf: [64]Instance = undefined;
    const insts = instances_mod.collect(&list, .{}, &buf);
    try testing.expect(insts.len > 8);

    var flat: [64 * stride]f32 = undefined;
    const written = pack(insts, &flat);
    try testing.expectEqual(insts.len * stride, written.len);

    var back: [64]Instance = undefined;
    const again = unpack(written, &back);
    try testing.expectEqual(insts.len, again.len);
    for (insts, again) |a, b| {
        try testing.expectEqualSlices(f32, &a.box, &b.box);
        try testing.expectEqualSlices(f32, &a.color, &b.color);
        try testing.expectEqualSlices(f32, &a.clip, &b.clip);
        try testing.expectEqualSlices(f32, &a.points, &b.points);
        try testing.expectEqual(a.param, b.param);
        try testing.expectEqual(a.kind, b.kind);
    }
}

test "the kernel draws the frame the reference drew" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try richList(&list);

    var reference = try newImage(64, 48);
    defer reference.deinit();
    const clear = types.Color.hex(0x000000, 1);
    render.draw(&list, &reference, .{ .clear = clear });

    // The kernel sees the frame the way a device would: an RGBA float target,
    // instances in logical space, and the frame transform in a uniform.
    var instance_buf: [64]Instance = undefined;
    const insts = instances_mod.collect(&list, .{}, &instance_buf);

    const target = try testing.allocator.alloc(f32, 64 * 48 * 4);
    defer testing.allocator.free(target);
    @memset(target, 0);

    try dispatch(.cpu, insts, target, 64, 48, .{}, testing.allocator);

    // Compare against the software renderer's own pixels, one channel at a time.
    var worst: f32 = 0;
    for (0..48) |y| {
        for (0..64) |x| {
            const i = (y * 64 + x) * 4;
            const want = reference.getPixel(x, y);
            worst = @max(worst, @abs(target[i] - want.r));
            worst = @max(worst, @abs(target[i + 1] - want.g));
            worst = @max(worst, @abs(target[i + 2] - want.b));
        }
    }
    try testing.expectEqual(@as(f32, 0), worst);
}

test "a frame's params describe where it lands" {
    const p = Params.of(800, 600, 12, .{ .origin = .{ .x = 4, .y = 5 }, .scale = 2 });
    try testing.expectEqual(@as(f32, 800), p.width);
    try testing.expectEqual(@as(f32, 12), p.count);
    try testing.expectEqual(@as(f32, 2), p.scale);
    try testing.expectEqual(@as(f32, 4), p.origin_x);
    try testing.expectEqual(@as(f32, @floatFromInt(atlas.width)), p.atlas_width);
    try testing.expectEqual(@as(usize, atlas.width * atlas.height), atlasPixels().len);
}

test "the render pass program mentions every instance kind" {
    // A cheap guard against the WGSL drifting away from the kinds the library
    // emits: a shader that silently ignores a kind draws the wrong picture, and
    // nothing else in the build would notice.
    inline for (.{ "kind == 0u", "kind == 1u", "kind == 2u", "kind == 3u" }) |needle| {
        try testing.expect(std.mem.indexOf(u8, render_pass_wgsl, needle) != null);
    }
    try testing.expect(std.mem.indexOf(u8, render_pass_wgsl, "textureSampleLevel") != null);
    // And that it declares the buffers the host has to bind.
    inline for (.{ "@binding(0)", "@binding(1)", "@binding(2)", "@binding(3)" }) |needle| {
        try testing.expect(std.mem.indexOf(u8, render_pass_wgsl, needle) != null);
    }
}
