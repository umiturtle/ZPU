//! The coverage maths. One function per shape, one shape per `Instance`.
//!
//! A frame reaches the screen as a **stream of instances** (`zui/frame.zig`),
//! and this file answers exactly one question about them: *how much of this
//! pixel does this instance cover?* — `coverage(inst, x, y)`, for a pixel centre
//! `(x, y)`, which is the same question a fragment shader is asked. Everything
//! here is therefore written in the shape a shader wants: no allocations, no
//! state, no clip stack, no spans — one pure function per kind.
//!
//! The two render paths in this library are then *thin*:
//!
//! * `render.zig` walks a display list, converts each command to instances in
//!   device space, and for each instance loops over the pixels its bounding box
//!   touches, calling `coverage` and compositing.
//! * `instances.zig` walks the same display list into instances in *logical*
//!   space, for upload to a backend — and `pipeline.zig` draws those through the
//!   very same loop (`emit`), which is what makes the uploaded frame and the
//!   directly drawn frame the same picture rather than two implementations that
//!   hope to agree.
//!
//! The shapes are all analytic: a rectangle is an area overlap, a rounded box a
//! signed distance field, a line a distance to a segment, a glyph a bilinear
//! sample of the atlas. Triangles are the exception — small, rare, and easy to
//! get wrong — so they are supersampled 4×4 inside the pixel.
const std = @import("std");
const zgf = @import("zgf");
const atlas = @import("atlas.zig");
const frame_mod = @import("frame.zig");
const types = @import("types.zig");

const Color = types.Color;
const Instance = frame_mod.Instance;
const Rect = types.Rect;
const Vec2 = types.Vec2;

pub fn toZgf(c: Color) zgf.Color {
    return .{ .r = c.r, .g = c.g, .b = c.b, .a = c.a };
}

/// The pixels a rasteriser writes into.
///
/// A `zgf.Image` is a `zpu.Tensor`, and going through the generic element
/// accessors costs a dtype dispatch per *component* — measured at ~80 ns per
/// write on this machine, which for a 320×200 frame is 15 ms before any shape is
/// drawn. A tensor's typed slice is already public API, so when the image is
/// `f32` and contiguous (what an image is in practice) the rasteriser takes the
/// slice once per frame and blends straight into it; the generic path stays as
/// the fallback, so a `f64` or non-contiguous image still renders correctly.
pub const Surface = struct {
    img: ?*zgf.Image,
    fast: ?[]f32,
    width: usize,
    height: usize,
    channels: usize,

    pub fn init(img: *zgf.Image) Surface {
        return .{
            .img = img,
            .fast = img.pixels.flatSliceMut(f32),
            .width = img.width(),
            .height = img.height(),
            .channels = img.channels(),
        };
    }

    /// A surface over a raw component buffer — the form a backend hands a
    /// kernel: no tensor, no dtype dispatch, just `width * height * channels`
    /// floats. This is the same code the software renderer runs, which is the
    /// point: a frame drawn on a device and a frame drawn here are one
    /// implementation.
    pub fn fromSlice(buf: []f32, width: usize, height: usize, channels: usize) Surface {
        return .{
            .img = null,
            .fast = buf[0..@min(buf.len, width * height * channels)],
            .width = width,
            .height = height,
            .channels = channels,
        };
    }

    pub fn inBounds(self: Surface, x: i64, y: i64) bool {
        return x >= 0 and y >= 0 and x < @as(i64, @intCast(self.width)) and y < @as(i64, @intCast(self.height));
    }

    /// Alpha-composites `src` over one pixel, with `coverage` scaling the
    /// source's alpha (see `zgf.Image.blendCoverage` for the contract).
    pub fn blend(self: Surface, x: i64, y: i64, src: Color, cov_in: f32) void {
        if (!self.inBounds(x, y)) return;
        const cov = std.math.clamp(cov_in, 0, 1);
        const a = src.a * cov;
        if (a <= 0) return;
        // A fully covered opaque pixel is a store, not a blend: a window
        // background is most of a frame's pixels, and skipping the read half of
        // the read-modify-write is worth it.
        if (a >= 1) return self.store(x, y, src);
        const buf = self.fast orelse {
            if (self.img) |im| im.blendCoverage(@intCast(x), @intCast(y), toZgf(src), cov);
            return;
        };
        const i = self.base(x, y);
        switch (self.channels) {
            1 => {
                const v = 0.299 * src.r + 0.587 * src.g + 0.114 * src.b;
                buf[i] += (v - buf[i]) * a;
            },
            3, 4 => {
                buf[i] += (src.r - buf[i]) * a;
                buf[i + 1] += (src.g - buf[i + 1]) * a;
                buf[i + 2] += (src.b - buf[i + 2]) * a;
                if (self.channels == 4) buf[i + 3] = @max(buf[i + 3], a + buf[i + 3] * (1 - a));
            },
            else => unreachable,
        }
    }

    /// Writes a pixel that needs no compositing.
    pub fn store(self: Surface, x: i64, y: i64, src: Color) void {
        const buf = self.fast orelse {
            if (self.img) |im| im.setPixel(@intCast(x), @intCast(y), toZgf(src));
            return;
        };
        const i = self.base(x, y);
        switch (self.channels) {
            1 => buf[i] = 0.299 * src.r + 0.587 * src.g + 0.114 * src.b,
            3, 4 => {
                buf[i] = src.r;
                buf[i + 1] = src.g;
                buf[i + 2] = src.b;
                if (self.channels == 4) buf[i + 3] = src.a;
            },
            else => unreachable,
        }
    }

    pub fn base(self: Surface, x: i64, y: i64) usize {
        return (@as(usize, @intCast(y)) * self.width + @as(usize, @intCast(x))) * self.channels;
    }
};

// ---------------------------------------------------------------------------
// coverage, one kind at a time
// ---------------------------------------------------------------------------

/// The pixels a shape can affect: its bounding box, padded by a pixel and
/// clipped to the instance's clip rectangle and to the image.
///
/// Clamping here is what makes a huge rectangle cost its visible pixels and
/// nothing more. The clip is deliberately *hard* — a pixel is in or out, with no
/// antialiased edge — because every clip this library pushes comes from a widget
/// box on an integer boundary, and a hard edge is one comparison instead of a
/// second coverage evaluation. Both render paths share this rule, so a clip can
/// never make them disagree.
pub const Span = struct { x0: i64, x1: i64, y0: i64, y1: i64 };

pub fn spanOf(surf: *const Surface, clip: Rect, bounds: Rect) ?Span {
    const vis = bounds.outset(1).intersect(clip);
    if (vis.w <= 0 or vis.h <= 0) return null;
    const x0 = @as(i64, @intFromFloat(@floor(vis.x)));
    const x1 = @as(i64, @intFromFloat(@ceil(vis.x + vis.w)));
    const y0 = @as(i64, @intFromFloat(@floor(vis.y)));
    const y1 = @as(i64, @intFromFloat(@ceil(vis.y + vis.h)));
    const w: i64 = @intCast(surf.width);
    const h: i64 = @intCast(surf.height);
    const cx0 = std.math.clamp(x0, 0, w);
    const cx1 = std.math.clamp(x1, 0, w);
    const cy0 = std.math.clamp(y0, 0, h);
    const cy1 = std.math.clamp(y1, 0, h);
    if (cx1 <= cx0 or cy1 <= cy0) return null;
    return .{ .x0 = cx0, .x1 = cx1, .y0 = cy0, .y1 = cy1 };
}

/// How much of the pixel centred at `(x, y)` the instance covers.
///
/// This is the whole of the rasteriser's shape knowledge, and it is the function
/// a fragment shader implements: `gl_FragCoord.xy` and `@builtin(position).xy`
/// are pixel centres too, so the same expression is valid on both sides.
pub fn coverage(inst: Instance, x: f32, y: f32) f32 {
    return switch (inst.kindOf()) {
        .rect => rectCoverage(inst.boxRect(), x, y),
        .rounded => distanceCoverage(roundedBoxDistance(
            .{ .x = x, .y = y },
            inst.boxRect().center(),
            .{ .x = inst.boxRect().w * 0.5, .y = inst.boxRect().h * 0.5 },
            inst.param,
        )),
        .glyph => glyphCoverage(inst, x, y),
        .line => capsuleCoverage(
            .{ .x = x, .y = y },
            inst.point(0),
            inst.point(1),
            inst.param,
        ),
        .triangle => triangleCoverage(inst, x, y),
    };
}

/// Area of the pixel square centred on `(x, y)` that `r` covers.
pub fn rectCoverage(r: Rect, x: f32, y: f32) f32 {
    const ox = @max(0, @min(r.x + r.w, x + 0.5) - @max(r.x, x - 0.5));
    const oy = @max(0, @min(r.y + r.h, y + 0.5) - @max(r.y, y - 0.5));
    return ox * oy;
}

/// Signed distance from `p` to a rounded box centred at `c` with half-extents
/// `b` and corner radius `radius`. Negative inside.
pub fn roundedBoxDistance(p: Vec2, c: Vec2, b: Vec2, radius: f32) f32 {
    const qx = @abs(p.x - c.x) - (b.x - radius);
    const qy = @abs(p.y - c.y) - (b.y - radius);
    const outside = @sqrt(@max(qx, 0) * @max(qx, 0) + @max(qy, 0) * @max(qy, 0));
    const inside = @min(@max(qx, qy), 0);
    return outside + inside - radius;
}

/// Coverage of a shape described by a signed distance: one pixel of ramp
/// around the edge, which is what a 1-pixel-wide antialiased edge is.
pub fn distanceCoverage(d: f32) f32 {
    return std.math.clamp(0.5 - d, 0, 1);
}

/// Distance from `p` to the segment `a..b`, and the coverage of a stroke of
/// `width` around it. Clamping the parameter to the segment gives the ends round
/// caps, which is what a hairline separator and a checkbox tick both want.
pub fn capsuleCoverage(p: Vec2, a: Vec2, b: Vec2, width: f32) f32 {
    const ab = Vec2.sub(b, a);
    const len2 = ab.x * ab.x + ab.y * ab.y;
    var t: f32 = 0;
    if (len2 > 0) {
        const ap = Vec2.sub(p, a);
        t = std.math.clamp((ap.x * ab.x + ap.y * ab.y) / len2, 0, 1);
    }
    const closest = Vec2{ .x = a.x + ab.x * t, .y = a.y + ab.y * t };
    const d = @sqrt((p.x - closest.x) * (p.x - closest.x) + (p.y - closest.y) * (p.y - closest.y));
    return std.math.clamp(width * 0.5 + 0.5 - d, 0, 1);
}

/// A glyph: the atlas sampled through the instance's `uv`, with the texture
/// coordinate carried by the screen position. `du`/`dv` turn a screen pixel into
/// the uv step the box's width implies, so the same glyph box renders the ink at
/// any scale — and because the atlas has a texel of padding that reads zero, the
/// ink fades out at the box edge instead of clipping.
fn glyphCoverage(inst: Instance, x: f32, y: f32) f32 {
    const r = inst.boxRect();
    if (r.w <= 0 or r.h <= 0) return 0;
    const uv = inst.points;
    const u = uv[0] + (x - r.x) * (uv[2] - uv[0]) / r.w;
    const v = uv[1] + (y - r.y) * (uv[3] - uv[1]) / r.h;
    return atlas.sample(u, v);
}

/// Triangles are small (an arrow, a caret) and rare, so they are supersampled
/// 4×4 rather than given their own exact coverage maths: 16 samples per pixel
/// across a 9-pixel arrow is nothing, and it cannot get the winding wrong.
fn triangleCoverage(inst: Instance, x: f32, y: f32) f32 {
    const a = inst.point(0);
    const b = inst.point(1);
    const c = inst.point(2);
    const sub = 4;
    const step = 1.0 / @as(f32, @floatFromInt(sub));
    var hits: usize = 0;
    for (0..sub) |sy| {
        for (0..sub) |sx| {
            const p = Vec2{
                .x = x + (@as(f32, @floatFromInt(sx)) + 0.5) * step - 0.5,
                .y = y + (@as(f32, @floatFromInt(sy)) + 0.5) * step - 0.5,
            };
            if (insideTriangle(p, a, b, c)) hits += 1;
        }
    }
    if (hits == 0) return 0;
    return @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(sub * sub));
}

fn insideTriangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) bool {
    const d1 = edgeSign(p, a, b);
    const d2 = edgeSign(p, b, c);
    const d3 = edgeSign(p, c, a);
    const has_neg = d1 < 0 or d2 < 0 or d3 < 0;
    const has_pos = d1 > 0 or d2 > 0 or d3 > 0;
    return !(has_neg and has_pos);
}

fn edgeSign(p: Vec2, a: Vec2, b: Vec2) f32 {
    return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
}

// ---------------------------------------------------------------------------
// the loop every render path runs
// ---------------------------------------------------------------------------

/// Draws one instance: for every pixel its box touches inside the clip, take the
/// coverage and composite.
///
/// This is the software rasteriser in full. It is also the reference the GPU
/// path is checked against, because it computes exactly what the fragment shader
/// computes and composites in exactly the order the instances arrive in.
pub fn emit(surf: *Surface, inst: Instance) void {
    if (inst.color[3] <= 0) return;
    const span = spanOf(surf, inst.clipRect(), inst.bounds()) orelse return;
    const color = inst.colorOf();
    var y = span.y0;
    while (y < span.y1) : (y += 1) {
        var x = span.x0;
        while (x < span.x1) : (x += 1) {
            const cov = coverage(inst, @as(f32, @floatFromInt(x)) + 0.5, @as(f32, @floatFromInt(y)) + 0.5);
            if (cov > 0) surf.blend(x, y, color, cov);
        }
    }
}

/// The corner radius a box can actually use: never more than half its smaller
/// side, so a 30×8 chip cannot ask for a 12-pixel radius.
pub fn clampRadius(radius: f32, r: Rect) f32 {
    return std.math.clamp(radius, 0, @min(r.w, r.h) * 0.5);
}

/// The box a border of `border` pixels leaves for the fill.
pub fn innerBox(r: Rect, border: f32) Rect {
    return if (border > 0) r.shrink(border) else r;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const zpu = @import("zpu");

fn newImage(w: usize, h: usize) !zgf.Image {
    try zpu.init(testing.allocator, .{});
    return zgf.Image.init(testing.allocator, w, h, 3);
}

fn rectInstance(r: Rect, color: Color) Instance {
    return .{ .box = r.toArray(), .color = color.toArray() };
}

test "a rectangle's coverage is its overlap with the pixel" {
    const r = Rect.init(1, 0, 0.5, 1);
    try testing.expectApproxEqAbs(@as(f32, 0.5), rectCoverage(r, 1.5, 0.5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), rectCoverage(r, 2.5, 0.5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), rectCoverage(Rect.init(0, 0, 4, 4), 2.5, 2.5), 1e-6);
}

test "a rounded box distance is negative inside and positive outside" {
    const c = Vec2{ .x = 5, .y = 5 };
    const b = Vec2{ .x = 5, .y = 5 };
    try testing.expect(roundedBoxDistance(c, c, b, 2) < 0);
    try testing.expect(roundedBoxDistance(.{ .x = 11, .y = 5 }, c, b, 2) > 0);
    // On the flat edge the distance is the distance.
    try testing.expectApproxEqAbs(@as(f32, 1), roundedBoxDistance(.{ .x = 11, .y = 5 }, c, b, 0), 1e-6);
}

test "a rectangle is solid inside, empty outside and half at the edge" {
    var img = try newImage(8, 8);
    defer img.deinit();
    var surf = Surface.init(&img);
    const white = Color.hex(0xffffff, 1);

    emit(&surf, rectInstance(Rect.init(2, 3, 4, 2), white));
    try testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(2, 3).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(5, 4).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(1, 3).r, 1e-6);
    // A half-covered left edge pixel.
    emit(&surf, rectInstance(Rect.init(2.5, 6, 1, 1), white));
    try testing.expectApproxEqAbs(@as(f32, 0.5), img.getPixel(2, 6).r, 1e-3);
}

test "the clip is a hard edge, and it is the instance's own" {
    var img = try newImage(8, 8);
    defer img.deinit();
    var surf = Surface.init(&img);
    const white = Color.hex(0xffffff, 1);
    var inst = rectInstance(Rect.init(0, 0, 8, 8), white);
    inst.clip = .{ 0, 0, 4, 8 };
    emit(&surf, inst);
    try testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(3, 3).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(4, 3).r, 1e-6);
}

test "a line covers its path and nothing else" {
    var img = try newImage(16, 16);
    defer img.deinit();
    var surf = Surface.init(&img);
    const white = Color.hex(0xffffff, 1);
    const a = Vec2{ .x = 2, .y = 8 };
    const b = Vec2{ .x = 13, .y = 8 };
    emit(&surf, lineInstance(a, b, 2, white));
    try testing.expect(img.getPixel(7, 8).r > 0.9);
    try testing.expect(img.getPixel(7, 7).r > 0.4);
    try testing.expect(img.getPixel(7, 5).r < 0.05);
    try testing.expect(img.getPixel(14, 8).r < 0.05);
}

test "a glyph quad paints the font's ink" {
    var img = try newImage(16, 16);
    defer img.deinit();
    var surf = Surface.init(&img);
    const white = Color.hex(0xffffff, 1);

    // 'H' at the font's own size, in a box at (1, 1).
    const g = @import("font.zig").glyphOf('H');
    const idx = 'H' - 32;
    const uv = atlas.uvOf(idx);
    const inst = Instance{
        .box = .{ 1, 1, @floatFromInt(g.w), @floatFromInt(g.h) },
        .color = white.toArray(),
        .kind = @backingInt(Instance.Kind.glyph),
        .points = .{ uv[0], uv[1], uv[2], uv[3], 0, 0 },
    };
    emit(&surf, inst);

    // Every ink pixel matches the font byte, exactly.
    var ink: usize = 0;
    for (0..g.h) |row| {
        for (0..g.w) |col| {
            const want = @as(f32, @floatFromInt(@import("font.zig").coverage(g, @intCast(col), @intCast(row)))) / 255.0;
            const got = img.getPixel(1 + col, 1 + row).r;
            try testing.expectApproxEqAbs(want, got, 1e-3);
            if (want > 0) ink += 1;
        }
    }
    try testing.expect(ink > 10);
    // And nothing outside the box.
    try testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(0, 0).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(15, 15).r, 1e-6);
}

fn lineInstance(a: Vec2, b: Vec2, width: f32, color: Color) Instance {
    const pad = width * 0.5;
    return .{
        .box = .{
            @min(a.x, b.x) - pad,
            @min(a.y, b.y) - pad,
            @abs(b.x - a.x) + width,
            @abs(b.y - a.y) + width,
        },
        .color = color.toArray(),
        .param = width,
        .kind = @backingInt(Instance.Kind.line),
        .points = .{ a.x, a.y, b.x, b.y, 0, 0 },
    };
}
