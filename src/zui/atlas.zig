//! The glyph atlas: every glyph's coverage packed into one texture, at comptime.
//!
//! The font data is 95 separate bitmaps. A GPU cannot sample 95 textures, and a
//! shader cannot know where a glyph's bytes start without a lookup table — so
//! the glyphs are packed once, into a single 8-bit image, and every glyph draws
//! as one textured quad whose `uv` points into it.
//!
//! **This is computed at compile time.** The glyph boxes are `comptime`
//! constants, so the packing is too: no allocation, no initialisation order, no
//! global mutable state, and the texture is a `const` array that a backend can
//! upload directly. It is the same packing a font tool would do, run by the
//! compiler instead of by a startup function.
//!
//! The layout is deliberately dumb (shelves, one pixel of padding) because its
//! only job is to be *stable* and *correct at the edges*:
//!
//! * One texel of padding around every glyph, so bilinear sampling just outside
//!   the ink box reads a zero instead of the neighbouring glyph. That is exactly
//!   what `font.coverage` returns for coordinates outside a glyph, which is why
//!   the direct renderer and the instance path produce the same pixels.
//! * Shelf order follows the glyph table, so the atlas is a pure function of
//!   `font_data.zig` and two builds of the same source produce the same bytes.
const std = @import("std");
const data = @import("font_data.zig");

/// Width in texels. 128 is a power of two (a friendly GPU texture width) and
/// wide enough that the tall glyphs do not fragment the packing.
pub const width: usize = 128;
/// Blank texels between glyphs, and at the left/top edge of the image.
pub const padding: usize = 1;

const Placement = struct { x: usize, y: usize, w: usize, h: usize };

/// Shelf packing: left to right, top to bottom, new shelf when the row is full.
fn place() [data.glyphs.len]Placement {
    var out: [data.glyphs.len]Placement = undefined;
    var x: usize = padding;
    var y: usize = padding;
    var shelf_h: usize = 0;
    for (data.glyphs, 0..) |g, i| {
        if (g.w == 0 or g.h == 0) {
            // Blank glyph (a space): no ink, no texels. The reader never samples
            // it — `zui` skips empty glyphs before drawing — but keeping the
            // entry means `rects` stays indexed by glyph, like the font table.
            out[i] = .{ .x = 0, .y = 0, .w = 0, .h = 0 };
            continue;
        }
        if (x + g.w + padding > width) {
            x = padding;
            y += shelf_h + padding;
            shelf_h = 0;
        }
        out[i] = .{ .x = x, .y = y, .w = g.w, .h = g.h };
        x += g.w + padding;
        shelf_h = @max(shelf_h, g.h);
    }
    return out;
}

const placements = place();

/// Height in texels, measured from the packing above.
pub const height: usize = blk: {
    var bottom: usize = 0;
    for (placements) |p| bottom = @max(bottom, p.y + p.h);
    break :blk bottom + padding;
};

/// The texture: 8-bit coverage, row major, origin top-left.
pub const pixels: [width * height]u8 = blk: {
    // The packing is a few thousand comptime steps; the default quota (1000)
    // is far too small for it.
    @setEvalBranchQuota(1 << 20);
    var img: [width * height]u8 = @splat(0);
    for (data.glyphs, 0..) |g, i| {
        const p = placements[i];
        for (0..p.h) |row| {
            for (0..p.w) |col| {
                img[(p.y + row) * width + p.x + col] = data.alpha[g.offset + row * p.w + col];
            }
        }
    }
    break :blk img;
};

/// Where each glyph's ink sits in the atlas, in texels. Indexed like
/// `font_data.glyphs` (ASCII 32 upwards).
pub const rects: [data.glyphs.len]Placement = placements;

/// The source rectangle `uv` for glyph `i`, normalised to `0..1` — the form a
/// shader wants, and the form `instances.Instance` carries.
pub fn uvOf(i: usize) [4]f32 {
    const p = rects[i];
    if (p.w == 0 or p.h == 0) return .{ 0, 0, 0, 0 };
    const w: f32 = @floatFromInt(width);
    const h: f32 = @floatFromInt(height);
    return .{
        @as(f32, @floatFromInt(p.x)) / w,
        @as(f32, @floatFromInt(p.y)) / h,
        @as(f32, @floatFromInt(p.x + p.w)) / w,
        @as(f32, @floatFromInt(p.y + p.h)) / h,
    };
}

/// Bilinear sample of the atlas at normalised `(u, v)`, using the texel-centre
/// convention: `u * width - 0.5` is the texel coordinate, so a sample that lands
/// exactly on a texel centre returns that texel unchanged. Outside the image the
/// value is zero, which is what makes the padding behave like empty space.
pub fn sample(u: f32, v: f32) f32 {
    const x = u * @as(f32, @floatFromInt(width)) - 0.5;
    const y = v * @as(f32, @floatFromInt(height)) - 0.5;
    const fx = @floor(x);
    const fy = @floor(y);
    const tx = x - fx;
    const ty = y - fy;
    const ix: i32 = @intFromFloat(fx);
    const iy: i32 = @intFromFloat(fy);
    const c00 = texel(ix, iy);
    const c10 = texel(ix + 1, iy);
    const c01 = texel(ix, iy + 1);
    const c11 = texel(ix + 1, iy + 1);
    const top = c00 + (c10 - c00) * tx;
    const bottom = c01 + (c11 - c01) * tx;
    return top + (bottom - top) * ty;
}

fn texel(x: i32, y: i32) f32 {
    if (x < 0 or y < 0) return 0;
    const ux: usize = @intCast(x);
    const uy: usize = @intCast(y);
    if (ux >= width or uy >= height) return 0;
    return @as(f32, @floatFromInt(pixels[uy * width + ux])) / 255.0;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "the atlas is the right size for its contents" {
    // Every glyph fits, nothing overlaps, and the padding is real.
    var occupied: usize = 0;
    for (data.glyphs, 0..) |g, i| {
        const p = rects[i];
        if (g.w == 0 or g.h == 0) {
            try testing.expectEqual(@as(usize, 0), p.w + p.h);
            continue;
        }
        try testing.expectEqual(g.w, p.w);
        try testing.expectEqual(g.h, p.h);
        try testing.expect(p.x + p.w <= width);
        try testing.expect(p.y + p.h <= height);
        occupied += p.w * p.h;
    }
    // The atlas is bigger than its ink (padding) and not absurdly so.
    try testing.expect(occupied < width * height);
    try testing.expect(width * height < occupied * 3);
}

test "no two glyphs touch" {
    for (rects, 0..) |a, i| {
        if (a.w == 0 or a.h == 0) continue;
        for (rects[i + 1 ..], i + 1..) |b, j| {
            if (b.w == 0 or b.h == 0) continue;
            const apart = a.x + a.w + padding <= b.x or
                b.x + b.w + padding <= a.x or
                a.y + a.h + padding <= b.y or
                b.y + b.h + padding <= a.y;
            try testing.expect(apart);
            _ = j;
        }
    }
}

test "the atlas carries the same bytes as the font table" {
    for (data.glyphs, 0..) |g, i| {
        const p = rects[i];
        for (0..p.h) |row| {
            for (0..p.w) |col| {
                try testing.expectEqual(
                    data.alpha[g.offset + row * p.w + col],
                    pixels[(p.y + row) * width + p.x + col],
                );
            }
        }
    }
}

test "sampling the uv of a glyph reproduces the glyph" {
    // At one sample per texel centre, the atlas sample must equal the coverage
    // byte exactly — that is the property the instance path relies on.
    for (data.glyphs, 0..) |g, i| {
        const p = rects[i];
        if (p.w == 0 or p.h == 0) continue;
        const uv = uvOf(i);
        for (0..p.h) |row| {
            for (0..p.w) |col| {
                const u = uv[0] + (uv[2] - uv[0]) * (@as(f32, @floatFromInt(col)) + 0.5) / @as(f32, @floatFromInt(p.w));
                const v = uv[1] + (uv[3] - uv[1]) * (@as(f32, @floatFromInt(row)) + 0.5) / @as(f32, @floatFromInt(p.h));
                const want = @as(f32, @floatFromInt(data.alpha[g.offset + row * p.w + col])) / 255.0;
                // `2e-3` and not `1e-6`: the uv round trip divides by the atlas
                // size and multiplies back in f32, so a sample can sit a
                // thousandth of a texel off centre. Over the bilinear ramp that
                // is a thousandth of a coverage step — a quarter of one 8-bit
                // level, which is why the two render paths can be compared at
                // one 8-bit level and still agree.
                try testing.expectApproxEqAbs(want, sample(u, v), 2e-3);
            }
        }
    }
}

test "just outside a glyph's box the atlas reads zero" {
    // Half a texel outside the ink is still padding; that is what makes a glyph
    // fade out at its edge instead of sampling its neighbour.
    const e = 'e' - 32;
    const p = rects[e];
    const uv = uvOf(e);
    const one_w = (uv[2] - uv[0]) / @as(f32, @floatFromInt(p.w));
    const one_h = (uv[3] - uv[1]) / @as(f32, @floatFromInt(p.h));
    try testing.expectApproxEqAbs(@as(f32, 0), sample(uv[0] - one_w * 0.25, uv[1] + one_h), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), sample(uv[2] + one_w * 0.25, uv[3] - one_h), 1e-6);
}
