//! The software backend: a display list, rasterised into a `zgf.Image`.
//!
//! The whole renderer is the three lines at the top of `draw`: clear the image,
//! walk the display list into instances (in device space), and hand each one to
//! `raster.emit`. Everything `zui` knows about *shapes* lives in
//! `zui/raster.zig`; everything it knows about *turning a display list into
//! shapes* — the clip stack, a border becoming two boxes, a text run becoming
//! one quad per glyph — lives in `zui/instances.zig`. This file is the software
//! consumer of that stream, and nothing else.
//!
//! That is deliberate. `zui/pipeline.zig` rasterises the *same* stream from
//! instances in logical space, and `instances.zig` is the single place the two
//! could ever disagree; its test suite draws a frame both ways and compares the
//! pixels. A renderer that owned its own copy of the shape maths would make that
//! comparison a matter of opinion.
//!
//! This is the only file in `zui` that imports `zgf` for rendering, and the
//! conversion from `zui.Color` to `zgf.Color` happens in `raster.zig`.
const std = @import("std");
const zgf = @import("zgf");
const frame_mod = @import("../zui/frame.zig");
const instances = @import("../zui/instances.zig");
const raster = @import("../zui/raster.zig");
const types = @import("../zui/types.zig");

const Command = frame_mod.Command;
const DisplayList = frame_mod.DisplayList;
const Instance = frame_mod.Instance;
const Rect = types.Rect;
const Surface = raster.Surface;
const Vec2 = types.Vec2;

pub const Options = struct {
    /// Where the list is drawn, in image pixels (the UI's coordinates are
    /// scaled by `scale`).
    origin: Vec2 = .{},
    /// Device pixels per logical pixel: 1 is a plain screen, 2 is a HiDPI one.
    /// The display list is in logical pixels, so scaling happens exactly here —
    /// in the transform the walk applies, never in the shapes themselves.
    scale: f32 = 1,
    /// Clear colour, or `null` to draw over whatever the image already holds.
    clear: ?types.Color = null,
};

/// Rasterises `list` into `img`.
pub fn draw(list: *const DisplayList, img: *zgf.Image, opts: Options) void {
    if (opts.clear) |c| img.fill(raster.toZgf(c));
    var ctx = Ctx{ .surf = Surface.init(img) };
    instances.each(list, .{ .origin = opts.origin, .scale = opts.scale }, &ctx, Ctx.emit);
}

/// The sink the walk calls once per instance. A pointer to this struct is the
/// whole state a frame of software rendering needs.
const Ctx = struct {
    surf: Surface,

    fn emit(self: *Ctx, inst: Instance) void {
        raster.emit(&self.surf, inst);
    }
};

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const zpu = @import("zpu");

fn newImage(w: usize, h: usize) !zgf.Image {
    try zpu.init(testing.allocator, .{});
    return zgf.Image.init(testing.allocator, w, h, 3);
}

fn pixel(img: *const zgf.Image, x: usize, y: usize) zgf.Color {
    return img.getPixel(x, y);
}

fn fillList(list: *DisplayList, cmd: Command) !void {
    try list.push(testing.allocator, cmd);
}

test "an integer rectangle lands on exactly those pixels" {
    var img = try newImage(8, 8);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .rect = .{ .r = Rect.init(2, 3, 4, 2), .color = white } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    try testing.expectApproxEqAbs(@as(f32, 1.0), pixel(&img, 2, 3).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), pixel(&img, 5, 4).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 1, 3).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 6, 3).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 2, 5).r, 1e-6);

    // Half of one pixel is half as bright.
    var half = try newImage(4, 4);
    defer half.deinit();
    var list2 = DisplayList{};
    defer list2.deinit(testing.allocator);
    try fillList(&list2, .{ .rect = .{ .r = Rect.init(1, 0, 0.5, 1), .color = white } });
    draw(&list2, &half, .{ .clear = types.Color.hex(0x000000, 1) });
    try testing.expectApproxEqAbs(@as(f32, 0.5), pixel(&half, 1, 0).r, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&half, 2, 0).r, 1e-6);
}

test "clips are respected, including nesting" {
    var img = try newImage(8, 8);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .push_clip = Rect.init(0, 0, 4, 8) });
    try fillList(&list, .{ .rect = .{ .r = Rect.init(0, 0, 8, 8), .color = white } });
    try fillList(&list, .{ .pop_clip = {} });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    try testing.expectApproxEqAbs(@as(f32, 1.0), pixel(&img, 3, 3).r, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 4, 3).r, 1e-6);
    try testing.expect(list.isBalanced());
}

test "a rounded corner is antialiased" {
    var img = try newImage(12, 12);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .rounded = .{
        .r = Rect.init(1, 1, 10, 10),
        .radius = 4,
        .color = white,
        .border = 0,
        .border_color = types.Color.transparent,
    } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    // The middle of an edge is solid, the very corner is not, and the pixel
    // just inside the corner has partial coverage.
    try testing.expect(pixel(&img, 6, 1).r > 0.95);
    try testing.expect(pixel(&img, 1, 1).r < 0.5);
    const inside_corner = pixel(&img, 2, 2).r;
    try testing.expect(inside_corner > 0.05 and inside_corner < 0.99);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 0, 0).r, 1e-6);
}

test "a border leaves the inside alone" {
    var img = try newImage(12, 12);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const red = types.Color.hex(0xff0000, 1);
    const blue = types.Color.hex(0x0000ff, 1);
    try fillList(&list, .{ .rounded = .{
        .r = Rect.init(1, 1, 10, 10),
        .radius = 0,
        .color = red,
        .border = 2,
        .border_color = blue,
    } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    const edge = pixel(&img, 1, 6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), edge.r, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 1.0), edge.b, 1e-3);
    const middle = pixel(&img, 6, 6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), middle.r, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), middle.b, 1e-3);
}

test "lines cover their path and nothing else" {
    var img = try newImage(16, 16);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .line = .{ .a = .{ .x = 2, .y = 8 }, .b = .{ .x = 13, .y = 8 }, .width = 2, .color = white } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    try testing.expect(pixel(&img, 7, 8).r > 0.9);
    try testing.expect(pixel(&img, 7, 7).r > 0.4);
    try testing.expect(pixel(&img, 7, 5).r < 0.05);
    try testing.expect(pixel(&img, 14, 8).r < 0.05);

    // A diagonal line: the midpoint of the path is covered.
    var img2 = try newImage(16, 16);
    defer img2.deinit();
    var list2 = DisplayList{};
    defer list2.deinit(testing.allocator);
    try fillList(&list2, .{ .line = .{ .a = .{ .x = 2, .y = 2 }, .b = .{ .x = 12, .y = 12 }, .width = 2, .color = white } });
    draw(&list2, &img2, .{ .clear = types.Color.hex(0x000000, 1) });
    try testing.expect(pixel(&img2, 7, 7).r > 0.7);
    try testing.expect(pixel(&img2, 7, 3).r < 0.05);
}

test "triangles fill their interior" {
    var img = try newImage(16, 16);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .triangle = .{
        .a = .{ .x = 2, .y = 2 },
        .b = .{ .x = 14, .y = 2 },
        .c = .{ .x = 8, .y = 14 },
        .color = white,
    } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    try testing.expect(pixel(&img, 8, 4).r > 0.9);
    try testing.expect(pixel(&img, 3, 2).r > 0.3); // the top edge cuts this pixel
    // Outside the slanted edges, and below the apex.
    try testing.expect(pixel(&img, 1, 8).r < 0.05);
    try testing.expect(pixel(&img, 2, 12).r < 0.05);
    try testing.expect(pixel(&img, 15, 15).r < 0.05);
}

test "text puts ink where the metrics say" {
    var img = try newImage(120, 40);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .text = .{ .pos = .{ .x = 4, .y = 20 }, .size = 14, .color = white, .text = "Hg" } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1) });

    var ink: usize = 0;
    var rightmost: usize = 0;
    for (0..40) |y| {
        for (0..120) |x| {
            if (pixel(&img, x, y).r > 0.2) {
                ink += 1;
                rightmost = @max(rightmost, x);
            }
        }
    }
    try testing.expect(ink > 30);
    // 'H' and 'g' together are about 20 px wide at size 14, starting at x = 4.
    try testing.expect(rightmost < 4 + 30);
    try testing.expect(rightmost > 4 + 5);
    // Nothing above the ascender or below the descender.
    for (0..40) |y| {
        if (y < 5 or y > 26) {
            for (0..120) |x| try testing.expect(pixel(&img, x, y).r < 0.05);
        }
    }
}

test "scaling a display list scales the picture" {
    var img = try newImage(40, 40);
    defer img.deinit();
    var list = DisplayList{};
    defer list.deinit(testing.allocator);

    const white = types.Color.hex(0xffffff, 1);
    try fillList(&list, .{ .rect = .{ .r = Rect.init(2, 2, 4, 4), .color = white } });
    draw(&list, &img, .{ .clear = types.Color.hex(0x000000, 1), .scale = 2, .origin = .{ .x = 1, .y = 1 } });

    // Logical (2,2)..(6,6) becomes device (5,5)..(13,13).
    try testing.expectApproxEqAbs(@as(f32, 1.0), pixel(&img, 6, 6).r, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 1.0), pixel(&img, 12, 12).r, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 4, 5).r, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pixel(&img, 14, 12).r, 1e-3);
}
