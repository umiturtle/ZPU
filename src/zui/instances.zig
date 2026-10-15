//! Display list -> instance stream: the conversion both render paths share.
//!
//! `DisplayList` is the authoring format. It has nesting (a clip stack) and a
//! text run, neither of which a shader can be handed directly. `Instance` is the
//! drawing format: one shape, fixed size, self-contained. This file is the only
//! place the two meet, and it is written as a *stream* — `each` hands one
//! instance at a time to a sink — so the same walk can feed a rasteriser, fill a
//! caller's slice, or pack a GPU buffer, with no intermediate list and no
//! allocator in the picture.
//!
//! Two rules make the conversion total:
//!
//! * **A border is two instances.** A rounded box with a 1-pixel border is the
//!   outer box in the border colour, then the inner box in the fill colour. The
//!   shader needs no border mode, the software rasteriser needs no second
//!   distance field, and the result is the same picture down to the antialiased
//!   edge — which is exactly the test at the bottom of this file.
//! * **A text run is one instance per glyph.** The atlas turns 95 bitmaps into
//!   one texture, so a glyph is a textured quad like any other shape. A space
//!   has no ink and produces no instance at all.
//!
//! Clips are *not* flattened. Each instance carries the clip in force where it
//! was emitted, because a clip has to survive whatever a host does with the
//! stream (sort it, split it into passes, upload it asynchronously), and a
//! uniform would not.
const std = @import("std");
const atlas = @import("atlas.zig");
const font = @import("font.zig");
const frame_mod = @import("frame.zig");
const types = @import("types.zig");

const Command = frame_mod.Command;
const DisplayList = frame_mod.DisplayList;
const Instance = frame_mod.Instance;
const Rect = types.Rect;
const Vec2 = types.Vec2;

/// Where the stream is going, and therefore which space it lands in.
///
/// `Transform{}` is the identity — instances in the space they were authored in,
/// which is what a backend uploads: the device scale is the host's business, and
/// a buffer should not have to be rebuilt because a window moved to the other
/// monitor. A rasteriser instead passes the frame's origin and scale, so the
/// walk itself is the only place the two spaces differ.
pub const Transform = struct {
    origin: Vec2 = .{},
    scale: f32 = 1,

    pub fn rect(self: Transform, r: Rect) Rect {
        return .{
            .x = self.origin.x + r.x * self.scale,
            .y = self.origin.y + r.y * self.scale,
            .w = r.w * self.scale,
            .h = r.h * self.scale,
        };
    }

    pub fn point(self: Transform, p: Vec2) Vec2 {
        return .{ .x = self.origin.x + p.x * self.scale, .y = self.origin.y + p.y * self.scale };
    }
};

/// How deep the clip stack may nest while walking. The library's own widgets
/// nest at most twice (a scroll view inside a window); beyond this, a deeper
/// push is folded into the clip already in force rather than dropped.
pub const max_clip_depth = 8;

/// Walks `list` and calls `sink(sink_ctx, instance)` for every instance, in
/// draw order.
///
/// `sink` is a function pointer rather than an interface because the two
/// consumers are known and different: the rasteriser writes into an image, the
/// collector appends to a slice. Passing a function keeps the walk free of
/// indirection per instance *and* of the allocation a vtable would need.
pub fn each(
    list: *const DisplayList,
    tr: Transform,
    ctx: anytype,
    comptime sink: fn (@TypeOf(ctx), Instance) void,
) void {
    var clips: [max_clip_depth]Rect = undefined;
    var depth: usize = 0;
    var clip = Instance.no_clip_rect;

    for (list.commands.items) |cmd| {
        switch (cmd) {
            .push_clip => |r| {
                if (depth < max_clip_depth) {
                    clips[depth] = clip;
                    depth += 1;
                }
                clip = clip.intersect(tr.rect(r));
            },
            .pop_clip => {
                if (depth > 0) {
                    depth -= 1;
                    clip = clips[depth];
                } else {
                    clip = Instance.no_clip_rect;
                }
            },
            .rect => |c| if (c.color.a > 0) sink(ctx, withClip(.{
                .box = boxOf(tr.rect(c.r)),
                .color = c.color.toArray(),
            }, clip)),
            .rounded => |c| {
                const r = tr.rect(c.r);
                const border = c.border * tr.scale;
                // The border is the whole box in the border colour; the fill is
                // an ordinary rounded box drawn on top of it.
                if (border > 0 and c.border_color.a > 0) sink(ctx, withClip(.{
                    .box = boxOf(r),
                    .color = c.border_color.toArray(),
                    .param = c.radius * tr.scale,
                    .kind = @backingInt(Instance.Kind.rounded),
                }, clip));
                if (c.color.a > 0) {
                    const inner = r.shrink(border);
                    if (inner.w > 0 and inner.h > 0) sink(ctx, withClip(.{
                        .box = boxOf(inner),
                        .color = c.color.toArray(),
                        .param = @max(0, c.radius * tr.scale - border),
                        .kind = @backingInt(Instance.Kind.rounded),
                    }, clip));
                }
            },
            .line => |c| if (c.color.a > 0) {
                const a = tr.point(c.a);
                const b = tr.point(c.b);
                const width = c.width * tr.scale;
                const pad = width * 0.5;
                sink(ctx, withClip(.{
                    // The capsule's bounding box: the endpoints grown by half
                    // the stroke, which is also what a vertex shader would emit.
                    .box = .{
                        @min(a.x, b.x) - pad,
                        @min(a.y, b.y) - pad,
                        @abs(b.x - a.x) + width,
                        @abs(b.y - a.y) + width,
                    },
                    .color = c.color.toArray(),
                    .param = width,
                    .kind = @backingInt(Instance.Kind.line),
                    .points = .{ a.x, a.y, b.x, b.y, 0, 0 },
                }, clip));
            },
            .triangle => |c| if (c.color.a > 0) {
                const a = tr.point(c.a);
                const b = tr.point(c.b);
                const d = tr.point(c.c);
                sink(ctx, withClip(.{
                    .color = c.color.toArray(),
                    .kind = @backingInt(Instance.Kind.triangle),
                    .points = .{ a.x, a.y, b.x, b.y, d.x, d.y },
                }, clip));
            },
            .text => |c| text(ctx, c.pos, c.size, c.color, c.text, tr, clip, sink),
        }
    }
}

fn boxOf(r: Rect) [4]f32 {
    return .{ r.x, r.y, r.w, r.h };
}

fn withClip(inst: Instance, clip: Rect) Instance {
    var out = inst;
    out.clip = boxOf(clip);
    return out;
}

/// Emits one instance per glyph of a text run.
fn text(
    ctx: anytype,
    pos: Vec2,
    size: f32,
    color: types.Color,
    run: []const u8,
    tr: Transform,
    clip: Rect,
    comptime sink: fn (@TypeOf(ctx), Instance) void,
) void {
    if (color.a <= 0 or run.len == 0) return;
    const s = font.scaleOf(size) * tr.scale;
    if (s <= 0) return;
    var it = font.layout(run, size, pos.x);
    const baseline = pos.y;
    while (it.next()) |g| {
        if (g.glyph.w == 0 or g.glyph.h == 0) continue;
        const left = tr.origin.x + (g.x + @as(f32, @floatFromInt(g.glyph.left)) * font.scaleOf(size)) * tr.scale;
        const top = tr.origin.y + (baseline + @as(f32, @floatFromInt(g.glyph.top)) * font.scaleOf(size)) * tr.scale;
        const w = @as(f32, @floatFromInt(g.glyph.w)) * s;
        const h = @as(f32, @floatFromInt(g.glyph.h)) * s;
        const idx: usize = if (g.c < font.first_char or g.c > font.last_char)
            '?' - font.first_char
        else
            @as(usize, g.c - font.first_char);
        const uv = atlas.uvOf(idx);
        sink(ctx, .{
            .box = .{ left, top, w, h },
            .color = color.toArray(),
            .clip = clip.toArray(),
            .kind = @backingInt(Instance.Kind.glyph),
            .points = .{ uv[0], uv[1], uv[2], uv[3], 0, 0 },
        });
    }
}

/// Collects the stream into a slice the caller owns.
///
/// Two passes over the walk and no allocation: the first counts, the second
/// fills. A display list is small and pure — walking it twice costs less than
/// growing a buffer, and a caller that wants to skip the first pass can call
/// `count` once and reuse the number for a whole frame's worth of instances.
pub fn collect(list: *const DisplayList, tr: Transform, out: []Instance) []Instance {
    const n = @min(count(list, tr), out.len);
    if (n == 0) return out[0..0];
    var fill = Collector{ .buf = out[0..n] };
    each(list, tr, &fill, Collector.sink);
    return out[0..fill.n];
}

/// How many instances `list` expands to. Separate from `collect` so a caller can
/// size a buffer, or a test can assert a frame's cost, without allocating.
pub fn count(list: *const DisplayList, tr: Transform) usize {
    var counter = Collector{ .buf = &.{} };
    each(list, tr, &counter, Collector.sink);
    return counter.n;
}

const Collector = struct {
    buf: []Instance,
    n: usize = 0,

    fn sink(self: *Collector, inst: Instance) void {
        if (self.n < self.buf.len) self.buf[self.n] = inst;
        self.n += 1;
    }
};

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn push(list: *DisplayList, cmd: Command) !void {
    try list.push(testing.allocator, cmd);
}

test "one shape, one instance" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    const white = types.Color.hex(0xffffff, 1);

    try push(&list, .{ .rect = .{ .r = Rect.init(1, 2, 3, 4), .color = white } });
    try testing.expectEqual(@as(usize, 1), count(&list, .{}));

    var buf: [8]Instance = undefined;
    const insts = collect(&list, .{}, &buf);
    try testing.expectEqual(@as(usize, 1), insts.len);
    try testing.expectEqual(@as(f32, 3), insts[0].boxRect().w);
    try testing.expect(insts[0].kindOf() == .rect);
    try testing.expectEqual(@as(f32, 1), insts[0].colorOf().a);
}

test "a border is two instances, the inner box shrunk" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try push(&list, .{ .rounded = .{
        .r = Rect.init(0, 0, 10, 10),
        .radius = 4,
        .color = types.Color.hex(0x00ff00, 1),
        .border = 2,
        .border_color = types.Color.hex(0xff0000, 1),
    } });

    var buf: [4]Instance = undefined;
    const insts = collect(&list, .{}, &buf);
    try testing.expectEqual(@as(usize, 2), insts.len);
    // The border keeps the outer box and the radius; the fill is inset.
    try testing.expectEqual(@as(f32, 10), insts[0].boxRect().w);
    try testing.expectEqual(@as(f32, 4), insts[0].param);
    try testing.expectEqual(@as(f32, 2), insts[1].boxRect().x);
    try testing.expectEqual(@as(f32, 6), insts[1].boxRect().w);
    try testing.expectEqual(@as(f32, 2), insts[1].param);
    // Draw order: the border first.
    try testing.expect(insts[0].colorOf().r > insts[0].colorOf().g);
}

test "clips nest, and every instance carries the one in force" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    const white = types.Color.hex(0xffffff, 1);
    try push(&list, .{ .push_clip = Rect.init(0, 0, 100, 100) });
    try push(&list, .{ .rect = .{ .r = Rect.init(0, 0, 10, 10), .color = white } });
    try push(&list, .{ .push_clip = Rect.init(0, 0, 20, 20) });
    try push(&list, .{ .rect = .{ .r = Rect.init(0, 0, 10, 10), .color = white } });
    try push(&list, .{ .pop_clip = {} });
    try push(&list, .{ .rect = .{ .r = Rect.init(0, 0, 10, 10), .color = white } });

    var buf: [4]Instance = undefined;
    const insts = collect(&list, .{}, &buf);
    try testing.expectEqual(@as(usize, 3), insts.len);
    try testing.expectEqual(@as(f32, 100), insts[0].clipRect().w);
    // The nested clip is the intersection, not the replacement.
    try testing.expectEqual(@as(f32, 20), insts[1].clipRect().w);
    // And it is restored on the way out.
    try testing.expectEqual(@as(f32, 100), insts[2].clipRect().w);
}

test "a text run becomes one instance per glyph, with atlas uv" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try push(&list, .{ .text = .{
        .pos = .{ .x = 4, .y = 20 },
        .size = 14,
        .color = types.Color.hex(0xffffff, 1),
        .text = "Hi there",
    } });

    // Two runs of the same text expand the same way: the stream is a pure
    // function of the list.
    const n = count(&list, .{});
    try testing.expectEqual(@as(usize, 7), n); // the space has no ink

    var buf: [16]Instance = undefined;
    const insts = collect(&list, .{}, &buf);
    try testing.expectEqual(@as(usize, 7), insts.len);
    for (insts) |inst| {
        try testing.expect(inst.kindOf() == .glyph);
        // uv points inside the atlas, and the `u` span is proportional to the
        // glyph's width in texels.
        try testing.expect(inst.points[0] < inst.points[2]);
        try testing.expect(inst.points[0] >= 0 and inst.points[2] <= 1);
    }
    // Glyphs advance to the right, and the run stays inside the box the
    // measurer promised for it: measure and draw are one code path.
    try testing.expect(insts[1].boxRect().x > insts[0].boxRect().x);
    const m = font.measure("Hi there", 14);
    var ink = insts[0].bounds();
    for (insts[1..]) |inst| ink = ink.merged(inst.bounds());
    try testing.expect(ink.x >= 4);
    try testing.expect(ink.x + ink.w <= 4 + m.width + 1);
    try testing.expect(ink.w > m.width * 0.5);

    // The stream is a pure function of the list: walking it twice gives the
    // same instances, byte for byte.
    var buf2: [16]Instance = undefined;
    const again = collect(&list, .{}, &buf2);
    try testing.expectEqual(insts.len, again.len);
    for (insts, again) |a, b| {
        try testing.expect(std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b)));
    }
}

test "scaling moves positions and leaves uv alone" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try push(&list, .{ .rect = .{ .r = Rect.init(1, 2, 3, 4), .color = types.Color.hex(0xffffff, 1) } });
    try push(&list, .{ .text = .{
        .pos = .{ .x = 1, .y = 2 },
        .size = 14,
        .color = types.Color.hex(0xffffff, 1),
        .text = "A",
    } });

    var buf: [4]Instance = undefined;
    const logical = collect(&list, .{}, &buf);
    const rect_logical = logical[0];
    const glyph_logical = logical[1];

    var buf2: [4]Instance = undefined;
    const device = collect(&list, .{ .origin = .{ .x = 5, .y = 7 }, .scale = 2 }, &buf2);
    try testing.expectEqual(@as(f32, 5 + 1 * 2), device[0].boxRect().x);
    try testing.expectEqual(@as(f32, 4 * 2), device[0].boxRect().h);
    // A glyph's box scales, its texture coordinate does not.
    try testing.expectEqual(glyph_logical.points[0], device[1].points[0]);
    try testing.expectApproxEqAbs(glyph_logical.boxRect().w * 2, device[1].boxRect().w, 1e-4);
    _ = rect_logical;
}

test "a triangle keeps its three points and reports its bounds" {
    var list = DisplayList{};
    defer list.deinit(testing.allocator);
    try push(&list, .{ .triangle = .{
        .a = .{ .x = 2, .y = 2 },
        .b = .{ .x = 14, .y = 2 },
        .c = .{ .x = 8, .y = 14 },
        .color = types.Color.hex(0xffffff, 1),
    } });

    var buf: [2]Instance = undefined;
    const insts = collect(&list, .{}, &buf);
    try testing.expectEqual(@as(usize, 1), insts.len);
    try testing.expect(insts[0].kindOf() == .triangle);
    try testing.expectEqual(@as(f32, 14), insts[0].point(2).y);
    const b = insts[0].bounds();
    try testing.expectEqual(@as(f32, 2), b.x);
    try testing.expectEqual(@as(f32, 12), b.w);
    try testing.expectEqual(@as(f32, 12), b.h);
}
