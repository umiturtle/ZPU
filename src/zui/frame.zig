//! The display list: the one thing `zui` keeps from a frame.
//!
//! Widget calls do not draw. They *append* to a command list, which is the
//! interface between the UI layer and whatever turns pixels on: the software
//! renderer in `zui/render.zig` today, an instance buffer for a GPU backend
//! later. Keeping this boundary is what makes the library hybrid rather than
//! purely immediate: the list can be replayed, compared, cached, walked for a
//! test, or uploaded — none of which is possible if widgets draw directly.
//!
//! Every slice in a command points into the frame arena (owned by `Ui`), never
//! into caller memory: a command list stays valid after the caller's strings go
//! out of scope, and it costs nothing to build.
const std = @import("std");
const types = @import("types.zig");

const Color = types.Color;
const Rect = types.Rect;
const Vec2 = types.Vec2;

pub const Command = union(enum) {
    /// A solid axis-aligned rectangle.
    rect: struct { r: Rect, color: Color },
    /// A rectangle with rounded corners, and optionally a border of `border`
    /// pixels in `border_color` drawn inside it. Anti-aliased by the renderer,
    /// so `radius` may be fractional.
    rounded: struct { r: Rect, radius: f32, color: Color, border: f32, border_color: Color },
    line: struct { a: Vec2, b: Vec2, width: f32, color: Color },
    triangle: struct { a: Vec2, b: Vec2, c: Vec2, color: Color },
    /// A run of text: shaped by the renderer with the same metrics the layout
    /// used, so measuring and drawing cannot disagree.
    text: struct { pos: Vec2, size: f32, color: Color, text: []const u8 },
    /// Clips everything that follows to `r`, until the matching `pop_clip`.
    push_clip: Rect,
    pop_clip,

    pub fn kindName(self: Command) []const u8 {
        return @tagName(self);
    }
};

pub const DisplayList = struct {
    commands: std.ArrayListUnmanaged(Command) = .empty,
    /// Number of `push_clip` commands that have not been popped.
    clip_depth: usize = 0,
    /// Everything drawn, intersected with every clip in force.
    bounds: Rect = .{},
    /// Increments once per rebuild: the identity of this picture. Two frames
    /// with the same revision are the same list, which is how the idle path
    /// knows it may skip.
    revision: u64 = 0,
    /// Where the list was empty because the frame was skipped, this is the
    /// revision it was copied from.
    reused: bool = false,

    pub fn deinit(self: *DisplayList, alloc: std.mem.Allocator) void {
        self.commands.deinit(alloc);
        self.* = .{};
    }

    /// Keeps the capacity: a frame's worth of memory is reused forever.
    pub fn reset(self: *DisplayList) void {
        self.commands.clearRetainingCapacity();
        self.clip_depth = 0;
        self.bounds = .{};
    }

    pub fn push(self: *DisplayList, alloc: std.mem.Allocator, cmd: Command) !void {
        switch (cmd) {
            .push_clip => |r| {
                self.clip_depth += 1;
                const b = if (self.clip_depth == 1) r else self.bounds.intersect(r);
                self.bounds = b;
            },
            .pop_clip => {
                // The renderer keeps a stack for the nested case; `bounds` here
                // is only the coarse "is anything on screen at all" check, so
                // undoing a clip exactly is not required.
                self.clip_depth -|= 1;
            },
            else => {},
        }
        try self.commands.append(alloc, cmd);
    }

    pub fn len(self: *const DisplayList) usize {
        return self.commands.items.len;
    }

    pub fn isBalanced(self: *const DisplayList) bool {
        return self.clip_depth == 0;
    }

    /// How many commands of each kind the list holds: used by the tests to pin
    /// down "one button is exactly these commands", and by the benchmark to
    /// report a frame's shape.
    pub fn histogram(self: *const DisplayList, out: *[8]usize) void {
        @memset(out, 0);
        for (self.commands.items) |c| {
            const i: usize = switch (c) {
                .rect => 0,
                .rounded => 1,
                .line => 2,
                .triangle => 3,
                .text => 4,
                .push_clip => 5,
                .pop_clip => 6,
            };
            out[i] += 1;
        }
    }
};

/// One draw-ready instance: everything needed to paint one shape, in a record a
/// GPU can read as-is.
///
/// The display list is the *authoring* format — commands carry a clip stack and
/// a text run, which are not things a shader can be handed. The instance stream
/// is the *drawing* format: one shape, one bounding box, one clip rectangle, one
/// colour, fixed size, no pointers, no nesting. `zui/instances.zig` does the
/// conversion, `zui/raster.zig` consumes it on the CPU, and a backend can upload
/// it unchanged.
///
/// The record is 20 floats (80 bytes), which is 16-byte aligned — the shape a
/// `storage`/`vertex` buffer wants, with no packing step between here and a
/// shader.
pub const Instance = struct {
    /// Bounding box in the host's space: x, y, w, h. Always the true bounds of
    /// what is drawn — a text run's glyph box, a line's capsule — so a host can
    /// cull with it, and so a vertex shader can emit one quad from it.
    box: [4]f32 = .{ 0, 0, 0, 0 },
    color: [4]f32 = .{ 0, 0, 0, 0 },
    /// Clip rectangle in the same space, or `no_clip`. Carried per instance
    /// rather than as a frame-wide uniform because the display list nests clips
    /// (a scroll view inside a window) and a scrolling viewport is exactly the
    /// case where clipping is load-bearing rather than decorative.
    clip: [4]f32 = no_clip,
    /// `rounded`: the corner radius. `line`: the stroke width. Otherwise 0.
    param: f32 = 0,
    kind: u32 = @backingInt(Kind.rect),
    /// Extra geometry, in the same space as `box`, read per kind:
    /// `glyph` = the atlas `uv` (x0, y0, x1, y1, ·, ·); `line` = the two
    /// endpoints; `triangle` = the three vertices.
    points: [6]f32 = .{ 0, 0, 0, 0, 0, 0 },

    /// No clipping: a rectangle wider than any frame. Kept as data rather than
    /// an option so a shader needs no branch.
    pub const no_clip: [4]f32 = .{ -1e9, -1e9, 2e9, 2e9 };
    pub const no_clip_rect: Rect = .{ .x = -1e9, .y = -1e9, .w = 2e9, .h = 2e9 };

    pub const Kind = enum(u32) {
        /// A solid axis-aligned rectangle.
        rect = 0,
        /// A rectangle with rounded corners; `param` is the radius.
        rounded = 1,
        /// One glyph, sampled from the font atlas through `points[0..4]`.
        glyph = 2,
        /// A stroked segment; `param` is the width, `points[0..4]` the ends.
        line = 3,
        /// A triangle, `points[0..6]` its vertices.
        triangle = 4,
    };

    pub fn kindOf(self: Instance) Kind {
        return @fromBackingInt(@intCast(self.kind));
    }

    pub fn boxRect(self: Instance) Rect {
        return .{ .x = self.box[0], .y = self.box[1], .w = self.box[2], .h = self.box[3] };
    }

    pub fn clipRect(self: Instance) Rect {
        return .{ .x = self.clip[0], .y = self.clip[1], .w = self.clip[2], .h = self.clip[3] };
    }

    pub fn colorOf(self: Instance) Color {
        return Color.fromArray(self.color);
    }

    /// Point `i` of the payload, as a position. Only meaningful for `line` and
    /// `triangle`.
    pub fn point(self: Instance, i: usize) Vec2 {
        return .{ .x = self.points[i * 2], .y = self.points[i * 2 + 1] };
    }

    pub fn withClip(self: Instance, r: Rect) Instance {
        var out = self;
        out.clip = .{ r.x, r.y, r.w, r.h };
        return out;
    }

    /// The bounds a rasteriser must span. For everything but a triangle `box` is
    /// already it; a triangle carries three points and no box, because six
    /// floats of geometry is all the payload has room for and a triangle is not
    /// what a UI is made of.
    pub fn bounds(self: Instance) Rect {
        if (self.kindOf() != .triangle) return self.boxRect();
        const a = self.point(0);
        const b = self.point(1);
        const c = self.point(2);
        const x = @min(a.x, @min(b.x, c.x));
        const y = @min(a.y, @min(b.y, c.y));
        return .{ .x = x, .y = y, .w = @max(a.x, @max(b.x, c.x)) - x, .h = @max(a.y, @max(b.y, c.y)) - y };
    }

    /// The same instance in device pixels: `p' = origin + p * scale`, with `uv`
    /// and other non-positional payloads left alone.
    pub fn scaled(self: Instance, origin: Vec2, scale: f32) Instance {
        var out = self;
        out.box = scaleBox(self.box, origin, scale);
        out.clip = scaleBox(self.clip, origin, scale);
        switch (self.kindOf()) {
            .rect => {},
            .rounded => out.param = self.param * scale,
            .glyph => {}, // `points` is a texture coordinate, not a position.
            .line => {
                out.param = self.param * scale;
                for (0..2) |i| {
                    const p = self.point(i);
                    out.points[i * 2] = origin.x + p.x * scale;
                    out.points[i * 2 + 1] = origin.y + p.y * scale;
                }
            },
            .triangle => for (0..3) |i| {
                const p = self.point(i);
                out.points[i * 2] = origin.x + p.x * scale;
                out.points[i * 2 + 1] = origin.y + p.y * scale;
            },
        }
        return out;
    }

    fn scaleBox(b: [4]f32, origin: Vec2, scale: f32) [4]f32 {
        return .{
            origin.x + b[0] * scale,
            origin.y + b[1] * scale,
            b[2] * scale,
            b[3] * scale,
        };
    }
};

test "display list bookkeeping" {
    const alloc = std.testing.allocator;
    var list = DisplayList{};
    defer list.deinit(alloc);

    try list.push(alloc, .{ .rect = .{ .r = Rect.init(1, 2, 3, 4), .color = Color.hex(0xff0000, 1) } });
    try list.push(alloc, .{ .push_clip = Rect.init(0, 0, 10, 10) });
    try list.push(alloc, .{ .text = .{ .pos = .{ .x = 1, .y = 2 }, .size = 14, .color = Color.hex(0xffffff, 1), .text = "hi" } });
    try std.testing.expectEqual(@as(usize, 3), list.len());
    try std.testing.expect(!list.isBalanced());
    try std.testing.expectEqual(@as(f32, 10), list.bounds.w);

    try list.push(alloc, .{ .pop_clip = {} });
    try std.testing.expect(list.isBalanced());

    var hist: [8]usize = undefined;
    list.histogram(&hist);
    try std.testing.expectEqual(@as(usize, 1), hist[0]);
    try std.testing.expectEqual(@as(usize, 1), hist[4]);
    try std.testing.expectEqual(@as(usize, 0), hist[1]);

    list.reset();
    try std.testing.expectEqual(@as(usize, 0), list.len());
    try std.testing.expectEqual(@as(f32, 0), list.bounds.w);
}
