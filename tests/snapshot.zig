//! Snapshot testing: the reference is a file in the repository.
//!
//! A snapshot answers a different question from an assertion. "softmax sums to
//! one" is a property; "softmax of *this* tensor is *these* numbers" is a
//! reference — it catches a change in behaviour that no property notices, and it
//! catches it as a **diff a human reads in a pull request**, which is the whole
//! point.
//!
//! Two rules keep snapshots from becoming noise:
//!
//! * **They are text, in the repository, in a stable order.** `snapshots/` holds
//!   one file per case, written by this file and reviewed like any other source
//!   file. No binary blobs, no hashes of hashes: an image is a summary plus an
//!   ASCII rendition, a display list is its commands, a tensor is its numbers.
//! * **Numbers are compared with a tolerance, text exactly.** Floating point is
//!   not stable across machines and libm versions; a snapshot that demanded bit
//!   equality would be a snapshot people delete. Integers, shapes, dtypes,
//!   command streams and ASCII pictures *are* exact.
//!
//! Update mode is a build flag, so it is impossible to rewrite the references by
//! accident while running tests:
//!
//! ```
//! zig build test                       # compare against snapshots/
//! zig build test -Dupdate-snapshots    # rewrite them (then read the diff)
//! ```
const std = @import("std");
const build_options = @import("build_options");
const zgf = @import("zgf");
const zpu = @import("zpu");
const zui = @import("zui");

pub const root = "snapshots";

/// What a comparison found. `finish` fails the test if anything but `compared`
/// and `written` moved.
pub const Stats = struct {
    compared: usize = 0,
    written: usize = 0,
    missing: usize = 0,
    mismatched: usize = 0,
};

pub const Snap = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    update: bool,
    stats: Stats = .{},
    /// Mismatches are printed as they happen, so a failing run names every case
    /// rather than only the first.
    first_failure: ?[]const u8 = null,

    pub fn init(alloc: std.mem.Allocator) Snap {
        return .{
            .alloc = alloc,
            .io = std.testing.io,
            .update = build_options.snapshot_update,
        };
    }

    fn path(self: *Snap, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.alloc, "{s}/{s}.snap", .{ root, name });
    }

    fn read(self: *Snap, file: []const u8) ?[]u8 {
        return std.Io.Dir.cwd().readFileAlloc(self.io, file, self.alloc, max_snapshot_bytes) catch null;
    }

    fn write(self: *Snap, file: []const u8, text: []const u8) !void {
        if (std.fs.path.dirname(file)) |dir| std.Io.Dir.cwd().createDirPath(self.io, dir) catch {};
        try std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = file, .data = text });
    }

    fn record(self: *Snap, name: []const u8, why: []const u8) void {
        self.stats.mismatched += 1;
        if (self.first_failure == null) self.first_failure = name;
        std.debug.print("snapshot mismatch: {s}: {s}\n", .{ name, why });
    }

    /// Compares `text` with the stored snapshot, byte for byte.
    pub fn expectText(self: *Snap, name: []const u8, text: []const u8) !void {
        const file = try self.path(name);
        defer self.alloc.free(file);
        if (self.update) {
            try self.write(file, text);
            self.stats.written += 1;
            return;
        }
        const stored = self.read(file) orelse {
            self.stats.missing += 1;
            std.debug.print("snapshot missing: {s} — run `zig build test -Dupdate-snapshots`\n", .{name});
            return;
        };
        defer self.alloc.free(stored);
        if (!std.mem.eql(u8, stored, text)) {
            self.record(name, "text differs");
            printDiff(stored, text);
            return;
        }
        self.stats.compared += 1;
    }

    /// Formats with `fmt` and compares the result as text.
    pub fn expectFmt(self: *Snap, name: []const u8, comptime fmt: []const u8, args: anytype) !void {
        var buf = std.Io.Writer.Allocating.init(self.alloc);
        defer buf.deinit();
        try buf.writer.print(fmt, args);
        try self.expectText(name, buf.written());
    }

    /// Compares a tensor's values, elementwise, with a relative tolerance.
    ///
    /// The file keeps the shape and dtype in its header (exact) and the numbers
    /// below it (tolerant), so a change in *shape* is a hard failure and a change
    /// in the last few bits is not.
    pub fn expectTensor(self: *Snap, name: []const u8, t: zpu.Tensor, tol: f64) !void {
        var buf = std.Io.Writer.Allocating.init(self.alloc);
        defer buf.deinit();
        try writeTensor(&buf.writer, name, t);

        if (self.update) {
            const file = try self.path(name);
            defer self.alloc.free(file);
            try self.write(file, buf.written());
            self.stats.written += 1;
            return;
        }
        const file = try self.path(name);
        defer self.alloc.free(file);
        const stored = self.read(file) orelse {
            self.stats.missing += 1;
            std.debug.print("snapshot missing: {s} — run `zig build test -Dupdate-snapshots`\n", .{name});
            return;
        };
        defer self.alloc.free(stored);

        try self.compareNumbers(name, stored, buf.written(), t, tol);
    }

    fn compareNumbers(
        self: *Snap,
        name: []const u8,
        stored: []const u8,
        produced: []const u8,
        t: zpu.Tensor,
        tol: f64,
    ) !void {
        // The shape and dtype are written in the header and compared exactly: an
        // `f32` that became an `f64` is a real change even when every number
        // agrees.
        if (!std.mem.eql(u8, headerOf(stored), headerOf(produced))) {
            self.record(name, "shape or dtype differs");
            std.debug.print("  was: {s}\n  now: {s}\n", .{ headerOf(stored), headerOf(produced) });
            return;
        }
        // The values are compared against the *tensor*, not against the text this
        // run produced: the text is rounded for readability, the numbers are not.
        var want: std.ArrayList(f64) = .empty;
        defer want.deinit(self.alloc);
        try parseNumbers(self.alloc, &want, stored);

        const n = t.numel();
        if (want.items.len != n) {
            var msg: [128]u8 = undefined;
            const text = std.fmt.bufPrint(&msg, "snapshot holds {d} values, the tensor has {d}", .{
                want.items.len, n,
            }) catch "different number of values";
            self.record(name, text);
            return;
        }
        for (want.items, 0..) |w, i| {
            const g = t.readLogical(n, i);
            if (!close(w, g, tol)) {
                var msg: [160]u8 = undefined;
                const text = std.fmt.bufPrint(&msg, "element {d}: snapshot {e}, now {e}", .{ i, w, g }) catch "element differs";
                self.record(name, text);
                return;
            }
        }
        self.stats.compared += 1;
    }

    /// Compares an image: its size and channel count exactly, its statistics
    /// numerically, and a downsampled luminance picture glyph for glyph.
    pub fn expectImage(self: *Snap, name: []const u8, img: *zgf.Image, cols: usize, rows: usize) !void {
        var buf = std.Io.Writer.Allocating.init(self.alloc);
        defer buf.deinit();
        try writeImage(&buf.writer, name, img, cols, rows);
        try self.expectText(name, buf.written());
    }

    /// Compares a command list: one line per command, exactly.
    pub fn expectDisplayList(self: *Snap, name: []const u8, list: *const zui.DisplayList) !void {
        var buf = std.Io.Writer.Allocating.init(self.alloc);
        defer buf.deinit();
        _ = try buf.writer.print("# {s}\n", .{name});
        try writeDisplayList(&buf.writer, list);
        try self.expectText(name, buf.written());
    }

    /// Compares an instance stream: one line per instance, exactly. Clipped to
    /// `limit` lines so a large frame stays readable.
    pub fn expectInstances(self: *Snap, name: []const u8, insts: []const zui.Instance, limit: usize) !void {
        var buf = std.Io.Writer.Allocating.init(self.alloc);
        defer buf.deinit();
        _ = try buf.writer.print("# {s}\n", .{name});
        try writeInstances(&buf.writer, insts, limit);
        try self.expectText(name, buf.written());
    }

    /// Fails the test if any comparison above failed.
    pub fn finish(self: *Snap) !void {
        if (self.update) {
            if (self.stats.written > 0) {
                std.debug.print("snapshots: wrote {d} files under {s}/\n", .{ self.stats.written, root });
            }
            return;
        }
        if (self.stats.missing > 0) {
            std.debug.print("snapshots: {d} missing, {d} compared, {d} mismatched\n", .{
                self.stats.missing,
                self.stats.compared,
                self.stats.mismatched,
            });
            return error.SnapshotMissing;
        }
        if (self.stats.mismatched > 0) {
            std.debug.print("snapshots: {d} mismatched (first: {s})\n", .{
                self.stats.mismatched,
                self.first_failure orelse "?",
            });
            return error.SnapshotMismatch;
        }
    }
};

const max_snapshot_bytes: std.Io.Limit = .limited(8 << 20);

fn close(want: f64, got: f64, tol: f64) bool {
    if (std.math.isNan(want) or std.math.isNan(got)) return std.math.isNan(want) and std.math.isNan(got);
    if (std.math.isInf(want) or std.math.isInf(got)) return want == got;
    const scale = @max(1.0, @max(@abs(want), @abs(got)));
    return @abs(want - got) <= tol * scale;
}

/// The `# shape ... dtype ...` line, which is compared exactly.
fn headerOf(text: []const u8) []const u8 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (std.mem.indexOf(u8, line, "shape") != null) return line;
    }
    return "";
}

/// Collects the numbers of a snapshot body: every line that is not a comment.
fn parseNumbers(alloc: std.mem.Allocator, out: *std.ArrayList(f64), text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var toks = std.mem.tokenizeAny(u8, line, " \t\r");
        while (toks.next()) |tok| {
            const v = try std.fmt.parseFloat(f64, tok);
            try out.append(alloc, v);
        }
    }
}

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

/// `1.0000000` is `1`: trailing zeros are noise in a diff.
fn trimZeros(text: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, text, '.') == null) return text;
    var end = text.len;
    while (end > 0 and text[end - 1] == '0') end -= 1;
    if (end > 0 and text[end - 1] == '.') end -= 1;
    return text[0..end];
}

/// A tensor as text: a header the reader can check, then one row per line.
///
/// Rows are the *logical* rows of the last axis, so a snapshot diff shows which
/// row changed rather than which flat index did.
pub fn writeTensor(w: *std.Io.Writer, name: []const u8, t: zpu.Tensor) !void {
    try w.print("# {s}\n", .{name});
    const n = t.numel();
    if (n == 0) {
        try w.print("# shape (empty) dtype {s}\n", .{@tagName(t.dtype())});
        return;
    }
    const rank = t.rank();
    try w.print("# shape", .{});
    if (rank == 0) try w.print(" ()", .{});
    for (0..rank) |i| try w.print(" {d}", .{t.shape.dims[i]});
    try w.print(" dtype {s}\n", .{@tagName(t.dtype())});

    // A scalar prints as one value on one line; everything else as one logical
    // row of the last axis per line.
    const cols = if (rank == 0) 1 else t.shape.dims[rank - 1];
    for (0..n) |i| {
        try writeNumber(w, t.readLogical(n, i));
        if ((i + 1) % cols == 0) try w.writeByte('\n') else try w.writeByte(' ');
    }
}

/// Numbers are printed *exactly* (the text parses back to the same bits), but
/// in the shortest form that does: hand-written values like `0.5` or `2` stay
/// readable, and a computed value keeps all seventeen of its digits.
///
/// Writing the value exactly is what lets the comparison be tolerant without
/// being lenient: the tolerance absorbs a different libm or summation order,
/// while the file stays a faithful record of what the code produced.
pub fn writeNumber(w: *std.Io.Writer, v: f64) !void {
    if (std.math.isNan(v)) return w.writeAll("nan");
    if (std.math.isInf(v)) return w.writeAll(if (v > 0) "inf" else "-inf");
    if (v == 0) return w.writeAll("0");

    // `{d:.7}` is the shortest thing a reader wants to look at; if it does not
    // round-trip, the exact scientific form wins.
    var buf: [64]u8 = undefined;
    if (@abs(v) >= 1e-3 and @abs(v) < 1e7) {
        const printed = std.fmt.bufPrint(&buf, "{d:.7}", .{v}) catch unreachable;
        const short = trimZeros(printed);
        if (std.fmt.parseFloat(f64, short) catch null) |back| {
            if (@as(u64, @bitCast(back)) == @as(u64, @bitCast(v))) return w.writeAll(short);
        }
    }
    return w.print("{e}", .{v});
}

/// An image as text: size and channel count, then statistics, then a luminance
/// rendition — small enough to read in a diff, detailed enough that a moved line
/// or a changed colour shows up.
pub fn writeImage(w: *std.Io.Writer, name: []const u8, img: *zgf.Image, cols: usize, rows: usize) !void {
    try w.print("# {s}\n", .{name});
    try w.print("size {d} {d} channels {d}\n", .{ img.width(), img.height(), img.channels() });
    const W = img.width();
    const H = img.height();

    var sum: f64 = 0;
    var lo: f64 = std.math.inf(f64);
    var hi: f64 = -std.math.inf(f64);
    for (0..H) |y| {
        for (0..W) |x| {
            const v = luminance(img, x, y);
            sum += v;
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
    }
    const mean = if (W * H == 0) 0 else sum / @as(f64, @floatFromInt(W * H));
    try w.print("luma min {d:.6} max {d:.6} mean {d:.6}\n", .{ lo, hi, mean });

    // Box-downsample to `cols × rows`, then one glyph per cell. Adding the cells
    // up is not the same as the mean of the whole image (a rounded box is a
    // weighted average), but it is *stable*, which is what a snapshot needs.
    const ramp = " .:-=+*#%@";
    for (0..rows) |ry| {
        for (0..cols) |rx| {
            const x0 = rx * W / cols;
            const x1 = @max(x0 + 1, (rx + 1) * W / cols);
            const y0 = ry * H / rows;
            const y1 = @max(y0 + 1, (ry + 1) * H / rows);
            var acc: f64 = 0;
            var count: usize = 0;
            for (y0..@min(y1, H)) |y| {
                for (x0..@min(x1, W)) |x| {
                    acc += luminance(img, x, y);
                    count += 1;
                }
            }
            const v = if (count == 0) 0 else acc / @as(f64, @floatFromInt(count));
            const idx: usize = @intFromFloat(std.math.clamp(v, 0, 1) * @as(f64, @floatFromInt(ramp.len - 1)));
            try w.writeByte(ramp[@min(idx, ramp.len - 1)]);
        }
        try w.writeByte('\n');
    }
}

fn luminance(img: *zgf.Image, x: usize, y: usize) f64 {
    const c = img.getPixel(x, y);
    if (img.channels() == 1) return c.r;
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

/// A display list, one command per line, in the units the widgets used.
pub fn writeDisplayList(w: *std.Io.Writer, list: *const zui.DisplayList) !void {
    for (list.commands.items) |cmd| {
        switch (cmd) {
            .rect => |c| try w.print("rect {d:.2},{d:.2} {d:.2}x{d:.2} {s}\n", .{
                c.r.x, c.r.y, c.r.w, c.r.h, try colorText(c.color),
            }),
            .rounded => |c| try w.print("rounded {d:.2},{d:.2} {d:.2}x{d:.2} r={d:.2} {s} border={d:.2} {s}\n", .{
                c.r.x, c.r.y, c.r.w, c.r.h, c.radius, try colorText(c.color), c.border, try colorText(c.border_color),
            }),
            .line => |c| try w.print("line {d:.2},{d:.2} -> {d:.2},{d:.2} w={d:.2} {s}\n", .{
                c.a.x, c.a.y, c.b.x, c.b.y, c.width, try colorText(c.color),
            }),
            .triangle => |c| try w.print("triangle {d:.2},{d:.2} {d:.2},{d:.2} {d:.2},{d:.2} {s}\n", .{
                c.a.x, c.a.y, c.b.x, c.b.y, c.c.x, c.c.y, try colorText(c.color),
            }),
            .text => |c| try w.print("text \"{s}\" at {d:.2},{d:.2} size={d:.1} {s}\n", .{
                c.text, c.pos.x, c.pos.y, c.size, try colorText(c.color),
            }),
            .push_clip => |r| try w.print("push_clip {d:.2},{d:.2} {d:.2}x{d:.2}\n", .{ r.x, r.y, r.w, r.h }),
            .pop_clip => try w.print("pop_clip\n", .{}),
        }
    }
}

fn colorText(c: zui.Color) ![]const u8 {
    const S = struct {
        var buf: [16]u8 = undefined;
        fn hex(v: f32) u8 {
            return @intFromFloat(std.math.clamp(v, 0, 1) * 255.0 + 0.5);
        }
    };
    return std.fmt.bufPrint(&S.buf, "#{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
        S.hex(c.r), S.hex(c.g), S.hex(c.b), S.hex(c.a),
    }) catch "#????????";
}

/// A display list as a *tree*: containers open a clip, so the clip nesting is the
/// layout structure that a flat command stream hides.
///
/// This is the artifact to reach for when a frame moves: the display list says
/// which coordinates changed, the tree says which box did.
pub const Tree = struct {
    list: *const zui.DisplayList,
    name: []const u8 = "root",

    pub fn format(self: Tree, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const b = self.list.bounds;
        try w.print("{s}: {d} commands, bounds {d:.1},{d:.1} {d:.1}x{d:.1}\n", .{
            self.name, self.list.len(), b.x, b.y, b.w, b.h,
        });
        var depth: usize = 0;
        for (self.list.commands.items) |cmd| {
            if (cmd == .pop_clip) depth -|= 1;
            try writeIndent(w, depth);
            switch (cmd) {
                .rect => |c| try w.print("{s} {d:.1},{d:.1} {d:.1}x{d:.1}\n", .{ "rect", c.r.x, c.r.y, c.r.w, c.r.h }),
                .rounded => |c| try w.print("{s} {d:.1},{d:.1} {d:.1}x{d:.1} r={d:.1}\n", .{ "rounded", c.r.x, c.r.y, c.r.w, c.r.h, c.radius }),
                .line => |c| try w.print("{s} ({d:.1},{d:.1})->({d:.1},{d:.1})\n", .{ "line", c.a.x, c.a.y, c.b.x, c.b.y }),
                .triangle => |c| try w.print("{s} ({d:.1},{d:.1}) ({d:.1},{d:.1}) ({d:.1},{d:.1})\n", .{ "triangle", c.a.x, c.a.y, c.b.x, c.b.y, c.c.x, c.c.y }),
                .text => |c| try w.print("{s} \"{s}\" {d:.1},{d:.1}\n", .{ "text", c.text, c.pos.x, c.pos.y }),
                .push_clip => |r| {
                    try w.print("clip {d:.1},{d:.1} {d:.1}x{d:.1}\n", .{ r.x, r.y, r.w, r.h });
                    depth += 1;
                },
                .pop_clip => try w.print("{s}\n", .{"end"}),
            }
        }
    }
};

fn writeIndent(w: *std.Io.Writer, depth: usize) !void {
    for (0..depth) |_| try w.writeAll("  ");
}

pub fn writeInstances(w: *std.Io.Writer, insts: []const zui.Instance, limit: usize) !void {
    const n = @min(insts.len, limit);
    try w.print("instances {d}\n", .{insts.len});
    for (insts[0..n]) |inst| {
        const b = inst.box;
        try w.print("{s} {d:.2},{d:.2} {d:.2}x{d:.2} {s} clip {d:.0},{d:.0} {d:.0}x{d:.0} p={d:.2}\n", .{
            @tagName(inst.kindOf()), b[0], b[1], b[2], b[3], try colorText(inst.colorOf()),
            inst.clip[0], inst.clip[1], inst.clip[2], inst.clip[3], inst.param,
        });
    }
    if (insts.len > n) try w.print("... {d} more\n", .{insts.len - n});
}

/// Prints the first differing line of a text mismatch: enough to see what
/// changed, short enough to stay readable.
fn printDiff(want: []const u8, got: []const u8) void {
    var wi = std.mem.splitScalar(u8, want, '\n');
    var gi = std.mem.splitScalar(u8, got, '\n');
    var line: usize = 0;
    while (true) : (line += 1) {
        const wl = wi.next();
        const gl = gi.next();
        if (wl == null and gl == null) return;
        const wl_txt = wl orelse "<end>";
        const gl_txt = gl orelse "<end>";
        if (!std.mem.eql(u8, wl_txt, gl_txt)) {
            std.debug.print("  line {d}:\n    snapshot: {s}\n    current:  {s}\n", .{ line + 1, wl_txt, gl_txt });
            return;
        }
    }
}
