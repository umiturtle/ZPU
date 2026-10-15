//! ZGF — a general purpose graphics library for Zig.
//!
//! ZGF keeps images in the *same* tensor type ZPU computes with, so a filter is
//! just an expression and can be differentiated like any other function:
//!
//! ```zig
//! const zgf = @import("zgf");
//! const zpu = @import("zpu");
//!
//! var scope = try zpu.Scope.init(alloc);
//! defer scope.deinit();
//!
//! var img = try zgf.Image.init(alloc, 64, 64, 3);
//! defer img.deinit();
//! try img.fill(.{ 0.1, 0.2, 0.3 });
//! try img.drawCircle(32, 32, 10, .{ 1, 1, 0 });
//!
//! const soft = try img.blur(2);      // a plain tensor operation
//! const edged = try img.sobel();
//! const encoded = try img.toPpm(alloc);   // P6, no dependency
//! ```
//!
//! Layers already here:
//!
//! * `Image`     — RGBA/RGB/mono f32 image over a `zpu.Tensor` ([H, W, C] in 0..1)
//! * drawing    — `fill`, `drawRect`, `drawCircle`, `drawLine`, `setPixel`
//! * filters    — `blur`, `sobel`, `sharpen`, `brightness`, `contrast`
//! * transfer   — `toPixels`/`fromPixels` (u8 sRGB), `toPpm`/`savePpm`
//!
//! GPU execution will come through the shared `zbackend` layer
//! (`zbackend.Kernel`); the CPU path here is complete and testable headless.
const std = @import("std");
const zpu = @import("zpu");

pub const version = "0.1.0";

pub const Color = struct { r: f32, g: f32, b: f32, a: f32 = 1 };
pub const Error = error{ InvalidImage, OutOfRange, TooLarge, IoFailed } || zpu.tensor.Error || std.mem.Allocator.Error;

/// An image backed by a `zpu.Tensor` of shape `[H, W, C]` in `0..1`, row major,
/// origin at the top left.
///
/// The image owns its tensor: `untrack` is called on creation so the image
/// survives (and is released) independently of whatever `Scope` happens to be
/// active when it is created.
pub const Image = struct {
    pixels: zpu.Tensor,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, w: usize, h: usize, c: usize) Error!Image {
        if (w == 0 or h == 0) return Error.InvalidImage;
        if (c != 1 and c != 3 and c != 4) return Error.InvalidImage;
        const shape = try zpu.Shape.init(&.{ h, w, c });
        const t = try zpu.Tensor.zeros(shape, .f32);
        zpu.untrack(t); // the image owns it, not the ambient scope
        return .{ .pixels = t, .alloc = alloc };
    }

    pub fn deinit(self: *Image) void {
        self.pixels.deinit();
    }

    pub fn width(self: *const Image) usize {
        return self.pixels.shape.dims[1];
    }

    pub fn height(self: *const Image) usize {
        return self.pixels.shape.dims[0];
    }

    pub fn channels(self: *const Image) usize {
        return self.pixels.shape.dims[2];
    }

    pub fn numel(self: *const Image) usize {
        return self.pixels.numel();
    }

    fn index(self: *const Image, x: usize, y: usize, c: usize) usize {
        return (y * self.width() + x) * self.channels() + c;
    }

    /// Writes one pixel (out of range coordinates are ignored).
    pub fn setPixel(self: *Image, x: isize, y: isize, color: Color) void {
        if (x < 0 or y < 0) return;
        const ux: usize = @intCast(x);
        const uy: usize = @intCast(y);
        if (ux >= self.width() or uy >= self.height()) return;
        const c = self.channels();
        const base = self.index(ux, uy, 0);
        switch (c) {
            1 => self.pixels.writeF64(base, 0.299 * @as(f64, color.r) + 0.587 * @as(f64, color.g) + 0.114 * @as(f64, color.b)),
            3 => {
                self.pixels.writeF64(base, color.r);
                self.pixels.writeF64(base + 1, color.g);
                self.pixels.writeF64(base + 2, color.b);
            },
            else => {
                self.pixels.writeF64(base, color.r);
                self.pixels.writeF64(base + 1, color.g);
                self.pixels.writeF64(base + 2, color.b);
                self.pixels.writeF64(base + 3, color.a);
            },
        }
    }

    /// Alpha-composites `src` *over* the existing pixel: the operation a
    /// transparent UI needs and an opaque `setPixel` cannot express.
    ///
    /// Straight (non-premultiplied) alpha, because that is what a colour picker
    /// hands a programmer: `out = src.rgb * a + dst.rgb * (1 - a)`.
    pub fn blendPixel(self: *Image, x: usize, y: usize, src: Color) void {
        if (x >= self.width() or y >= self.height()) return;
        self.blendCoverage(x, y, src, 1);
    }

    /// `blendPixel` with the source alpha scaled by `coverage` in `0..1`: what
    /// an antialiased edge or a font's coverage byte needs. Coverage is a
    /// separate factor from the colour's own alpha, so a translucent shape still
    /// gets a soft edge.
    pub fn blendCoverage(self: *Image, x: usize, y: usize, src: Color, coverage: f32) void {
        const a = src.a * std.math.clamp(coverage, 0, 1);
        if (a <= 0) return;
        if (x >= self.width() or y >= self.height()) return;
        if (a >= 1) {
            self.setPixel(@intCast(x), @intCast(y), src);
            return;
        }
        const dst = self.getPixel(x, y);
        self.setPixel(@intCast(x), @intCast(y), .{
            .r = dst.r + (src.r - dst.r) * a,
            .g = dst.g + (src.g - dst.g) * a,
            .b = dst.b + (src.b - dst.b) * a,
            .a = @max(dst.a, a + dst.a * (1 - a)),
        });
    }

    pub fn getPixel(self: *const Image, x: usize, y: usize) Color {
        const c = self.channels();
        const base = self.index(x, y, 0);
        return switch (c) {
            1 => blk: {
                const v: f32 = @floatCast(self.pixels.readF64(base));
                break :blk Color{ .r = v, .g = v, .b = v };
            },
            3 => .{
                .r = @floatCast(self.pixels.readF64(base)),
                .g = @floatCast(self.pixels.readF64(base + 1)),
                .b = @floatCast(self.pixels.readF64(base + 2)),
            },
            else => .{
                .r = @floatCast(self.pixels.readF64(base)),
                .g = @floatCast(self.pixels.readF64(base + 1)),
                .b = @floatCast(self.pixels.readF64(base + 2)),
                .a = @floatCast(self.pixels.readF64(base + 3)),
            },
        };
    }

    pub fn fill(self: *Image, color: Color) void {
        for (0..self.height()) |y| {
            for (0..self.width()) |x| self.setPixel(@intCast(x), @intCast(y), color);
        }
    }

    /// Axis aligned rectangle from `(x0, y0)` inclusive to `(x1, y1)` exclusive.
    pub fn drawRect(self: *Image, x0: isize, y0: isize, x1: isize, y1: isize, color: Color) void {
        var y = y0;
        while (y < y1) : (y += 1) {
            var x = x0;
            while (x < x1) : (x += 1) self.setPixel(x, y, color);
        }
    }

    /// Filled circle (midpoint test, no anti-aliasing).
    pub fn drawCircle(self: *Image, cx: isize, cy: isize, radius: isize, color: Color) void {
        const r2 = radius * radius;
        var y = cy - radius;
        while (y <= cy + radius) : (y += 1) {
            var x = cx - radius;
            while (x <= cx + radius) : (x += 1) {
                const dx = x - cx;
                const dy = y - cy;
                if (dx * dx + dy * dy <= r2) self.setPixel(x, y, color);
            }
        }
    }

    /// Bresenham line, one pixel wide.
    pub fn drawLine(self: *Image, x0: isize, y0: isize, x1: isize, y1: isize, color: Color) void {
        self.drawLineThick(x0, y0, x1, y1, 1, color);
    }

    /// Bresenham line drawn with a square brush of `thickness` pixels.
    pub fn drawLineThick(self: *Image, x0: isize, y0: isize, x1: isize, y1: isize, thickness: usize, color: Color) void {
        const t: isize = @intCast(@max(thickness, 1));
        const lo = -@divTrunc(t - 1, 2); // brush offsets: lo .. lo + t - 1
        const hi = lo + t - 1;

        var x = x0;
        var y = y0;
        const dx: isize = @intCast(@abs(x1 - x0));
        const dy: isize = -@as(isize, @intCast(@abs(y1 - y0)));
        const sx: isize = if (x0 < x1) 1 else -1;
        const sy: isize = if (y0 < y1) 1 else -1;
        var err = dx + dy;
        while (true) {
            var oy = lo;
            while (oy <= hi) : (oy += 1) {
                var ox = lo;
                while (ox <= hi) : (ox += 1) self.setPixel(x + ox, y + oy, color);
            }
            if (x == x1 and y == y1) break;
            const e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y += sy;
            }
        }
    }

    /// A linear gradient between two colours (top to bottom).
    pub fn verticalGradient(self: *Image, top: Color, bottom: Color) void {
        const h = self.height();
        for (0..h) |y| {
            const t: f32 = if (h == 1) 0 else @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(h - 1));
            const color = Color{
                .r = top.r + (bottom.r - top.r) * t,
                .g = top.g + (bottom.g - top.g) * t,
                .b = top.b + (bottom.b - top.b) * t,
                .a = top.a + (bottom.a - top.a) * t,
            };
            for (0..self.width()) |x| self.setPixel(@intCast(x), @intCast(y), color);
        }
    }

    // ------------------------------------------------------------- filters
    //
    // Filters go through `zpu` operations, so they take a `Scope` (or the
    // caller keeps whatever the ops produce) and they are differentiable.

    /// Gaussian blur with a separable kernel of radius `radius` (σ = radius/2).
    pub fn blur(self: *const Image, radius: usize) Error!zpu.Tensor {
        if (radius == 0) return zpu.addScalar(self.pixels, 0);
        const kernel = try gaussianKernel(self.alloc, radius);
        defer zpu.releaseTemp(kernel);
        const h = try self.convolve1d(kernel, 0);
        defer zpu.releaseTemp(h);
        return self.convolve1d(kernel, 1);
    }

    /// Sobel edge magnitude, per channel.
    pub fn sobel(self: *const Image) Error!zpu.Tensor {
        const gx = try zpu.Tensor.fromArray(f32, &.{ -1, 0, 1, -2, 0, 2, -1, 0, 1 }, zpu.Shape.initComptime(&.{ 3, 3 }));
        defer zpu.releaseTemp(gx);
        const gy = try zpu.Tensor.fromArray(f32, &.{ -1, -2, -1, 0, 0, 0, 1, 2, 1 }, zpu.Shape.initComptime(&.{ 3, 3 }));
        defer zpu.releaseTemp(gy);
        const dx = try self.convolve2d(gx);
        defer zpu.releaseTemp(dx);
        const dy = try self.convolve2d(gy);
        defer zpu.releaseTemp(dy);
        const sx = try zpu.square(dx);
        defer zpu.releaseTemp(sx);
        const sy = try zpu.square(dy);
        defer zpu.releaseTemp(sy);
        const sum = try zpu.add(sx, sy);
        defer zpu.releaseTemp(sum);
        return zpu.sqrt(sum);
    }

    /// Unsharp mask: `x + amount * (x - blur(x))`.
    pub fn sharpen(self: *const Image, radius: usize, amount: f32) Error!zpu.Tensor {
        const soft = try self.blur(radius);
        defer zpu.releaseTemp(soft);
        const diff = try zpu.sub(self.pixels, soft);
        defer zpu.releaseTemp(diff);
        const scaled = try zpu.mulScalar(diff, amount);
        defer zpu.releaseTemp(scaled);
        const sharp = try zpu.add(self.pixels, scaled);
        defer zpu.releaseTemp(sharp);
        return zpu.clamp(sharp, 0, 1);
    }

    pub fn brightness(self: *const Image, factor: f32) Error!zpu.Tensor {
        const scaled = try zpu.mulScalar(self.pixels, factor);
        defer zpu.releaseTemp(scaled);
        return zpu.clamp(scaled, 0, 1);
    }

    /// `(x - 0.5) * factor + 0.5`, clamped to `0..1`.
    pub fn contrast(self: *const Image, factor: f32) Error!zpu.Tensor {
        const shifted = try zpu.subScalar(self.pixels, 0.5);
        defer zpu.releaseTemp(shifted);
        const scaled = try zpu.mulScalar(shifted, factor);
        defer zpu.releaseTemp(scaled);
        const back = try zpu.addScalar(scaled, 0.5);
        defer zpu.releaseTemp(back);
        return zpu.clamp(back, 0, 1);
    }

    /// Separable convolution over the x axis (`axis = 1`) or y axis (`axis = 0`)
    /// with zero padding, applied to every channel.
    fn convolve1d(self: *const Image, kernel: zpu.Tensor, axis: usize) Error!zpu.Tensor {
        const k = kernel.numel();
        const r = (k - 1) / 2;
        const w = self.width();
        const h = self.height();
        const c = self.channels();

        // pad the axis and gather the shifted windows with `sliceRanges`
        var pads: [3][2]usize = @splat(.{ 0, 0 });
        pads[axis] = .{ r, r };
        const padded = try zpu.pad(self.pixels, &pads, 0);
        defer zpu.releaseTemp(padded);

        var acc: ?zpu.Tensor = null;
        errdefer if (acc) |a| zpu.releaseTemp(a);
        var i: usize = 0;
        while (i < k) : (i += 1) {
            const start: i64 = @intCast(i);
            const ranges = switch (axis) {
                0 => [_]zpu.Range{
                    .{ .start = start, .end = start + @as(i64, @intCast(h)) },
                    zpu.Range.all(),
                    zpu.Range.all(),
                },
                else => [_]zpu.Range{
                    zpu.Range.all(),
                    .{ .start = start, .end = start + @as(i64, @intCast(w)) },
                    zpu.Range.all(),
                },
            };
            const window = try zpu.sliceRanges(padded, &ranges);
            const weight = kernel.readF64(i);
            const scaled = try zpu.mulScalar(window, weight);
            zpu.releaseTemp(window);
            if (acc) |a| {
                const sum = try zpu.add(a, scaled);
                zpu.releaseTemp(a);
                zpu.releaseTemp(scaled);
                acc = sum;
            } else {
                acc = scaled; // the accumulator takes the creation reference
            }
        }
        _ = c;
        return acc.?;
    }

    /// Full 2-D convolution over a `[H, W]` kernel (one or three channels).
    fn convolve2d(self: *const Image, kernel: zpu.Tensor) Error!zpu.Tensor {
        const w = self.width();
        const h = self.height();
        const c = self.channels();
        const kh = kernel.shape.dims[0];
        const kw = kernel.shape.dims[1];
        const r = kh / 2;

        // build a depthwise NCHW kernel: (C, C, kh, kw)
        const kshape = try zpu.Shape.init(&.{ c, c, kh, kw });
        const k4 = try zpu.Tensor.zeros(kshape, .f32);
        defer zpu.releaseTemp(k4);
        for (0..c) |ch| {
            for (0..kh) |y| {
                for (0..kw) |x| {
                    const v = kernel.readF64(y * kw + x);
                    const idx = ((ch * c + ch) * kh + y) * kw + x;
                    k4.writeF64(idx, v);
                }
            }
        }

        // image -> (1, C, H, W)
        const perm = [_]usize{ 2, 0, 1 };
        const nchw = try zpu.transpose(self.pixels, &perm);
        defer zpu.releaseTemp(nchw);
        const flat_shape = try zpu.Shape.init(&.{ 1, c, h, w });
        const flat = try zpu.reshape(nchw, flat_shape);
        defer zpu.releaseTemp(flat);

        const conv = try zpu.nn.conv2d(flat, k4, 1, r);
        defer zpu.releaseTemp(conv);
        // (1, C, H, W) -> [H, W, C]
        const again = try zpu.reshape(conv, try zpu.Shape.init(&.{ c, h, w }));
        defer zpu.releaseTemp(again);
        const back = [_]usize{ 1, 2, 0 };
        return zpu.transpose(again, &back);
    }

    // ------------------------------------------------------------- transfer

    /// 8 bit sRGB view of the image, `[H, W, C]` bytes.
    pub fn toPixels(self: *const Image, alloc: std.mem.Allocator) Error![]u8 {
        const n = self.numel();
        const out = try alloc.alloc(u8, n);
        errdefer alloc.free(out);
        for (0..n) |i| {
            const v = self.pixels.readF64(i);
            out[i] = @intFromFloat(@round(std.math.clamp(v, 0, 1) * 255.0));
        }
        return out;
    }

    pub fn fromPixels(alloc: std.mem.Allocator, w: usize, h: usize, c: usize, bytes: []const u8) Error!Image {
        if (bytes.len != w * h * c) return Error.InvalidImage;
        var img = try Image.init(alloc, w, h, c);
        errdefer img.deinit();
        for (bytes, 0..) |b, i| img.pixels.writeF64(i, @as(f64, @floatFromInt(b)) / 255.0);
        return img;
    }

    /// Takes a computed tensor as the new image ("what I just calculated is what
    /// I want to look at"): `[H, W, C]` with `C` in 1, 3 or 4, clamped to
    /// `0..1` and quantised to 8 bit. The tensor is *copied*, so the caller
    /// keeps owning it — and anything a filter or a gradient produced can be
    /// saved without a detour through a byte buffer.
    pub fn fromTensor(alloc: std.mem.Allocator, t: zpu.Tensor) Error!Image {
        if (t.rank() != 3) return Error.InvalidImage;
        const h = t.shape.dims[0];
        const w = t.shape.dims[1];
        const c = t.shape.dims[2];
        if (c != 1 and c != 3 and c != 4) return Error.InvalidImage;
        var img = try Image.init(alloc, w, h, c);
        errdefer img.deinit();
        for (0..t.numel()) |i| img.pixels.writeF64(i, std.math.clamp(t.readLogical(t.numel(), i), 0, 1));
        return img;
    }

    /// Binary PPM/PGM (P6/P5). The header is text, the pixels are 8 bit sRGB.
    pub fn toPpm(self: *const Image, alloc: std.mem.Allocator) Error![]u8 {
        const body = try self.toPixels(alloc);
        defer alloc.free(body);
        const magic: []const u8 = if (self.channels() == 1) "P5" else "P6";
        const header = try std.fmt.allocPrint(alloc, "{s}\n{d} {d}\n255\n", .{ magic, self.width(), self.height() });
        defer alloc.free(header);
        const out = try alloc.alloc(u8, header.len + body.len);
        @memcpy(out[0..header.len], header);
        @memcpy(out[header.len..], body);
        return out;
    }

    /// Writes the image as a binary PPM/PGM file (`io` comes from `std.process.Init`).
    pub fn savePpm(self: *const Image, io: std.Io, path: []const u8) Error!void {
        const bytes = try self.toPpm(self.alloc);
        defer self.alloc.free(bytes);
        const file = std.Io.Dir.cwd().createFile(io, path, .{}) catch return Error.IoFailed;
        defer file.close(io);
        file.writeStreamingAll(io, bytes) catch return Error.IoFailed;
    }
};

/// Normalised 1-D Gaussian kernel of length `2 * radius + 1` (σ = radius / 2).
pub fn gaussianKernel(alloc: std.mem.Allocator, radius: usize) Error!zpu.Tensor {
    const n = radius * 2 + 1;
    const values = try alloc.alloc(f64, n);
    defer alloc.free(values);
    const sigma = @max(@as(f64, @floatFromInt(radius)) / 2.0, 0.5);
    var sum: f64 = 0;
    for (values, 0..) |*v, i| {
        const x = @as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(radius));
        v.* = @exp(-(x * x) / (2 * sigma * sigma));
        sum += v.*;
    }
    for (values) |*v| v.* /= sum;
    const shape = try zpu.Shape.init(&.{n});
    return zpu.Tensor.fromArray(f64, values, shape);
}

/// Loads a binary PPM/PGM file into an image.
pub fn loadPpm(alloc: std.mem.Allocator, io: std.Io, path: []const u8) Error!Image {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return Error.IoFailed;
    defer file.close(io);
    const len: usize = @intCast(file.length(io) catch return Error.IoFailed);
    if (len > 64 * 1024 * 1024) return Error.TooLarge;
    const bytes = try alloc.alloc(u8, len);
    defer alloc.free(bytes);
    _ = file.readPositionalAll(io, bytes, 0) catch return Error.IoFailed;
    return decodePpm(alloc, bytes);
}

/// Decodes the bytes of a binary PPM/PGM file (`loadPpm` minus the filesystem).
///
/// The header is walked with an explicit cursor rather than a whitespace
/// tokenizer: a tokenizer that also skips *leading* delimiters would run past
/// the header's newline whenever the first pixel byte happens to be `\r`, `\n`,
/// `\t` or a space — which is about one pixel in sixteen, so it is not a corner
/// case but a coin flip. Only one whitespace byte may separate the max value
/// from the samples, which is what `toPpm` writes.
pub fn decodePpm(alloc: std.mem.Allocator, bytes: []const u8) Error!Image {
    var pos: usize = 0;
    var field: [4][]const u8 = undefined;
    for (&field) |*f| {
        while (pos < bytes.len and isPpmSpace(bytes[pos])) pos += 1;
        const start = pos;
        while (pos < bytes.len and !isPpmSpace(bytes[pos])) pos += 1;
        if (pos == start) return Error.InvalidImage;
        f.* = bytes[start..pos];
    }
    const magic = field[0];
    if (!std.mem.eql(u8, magic, "P5") and !std.mem.eql(u8, magic, "P6")) return Error.InvalidImage;
    const w = std.fmt.parseUnsigned(usize, field[1], 10) catch return Error.InvalidImage;
    const h = std.fmt.parseUnsigned(usize, field[2], 10) catch return Error.InvalidImage;
    const max = std.fmt.parseUnsigned(usize, field[3], 10) catch return Error.InvalidImage;
    if (max != 255 or w == 0 or h == 0) return Error.InvalidImage;

    if (pos >= bytes.len or !isPpmSpace(bytes[pos])) return Error.InvalidImage;
    pos += 1;
    const channels: usize = if (magic[1] == '5') 1 else 3;
    const need = w * h * channels;
    if (bytes.len < pos + need) return Error.InvalidImage;
    return Image.fromPixels(alloc, w, h, channels, bytes[pos .. pos + need]);
}

fn isPpmSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

test "image basics" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var img = try Image.init(alloc, 4, 3, 3);
    defer img.deinit();
    try std.testing.expectEqual(@as(usize, 4), img.width());
    try std.testing.expectEqual(@as(usize, 3), img.height());
    try std.testing.expectEqual(@as(usize, 3), img.channels());
    try std.testing.expectEqual(@as(usize, 36), img.numel());

    img.fill(.{ .r = 0.5, .g = 0.25, .b = 0 });
    const p = img.getPixel(2, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), p.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), p.g, 1e-6);

    img.setPixel(0, 0, .{ .r = 1, .g = 1, .b = 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(0, 0).r, 1e-6);
    // out of range writes are ignored, not a crash
    img.setPixel(-1, 5, .{ .r = 1, .g = 0, .b = 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), img.getPixel(3, 2).r, 1e-6);
}

test "alpha blending is src over dst" {
    const alloc = std.testing.allocator;
    var img = try Image.init(alloc, 2, 1, 3);
    defer img.deinit();
    img.setPixel(0, 0, .{ .r = 1, .g = 1, .b = 1 });
    img.setPixel(1, 0, .{ .r = 0, .g = 0, .b = 0 });

    // Half of pure red over white, and a quarter of it over black.
    img.blendPixel(0, 0, .{ .r = 1, .g = 0, .b = 0, .a = 0.5 });
    const over_white = img.getPixel(0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), over_white.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), over_white.g, 1e-6);

    img.blendCoverage(1, 0, .{ .r = 1, .g = 0, .b = 0, .a = 1 }, 0.25);
    const over_black = img.getPixel(1, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), over_black.r, 1e-6);

    // Zero coverage is a no-op, and out-of-range writes are ignored.
    const before = img.getPixel(1, 0);
    img.blendCoverage(1, 0, .{ .r = 0, .g = 1, .b = 0, .a = 1 }, 0);
    try std.testing.expectEqual(before.r, img.getPixel(1, 0).r);
    img.blendPixel(99, 99, .{ .r = 1, .g = 1, .b = 1 });
}

test "pixels round trip through 8 bit" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var img = try Image.init(alloc, 2, 1, 3);
    defer img.deinit();
    img.setPixel(0, 0, .{ .r = 1, .g = 0, .b = 0.5 });
    img.setPixel(1, 0, .{ .r = 0, .g = 0.25, .b = 0 });

    const bytes = try img.toPixels(alloc);
    defer alloc.free(bytes);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 128, 0, 64, 0 }, bytes);

    var back = try Image.fromPixels(alloc, 2, 1, 3, bytes);
    defer back.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), back.getPixel(0, 0).r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0 / 255.0), back.getPixel(0, 0).b, 1e-6);
}

test "ppm bytes are a valid P6 file" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var img = try Image.init(alloc, 1, 2, 3);
    defer img.deinit();
    img.setPixel(0, 0, .{ .r = 1, .g = 1, .b = 1 });
    const ppm = try img.toPpm(alloc);
    defer alloc.free(ppm);
    try std.testing.expectEqualStrings("P6\n1 2\n255\n", ppm[0..11]);
    try std.testing.expectEqual(@as(usize, 11 + 6), ppm.len);
    try std.testing.expectEqual(@as(u8, 255), ppm[11]);
    try std.testing.expectEqual(@as(u8, 0), ppm[16]);
}

test "a tensor becomes an image without a byte detour" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // A filter's output (or a gradient, or anything else the core produced) goes
    // straight in; values outside 0..1 are clamped rather than wrapping.
    var img = try Image.init(alloc, 2, 2, 3);
    defer img.deinit();
    img.fill(.{ .r = 0.25, .g = 0.5, .b = 0.75 });
    const blurry = try img.blur(1);
    var soft = try Image.fromTensor(alloc, blurry);
    defer soft.deinit();
    try std.testing.expectEqual(img.width(), soft.width());
    try std.testing.expectEqual(img.channels(), soft.channels());
    // Blurring pads with zeros, so border pixels can only get darker than the
    // uniform colour they started from — and the channel ordering survives.
    try std.testing.expect(soft.pixels.readF64(0) > 0.0);
    try std.testing.expect(soft.pixels.readF64(0) <= 0.25 + 1e-6);
    try std.testing.expect(soft.pixels.readF64(1) > soft.pixels.readF64(0));
    try std.testing.expect(soft.pixels.readF64(2) > soft.pixels.readF64(1));

    const out_of_range = try zpu.Tensor.full(zpu.Shape.initComptime(&.{ 2, 2, 3 }), .f32, 4.0);
    var clamped = try Image.fromTensor(alloc, out_of_range);
    defer clamped.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), clamped.pixels.readF64(0), 1e-6);
}

test "ppm decodes back into the image it came from" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    // One of the pixels is chosen so that its first byte is 0x0d (`\r`): a
    // header parser that tokenizes on whitespace runs past the header's newline
    // on exactly this input. Every default-filled image is like this, because
    // the background bytes are small integers.
    var img = try Image.init(alloc, 3, 2, 3);
    defer img.deinit();
    img.fill(.{ .r = 13.0 / 255.0, .g = 10.0 / 255.0, .b = 32.0 / 255.0 });
    img.setPixel(1, 1, .{ .r = 1, .g = 0.5, .b = 0.25 });

    const ppm = try img.toPpm(alloc);
    defer alloc.free(ppm);
    var back = try decodePpm(alloc, ppm);
    defer back.deinit();
    try std.testing.expectEqual(img.width(), back.width());
    try std.testing.expectEqual(img.height(), back.height());
    try std.testing.expectEqual(img.channels(), back.channels());
    for (0..img.numel()) |i| {
        // 8-bit rounding is the only loss allowed on the way through
        try std.testing.expectApproxEqAbs(img.pixels.readF64(i), back.pixels.readF64(i), 1.0 / 255.0);
    }

    // A colour image written as P6 with a mono body must be rejected, not
    // silently reinterpreted.
    try std.testing.expectError(Error.InvalidImage, decodePpm(alloc, "P6\n3 2\n255\n\x00\x01"));
    try std.testing.expectError(Error.InvalidImage, decodePpm(alloc, "P3\n3 2\n255\n0 0 0"));
    try std.testing.expectError(Error.InvalidImage, decodePpm(alloc, "P5\n0 2\n255\n"));
}

test "filters behave" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    var img = try Image.init(alloc, 16, 16, 1);
    defer img.deinit();
    img.fill(.{ .r = 0.5, .g = 0.5, .b = 0.5 });

    // blurring a constant image gives the same constant
    const soft = try img.blur(2);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), soft.readF64(8 * 16 + 8), 1e-5);
    // a flat image has no edges
    const edges = try img.sobel();
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), edges.readF64(8 * 16 + 8), 1e-6);

    // a bright line in the middle: sobel lights up next to it
    img.fill(.{ .r = 0, .g = 0, .b = 0 });
    img.drawRect(8, 0, 9, 16, .{ .r = 1, .g = 1, .b = 1 });
    const edges2 = try img.sobel();
    try std.testing.expect(edges2.readF64(8 * 16 + 7) > 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), edges2.readF64(2 * 16 + 2), 1e-6);

    // brightness and contrast clamp
    const bright = try img.brightness(3);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bright.readF64(8 * 16 + 8), 1e-6);
    const flat = try img.contrast(0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), flat.readF64(2 * 16 + 2), 1e-6);
}

test "drawing primitives" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    var img = try Image.init(alloc, 20, 20, 3);
    defer img.deinit();
    img.fill(.{ .r = 0, .g = 0, .b = 0 });
    img.drawRect(2, 2, 5, 5, .{ .r = 1, .g = 0, .b = 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(3, 3).r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(6, 6).r, 1e-6);

    img.drawCircle(14, 14, 3, .{ .r = 0, .g = 1, .b = 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(14, 14).g, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(18, 18).g, 1e-6);

    img.drawLine(0, 0, 19, 19, .{ .r = 0, .g = 0, .b = 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), img.getPixel(10, 10).b, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), img.getPixel(10, 0).b, 1e-6);
}

test "gaussian kernel is normalised" {
    const alloc = std.testing.allocator;
    try zpu.init(alloc, .{});
    const k = try gaussianKernel(alloc, 3);
    defer zpu.releaseTemp(k);
    try std.testing.expectEqual(@as(usize, 7), k.numel());
    var sum: f64 = 0;
    for (0..7) |i| sum += k.readF64(i);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
    // symmetric and peaked in the middle
    try std.testing.expectApproxEqAbs(k.readF64(0), k.readF64(6), 1e-12);
    try std.testing.expect(k.readF64(3) > k.readF64(0));
}
