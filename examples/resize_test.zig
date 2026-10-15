//! リサイズテスト — 100x100 から 4K (3840x2160) まで
//! スクリーンショットは snapshots/zapp/resize/ に PPM で保存される。

const std = @import("std");
const zapp = @import("zapp");
const zgf = @import("zgf");

const Dir = std.Io.Dir;

const TestSize = struct { width: u32, height: u32, name: []const u8 };
const test_sizes = [_]TestSize{
    .{ .width = 100, .height = 100, .name = "100x100" },
    .{ .width = 320, .height = 240, .name = "320x240_QVGA" },
    .{ .width = 640, .height = 480, .name = "640x480_VGA" },
    .{ .width = 800, .height = 600, .name = "800x600_SVGA" },
    .{ .width = 1024, .height = 768, .name = "1024x768_XGA" },
    .{ .width = 1280, .height = 720, .name = "1280x720_HD" },
    .{ .width = 1366, .height = 768, .name = "1366x768_WXGA" },
    .{ .width = 1920, .height = 1080, .name = "1920x1080_FullHD" },
    .{ .width = 2560, .height = 1440, .name = "2560x1440_QHD" },
    .{ .width = 3840, .height = 2160, .name = "3840x2160_4K" },
};

fn renderPattern(fb: *zgf.Image) void {
    const w = fb.width();
    const h = fb.height();
    for (0..h) |y| for (0..w) |x| {
        const r = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(w));
        const g = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(h));
        fb.setPixel(@intCast(x), @intCast(y), .{ .r = r, .g = g, .b = 1.0 - (r + g) / 2.0 });
    };
    const cs = @max(w / 20, 10);
    for (0..h) |y| for (0..w) |x| {
        if ((x / cs + y / cs) % 2 == 0 and x > w / 4 and x < w * 3 / 4 and y > h / 4 and y < h * 3 / 4)
            fb.setPixel(@intCast(x), @intCast(y), .{ .r = 1, .g = 1, .b = 1, .a = 0.5 });
    };
    const lc = zgf.Color{ .r = 1, .g = 1, .b = 0, .a = 1 };
    const m = @min(w, h);
    for (0..m) |i| {
        fb.setPixel(@intCast(i * w / m), @intCast(i * h / m), lc);
        fb.setPixel(@intCast(w - 1 - i * w / m), @intCast(i * h / m), lc);
    }
    const bc = zgf.Color{ .r = 1, .g = 0, .b = 0, .a = 1 };
    for (0..w) |x| { fb.setPixel(@intCast(x), 0, bc); fb.setPixel(@intCast(x), @intCast(h - 1), bc); }
    for (0..h) |y| { fb.setPixel(0, @intCast(y), bc); fb.setPixel(@intCast(w - 1), @intCast(y), bc); }
    const cc = zgf.Color{ .r = 0, .g = 1, .b = 0, .a = 1 };
    for (0..w) |x| fb.setPixel(@intCast(x), @intCast(h / 2), cc);
    for (0..h) |y| fb.setPixel(@intCast(w / 2), @intCast(y), cc);
}

fn save(io: std.Io, alloc: std.mem.Allocator, fb: *const zgf.Image, name: []const u8) !void {
    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);
    const path = try std.fmt.allocPrint(alloc, "snapshots/zapp/resize/{s}.ppm", .{name});
    defer alloc.free(path);
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);
    std.debug.print("  ✓ {s}\n", .{path});
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;
    std.debug.print("\n=== ZAPP リサイズテスト ===\n\n", .{});
    Dir.cwd().createDirPath(io, "snapshots/zapp/resize") catch {};
    var win = zapp.Window.init(.{ .title = "Resize", .width = 100, .height = 100, .backend = .software }) catch |e| {
        std.debug.print("エラー: {}\n", .{e});
        return e;
    };
    defer win.deinit();
    for (test_sizes) |ts| {
        std.debug.print("{s} ({d}x{d})... ", .{ ts.name, ts.width, ts.height });
        win.resize(ts.width, ts.height) catch |e| { std.debug.print("✗ {}\n", .{e}); continue; };
        const sz = win.getSize();
        if (sz.width != ts.width or sz.height != ts.height) { std.debug.print("✗ size mismatch\n", .{}); continue; }
        const fb = win.getFramebuffer();
        renderPattern(fb);
        save(io, alloc, fb, ts.name) catch |e| { std.debug.print("✗ {}\n", .{e}); continue; };
    }
    std.debug.print("\n=== 完了 ===\n", .{});
}
