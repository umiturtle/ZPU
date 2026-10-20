//! ZAPP Vector Rendering Demo
//!
//! ベクターレンダリングライブラリのデモ。

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");

const Dir = std.Io.Dir;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    std.debug.print("\n=== ZAPP Vector Rendering Demo ===\n\n", .{});

    Dir.cwd().createDirPath(io, "snapshots/zapp/vector") catch {};

    // デモ1: 線描画
    try demoLines(io, alloc);

    // デモ2: 図形描画
    try demoShapes(io, alloc);

    // デモ3: 複合描画
    try demoComposite(io, alloc);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

fn demoLines(io: std.Io, alloc: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(alloc, 400, 300, 3);
    defer fb.deinit();

    fb.fill(.{ .r = 0.1, .g = 0.1, .b = 0.15, .a = 1.0 });

    // 複数の線を異なる色で描画
    const colors = [_]zgf.Color{
        .{ .r = 1.0, .g = 0.0, .b = 0.0 },
        .{ .r = 0.0, .g = 1.0, .b = 0.0 },
        .{ .r = 0.0, .g = 0.0, .b = 1.0 },
        .{ .r = 1.0, .g = 1.0, .b = 0.0 },
        .{ .r = 1.0, .g = 0.0, .b = 1.0 },
    };

    for (colors, 0..) |color, i| {
        const y: f32 = 50 + @as(f32, @floatFromInt(i)) * 40;
        zapp.vector.drawLineFast(&fb, 50, y, 350, y + 20, color);
    }

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = "snapshots/zapp/vector/lines.ppm";
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} (400x300)\n", .{path});
}

fn demoShapes(io: std.Io, alloc: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(alloc, 400, 300, 3);
    defer fb.deinit();

    fb.fill(.{ .r = 0.1, .g = 0.1, .b = 0.15, .a = 1.0 });

    // 矩形
    zapp.vector.fillRectOptimized(&fb, 50, 50, 150, 100, .{ .r = 0.3, .g = 0.6, .b = 0.9 });

    // 円
    zapp.vector.drawCircleOptimized(&fb, 250, 75, 50, .{ .r = 0.9, .g = 0.3, .b = 0.3 });

    // 別の円
    zapp.vector.drawCircleOptimized(&fb, 200, 200, 80, .{ .r = 0.3, .g = 0.9, .b = 0.3 });

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = "snapshots/zapp/vector/shapes.ppm";
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} (400x300)\n", .{path});
}

fn demoComposite(io: std.Io, alloc: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(alloc, 400, 300, 3);
    defer fb.deinit();

    fb.fill(.{ .r = 0.05, .g = 0.05, .b = 0.1, .a = 1.0 });

    // 背景の円
    for (0..5) |i| {
        const radius: f32 = 150 - @as(f32, @floatFromInt(i)) * 25;
        const brightness: f32 = 0.2 + @as(f32, @floatFromInt(i)) * 0.15;
        zapp.vector.drawCircleOptimized(&fb, 200, 150, radius, .{
            .r = brightness,
            .g = brightness * 0.8,
            .b = brightness * 1.2,
        });
    }

    // 前景の線
    for (0..8) |i| {
        const angle: f32 = @as(f32, @floatFromInt(i)) * 0.785; // 45度
        const x1: f32 = 200 + @cos(angle) * 100;
        const y1: f32 = 150 + @sin(angle) * 100;
        const x2: f32 = 200 - @cos(angle) * 100;
        const y2: f32 = 150 - @sin(angle) * 100;
        zapp.vector.drawLineFast(&fb, x1, y1, x2, y2, .{
            .r = 0.9,
            .g = 0.9,
            .b = 0.9,
        });
    }

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = "snapshots/zapp/vector/composite.ppm";
    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} (400x300)\n", .{path});
}
