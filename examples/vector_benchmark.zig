// ベクターレンダリングのパフォーマンスベンチマーク
const std = @import("std");
const zapp = @import("zapp");
const zgf = @import("zgf");
const vector = zapp.vector;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    
    std.debug.print("\n=== Vector Rendering Performance Benchmark ===\n\n", .{});
    
    // ベンチマーク1: 高速ライン描画
    try benchmarkDrawLineFast(allocator);
    
    // ベンチマーク2: バッチライン描画
    try benchmarkDrawLinesBatch(allocator);
    
    // ベンチマーク3: 最適化された矩形塗りつぶし
    try benchmarkFillRectOptimized(allocator);
    
    // ベンチマーク4: 最適化された円描画
    try benchmarkDrawCircleOptimized(allocator);
    
    // ベンチマーク5: 最適化された円塗りつぶし
    try benchmarkFillCircleOptimized(allocator);
    
    // ベンチマーク6: ベジェ曲線
    try benchmarkBezierCurves(allocator);
    
    // ベンチマーク7: パス描画
    try benchmarkPathRendering(allocator);
    
    // ベンチマーク8: メモリプール
    try benchmarkMemoryPool(allocator);
    
    std.debug.print("\n=== Benchmark Complete ===\n\n", .{});
}

fn benchmarkDrawLineFast(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const iterations = 10000;
    
    for (0..iterations) |i| {
        const x1: f32 = @floatFromInt(i % 1000);
        const y1: f32 = @floatFromInt((i * 7) % 1000);
        const x2: f32 = @floatFromInt((i * 13) % 1000);
        const y2: f32 = @floatFromInt((i * 19) % 1000);
        vector.drawLineFast(&fb, x1, y1, x2, y2, .{ .r = 1.0, .g = 0.0, .b = 0.0 });
    }
    
    std.debug.print("drawLineFast: {d} lines - ✓ COMPLETE\n", .{iterations});
}

fn benchmarkDrawLinesBatch(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const line_count = 1000;
    var lines: [1000]vector.Line = undefined;
    
    for (0..line_count) |i| {
        lines[i] = .{
            .x0 = @floatFromInt(i % 1000),
            .y0 = @floatFromInt((i * 7) % 1000),
            .x1 = @floatFromInt((i * 13) % 1000),
            .y1 = @floatFromInt((i * 19) % 1000),
        };
    }
    
    const iterations = 10;
    
    for (0..iterations) |_| {
        vector.drawLinesBatch(&fb, &lines, .{ .r = 0.0, .g = 1.0, .b = 0.0 });
    }
    
    std.debug.print("drawLinesBatch: {d} batches of {d} lines - ✓ COMPLETE\n", .{iterations, line_count});
}

fn benchmarkFillRectOptimized(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const iterations = 10000;
    
    for (0..iterations) |i| {
        const x: usize = i % 900;
        const y: usize = (i * 7) % 900;
        vector.fillRectOptimized(&fb, x, y, x + 100, y + 100, .{ .r = 0.0, .g = 0.0, .b = 1.0 });
    }
    
    std.debug.print("fillRectOptimized: {d} rects - ✓ COMPLETE\n", .{iterations});
}

fn benchmarkDrawCircleOptimized(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const iterations = 10000;
    
    for (0..iterations) |i| {
        const cx: f32 = @floatFromInt(100 + (i % 800));
        const cy: f32 = @floatFromInt(100 + ((i * 7) % 800));
        vector.drawCircleOptimized(&fb, cx, cy, 50, .{ .r = 1.0, .g = 1.0, .b = 0.0 });
    }
    
    std.debug.print("drawCircleOptimized: {d} circles - ✓ COMPLETE\n", .{iterations});
}

fn benchmarkFillCircleOptimized(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const iterations = 5000;
    
    for (0..iterations) |i| {
        const cx: f32 = @floatFromInt(100 + (i % 800));
        const cy: f32 = @floatFromInt(100 + ((i * 7) % 800));
        vector.fillCircleOptimized(&fb, cx, cy, 50, .{ .r = 0.0, .g = 1.0, .b = 1.0 });
    }
    
    std.debug.print("fillCircleOptimized: {d} circles - ✓ COMPLETE\n", .{iterations});
}

fn benchmarkBezierCurves(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const iterations = 1000;
    
    for (0..iterations) |i| {
        const offset: f32 = @floatFromInt(i % 100);
        vector.drawQuadraticBezier(
            &fb,
            vector.Vec2.init(100 + offset, 100),
            vector.Vec2.init(500, 900 - offset),
            vector.Vec2.init(900 - offset, 100),
            .{ .r = 1.0, .g = 0.0, .b = 1.0 },
        );
    }
    
    std.debug.print("drawQuadraticBezier: {d} curves - ✓ COMPLETE\n", .{iterations});
}

fn benchmarkPathRendering(allocator: std.mem.Allocator) !void {
    var fb = try zgf.Image.init(allocator, 1000, 1000, 3);
    defer fb.deinit();
    
    const iterations = 100;
    
    for (0..iterations) |_| {
        var path = vector.Path.init(allocator);
        defer path.deinit();
        
        try path.moveTo(@floatFromInt(100), @floatFromInt(100));
        try path.lineTo(@floatFromInt(500), @floatFromInt(100));
        try path.lineTo(@floatFromInt(500), @floatFromInt(500));
        try path.lineTo(@floatFromInt(100), @floatFromInt(500));
        
        path.stroke(&fb, .{ .r = 1.0, .g = 0.5, .b = 0.0 });
    }
    
    std.debug.print("path.stroke: {d} paths - ✓ COMPLETE\n", .{iterations});
}

fn benchmarkMemoryPool(allocator: std.mem.Allocator) !void {
    _ = allocator;
    const iterations = 10000;
    
    for (0..iterations) |_| {
        var pool = vector.MemoryPool(u32, 1024).init();
        
        var i: usize = 0;
        while (i < 100) : (i += 1) {
            _ = pool.alloc() orelse break;
        }
    }
    
    std.debug.print("MemoryPool: {d} allocations - ✓ COMPLETE\n", .{iterations * 100});
}
