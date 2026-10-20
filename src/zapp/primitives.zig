//! ZAPP Primitive Rendering
//!
//! 最小限の描画プリミティブ。ピクセルレベルの操作のみを提供。
//! このモジュールは geometry.zig と zgf.Image に依存するが、
//! 他の高レベルモジュールには依存しない。
//!
//! ## 設計原則
//! - **最小限のAPI**: 点、線、矩形、円のみ
//! - **高速**: 最適化されたアルゴリズムを使用
//! - **個別テスト**: 各プリミティブを独立してテスト
//! - **組み合わせ可能**: 複雑な描画はこれらの組み合わせ

const std = @import("std");
const zgf = @import("zgf");
const geo = @import("geometry.zig");

// ============================================================================
// ピクセル操作
// ============================================================================

/// 単一ピクセルを描画（境界チェック付き）
pub fn setPixel(fb: *zgf.Image, x: i32, y: i32, color: zgf.Color) void {
    if (x >= 0 and x < @as(i32, @intCast(fb.width())) and
        y >= 0 and y < @as(i32, @intCast(fb.height()))) {
        fb.setPixel(@intCast(x), @intCast(y), color);
    }
}

/// 単一ピクセルを描画（境界チェックなし、高速）
pub fn setPixelFast(fb: *zgf.Image, x: u32, y: u32, color: zgf.Color) void {
    fb.setPixel(x, y, color);
}

// ============================================================================
// 線描画
// ============================================================================

/// 線を描画（Bresenham アルゴリズム）
pub fn drawLine(fb: *zgf.Image, p0: geo.Vec2, p1: geo.Vec2, color: zgf.Color) void {
    const width: i32 = @intCast(fb.width());
    const height: i32 = @intCast(fb.height());
    
    var x0: i32 = @intFromFloat(@round(p0.x));
    var y0: i32 = @intFromFloat(@round(p0.y));
    const x1: i32 = @intFromFloat(@round(p1.x));
    const y1: i32 = @intFromFloat(@round(p1.y));
    
    const dx: i32 = @intCast(@abs(x1 - x0));
    const dy: i32 = @intCast(@abs(y1 - y0));
    const sx: i32 = if (x0 < x1) 1 else -1;
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err = dx - dy;
    
    while (true) {
        setPixel(fb, x0, y0, color);
        
        if (x0 == x1 and y0 == y1) break;
        
        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
    
    _ = width;
    _ = height;
}

/// 太い線を描画（線幅指定）
pub fn drawThickLine(fb: *zgf.Image, p0: geo.Vec2, p1: geo.Vec2, width: f32, color: zgf.Color) void {
    if (width <= 1.0) {
        drawLine(fb, p0, p1, color);
        return;
    }
    
    // 線の方向に垂直なベクトルを計算
    const dir = p1.sub(p0).normalize();
    const perp = dir.perpendicular().scale(width * 0.5);
    
    // 4つの角を計算
    const p0a = p0.add(perp);
    const p0b = p0.sub(perp);
    const p1a = p1.add(perp);
    const p1b = p1.sub(perp);
    
    // 2つの三角形として描画（簡易実装）
    drawLine(fb, p0a, p1a, color);
    drawLine(fb, p0b, p1b, color);
    drawLine(fb, p0a, p0b, color);
    drawLine(fb, p1a, p1b, color);
}

// ============================================================================
// 矩形描画
// ============================================================================

/// 矩形の輪郭を描画
pub fn drawRect(fb: *zgf.Image, rect: geo.BoundingBox, color: zgf.Color) void {
    const min_x: i32 = @intFromFloat(@round(rect.min.x));
    const min_y: i32 = @intFromFloat(@round(rect.min.y));
    const max_x: i32 = @intFromFloat(@round(rect.max.x));
    const max_y: i32 = @intFromFloat(@round(rect.max.y));
    
    // 4辺を描画
    drawLine(fb, .{ .x = @floatFromInt(min_x), .y = @floatFromInt(min_y) },
                .{ .x = @floatFromInt(max_x), .y = @floatFromInt(min_y) }, color);
    drawLine(fb, .{ .x = @floatFromInt(max_x), .y = @floatFromInt(min_y) },
                .{ .x = @floatFromInt(max_x), .y = @floatFromInt(max_y) }, color);
    drawLine(fb, .{ .x = @floatFromInt(max_x), .y = @floatFromInt(max_y) },
                .{ .x = @floatFromInt(min_x), .y = @floatFromInt(max_y) }, color);
    drawLine(fb, .{ .x = @floatFromInt(min_x), .y = @floatFromInt(max_y) },
                .{ .x = @floatFromInt(min_x), .y = @floatFromInt(min_y) }, color);
}

/// 矩形を塗りつぶし
pub fn fillRect(fb: *zgf.Image, rect: geo.BoundingBox, color: zgf.Color) void {
    const width = fb.width();
    const height = fb.height();
    
    const start_x: u32 = @intCast(@max(0, @min(@as(i32, @intFromFloat(@round(rect.min.x))), @as(i32, @intCast(width)))));
    const start_y: u32 = @intCast(@max(0, @min(@as(i32, @intFromFloat(@round(rect.min.y))), @as(i32, @intCast(height)))));
    const end_x: u32 = @intCast(@max(0, @min(@as(i32, @intFromFloat(@round(rect.max.x))), @as(i32, @intCast(width)))));
    const end_y: u32 = @intCast(@max(0, @min(@as(i32, @intFromFloat(@round(rect.max.y))), @as(i32, @intCast(height)))));
    
    var y = start_y;
    while (y < end_y) : (y += 1) {
        var x = start_x;
        while (x < end_x) : (x += 1) {
            setPixelFast(fb, x, y, color);
        }
    }
}

// ============================================================================
// 円描画
// ============================================================================

/// 円の輪郭を描画（中間点アルゴリズム）
pub fn drawCircle(fb: *zgf.Image, center: geo.Vec2, radius: f32, color: zgf.Color) void {
    const r: usize = @intCast(@as(usize, @intFromFloat(@round(radius))));
    if (r == 0) return;
    
    const cx: i32 = @intFromFloat(@round(center.x));
    const cy: i32 = @intFromFloat(@round(center.y));
    
    var x: i32 = 0;
    var y: i32 = @intCast(r);
    var d: i32 = 3 - 2 * @as(i32, @intCast(r));
    
    while (y >= x) {
        plotCirclePoints(fb, cx, cy, x, y, color);
        
        x += 1;
        if (d > 0) {
            y -= 1;
            d += 4 * (x - y) + 10;
        } else {
            d += 4 * x + 6;
        }
    }
}

fn plotCirclePoints(fb: *zgf.Image, cx: i32, cy: i32, x: i32, y: i32, color: zgf.Color) void {
    const points = [_][2]i32{
        .{ cx + x, cy + y }, .{ cx - x, cy + y },
        .{ cx + x, cy - y }, .{ cx - x, cy - y },
        .{ cx + y, cy + x }, .{ cx - y, cy + x },
        .{ cx + y, cy - x }, .{ cx - y, cy - x },
    };
    
    for (points) |p| {
        setPixel(fb, p[0], p[1], color);
    }
}

/// 円を塗りつぶし（スキャンライン）
pub fn fillCircle(fb: *zgf.Image, center: geo.Vec2, radius: f32, color: zgf.Color) void {
    const r: usize = @intCast(@as(usize, @intFromFloat(@round(radius))));
    if (r == 0) return;
    
    const cx: i32 = @intFromFloat(@round(center.x));
    const cy: i32 = @intFromFloat(@round(center.y));
    const fh: i32 = @intCast(fb.height());
    
    var dy: i32 = -@as(i32, @intCast(r));
    while (dy <= @as(i32, @intCast(r))) : (dy += 1) {
        const dy_f32: f32 = @floatFromInt(dy);
        const r_f32: f32 = @floatFromInt(r);
        const dx_f32: f32 = @sqrt(r_f32 * r_f32 - dy_f32 * dy_f32);
        const dx: i32 = @intFromFloat(@round(dx_f32));
        
        const y = cy + dy;
        if (y < 0 or y >= fh) continue;
        
        const x_start = cx - dx;
        const x_end = cx + dx + 1;
        var x = x_start;
        while (x < x_end) : (x += 1) {
            setPixel(fb, x, y, color);
        }
    }
}

// ============================================================================
// ベジェ曲線
// ============================================================================

/// 2次ベジェ曲線を描画
pub fn drawQuadraticBezier(
    fb: *zgf.Image,
    p0: geo.Vec2,
    p1: geo.Vec2,
    p2: geo.Vec2,
    color: zgf.Color,
    segments: u32,
) void {
    if (segments == 0) return;
    
    var prev = p0;
    var i: u32 = 1;
    while (i <= segments) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
        const next = geo.quadraticBezier(p0, p1, p2, t);
        drawLine(fb, prev, next, color);
        prev = next;
    }
}

/// 3次ベジェ曲線を描画
pub fn drawCubicBezier(
    fb: *zgf.Image,
    p0: geo.Vec2,
    p1: geo.Vec2,
    p2: geo.Vec2,
    p3: geo.Vec2,
    color: zgf.Color,
    segments: u32,
) void {
    if (segments == 0) return;
    
    var prev = p0;
    var i: u32 = 1;
    while (i <= segments) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
        const next = geo.cubicBezier(p0, p1, p2, p3, t);
        drawLine(fb, prev, next, color);
        prev = next;
    }
}

// ============================================================================
// パス描画
// ============================================================================

/// パスの輪郭を描画
pub fn strokePath(fb: *zgf.Image, path: geo.Path, color: zgf.Color, segments: u32) void {
    var current = geo.Vec2.init(0, 0);
    
    for (path.commands) |cmd| {
        switch (cmd) {
            .moveTo => |p| {
                current = p;
            },
            .lineTo => |p| {
                drawLine(fb, current, p, color);
                current = p;
            },
            .quadTo => |q| {
                drawQuadraticBezier(fb, current, q.control, q.end, color, segments);
                current = q.end;
            },
            .cubicTo => |c| {
                drawCubicBezier(fb, current, c.control1, c.control2, c.end, color, segments);
                current = c.end;
            },
            .close => {},
        }
    }
}

// ============================================================================
// テスト
// ============================================================================

test "primitive - setPixel" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    setPixel(&fb, 50, 50, .{ .r = 1, .g = 0, .b = 0 });
    
    // 境界外は描画されない
    setPixel(&fb, -1, 50, .{ .r = 0, .g = 1, .b = 0 });
    setPixel(&fb, 50, 200, .{ .r = 0, .g = 0, .b = 1 });
}

test "primitive - drawLine" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    drawLine(&fb, .{ .x = 10, .y = 10 }, .{ .x = 90, .y = 90 }, .{ .r = 1, .g = 0, .b = 0 });
}

test "primitive - fillRect" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    const rect = geo.BoundingBox.init(.{ .x = 20, .y = 20 }, .{ .x = 80, .y = 80 });
    fillRect(&fb, rect, .{ .r = 0, .g = 1, .b = 0 });
}

test "primitive - drawCircle" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    drawCircle(&fb, .{ .x = 50, .y = 50 }, 30, .{ .r = 0, .g = 0, .b = 1 });
}

test "primitive - fillCircle" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    fillCircle(&fb, .{ .x = 50, .y = 50 }, 30, .{ .r = 1, .g = 1, .b = 0 });
}

test "primitive - drawQuadraticBezier" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    drawQuadraticBezier(
        &fb,
        .{ .x = 10, .y = 90 },
        .{ .x = 50, .y = 10 },
        .{ .x = 90, .y = 90 },
        .{ .r = 1, .g = 0, .b = 1 },
        32,
    );
}

test "primitive - strokePath" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    const commands = [_]geo.PathCommand{
        .{ .moveTo = .{ .x = 10, .y = 10 } },
        .{ .lineTo = .{ .x = 90, .y = 10 } },
        .{ .lineTo = .{ .x = 90, .y = 90 } },
        .{ .lineTo = .{ .x = 10, .y = 90 } },
    };
    
    const path = geo.Path{ .commands = &commands };
    strokePath(&fb, path, .{ .r = 0, .g = 1, .b = 1 }, 16);
}
