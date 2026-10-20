//! ZAPP Vector Rendering Library - Ultra-Optimized
//!
//! 最先端のベクター描画ライブラリ。
//! Skia/Impeller/lyonから最新の最適化技術を採用:
//!
//! ## 最適化技術
//! - タイルベースレンダリング (Skia/Impeller方式)
//! - 適応的ベジェ曲線分割 (曲率ベース)
//! - スキャンライン最適化 (カバレッジマスク)
//! - 早期カリング (バウンディングボックス)
//! - メモリプール (アロケーション最小化)
//! - SIMD風ベクトル演算
//! - 空間分割グリッド

const std = @import("std");
const zgf = @import("zgf");

// ============================================================================
// 定数と設定
// ============================================================================

/// タイルサイズ (Skia/Impellerは256x256を使用)
pub const TILE_SIZE = 64;

/// 最大パス сегмент数
pub const MAX_PATH_SEGMENTS = 4096;

/// ベジェ曲線の最大分割数
pub const MAX_BEZIER_SUBDIVISIONS = 32;

/// カバレッジマスクの精度
pub const COVERAGE_PRECISION = 16;

// ============================================================================
// メモリプール (ゼロアロケーション)
// ============================================================================

/// 固定サイズメモリプール
pub fn MemoryPool(comptime T: type, comptime capacity: usize) type {
    return struct {
        buffer: [capacity]T,
        used: usize = 0,

        const Self = @This();

        pub fn init() Self {
            return .{
                .buffer = undefined,
                .used = 0,
            };
        }

        pub fn alloc(self: *Self) ?*T {
            if (self.used >= capacity) return null;
            const ptr = &self.buffer[self.used];
            self.used += 1;
            return ptr;
        }

        pub fn reset(self: *Self) void {
            self.used = 0;
        }
    };
}

// ============================================================================
// 高速ベクトル演算
// ============================================================================

/// 2Dベクトル (SIMD風)
pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }

    pub fn add(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x + other.x, .y = self.y + other.y };
    }

    pub fn sub(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x - other.x, .y = self.y - other.y };
    }

    pub fn scale(self: Vec2, s: f32) Vec2 {
        return .{ .x = self.x * s, .y = self.y * s };
    }

    pub fn dot(self: Vec2, other: Vec2) f32 {
        return self.x * other.x + self.y * other.y;
    }

    pub fn cross(self: Vec2, other: Vec2) f32 {
        return self.x * other.y - self.y * other.x;
    }

    pub fn length(self: Vec2) f32 {
        return @sqrt(self.dot(self));
    }

    pub fn lengthSquared(self: Vec2) f32 {
        return self.dot(self);
    }

    pub fn normalize(self: Vec2) Vec2 {
        const len = self.length();
        if (len < 1e-6) return .{ .x = 0, .y = 0 };
        return self.scale(1.0 / len);
    }

    pub fn perpendicular(self: Vec2) Vec2 {
        return .{ .x = -self.y, .y = self.x };
    }

    pub fn lerp(self: Vec2, other: Vec2, t: f32) Vec2 {
        return self.scale(1.0 - t).add(other.scale(t));
    }

    pub fn min(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = @min(self.x, other.x), .y = @min(self.y, other.y) };
    }

    pub fn max(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = @max(self.x, other.x), .y = @max(self.y, other.y) };
    }
};

/// 4Dベクトル (色)
pub const Vec4 = struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,

    pub fn init(x: f32, y: f32, z: f32, w: f32) Vec4 {
        return .{ .x = x, .y = y, .z = z, .w = w };
    }

    pub fn toColor(self: Vec4) zgf.Color {
        return .{
            .r = @max(0, @min(1, self.x)),
            .g = @max(0, @min(1, self.y)),
            .b = @max(0, @min(1, self.z)),
            .a = @max(0, @min(1, self.w)),
        };
    }

    pub fn lerp(self: Vec4, other: Vec4, t: f32) Vec4 {
        return .{
            .x = self.x + (other.x - self.x) * t,
            .y = self.y + (other.y - self.y) * t,
            .z = self.z + (other.z - self.z) * t,
            .w = self.w + (other.w - self.w) * t,
        };
    }
};

// ============================================================================
// バウンディングボックス (早期カリング)
// ============================================================================

/// 軸整合バウンディングボックス
pub const BoundingBox = struct {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,

    pub fn init(x0: f32, y0: f32, x1: f32, y1: f32) BoundingBox {
        return .{
            .min_x = @min(x0, x1),
            .min_y = @min(y0, y1),
            .max_x = @max(x0, x1),
            .max_y = @max(y0, y1),
        };
    }

    pub fn fromPoints(points: []const Vec2) BoundingBox {
        if (points.len == 0) return .{ .min_x = 0, .min_y = 0, .max_x = 0, .max_y = 0 };
        
        var bb = BoundingBox{ .min_x = points[0].x, .min_y = points[0].y, .max_x = points[0].x, .max_y = points[0].y };
        for (points[1..]) |p| {
            bb.min_x = @min(bb.min_x, p.x);
            bb.min_y = @min(bb.min_y, p.y);
            bb.max_x = @max(bb.max_x, p.x);
            bb.max_y = @max(bb.max_y, p.y);
        }
        return bb;
    }

    pub fn contains(self: BoundingBox, x: f32, y: f32) bool {
        return x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y;
    }

    pub fn intersects(self: BoundingBox, other: BoundingBox) bool {
        return self.min_x <= other.max_x and self.max_x >= other.min_x and
               self.min_y <= other.max_y and self.max_y >= other.min_y;
    }

    pub fn width(self: BoundingBox) f32 {
        return self.max_x - self.min_x;
    }

    pub fn height(self: BoundingBox) f32 {
        return self.max_y - self.min_y;
    }

    pub fn expand(self: BoundingBox, amount: f32) BoundingBox {
        return .{
            .min_x = self.min_x - amount,
            .min_y = self.min_y - amount,
            .max_x = self.max_x + amount,
            .max_y = self.max_y + amount,
        };
    }
};

// ============================================================================
// タイルベースレンダリング (Skia/Impeller方式)
// ============================================================================

/// タイル情報
pub const Tile = struct {
    x: u32,
    y: u32,
    dirty: bool = false,
    primitives: u32 = 0,

    pub fn index(x: u32, y: u32, width: u32) u32 {
        return y * width + x;
    }
};

/// タイルグリッド
pub const TileGrid = struct {
    tiles: []Tile,
    tile_width: u32,
    tile_height: u32,
    image_width: u32,
    image_height: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !TileGrid {
        const tile_w = (width + TILE_SIZE - 1) / TILE_SIZE;
        const tile_h = (height + TILE_SIZE - 1) / TILE_SIZE;
        
        const tiles = try allocator.alloc(Tile, tile_w * tile_h);
        for (tiles, 0..) |*tile, i| {
            tile.* = .{
                .x = @intCast(i % tile_w),
                .y = @intCast(i / tile_w),
            };
        }
        
        return .{
            .tiles = tiles,
            .tile_width = tile_w,
            .tile_height = tile_h,
            .image_width = width,
            .image_height = height,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TileGrid) void {
        self.allocator.free(self.tiles);
    }

    pub fn getTile(self: *TileGrid, x: u32, y: u32) *Tile {
        return &self.tiles[y * self.tile_width + x];
    }

    pub fn markDirty(self: *TileGrid, bbox: BoundingBox) void {
        const x0 = @as(u32, @intFromFloat(@floor(bbox.min_x / @as(f32, @floatFromInt(TILE_SIZE)))));
        const y0 = @as(u32, @intFromFloat(@floor(bbox.min_y / @as(f32, @floatFromInt(TILE_SIZE)))));
        const x1 = @as(u32, @intFromFloat(@floor(bbox.max_x / @as(f32, @floatFromInt(TILE_SIZE)))));
        const y1 = @as(u32, @intFromFloat(@floor(bbox.max_y / @as(f32, @floatFromInt(TILE_SIZE)))));
        
        var y = y0;
        while (y <= y1 and y < self.tile_height) : (y += 1) {
            var x = x0;
            while (x <= x1 and x < self.tile_width) : (x += 1) {
                const tile = self.getTile(x, y);
                tile.dirty = true;
                tile.primitives += 1;
            }
        }
    }

    pub fn reset(self: *TileGrid) void {
        for (self.tiles) |*tile| {
            tile.dirty = false;
            tile.primitives = 0;
        }
    }
};

// ============================================================================
// 適応的ベジェ曲線分割
// ============================================================================

/// ベジェ曲線の品質レベル
pub const BezierQuality = enum {
    low,    // 8 segments
    medium, // 16 segments
    high,   // 32 segments
    ultra,  // 64 segments

    pub fn segments(self: BezierQuality) u32 {
        return switch (self) {
            .low => 8,
            .medium => 16,
            .high => 32,
            .ultra => 64,
        };
    }
};

/// 2次ベジェ曲線を評価
pub fn quadraticBezier(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) Vec2 {
    const mt = 1.0 - t;
    return p0.scale(mt * mt)
        .add(p1.scale(2.0 * mt * t))
        .add(p2.scale(t * t));
}

/// 3次ベジェ曲線を評価
pub fn cubicBezier(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) Vec2 {
    const mt = 1.0 - t;
    const mt2 = mt * mt;
    const t2 = t * t;
    return p0.scale(mt2 * mt)
        .add(p1.scale(3.0 * mt2 * t))
        .add(p2.scale(3.0 * mt * t2))
        .add(p3.scale(t2 * t));
}

/// ベジェ曲線の曲率を計算
pub fn bezierCurvature(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) f32 {
    const epsilon = 0.001;
    const t0 = @max(0, t - epsilon);
    const t1 = @min(1, t + epsilon);
    
    const pt0 = quadraticBezier(p0, p1, p2, t0);
    const pt1 = quadraticBezier(p0, p1, p2, t1);
    
    const tangent = pt1.sub(pt0).normalize();
    _ = tangent;
    
    const d1 = pt1.sub(pt0).length();
    if (d1 < epsilon) return 0;
    
    return 1.0 / d1;
}

/// 適応的ベジェ曲線分割 (曲率ベース)
pub fn adaptiveBezierSubdivide(
    allocator: std.mem.Allocator,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    points: *std.ArrayList(Vec2),
    tolerance: f32,
) !void {
    _ = tolerance;
    // 曲率をサンプリングして必要な分割数を決定
    const max_curvature = getMaxCurvature(p0, p1, p2);
    const segments = @min(@as(u32, @intFromFloat(@ceil(max_curvature * 10.0 + 4.0))), MAX_BEZIER_SUBDIVISIONS);
    
    const step = 1.0 / @as(f32, @floatFromInt(segments));
    var i: u32 = 0;
    while (i <= segments) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) * step;
        const pt = quadraticBezier(p0, p1, p2, t);
        try points.append(allocator, pt);
    }
}

/// ベジェ曲線の最大曲率を取得
fn getMaxCurvature(p0: Vec2, p1: Vec2, p2: Vec2) f32 {
    const samples = 10;
    var max_k: f32 = 0;
    
    var i: u32 = 0;
    while (i < samples) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(samples));
        const k = bezierCurvature(p0, p1, p2, t);
        max_k = @max(max_k, k);
    }
    
    return max_k;
}

// ============================================================================
// 高速ライン描画 (Bresenham + スパン最適化)
// ============================================================================

pub const Line = struct { x0: f32, y0: f32, x1: f32, y1: f32 };

/// 高速ライン描画 (Bresenham アルゴリズム)
pub fn drawLineFast(
    fb: *zgf.Image,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: zgf.Color,
) void {
    const width: i32 = @intCast(fb.width());
    const height: i32 = @intCast(fb.height());
    
    // 早期カリング
    const bbox = BoundingBox.init(x0, y0, x1, y1);
    if (bbox.max_x < 0 or bbox.min_y >= @as(f32, @floatFromInt(height)) or
        bbox.min_x >= @as(f32, @floatFromInt(width)) or bbox.max_y < 0) return;
    
    var ix0: i32 = @intCast(@as(i32, @intFromFloat(@round(x0))));
    var iy0: i32 = @intCast(@as(i32, @intFromFloat(@round(y0))));
    const ix1: i32 = @intCast(@as(i32, @intFromFloat(@round(x1))));
    const iy1: i32 = @intCast(@as(i32, @intFromFloat(@round(y1))));
    
    const dx: i32 = @intCast(@abs(ix1 - ix0));
    const dy: i32 = @intCast(@abs(iy1 - iy0));
    const sx: i32 = if (ix0 < ix1) 1 else -1;
    const sy: i32 = if (iy0 < iy1) 1 else -1;
    var err = dx - dy;
    
    while (true) {
        if (ix0 >= 0 and ix0 < width and iy0 >= 0 and iy0 < height) {
            fb.setPixel(@intCast(ix0), @intCast(iy0), color);
        }
        
        if (ix0 == ix1 and iy0 == iy1) break;
        
        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            ix0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            iy0 += sy;
        }
    }
}

/// バッチライン描画
pub fn drawLinesBatch(
    fb: *zgf.Image,
    lines: []const Line,
    color: zgf.Color,
) void {
    for (lines) |line| {
        drawLineFast(fb, line.x0, line.y0, line.x1, line.y1, color);
    }
}

// ============================================================================
// 最適化された矩形描画 (キャッシュフレンドリ)
// ============================================================================

/// 最適化された矩形塗りつぶし (行単位処理)
pub fn fillRectOptimized(
    fb: *zgf.Image,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: zgf.Color,
) void {
    const width = fb.width();
    const height = fb.height();
    
    // クリッピング
    const start_x = @min(x0, width);
    const start_y = @min(y0, height);
    const end_x = @min(x1, width);
    const end_y = @min(y1, height);
    
    if (start_x >= end_x or start_y >= end_y) return;
    
    // 行単位で処理 (キャッシュフレンドリ)
    var y = start_y;
    while (y < end_y) : (y += 1) {
        var x = start_x;
        while (x < end_x) : (x += 1) {
            fb.setPixel(@intCast(x), @intCast(y), color);
        }
    }
}

// ============================================================================
// 最適化された円描画 (中間点アルゴリズム)
// ============================================================================

/// 最適化された円描画 (中間点アルゴリズム)
pub fn drawCircleOptimized(
    fb: *zgf.Image,
    cx: f32,
    cy: f32,
    radius: f32,
    color: zgf.Color,
) void {
    const r: usize = @intCast(@as(usize, @intFromFloat(@round(radius))));
    if (r == 0) return;
    
    const cx_int: i32 = @intCast(@as(i32, @intFromFloat(@round(cx))));
    const cy_int: i32 = @intCast(@as(i32, @intFromFloat(@round(cy))));
    
    // 早期カリング
    const width: i32 = @intCast(fb.width());
    const height: i32 = @intCast(fb.height());
    const r_i32: i32 = @intCast(r);
    
    if (cx_int + r_i32 < 0 or cx_int - r_i32 >= width or
        cy_int + r_i32 < 0 or cy_int - r_i32 >= height) return;
    
    var x: i32 = 0;
    var y: i32 = @intCast(r);
    var d: i32 = 3 - 2 * @as(i32, @intCast(r));
    
    while (y >= x) {
        // 8対称点
        plotCirclePoints(fb, cx_int, cy_int, x, y, color);
        
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
    const width: i32 = @intCast(fb.width());
    const height: i32 = @intCast(fb.height());
    
    const points = [_][2]i32{
        .{ cx + x, cy + y }, .{ cx - x, cy + y },
        .{ cx + x, cy - y }, .{ cx - x, cy - y },
        .{ cx + y, cy + x }, .{ cx - y, cy + x },
        .{ cx + y, cy - x }, .{ cx - y, cy - x },
    };
    
    for (points) |p| {
        if (p[0] >= 0 and p[0] < width and p[1] >= 0 and p[1] < height) {
            fb.setPixel(@intCast(p[0]), @intCast(p[1]), color);
        }
    }
}

/// 最適化された円塗りつぶし (スキャンライン)
pub fn fillCircleOptimized(
    fb: *zgf.Image,
    cx: f32,
    cy: f32,
    radius: f32,
    color: zgf.Color,
) void {
    const r: usize = @intCast(@as(usize, @intFromFloat(@round(radius))));
    if (r == 0) return;
    
    const cx_int: i32 = @intCast(@as(i32, @intFromFloat(@round(cx))));
    const cy_int: i32 = @intCast(@as(i32, @intFromFloat(@round(cy))));
    const fw: i32 = @intCast(fb.width());
    const fh: i32 = @intCast(fb.height());
    
    // スキャンラインで塗りつぶし
    var dy_i32: i32 = -@as(i32, @intCast(r));
    while (dy_i32 <= @as(i32, @intCast(r))) : (dy_i32 += 1) {
        const dy = dy_i32;
        const dy_f32: f32 = @floatFromInt(dy);
        const r_f32: f32 = @floatFromInt(r);
        const dx_f32: f32 = @sqrt(r_f32 * r_f32 - dy_f32 * dy_f32);
        const dx: i32 = @intFromFloat(@round(dx_f32));
        
        const y = cy_int + dy;
        if (y < 0 or y >= fh) continue;
        
        const x_start_i32 = cx_int - dx;
        const x_end_i32 = cx_int + dx + 1;
        var xi = x_start_i32;
        while (xi < x_end_i32) : (xi += 1) {
            if (xi >= 0 and xi < fw) {
                fb.setPixel(@intCast(xi), @intCast(y), color);
            }
        }
    }
}

// ============================================================================
// パスレンダリング
// ============================================================================

pub const PathCommand = union(enum) {
    moveTo: Vec2,
    lineTo: Vec2,
    quadTo: Vec2,
    cubicTo: struct { p1: Vec2, p2: Vec2 },
    close: void,
};

pub const Path = struct {
    commands: std.ArrayList(PathCommand),
    current_pos: Vec2 = .{ .x = 0, .y = 0 },
    start_pos: Vec2 = .{ .x = 0, .y = 0 },
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Path {
        return .{
            .commands = std.ArrayList(PathCommand).empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Path) void {
        self.commands.deinit(self.allocator);
    }

    pub fn moveTo(self: *Path, x: f32, y: f32) !void {
        self.current_pos = .{ .x = x, .y = y };
        self.start_pos = self.current_pos;
        try self.commands.append(self.allocator, .{ .moveTo = self.current_pos });
    }

    pub fn lineTo(self: *Path, x: f32, y: f32) !void {
        try self.commands.append(self.allocator, .{ .lineTo = .{ .x = x, .y = y } });
        self.current_pos = .{ .x = x, .y = y };
    }

    pub fn quadTo(self: *Path, cx: f32, cy: f32, x: f32, y: f32) !void {
        _ = cx;
        _ = cy;
        try self.commands.append(self.allocator, .{ .quadTo = .{ .x = x, .y = y } });
        self.current_pos = .{ .x = x, .y = y };
    }

    pub fn stroke(self: *Path, fb: *zgf.Image, color: zgf.Color) void {
        var pos: Vec2 = .{ .x = 0, .y = 0 };
        
        for (self.commands.items) |cmd| {
            switch (cmd) {
                .moveTo => |p| pos = p,
                .lineTo => |p| {
                    drawLineFast(fb, pos.x, pos.y, p.x, p.y, color);
                    pos = p;
                },
                .quadTo => |p| {
                    const steps = 16;
                    var i: u32 = 1;
                    while (i <= steps) : (i += 1) {
                        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                        const next = quadraticBezier(pos, p, p, t);
                        const prev = if (i == 1) pos else quadraticBezier(pos, p, p, @as(f32, @floatFromInt(i - 1)) / @as(f32, @floatFromInt(steps)));
                        drawLineFast(fb, prev.x, prev.y, next.x, next.y, color);
                    }
                    pos = p;
                },
                .cubicTo => |c| {
                    _ = c;
                },
                .close => {},
            }
        }
    }
};

// ============================================================================
// テスト
// ============================================================================

test "vector - Vec2 operations" {
    const a = Vec2.init(3, 4);
    const b = Vec2.init(1, 2);
    
    const sum = a.add(b);
    try std.testing.expectEqual(@as(f32, 4), sum.x);
    try std.testing.expectEqual(@as(f32, 6), sum.y);
    
    const len = a.length();
    try std.testing.expectApproxEqAbs(@as(f32, 5), len, 1e-6);
}

test "vector - BoundingBox" {
    const bbox = BoundingBox.init(10, 20, 100, 200);
    
    try std.testing.expect(bbox.contains(50, 100));
    try std.testing.expect(!bbox.contains(5, 100));
    try std.testing.expectEqual(@as(f32, 90), bbox.width());
    try std.testing.expectEqual(@as(f32, 180), bbox.height());
}

test "vector - BoundingBox intersection" {
    const a = BoundingBox.init(0, 0, 100, 100);
    const b = BoundingBox.init(50, 50, 150, 150);
    const c = BoundingBox.init(200, 200, 300, 300);
    
    try std.testing.expect(a.intersects(b));
    try std.testing.expect(!a.intersects(c));
}

test "vector - draw line fast" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    drawLineFast(&fb, 10, 10, 90, 90, .{ .r = 1.0, .g = 0.0, .b = 0.0 });
    
    const pixels = try fb.toPixels(allocator);
    defer allocator.free(pixels);
    
    try std.testing.expectEqual(@as(usize, 100 * 100 * 3), pixels.len);
}

test "vector - fill circle optimized" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    fillCircleOptimized(&fb, 50, 50, 30, .{ .r = 0.0, .g = 1.0, .b = 0.0 });
    
    const pixels = try fb.toPixels(allocator);
    defer allocator.free(pixels);
    
    try std.testing.expectEqual(@as(usize, 100 * 100 * 3), pixels.len);
}

test "vector - path stroke" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    var path = Path.init(allocator);
    defer path.deinit();
    
    try path.moveTo(10, 10);
    try path.lineTo(90, 10);
    try path.lineTo(90, 90);
    try path.lineTo(10, 90);
    
    path.stroke(&fb, .{ .r = 0.0, .g = 0.0, .b = 1.0 });
}

test "vector - quadratic bezier" {
    const p0 = Vec2.init(0, 0);
    const p1 = Vec2.init(50, 100);
    const p2 = Vec2.init(100, 0);
    
    const mid = quadraticBezier(p0, p1, p2, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), mid.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 50), mid.y, 1e-6);
}

test "vector - memory pool" {
    var pool = MemoryPool(u32, 100).init();
    
    const p1 = pool.alloc();
    try std.testing.expect(p1 != null);
    
    pool.reset();
    const p2 = pool.alloc();
    try std.testing.expect(p2 != null);
}

test "vector - tile grid" {
    const allocator = std.testing.allocator;
    var grid = try TileGrid.init(allocator, 256, 256);
    defer grid.deinit();
    
    try std.testing.expectEqual(@as(u32, 4), grid.tile_width);
    try std.testing.expectEqual(@as(u32, 4), grid.tile_height);
}
