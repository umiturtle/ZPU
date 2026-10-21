//! ZAPP Geometry Primitives
//!
//! 純粋なジオメトリ定義。描画機能は含まない。
//! このモジュールは他のモジュールに依存せず、独立してテスト・最適化できる。
//!
//! ## 設計原則
//! - **純粋なデータ構造**: 描画ロジックを含まない
//! - **ゼロ依存**: 外部モジュールに依存しない
//! - **comptime最適化**: コンパイル時計算を積極活用
//! - **個別テスト**: 独立して完全にテスト可能

const std = @import("std");

// ============================================================================
// 基本ベクトル
// ============================================================================

/// 2Dベクトル（位置、方向、サイズ）
pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }

    // 演算
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

    pub fn distance(self: Vec2, other: Vec2) f32 {
        return self.sub(other).length();
    }

    pub fn distanceSquared(self: Vec2, other: Vec2) f32 {
        return self.sub(other).lengthSquared();
    }

    // 変換
    pub fn rotate(self: Vec2, radians: f32) Vec2 {
        const c = @cos(radians);
        const s = @sin(radians);
        return .{
            .x = self.x * c - self.y * s,
            .y = self.x * s + self.y * c,
        };
    }

    pub fn angle(self: Vec2) f32 {
        return std.math.atan2(self.y, self.x);
    }
};

// ============================================================================
// バウンディングボックス
// ============================================================================

/// 軸整合バウンディングボックス（AABB）
pub const BoundingBox = struct {
    min: Vec2,
    max: Vec2,

    pub fn init(min: Vec2, max: Vec2) BoundingBox {
        return .{ .min = min, .max = max };
    }

    pub fn fromPoints(points: []const Vec2) BoundingBox {
        if (points.len == 0) return .{ .min = .{ .x = 0, .y = 0 }, .max = .{ .x = 0, .y = 0 } };
        
        var result = BoundingBox{ .min = points[0], .max = points[0] };
        for (points[1..]) |p| {
            result.min = result.min.min(p);
            result.max = result.max.max(p);
        }
        return result;
    }

    pub fn fromCenterSize(center_point: Vec2, dimensions: Vec2) BoundingBox {
        const half = dimensions.scale(0.5);
        return .{
            .min = center_point.sub(half),
            .max = center_point.add(half),
        };
    }

    pub fn contains(self: BoundingBox, point: Vec2) bool {
        return point.x >= self.min.x and point.x <= self.max.x and
               point.y >= self.min.y and point.y <= self.max.y;
    }

    pub fn intersects(self: BoundingBox, other: BoundingBox) bool {
        return self.min.x <= other.max.x and self.max.x >= other.min.x and
               self.min.y <= other.max.y and self.max.y >= other.min.y;
    }

    pub fn width(self: BoundingBox) f32 {
        return self.max.x - self.min.x;
    }

    pub fn height(self: BoundingBox) f32 {
        return self.max.y - self.min.y;
    }

    pub fn size(self: BoundingBox) Vec2 {
        return self.max.sub(self.min);
    }

    pub fn center(self: BoundingBox) Vec2 {
        return self.min.add(self.max).scale(0.5);
    }

    pub fn expand(self: BoundingBox, amount: f32) BoundingBox {
        const offset = Vec2.init(amount, amount);
        return .{
            .min = self.min.sub(offset),
            .max = self.max.add(offset),
        };
    }

    pub fn merge(self: BoundingBox, other: BoundingBox) BoundingBox {
        return .{
            .min = self.min.min(other.min),
            .max = self.max.max(other.max),
        };
    }

    pub fn area(self: BoundingBox) f32 {
        return self.width() * self.height();
    }
};

// ============================================================================
// ベジェ曲線
// ============================================================================

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
    
    const d = pt1.sub(pt0).length();
    if (d < epsilon) return 0;
    
    return 1.0 / d;
}

// ============================================================================
// パスコマンド
// ============================================================================

/// パスコマンド（描画命令）
pub const PathCommand = union(enum) {
    moveTo: Vec2,
    lineTo: Vec2,
    quadTo: struct { control: Vec2, end: Vec2 },
    cubicTo: struct { control1: Vec2, control2: Vec2, end: Vec2 },
    close: void,
};

/// パス（図形の輪郭）
pub const Path = struct {
    commands: []const PathCommand,
    
    pub fn getBounds(self: Path) BoundingBox {
        // パスのバウンディングボックスを計算
        var points = std.ArrayList(Vec2).init(std.heap.page_allocator);
        defer points.deinit();
        
        var current = Vec2.init(0, 0);
        
        for (self.commands) |cmd| {
            switch (cmd) {
                .moveTo => |p| {
                    current = p;
                    _ = points.append(p);
                },
                .lineTo => |p| {
                    current = p;
                    _ = points.append(p);
                },
                .quadTo => |q| {
                    // ベジェ曲線をサンプリング
                    const steps = 16;
                    var i: usize = 1;
                    while (i <= steps) : (i += 1) {
                        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
                        const p = quadraticBezier(current, q.control, q.end, t);
                        _ = points.append(p);
                    }
                    current = q.end;
                },
                .cubicTo => |c| {
                    _ = c;
                    // TODO: 3次ベジェの実装
                },
                .close => {},
            }
        }
        
        return BoundingBox.fromPoints(points.items);
    }
};

// ============================================================================
// コンパイル時図形生成
// ============================================================================

/// コンパイル時に正多角形を生成
pub fn regularPolygon(comptime sides: u32) type {
    if (sides < 3) @compileError("Polygon must have at least 3 sides");
    
    return struct {
        pub fn generate(radius: f32) [sides]Vec2 {
            var points: [sides]Vec2 = undefined;
            const angle_step = 2.0 * std.math.pi / @as(f32, @floatFromInt(sides));
            
            inline for (0..sides) |i| {
                const angle = @as(f32, @floatFromInt(i)) * angle_step - std.math.pi / 2.0;
                points[i] = .{
                    .x = radius * @cos(angle),
                    .y = radius * @sin(angle),
                };
            }
            
            return points;
        }
        
        pub fn generateCentered(center: Vec2, radius: f32) [sides]Vec2 {
            var points = generate(radius);
            for (&points) |*p| {
                p.* = p.add(center);
            }
            return points;
        }
    };
}

/// コンパイル時に星形を生成
pub fn star(comptime points: u32, comptime inner_ratio: f32) type {
    if (points < 3) @compileError("Star must have at least 3 points");
    if (inner_ratio <= 0 or inner_ratio >= 1) @compileError("Inner ratio must be between 0 and 1");
    
    return struct {
        pub fn generate(radius: f32) [points * 2]Vec2 {
            var result: [points * 2]Vec2 = undefined;
            const angle_step = std.math.pi / @as(f32, @floatFromInt(points));
            const inner_radius = radius * inner_ratio;
            
            inline for (0..points * 2) |i| {
                const angle = @as(f32, @floatFromInt(i)) * angle_step - std.math.pi / 2.0;
                const r = if (i % 2 == 0) radius else inner_radius;
                result[i] = .{
                    .x = r * @cos(angle),
                    .y = r * @sin(angle),
                };
            }
            
            return result;
        }
    };
}

// ============================================================================
// テスト
// ============================================================================

test "geometry - Vec2 operations" {
    const a = Vec2.init(3, 4);
    const b = Vec2.init(1, 2);
    
    const sum = a.add(b);
    try std.testing.expectEqual(@as(f32, 4), sum.x);
    try std.testing.expectEqual(@as(f32, 6), sum.y);
    
    const len = a.length();
    try std.testing.expectApproxEqAbs(@as(f32, 5), len, 1e-6);
    
    const normalized = a.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1), normalized.length(), 1e-6);
}

test "geometry - Vec2 rotation" {
    const v = Vec2.init(1, 0);
    const rotated = v.rotate(std.math.pi / 2.0);
    
    try std.testing.expectApproxEqAbs(@as(f32, 0), rotated.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), rotated.y, 1e-6);
}

test "geometry - BoundingBox" {
    const bbox = BoundingBox.init(.{ .x = 10, .y = 20 }, .{ .x = 100, .y = 200 });
    
    try std.testing.expect(bbox.contains(.{ .x = 50, .y = 100 }));
    try std.testing.expect(!bbox.contains(.{ .x = 5, .y = 100 }));
    
    try std.testing.expectEqual(@as(f32, 90), bbox.width());
    try std.testing.expectEqual(@as(f32, 180), bbox.height());
    
    const center = bbox.center();
    try std.testing.expectApproxEqAbs(@as(f32, 55), center.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 110), center.y, 1e-6);
}

test "geometry - BoundingBox intersection" {
    const a = BoundingBox.init(.{ .x = 0, .y = 0 }, .{ .x = 100, .y = 100 });
    const b = BoundingBox.init(.{ .x = 50, .y = 50 }, .{ .x = 150, .y = 150 });
    const c = BoundingBox.init(.{ .x = 200, .y = 200 }, .{ .x = 300, .y = 300 });
    
    try std.testing.expect(a.intersects(b));
    try std.testing.expect(!a.intersects(c));
}

test "geometry - quadratic bezier" {
    const p0 = Vec2.init(0, 0);
    const p1 = Vec2.init(50, 100);
    const p2 = Vec2.init(100, 0);
    
    const mid = quadraticBezier(p0, p1, p2, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), mid.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 50), mid.y, 1e-6);
}

test "geometry - cubic bezier" {
    const p0 = Vec2.init(0, 0);
    const p1 = Vec2.init(33, 100);
    const p2 = Vec2.init(66, 100);
    const p3 = Vec2.init(100, 0);
    
    const mid = cubicBezier(p0, p1, p2, p3, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 49.625), mid.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 75), mid.y, 1e-5);
}

test "geometry - regular polygon (hexagon)" {
    const Hexagon = regularPolygon(6);
    const points = Hexagon.generate(50.0);
    
    try std.testing.expectEqual(@as(usize, 6), points.len);
    
    // 最初の点は上向き（-90度）
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), points[0].x, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -50.0), points[0].y, 0.01);
}

test "geometry - regular polygon centered" {
    const Triangle = regularPolygon(3);
    const center = Vec2.init(100, 100);
    const points = Triangle.generateCentered(center, 50.0);
    
    try std.testing.expectEqual(@as(usize, 3), points.len);
    
    // 頂点の平均（重心）が(100, 100)であることを確認
    var sum_x: f32 = 0;
    var sum_y: f32 = 0;
    for (points) |p| {
        sum_x += p.x;
        sum_y += p.y;
    }
    const centroid_x = sum_x / @as(f32, @floatFromInt(points.len));
    const centroid_y = sum_y / @as(f32, @floatFromInt(points.len));
    try std.testing.expectApproxEqAbs(@as(f32, 100), centroid_x, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 100), centroid_y, 1.0);
}

test "geometry - star" {
    const Star5 = star(5, 0.5);
    const points = Star5.generate(50.0);
    
    try std.testing.expectEqual(@as(usize, 10), points.len);
}
