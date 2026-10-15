//! ZAPP 座標系
//!
//! 論理座標（VirtualRect）と物理座標（PhysicalRect）を分離し、
//! HiDPI対応や可変ウィンドウサイズを容易にする。

/// 論理座標系（レイアウト計算用）
///
/// デバイス非依存の座標系。ポイント単位で指定し、
/// レンダリング時に物理ピクセルに変換される。
pub const VirtualRect = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,

    /// 原点（左上）
    pub const origin = VirtualRect{ .x = 0, .y = 0, .w = 0, .h = 0 };

    /// 矩形が空（幅または高さが0以下）かチェック
    pub fn isEmpty(self: VirtualRect) bool {
        return self.w <= 0 or self.h <= 0;
    }

    /// 矩形が指定点を内包するかチェック
    pub fn contains(self: VirtualRect, px: f32, py: f32) bool {
        return px >= self.x and px < self.x + self.w and
            py >= self.y and py < self.y + self.h;
    }

    /// 矩形の交差部分を計算
    pub fn intersection(self: VirtualRect, other: VirtualRect) VirtualRect {
        const x0 = @max(self.x, other.x);
        const y0 = @max(self.y, other.y);
        const x1 = @min(self.x + self.w, other.x + other.w);
        const y1 = @min(self.y + self.h, other.y + other.h);

        if (x0 >= x1 or y0 >= y1) {
            return origin;
        }

        return .{
            .x = x0,
            .y = y0,
            .w = x1 - x0,
            .h = y1 - y0,
        };
    }

    /// 矩形の和（両方を含む最小矩形）を計算
    pub fn unionRect(self: VirtualRect, other: VirtualRect) VirtualRect {
        const x0 = @min(self.x, other.x);
        const y0 = @min(self.y, other.y);
        const x1 = @max(self.x + self.w, other.x + other.w);
        const y1 = @max(self.y + self.h, other.y + other.h);

        return .{
            .x = x0,
            .y = y0,
            .w = x1 - x0,
            .h = y1 - y0,
        };
    }

    /// 矩形を移動
    pub fn translate(self: VirtualRect, dx: f32, dy: f32) VirtualRect {
        return .{
            .x = self.x + dx,
            .y = self.y + dy,
            .w = self.w,
            .h = self.h,
        };
    }

    /// 矩形をスケール
    pub fn scale(self: VirtualRect, sx: f32, sy: f32) VirtualRect {
        return .{
            .x = self.x * sx,
            .y = self.y * sy,
            .w = self.w * sx,
            .h = self.h * sy,
        };
    }

    /// 矩形の中心点を取得
    pub fn center(self: VirtualRect) struct { x: f32, y: f32 } {
        return .{
            .x = self.x + self.w / 2,
            .y = self.y + self.h / 2,
        };
    }

    /// 矩形の右上を取得
    pub fn topRight(self: VirtualRect) struct { x: f32, y: f32 } {
        return .{
            .x = self.x + self.w,
            .y = self.y,
        };
    }

    /// 矩形の左下を取得
    pub fn bottomLeft(self: VirtualRect) struct { x: f32, y: f32 } {
        return .{
            .x = self.x,
            .y = self.y + self.h,
        };
    }

    /// 矩形の右下を取得
    pub fn bottomRight(self: VirtualRect) struct { x: f32, y: f32 } {
        return .{
            .x = self.x + self.w,
            .y = self.y + self.h,
        };
    }
};

/// 物理座標系（レンダリング用）
///
/// 実際のピクセル座標。整数値で指定する。
pub const PhysicalRect = struct {
    x: u32,
    y: u32,
    w: u32,
    h: u32,

    /// 原点（左上）
    pub const origin = PhysicalRect{ .x = 0, .y = 0, .w = 0, .h = 0 };

    /// 矩形が空かチェック
    pub fn isEmpty(self: PhysicalRect) bool {
        return self.w == 0 or self.h == 0;
    }

    /// 矩形が指定点を内包するかチェック
    pub fn contains(self: PhysicalRect, px: u32, py: u32) bool {
        return px >= self.x and px < self.x + self.w and
            py >= self.y and py < self.y + self.h;
    }

    /// 矩形の交差部分を計算
    pub fn intersection(self: PhysicalRect, other: PhysicalRect) PhysicalRect {
        const x0 = @max(self.x, other.x);
        const y0 = @max(self.y, other.y);
        const x1 = @min(self.x + self.w, other.x + other.w);
        const y1 = @min(self.y + self.h, other.y + other.h);

        if (x0 >= x1 or y0 >= y1) {
            return origin;
        }

        return .{
            .x = x0,
            .y = y0,
            .w = x1 - x0,
            .h = y1 - y0,
        };
    }

    /// 矩形の和を計算
    pub fn unionRect(self: PhysicalRect, other: PhysicalRect) PhysicalRect {
        const x0 = @min(self.x, other.x);
        const y0 = @min(self.y, other.y);
        const x1 = @max(self.x + self.w, other.x + other.w);
        const y1 = @max(self.y + self.h, other.y + other.h);

        return .{
            .x = x0,
            .y = y0,
            .w = x1 - x0,
            .h = y1 - y0,
        };
    }

    /// 矩形を移動
    pub fn translate(self: PhysicalRect, dx: u32, dy: u32) PhysicalRect {
        return .{
            .x = self.x + dx,
            .y = self.y + dy,
            .w = self.w,
            .h = self.h,
        };
    }

    /// 矩形の中心点を取得
    pub fn center(self: PhysicalRect) struct { x: u32, y: u32 } {
        return .{
            .x = self.x + self.w / 2,
            .y = self.y + self.h / 2,
        };
    }
};

/// 座標変換コンテキスト
///
/// 論理座標から物理座標への変換を管理する。
/// スケーリング係数（DPI比）を保持し、一貫した変換を提供する。
pub const CoordinateTransform = struct {
    /// スケーリング係数（物理/論理）
    scale_factor: f32,

    /// デフォルトのスケーリング係数（1.0 = 等倍）
    pub const default_scale: f32 = 1.0;

    /// 変換コンテキストを初期化
    pub fn init(scale_factor: f32) CoordinateTransform {
        return .{ .scale_factor = scale_factor };
    }

    /// デフォルトの変換コンテキスト（1倍）
    pub fn default() CoordinateTransform {
        return .{ .scale_factor = default_scale };
    }

    /// 論理座標を物理座標に変換
    pub fn toPhysical(self: CoordinateTransform, v: VirtualRect) PhysicalRect {
        return .{
            .x = @intFromFloat(@round(v.x * self.scale_factor)),
            .y = @intFromFloat(@round(v.y * self.scale_factor)),
            .w = @intFromFloat(@round(v.w * self.scale_factor)),
            .h = @intFromFloat(@round(v.h * self.scale_factor)),
        };
    }

    /// 物理座標を論理座標に変換
    pub fn toVirtual(self: CoordinateTransform, p: PhysicalRect) VirtualRect {
        return .{
            .x = @as(f32, @floatFromInt(p.x)) / self.scale_factor,
            .y = @as(f32, @floatFromInt(p.y)) / self.scale_factor,
            .w = @as(f32, @floatFromInt(p.w)) / self.scale_factor,
            .h = @as(f32, @floatFromInt(p.h)) / self.scale_factor,
        };
    }

    /// 論理距離を物理距離に変換
    pub fn toPhysicalLength(self: CoordinateTransform, length: f32) u32 {
        return @intFromFloat(@round(length * self.scale_factor));
    }

    /// 物理距離を論理距離に変換
    pub fn toVirtualLength(self: CoordinateTransform, length: u32) f32 {
        return @as(f32, @floatFromInt(length)) / self.scale_factor;
    }
};

// テスト

test "VirtualRect - isEmpty" {
    const empty = VirtualRect.origin;
    try @import("std").testing.expect(empty.isEmpty());

    const valid = VirtualRect{ .x = 0, .y = 0, .w = 10, .h = 10 };
    try @import("std").testing.expect(!valid.isEmpty());
}

test "VirtualRect - contains" {
    const rect = VirtualRect{ .x = 10, .y = 10, .w = 20, .h = 20 };

    try @import("std").testing.expect(rect.contains(15, 15));
    try @import("std").testing.expect(!rect.contains(5, 5));
    try @import("std").testing.expect(!rect.contains(30, 30));
}

test "VirtualRect - intersection" {
    const a = VirtualRect{ .x = 0, .y = 0, .w = 20, .h = 20 };
    const b = VirtualRect{ .x = 10, .y = 10, .w = 20, .h = 20 };

    const inter = a.intersection(b);
    try @import("std").testing.expectEqual(@as(f32, 10), inter.x);
    try @import("std").testing.expectEqual(@as(f32, 10), inter.y);
    try @import("std").testing.expectEqual(@as(f32, 10), inter.w);
    try @import("std").testing.expectEqual(@as(f32, 10), inter.h);
}

test "VirtualRect - intersection no overlap" {
    const a = VirtualRect{ .x = 0, .y = 0, .w = 10, .h = 10 };
    const b = VirtualRect{ .x = 20, .y = 20, .w = 10, .h = 10 };

    const inter = a.intersection(b);
    try @import("std").testing.expect(inter.isEmpty());
}

test "VirtualRect - unionRect" {
    const a = VirtualRect{ .x = 0, .y = 0, .w = 10, .h = 10 };
    const b = VirtualRect{ .x = 5, .y = 5, .w = 10, .h = 10 };

    const u = a.unionRect(b);
    try @import("std").testing.expectEqual(@as(f32, 0), u.x);
    try @import("std").testing.expectEqual(@as(f32, 0), u.y);
    try @import("std").testing.expectEqual(@as(f32, 15), u.w);
    try @import("std").testing.expectEqual(@as(f32, 15), u.h);
}

test "VirtualRect - translate" {
    const rect = VirtualRect{ .x = 10, .y = 10, .w = 20, .h = 20 };
    const moved = rect.translate(5, -5);

    try @import("std").testing.expectEqual(@as(f32, 15), moved.x);
    try @import("std").testing.expectEqual(@as(f32, 5), moved.y);
    try @import("std").testing.expectEqual(@as(f32, 20), moved.w);
    try @import("std").testing.expectEqual(@as(f32, 20), moved.h);
}

test "VirtualRect - scale" {
    const rect = VirtualRect{ .x = 10, .y = 10, .w = 20, .h = 20 };
    const scaled = rect.scale(2, 2);

    try @import("std").testing.expectEqual(@as(f32, 20), scaled.x);
    try @import("std").testing.expectEqual(@as(f32, 20), scaled.y);
    try @import("std").testing.expectEqual(@as(f32, 40), scaled.w);
    try @import("std").testing.expectEqual(@as(f32, 40), scaled.h);
}

test "PhysicalRect - isEmpty" {
    const empty = PhysicalRect.origin;
    try @import("std").testing.expect(empty.isEmpty());

    const valid = PhysicalRect{ .x = 0, .y = 0, .w = 10, .h = 10 };
    try @import("std").testing.expect(!valid.isEmpty());
}

test "PhysicalRect - contains" {
    const rect = PhysicalRect{ .x = 10, .y = 10, .w = 20, .h = 20 };

    try @import("std").testing.expect(rect.contains(15, 15));
    try @import("std").testing.expect(!rect.contains(5, 5));
}

test "CoordinateTransform - default" {
    const ctx = CoordinateTransform.default();
    try @import("std").testing.expectEqual(@as(f32, 1.0), ctx.scale_factor);
}

test "CoordinateTransform - toPhysical" {
    const ctx = CoordinateTransform.init(2.0);
    const v = VirtualRect{ .x = 10, .y = 20, .w = 30, .h = 40 };
    const p = ctx.toPhysical(v);

    try @import("std").testing.expectEqual(@as(u32, 20), p.x);
    try @import("std").testing.expectEqual(@as(u32, 40), p.y);
    try @import("std").testing.expectEqual(@as(u32, 60), p.w);
    try @import("std").testing.expectEqual(@as(u32, 80), p.h);
}

test "CoordinateTransform - toVirtual" {
    const ctx = CoordinateTransform.init(2.0);
    const p = PhysicalRect{ .x = 20, .y = 40, .w = 60, .h = 80 };
    const v = ctx.toVirtual(p);

    try @import("std").testing.expectEqual(@as(f32, 10), v.x);
    try @import("std").testing.expectEqual(@as(f32, 20), v.y);
    try @import("std").testing.expectEqual(@as(f32, 30), v.w);
    try @import("std").testing.expectEqual(@as(f32, 40), v.h);
}

test "CoordinateTransform - roundtrip" {
    const ctx = CoordinateTransform.init(1.5);
    const v = VirtualRect{ .x = 10, .y = 20, .w = 30, .h = 40 };
    const p = ctx.toPhysical(v);
    const v2 = ctx.toVirtual(p);

    // 丸め誤差があるが、近い値になるはず
    try @import("std").testing.expectApproxEqAbs(@as(f32, 10), v2.x, 1);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 20), v2.y, 1);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 30), v2.w, 1);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 40), v2.h, 1);
}
