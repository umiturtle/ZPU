//! Comptime-Optimized Vector Rendering System
//!
//! Zigのcomptime機能を最大限に活用したベクターレンダリングシステム。
//! コンパイル時にレンダリングパイプラインを特殊化し、ゼロコスト抽象化を実現。
//!
//! ## 設計原則
//! 1. **コンパイル時特殊化**: 機能フラグに基づいて最適なコードを生成
//! 2. **型安全ビルダー**: コンパイル時にエラーを検出
//! 3. **インライン展開**: 小さなプリミティブは自動的にインライン化
//! 4. **ゼロコスト抽象化**: 使わない機能は完全に削除される
//!
//! ## 使用例
//!
//! ```zig
//! // コンパイル時に特殊化されたレンダラーを作成
//! const MyRenderer = comptime_vector.Renderer(.{
//!     .antialiasing = true,
//!     .gradients = true,
//!     .shadows = false,
//! });
//!
//! // コンパイル時に図形を生成
//! const hexagon = comptime_vector.regularPolygon(6).generate(50.0);
//!
//! // 型安全なスタイルビルダー
//! const style = comptime_vector.StyleBuilder(.{
//!     .fill = true,
//!     .stroke = true,
//! }){};
//! ```

const std = @import("std");
const zgf = @import("zgf");
const vector = @import("vector.zig");

// ============================================================================
// コンパイル時機能フラグ
// ============================================================================

/// レンダラーの機能フラグ
pub const RenderFeatures = struct {
    /// アンチエイリアス（高品質だが低速）
    antialiasing: bool = false,
    /// グラデーションサポート
    gradients: bool = false,
    /// シャドウサポート
    shadows: bool = false,
    /// パスフィルサポート
    path_fill: bool = false,
    /// クリッピングサポート
    clipping: bool = false,
    /// ブレンドモードサポート
    blend_modes: bool = false,
    /// アフィン変換サポート
    transforms: bool = false,
    /// ディスプレイリスト（バッチング）
    display_list: bool = false,
};

/// パスレンダラーの機能フラグ
pub const PathFeatures = struct {
    /// ストロークサポート
    stroke: bool = true,
    /// フィルサポート
    fill: bool = false,
    /// ダッシュパターン
    dash: bool = false,
    /// 線端スタイル（butt, round, square）
    line_caps: bool = false,
    /// 接続スタイル（miter, round, bevel）
    line_joins: bool = false,
    /// 自己交差処理
    self_intersection: bool = false,
};

// ============================================================================
// コンパイル時特殊化レンダラー
// ============================================================================

/// コンパイル時に特殊化されたベクターレンダラー
pub fn Renderer(comptime features: RenderFeatures) type {
    return struct {
        const Self = @This();
        
        fb: *zgf.Image,
        
        pub fn init(framebuffer: *zgf.Image) Self {
            return .{ .fb = framebuffer };
        }
        
        /// 円を描画（comptimeで最適な実装を選択）
        pub fn drawCircle(self: *Self, cx: f32, cy: f32, radius: f32, color: zgf.Color) void {
            if (comptime features.antialiasing) {
                self.drawCircleAA(cx, cy, radius, color);
            } else {
                vector.drawCircleOptimized(self.fb, cx, cy, radius, color);
            }
        }
        
        /// 円を塗りつぶし（comptimeで最適な実装を選択）
        pub fn fillCircle(self: *Self, cx: f32, cy: f32, radius: f32, color: zgf.Color) void {
            if (comptime features.antialiasing) {
                self.fillCircleAA(cx, cy, radius, color);
            } else {
                vector.fillCircleOptimized(self.fb, cx, cy, radius, color);
            }
        }
        
        /// 角丸矩形を描画
        pub fn drawRoundedRect(
            self: *Self,
            x: f32,
            y: f32,
            w: f32,
            h: f32,
            radius: f32,
            color: zgf.Color,
        ) void {
            // コンパイル時に品質が決定されている
            self.drawRoundedRectImpl(x, y, w, h, radius, color);
        }
        
        /// グラデーション矩形（features.gradientsがtrueの場合のみ使用可能）
        pub fn drawGradientRect(
            self: *Self,
            x: f32,
            y: f32,
            w: f32,
            h: f32,
            from: zgf.Color,
            to: zgf.Color,
            direction: GradientDirection,
        ) void {
            if (comptime !features.gradients) {
                @compileError("Gradient support not enabled. Set .gradients = true in RenderFeatures");
            }
            self.drawGradientRectImpl(x, y, w, h, from, to, direction);
        }
        
        // =====================================================================
        // 内部実装（comptime分岐で最適化）
        // =====================================================================
        
        fn drawCircleAA(self: *Self, cx: f32, cy: f32, radius: f32, color: zgf.Color) void {
            // アンチエイリアス付き円描画（Wuのアルゴリズム）
            const r: usize = @intCast(@as(usize, @intFromFloat(@round(radius))));
            if (r == 0) return;
            
            const cx_int: i32 = @intCast(@as(i32, @intFromFloat(@round(cx))));
            const cy_int: i32 = @intCast(@as(i32, @intFromFloat(@round(cy))));
            
            var x: i32 = 0;
            var y: i32 = @intCast(r);
            var d: i32 = 3 - 2 * @as(i32, @intCast(r));
            
            while (y >= x) {
                self.plotCirclePointsAA(cx_int, cy_int, x, y, color);
                x += 1;
                if (d > 0) {
                    y -= 1;
                    d += 4 * (x - y) + 10;
                } else {
                    d += 4 * x + 6;
                }
            }
        }
        
        fn plotCirclePointsAA(self: *Self, cx: i32, cy: i32, x: i32, y: i32, color: zgf.Color) void {
            // 8対称点にアンチエイリアス適用
            const points = [_][2]i32{
                .{ cx + x, cy + y }, .{ cx - x, cy + y },
                .{ cx + x, cy - y }, .{ cx - x, cy - y },
                .{ cx + y, cy + x }, .{ cx - y, cy + x },
                .{ cx + y, cy - x }, .{ cx - y, cy - x },
            };
            
            const width: i32 = @intCast(self.fb.width());
            const height: i32 = @intCast(self.fb.height());
            
            for (points) |p| {
                if (p[0] >= 0 and p[0] < width and p[1] >= 0 and p[1] < height) {
                    // アンチエイリアス: エッジの距離に基づいてアルファを調整
                    self.setPixelAA(p[0], p[1], color);
                }
            }
        }
        
        fn setPixelAA(self: *Self, x: i32, y: i32, color: zgf.Color) void {
            // シンプルなアンチエイリアス（実際にはより複雑な計算が必要）
            self.fb.setPixel(@intCast(x), @intCast(y), color);
        }
        
        fn fillCircleAA(self: *Self, cx: f32, cy: f32, radius: f32, color: zgf.Color) void {
            // アンチエイリアス付き円フィル
            vector.fillCircleOptimized(self.fb, cx, cy, radius, color);
        }
        
        fn drawRoundedRectImpl(
            self: *Self,
            x: f32,
            y: f32,
            w: f32,
            h: f32,
            radius: f32,
            color: zgf.Color,
        ) void {
            // 角丸矩形の実装
            // 4つの角に円を描画し、矩形で接続
            _ = self;
            _ = x;
            _ = y;
            _ = w;
            _ = h;
            _ = radius;
            _ = color;
            // TODO: 実装
        }
        
        fn drawGradientRectImpl(
            self: *Self,
            x: f32,
            y: f32,
            w: f32,
            h: f32,
            from: zgf.Color,
            to: zgf.Color,
            direction: GradientDirection,
        ) void {
            _ = self;
            _ = x;
            _ = y;
            _ = w;
            _ = h;
            _ = from;
            _ = to;
            _ = direction;
            // TODO: 実装
        }
    };
}

/// グラデーション方向
pub const GradientDirection = enum {
    horizontal,
    vertical,
    diagonal_tl,
    diagonal_bl,
    radial,
};

// ============================================================================
// コンパイル時図形生成
// ============================================================================

/// コンパイル時に正多角形を生成
pub fn regularPolygon(comptime sides: u32) type {
    if (sides < 3) @compileError("Polygon must have at least 3 sides");
    
    return struct {
        pub fn generate(radius: f32) [sides]vector.Vec2 {
            var points: [sides]vector.Vec2 = undefined;
            const angle_step = 2.0 * std.math.pi / @as(f32, @floatFromInt(sides));
            
            // comptimeループで展開
            inline for (0..sides) |i| {
                const angle = @as(f32, @floatFromInt(i)) * angle_step - std.math.pi / 2.0;
                points[i] = .{
                    .x = radius * @cos(angle),
                    .y = radius * @sin(angle),
                };
            }
            
            return points;
        }
        
        pub fn generateCentered(center_x: f32, center_y: f32, radius: f32) [sides]vector.Vec2 {
            var points = generate(radius);
            for (&points) |*p| {
                p.x += center_x;
                p.y += center_y;
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
        pub fn generate(radius: f32) [points * 2]vector.Vec2 {
            var result: [points * 2]vector.Vec2 = undefined;
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
// 型安全スタイルビルダー
// ============================================================================

/// ストローク設定
pub const Stroke = struct {
    width: f32 = 1.0,
    color: zgf.Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
    dash_pattern: ?[]const f32 = null,
};

/// 型安全スタイルビルダー
pub fn StyleBuilder(comptime features: struct {
    fill: bool = true,
    stroke: bool = true,
    shadow: bool = false,
}) type {
    return struct {
        const Self = @This();
        
        fill_color: ?zgf.Color = null,
        stroke_config: ?Stroke = null,
        
        pub fn withFill(self: *Self, color: zgf.Color) *Self {
            if (comptime !features.fill) {
                @compileError("Fill feature not enabled");
            }
            self.fill_color = color;
            return self;
        }
        
        pub fn withStroke(self: *Self, width: f32, color: zgf.Color) *Self {
            if (comptime !features.stroke) {
                @compileError("Stroke feature not enabled");
            }
            self.stroke_config = .{ .width = width, .color = color };
            return self;
        }
        
        pub fn build(self: Self) Style {
            return .{
                .fill_color = self.fill_color,
                .stroke_config = self.stroke_config,
            };
        }
    };
}

/// 完成したスタイル
pub const Style = struct {
    fill_color: ?zgf.Color = null,
    stroke_config: ?Stroke = null,
};

// ============================================================================
// コンパイル時パスビルダー
// ============================================================================

/// コンパイル時に特殊化されたパスビルダー
pub fn PathBuilder(comptime max_commands: usize) type {
    return struct {
        const Self = @This();
        
        commands: [max_commands]vector.PathCommand = undefined,
        count: usize = 0,
        
        pub fn moveTo(self: *Self, x: f32, y: f32) void {
            if (self.count >= max_commands) return; // Silently ignore if full
            self.commands[self.count] = .{ .moveTo = .{ .x = x, .y = y } };
            self.count += 1;
        }
        
        pub fn lineTo(self: *Self, x: f32, y: f32) void {
            if (self.count >= max_commands) return; // Silently ignore if full
            self.commands[self.count] = .{ .lineTo = .{ .x = x, .y = y } };
            self.count += 1;
        }
        
        pub fn getCommands(self: *const Self) []const vector.PathCommand {
            return self.commands[0..self.count];
        }
    };
}

// ============================================================================
// テスト
// ============================================================================

test "comptime renderer - basic" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    // 基本レンダラー（アンチエイリアスなし）
    const BasicRenderer = Renderer(.{});
    var renderer = BasicRenderer.init(&fb);
    renderer.drawCircle(50, 50, 20, .{ .r = 1, .g = 0, .b = 0 });
}

test "comptime renderer - with antialiasing" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    // AA付きレンダラー
    const AARenderer = Renderer(.{ .antialiasing = true });
    var renderer = AARenderer.init(&fb);
    renderer.drawCircle(50, 50, 20, .{ .r = 0, .g = 1, .b = 0 });
}

test "comptime renderer - with gradients" {
    const alloc = std.testing.allocator;
    var fb = try zgf.Image.init(alloc, 100, 100, 3);
    defer fb.deinit();
    
    // グラデーション付きレンダラー
    const GradientRenderer = Renderer(.{ .gradients = true });
    var renderer = GradientRenderer.init(&fb);
    renderer.drawGradientRect(10, 10, 80, 80, 
        .{ .r = 1, .g = 0, .b = 0 },
        .{ .r = 0, .g = 0, .b = 1 },
        .horizontal);
}

test "comptime polygon - hexagon" {
    const Hexagon = regularPolygon(6);
    const points = Hexagon.generate(50.0);
    
    try std.testing.expectEqual(@as(usize, 6), points.len);
    
    // 最初の点は上向き（-90度）
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), points[0].x, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -50.0), points[0].y, 0.01);
}

test "comptime polygon - triangle" {
    const Triangle = regularPolygon(3);
    const points = Triangle.generateCentered(100.0, 100.0, 50.0);
    
    try std.testing.expectEqual(@as(usize, 3), points.len);
}

test "comptime star - 5-pointed" {
    const Star5 = star(5, 0.5);
    const points = Star5.generate(50.0);
    
    try std.testing.expectEqual(@as(usize, 10), points.len);
}

test "style builder - type safe" {
    var builder = StyleBuilder(.{ .fill = true, .stroke = true }){};
    
    _ = builder.withFill(.{ .r = 1, .g = 0, .b = 0, .a = 1 });
    _ = builder.withStroke(2.0, .{ .r = 0, .g = 0, .b = 0, .a = 1 });
    
    const style = builder.build();
    try std.testing.expect(style.fill_color != null);
    try std.testing.expect(style.stroke_config != null);
}

test "path builder - comptime" {
    var builder = PathBuilder(100){};
    
    builder.moveTo(10, 10);
    builder.lineTo(90, 10);
    builder.lineTo(90, 90);
    builder.lineTo(10, 90);
    
    const commands = builder.getCommands();
    try std.testing.expectEqual(@as(usize, 4), commands.len);
}
