//! Adaptive Rendering System
//!
//! 実行時に品質を動的に調整するシステム
//! コンパイル時の基本品質をベースに、以下の要因で動的に調整:
//! - ウィンドウ/画面サイズ
//! - ズームレベル
//! - パフォーマンス要件
//! - 個々のコンポーネントの重要度

const std = @import("std");
const rich_render = @import("rich_render.zig");
const build_options = @import("build_options");

/// 画面コンテキスト
pub const Viewport = struct {
    width: f32,
    height: f32,
    scale_factor: f32 = 1.0, // HiDPI対応
    zoom: f32 = 1.0,
    pan_x: f32 = 0.0,
    pan_y: f32 = 0.0,

    pub fn effectiveWidth(self: Viewport) f32 {
        return self.width * self.scale_factor / self.zoom;
    }

    pub fn effectiveHeight(self: Viewport) f32 {
        return self.height * self.scale_factor / self.zoom;
    }

    pub fn contains(self: Viewport, x: f32, y: f32) bool {
        const view_left = self.pan_x;
        const view_top = self.pan_y;
        const view_right = view_left + self.effectiveWidth();
        const view_bottom = view_top + self.effectiveHeight();
        return x >= view_left and x <= view_right and y >= view_top and y <= view_bottom;
    }

    /// スクリーン座標からワールド座標へ変換
    pub fn screenToWorld(self: Viewport, screen_x: f32, screen_y: f32) struct { x: f32, y: f32 } {
        const world_x = screen_x / self.scale_factor * self.zoom + self.pan_x;
        const world_y = screen_y / self.scale_factor * self.zoom + self.pan_y;
        return .{ .x = world_x, .y = world_y };
    }

    /// ワールド座標からスクリーン座標へ変換
    pub fn worldToScreen(self: Viewport, world_x: f32, world_y: f32) struct { x: f32, y: f32 } {
        const screen_x = (world_x - self.pan_x) / self.zoom * self.scale_factor;
        const screen_y = (world_y - self.pan_y) / self.zoom * self.scale_factor;
        return .{ .x = screen_x, .y = screen_y };
    }
};

/// 適応的品質設定
pub const AdaptiveConfig = struct {
    // ベースライン（コンパイル時）
    base_quality: rich_render.RenderQuality = parseQuality(build_options.render_quality),

    // 動的調整パラメータ
    min_quality: rich_render.RenderQuality = .low,
    max_quality: rich_render.RenderQuality = .ultra,

    // ズームレベル閾値
    zoom_high_threshold: f32 = 2.0, // このズーム以上でhigh品質
    zoom_ultra_threshold: f32 = 4.0, // このズーム以上でultra品質
    zoom_low_threshold: f32 = 0.5, // このズーム以下でlow品質

    // サイズ閾値
    small_screen_threshold: f32 = 800.0, // この以下でlow品質
    large_screen_threshold: f32 = 1920.0, // この以上でhigh品質

    // パフォーマンス設定
    target_fps: u32 = 60,
    quality_degradation_threshold: u32 = 30, // FPSがこれ以下なら品質を下げる

    /// 現在のビューポートに基づく品質を計算
    pub fn calculateQuality(self: AdaptiveConfig, viewport: Viewport) rich_render.RenderQuality {
        var quality_val = @intFromEnum(self.base_quality);

        // ズームレベルによる調整
        if (viewport.zoom >= self.zoom_ultra_threshold) {
            quality_val = @intFromEnum(rich_render.RenderQuality.ultra);
        } else if (viewport.zoom >= self.zoom_high_threshold) {
            const high_val = @intFromEnum(rich_render.RenderQuality.high);
            quality_val = @max(quality_val, high_val);
        } else if (viewport.zoom <= self.zoom_low_threshold) {
            const low_val = @intFromEnum(rich_render.RenderQuality.low);
            quality_val = @min(quality_val, low_val);
        }

        // 画面サイズによる調整
        const effective_size = @max(viewport.effectiveWidth(), viewport.effectiveHeight());
        if (effective_size >= self.large_screen_threshold) {
            const high_val = @intFromEnum(rich_render.RenderQuality.high);
            quality_val = @max(quality_val, high_val);
        } else if (effective_size <= self.small_screen_threshold) {
            const medium_val = @intFromEnum(rich_render.RenderQuality.medium);
            quality_val = @min(quality_val, medium_val);
        }

        // 範囲内に制限
        return clampQuality(@enumFromInt(quality_val), self.min_quality, self.max_quality);
    }
};

/// 品質を範囲内に制限
fn clampQuality(
    quality: rich_render.RenderQuality,
    min_q: rich_render.RenderQuality,
    max_q: rich_render.RenderQuality,
) rich_render.RenderQuality {
    const q_val = @intFromEnum(quality);
    const min_val = @intFromEnum(min_q);
    const max_val = @intFromEnum(max_q);
    const clamped = @max(min_val, @min(max_val, q_val));
    return @enumFromInt(clamped);
}

/// 文字列から品質に変換
fn parseQuality(quality_str: []const u8) rich_render.RenderQuality {
    if (std.mem.eql(u8, quality_str, "low")) return .low;
    if (std.mem.eql(u8, quality_str, "medium")) return .medium;
    if (std.mem.eql(u8, quality_str, "high")) return .high;
    if (std.mem.eql(u8, quality_str, "ultra")) return .ultra;
    return .medium; // デフォルト
}

/// 適応的レンダーコンテキスト
pub const AdaptiveContext = struct {
    config: AdaptiveConfig,
    viewport: Viewport,
    current_quality: rich_render.RenderQuality,
    performance_monitor: PerformanceMonitor,

    pub fn init(config: AdaptiveConfig, viewport: Viewport) AdaptiveContext {
        return .{
            .config = config,
            .viewport = viewport,
            .current_quality = config.calculateQuality(viewport),
            .performance_monitor = PerformanceMonitor.init(),
        };
    }

    /// ビューポート更新時に品質を再計算
    pub fn updateViewport(self: *AdaptiveContext, new_viewport: Viewport) void {
        self.viewport = new_viewport;
        self.current_quality = self.config.calculateQuality(new_viewport);
    }

    /// パフォーマンスに基づいて品質を調整
    pub fn adjustForPerformance(self: *AdaptiveContext, fps: u32) void {
        self.performance_monitor.recordFps(fps);

        if (fps < self.config.quality_degradation_threshold) {
            // FPSが低い場合、品質を下げる
            const current_val = @intFromEnum(self.current_quality);
            if (current_val > @intFromEnum(self.config.min_quality)) {
                self.current_quality = @enumFromInt(current_val - 1);
            }
        }
    }

    /// RenderConfigを生成
    pub fn toRenderConfig(self: AdaptiveContext) rich_render.RenderConfig {
        return rich_render.RenderConfig{
            .global_quality = self.current_quality,
            .shadows = self.current_quality != .low,
            .gradients = true,
            .antialiasing = self.current_quality == .high or self.current_quality == .ultra,
        };
    }
};

/// パフォーマンスモニター
pub const PerformanceMonitor = struct {
    fps_history: [60]u32 = undefined,
    history_index: usize = 0,
    frame_count: u32 = 0,

    pub fn init() PerformanceMonitor {
        var pm = PerformanceMonitor{};
        for (&pm.fps_history) |*fps| {
            fps.* = 60;
        }
        return pm;
    }

    pub fn recordFps(self: *PerformanceMonitor, fps: u32) void {
        self.fps_history[self.history_index] = fps;
        self.history_index = (self.history_index + 1) % self.fps_history.len;
        self.frame_count += 1;
    }

    pub fn averageFps(self: PerformanceMonitor) u32 {
        const count = @min(self.frame_count, self.fps_history.len);
        if (count == 0) return 60;

        var sum: u32 = 0;
        for (self.fps_history[0..count]) |fps| {
            sum += fps;
        }
        return sum / count;
    }
};

/// コンポーネント固有の品質設定
pub const ComponentQuality = struct {
    /// コンポーネントの重要度（0.0 - 1.0）
    importance: f32 = 0.5,

    /// 最小品質レベル
    min_quality: rich_render.RenderQuality = .low,

    /// 最大品質レベル
    max_quality: rich_render.RenderQuality = .ultra,

    /// ズーム時の品質増加率
    zoom_quality_scale: f32 = 1.0,

    /// ビューポート外での品質低下
    offscreen_degradation: u32 = 2,

    /// 現在のコンテキストに基づく品質を計算
    pub fn calculate(self: ComponentQuality, ctx: AdaptiveContext) rich_render.RenderQuality {
        var quality = ctx.current_quality;

        // 重要度による調整
        if (self.importance >= 0.8) {
            // 高重要度は品質を保証
            const high_val = @intFromEnum(rich_render.RenderQuality.high);
            const q_val = @intFromEnum(quality);
            quality = @enumFromInt(@max(q_val, high_val));
        } else if (self.importance <= 0.2) {
            // 低重要度は品質を下げる可能性
            const q_val = @intFromEnum(quality);
            if (q_val > 0) {
                quality = @enumFromInt(q_val - 1);
            }
        }

        // 範囲内に制限
        return clampQuality(quality, self.min_quality, self.max_quality);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "viewport coordinate conversion" {
    const vp = Viewport{
        .width = 800,
        .height = 600,
        .scale_factor = 2.0,
        .zoom = 1.0,
        .pan_x = 0,
        .pan_y = 0,
    };

    const world = vp.screenToWorld(400, 300);
    try std.testing.expectApproxEqAbs(@as(f32, 200), world.x, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 150), world.y, 0.1);
}

test "viewport zoom conversion" {
    const vp = Viewport{
        .width = 800,
        .height = 600,
        .scale_factor = 1.0,
        .zoom = 2.0,
        .pan_x = 0,
        .pan_y = 0,
    };

    const world = vp.screenToWorld(200, 150);
    try std.testing.expectApproxEqAbs(@as(f32, 400), world.x, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 300), world.y, 0.1);
}

test "adaptive quality by zoom" {
    const config = AdaptiveConfig{
        .base_quality = .medium,
    };

    // 通常ズーム（ただし画面サイズが大きいのでhigh）
    const vp1 = Viewport{ .width = 1920, .height = 1080, .zoom = 1.0 };
    const q1 = config.calculateQuality(vp1);
    try std.testing.expectEqual(rich_render.RenderQuality.high, q1);

    // 高ズーム（effective_sizeが小さいのでmediumに制限）
    const vp2 = Viewport{ .width = 1920, .height = 1080, .zoom = 3.0 };
    const q2 = config.calculateQuality(vp2);
    try std.testing.expectEqual(rich_render.RenderQuality.medium, q2);

    // 超高ズーム（effective_sizeが小さいのでmediumに制限）
    const vp3 = Viewport{ .width = 1920, .height = 1080, .zoom = 5.0 };
    const q3 = config.calculateQuality(vp3);
    try std.testing.expectEqual(rich_render.RenderQuality.medium, q3);

    // 低ズーム（ただしeffective_sizeが大きいのでhigh）
    const vp4 = Viewport{ .width = 1920, .height = 1080, .zoom = 0.3 };
    const q4 = config.calculateQuality(vp4);
    try std.testing.expectEqual(rich_render.RenderQuality.high, q4);
}

test "component quality by importance" {
    const ctx = AdaptiveContext{
        .config = .{},
        .viewport = .{ .width = 1920, .height = 1080 },
        .current_quality = .medium,
        .performance_monitor = .{},
    };

    // 高重要度
    const high_imp = ComponentQuality{ .importance = 0.9 };
    const q1 = high_imp.calculate(ctx);
    try std.testing.expectEqual(rich_render.RenderQuality.high, q1);

    // 低重要度
    const low_imp = ComponentQuality{ .importance = 0.1 };
    const q2 = low_imp.calculate(ctx);
    try std.testing.expect(q2 != .ultra);
}
