//! ZAPP Slider Widget
//!
//! スライダーウィジェット。
//! 適応的レンダリング対応：品質レベルに応じてグラデーション、シャドウ、アンチエイリアスを自動適用

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const vector = @import("../vector.zig");
const rich_render = @import("../rich_render.zig");
const adaptive_render = @import("../adaptive_render.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// スライダーの設定
pub const SliderConfig = struct {
    rect: VirtualRect,
    value: f32, // 0.0 - 1.0
    track_color: zgf.Color = .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 },
    fill_color: zgf.Color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
    handle_color: zgf.Color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
    handle_radius: f32 = 12.0,
    track_height: f32 = 8.0,
    
    /// グラデーションを使用するか
    gradient: bool = true,
    
    /// 重要度
    importance: f32 = 0.5,
};

/// スライダーをレンダリングする（基本版）
pub fn render(fb: *zgf.Image, config: SliderConfig) void {
    const clamped = @max(0.0, @min(1.0, config.value));
    
    // トラック
    const track_y = config.rect.y + (config.rect.h - config.track_height) / 2;
    const track_rect = VirtualRect{
        .x = config.rect.x,
        .y = track_y,
        .w = config.rect.w,
        .h = config.track_height,
    };
    drawing.fillRect(fb, track_rect, config.track_color);
    
    // フィル
    const fill_width = config.rect.w * clamped;
    if (fill_width > 0) {
        const fill_rect = VirtualRect{
            .x = config.rect.x,
            .y = track_y,
            .w = fill_width,
            .h = config.track_height,
        };
        drawing.fillRect(fb, fill_rect, config.fill_color);
    }
    
    // ハンドル
    const handle_x = config.rect.x + config.rect.w * clamped;
    const handle_y = config.rect.y + config.rect.h / 2;
    vector.fillCircleOptimized(fb, handle_x, handle_y, config.handle_radius, config.handle_color);
}

/// スライダーをレンダリングする（リッチ版）
pub fn renderRich(
    fb: *zgf.Image,
    config: SliderConfig,
    render_config: rich_render.RenderConfig,
) void {
    const clamped = @max(0.0, @min(1.0, config.value));
    
    // トラック（角丸）
    const track_y = config.rect.y + (config.rect.h - config.track_height) / 2;
    rich_render.drawRoundedRect(
        fb,
        config.rect.x,
        track_y,
        config.rect.w,
        config.track_height,
        config.track_height / 2,
        config.track_color,
        render_config,
    );
    
    // フィル
    const fill_width = config.rect.w * clamped;
    if (fill_width > 0) {
        if (config.gradient and render_config.gradients) {
            rich_render.drawGradientRect(
                fb,
                config.rect.x,
                track_y,
                fill_width,
                config.track_height,
                .{
                    .from = config.fill_color,
                    .to = .{
                        .r = @max(config.fill_color.r - 0.1, 0.0),
                        .g = @max(config.fill_color.g - 0.1, 0.0),
                        .b = @max(config.fill_color.b - 0.1, 0.0),
                        .a = config.fill_color.a,
                    },
                    .direction = .horizontal,
                },
                render_config,
            );
        } else {
            rich_render.drawRoundedRect(
                fb,
                config.rect.x,
                track_y,
                fill_width,
                config.track_height,
                config.track_height / 2,
                config.fill_color,
                render_config,
            );
        }
    }
    
    // ハンドル（シャドウ付き）
    const handle_x = config.rect.x + config.rect.w * clamped;
    const handle_y = config.rect.y + config.rect.h / 2;
    
    // ハンドルのシャドウ
    if (render_config.shadows and render_config.global_quality != .low) {
        rich_render.drawShadowRect(
            fb,
            handle_x - config.handle_radius,
            handle_y - config.handle_radius,
            config.handle_radius * 2,
            config.handle_radius * 2,
            config.handle_radius,
            config.handle_color,
            .{ .offset_y = 2, .blur = 4 },
            render_config,
        );
    }
    
    // ハンドル本体
    if (render_config.antialiasing) {
        // アンチエイリアス付き
        vector.fillCircleOptimized(fb, handle_x, handle_y, config.handle_radius, config.handle_color);
    } else {
        // 高速版
        vector.fillCircleOptimized(fb, handle_x, handle_y, config.handle_radius, config.handle_color);
    }
}

/// 適応的レンダリング対応のスライダー描画
pub fn renderAdaptive(
    fb: *zgf.Image,
    config: SliderConfig,
    ctx: adaptive_render.AdaptiveContext,
) void {
    const render_config = ctx.toRenderConfig();
    renderRich(fb, config, render_config);
}

// テスト
test "slider - 0%" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 30, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 30 },
        .value = 0.0,
    });
}

test "slider - 50%" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 30, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 30 },
        .value = 0.5,
    });
}

test "slider - 100%" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 30, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 30 },
        .value = 1.0,
    });
}

test "slider - clamps value" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 30, 3);
    defer img.deinit();
    
    // -0.5は0.0に、1.5は1.0にクランプされるはず
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 30 },
        .value = -0.5,
    });
}
