//! ZAPP Progress Bar Widget
//!
//! プログレスバーウィジェット。
//! 適応的レンダリング対応：品質レベルに応じてグラデーションと角丸を自動適用

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const rich_render = @import("../rich_render.zig");
const adaptive_render = @import("../adaptive_render.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// プログレスバーの設定
pub const ProgressBarConfig = struct {
    rect: VirtualRect,
    value: f32, // 0.0 - 1.0
    bg: zgf.Color = .{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 },
    fg: zgf.Color = .{ .r = 0.3, .g = 0.8, .b = 0.4, .a = 1.0 },
    radius: f32 = 10.0,
    
    /// グラデーションを使用するか
    gradient: bool = true,
    
    /// 重要度
    importance: f32 = 0.6,
};

/// プログレスバーをレンダリングする（基本版）
pub fn render(fb: *zgf.Image, config: ProgressBarConfig) void {
    const clamped = @max(0.0, @min(1.0, config.value));
    
    // 背景
    drawing.fillRect(fb, config.rect, config.bg);
    
    // フォアグラウンド（進捗部分）
    const fill_width = config.rect.w * clamped;
    if (fill_width > 0) {
        const fill_rect = VirtualRect{
            .x = config.rect.x,
            .y = config.rect.y,
            .w = fill_width,
            .h = config.rect.h,
        };
        drawing.fillRect(fb, fill_rect, config.fg);
    }
}

/// プログレスバーをレンダリングする（リッチ版）
pub fn renderRich(
    fb: *zgf.Image,
    config: ProgressBarConfig,
    render_config: rich_render.RenderConfig,
) void {
    const clamped = @max(0.0, @min(1.0, config.value));
    
    // 背景（角丸）
    rich_render.drawRoundedRect(
        fb,
        config.rect.x,
        config.rect.y,
        config.rect.w,
        config.rect.h,
        config.radius,
        config.bg,
        render_config,
    );
    
    // フォアグラウンド（進捗部分）
    const fill_width = config.rect.w * clamped;
    if (fill_width > 0) {
        if (config.gradient and render_config.gradients) {
            // グラデーション進捗
            rich_render.drawGradientRect(
                fb,
                config.rect.x,
                config.rect.y,
                fill_width,
                config.rect.h,
                .{
                    .from = config.fg,
                    .to = .{
                        .r = @max(config.fg.r - 0.1, 0.0),
                        .g = @max(config.fg.g - 0.1, 0.0),
                        .b = @max(config.fg.b - 0.1, 0.0),
                        .a = config.fg.a,
                    },
                    .direction = .horizontal,
                },
                render_config,
            );
        } else {
            rich_render.drawRoundedRect(
                fb,
                config.rect.x,
                config.rect.y,
                fill_width,
                config.rect.h,
                config.radius,
                config.fg,
                render_config,
            );
        }
    }
}

/// 適応的レンダリング対応のプログレスバー描画
pub fn renderAdaptive(
    fb: *zgf.Image,
    config: ProgressBarConfig,
    ctx: adaptive_render.AdaptiveContext,
) void {
    const render_config = ctx.toRenderConfig();
    renderRich(fb, config, render_config);
}

// テスト
test "progress bar - 0%" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 20, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 20 },
        .value = 0.0,
    });
}

test "progress bar - 50%" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 20, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 20 },
        .value = 0.5,
    });
}

test "progress bar - 100%" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 20, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 20 },
        .value = 1.0,
    });
}

test "progress bar - clamps value" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 200, 20, 3);
    defer img.deinit();
    
    // 1.0超は1.0にクランプされるはず
    render(&img, .{
        .rect = .{ .x = 0, .y = 0, .w = 200, .h = 20 },
        .value = 1.5,
    });
}
