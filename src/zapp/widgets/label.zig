//! ZAPP Label Widget
//!
//! テキストラベルを表示するウィジェット。

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// ラベルウィジェットの設定
pub const LabelConfig = struct {
    text: []const u8,
    rect: VirtualRect,
    color: zgf.Color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
    font_h: f32 = 14.0,
};

/// ラベルをレンダリングする
pub fn render(fb: *zgf.Image, config: LabelConfig) void {
    drawing.drawText(fb, config.text, config.rect.x, config.rect.y, config.color);
}

// テスト
test "label - basic render" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 100, 20, 3);
    defer img.deinit();
    
    // ラベルのレンダリング（現在はnoop）
    render(&img, .{
        .text = "Hello",
        .rect = .{ .x = 5, .y = 5, .w = 50, .h = 14 },
    });
}
