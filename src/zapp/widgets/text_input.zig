//! ZAPP TextInput Widget
//!
//! テキスト入力フィールドウィジェット。

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// テキスト入力の状態
pub const TextInputState = enum {
    normal,
    focused,
    disabled,
};

/// テキスト入力の設定
pub const TextInputConfig = struct {
    text: []const u8,
    placeholder: []const u8 = "",
    rect: VirtualRect,
    state: TextInputState = .normal,
    bg: zgf.Color = .{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 },
    fg: zgf.Color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
    placeholder_color: zgf.Color = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 },
    border: zgf.Color = .{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 },
    focus_border: zgf.Color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
};

/// テキスト入力をレンダリングする
pub fn render(fb: *zgf.Image, config: TextInputConfig) void {
    // 背景
    drawing.fillRect(fb, config.rect, config.bg);
    
    // 枠線（フォーカス時は強調）
    const border_color = switch (config.state) {
        .focused => config.focus_border,
        else => config.border,
    };
    drawing.drawRectBorder(fb, config.rect, border_color, if (config.state == .focused) 2.0 else 1.0);
    
    // テキストまたはプレースホルダー
    const display_text = if (config.text.len > 0) config.text else config.placeholder;
    const text_color = if (config.text.len > 0) config.fg else config.placeholder_color;
    
    drawing.drawText(fb, display_text, config.rect.x + 8, config.rect.y + 4, text_color);
}

// テスト
test "text_input - normal render" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 100, 30, 3);
    defer img.deinit();
    
    render(&img, .{
        .text = "Hello",
        .rect = .{ .x = 10, .y = 5, .w = 80, .h = 20 },
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}

test "text_input - placeholder" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 100, 30, 3);
    defer img.deinit();
    
    render(&img, .{
        .text = "",
        .placeholder = "Enter text...",
        .rect = .{ .x = 10, .y = 5, .w = 80, .h = 20 },
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}
