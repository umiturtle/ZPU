//! ZAPP Checkbox Widget
//!
//! チェックボックスウィジェット。

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// チェックボックスの設定
pub const CheckboxConfig = struct {
    checked: bool,
    rect: VirtualRect,
    bg: zgf.Color = .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 },
    border: zgf.Color = .{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 },
    check: zgf.Color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
};

/// チェックボックスをレンダリングする
pub fn render(fb: *zgf.Image, config: CheckboxConfig) void {
    // 背景
    drawing.fillRect(fb, config.rect, config.bg);
    
    // 枠線
    drawing.drawRectBorder(fb, config.rect, config.border, 1.0);
    
    // チェックマーク
    if (config.checked) {
        drawing.drawCheckmark(fb, config.rect, config.check);
    }
}

// テスト
test "checkbox - checked" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 20, 20, 3);
    defer img.deinit();
    
    render(&img, .{
        .checked = true,
        .rect = .{ .x = 2, .y = 2, .w = 16, .h = 16 },
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}

test "checkbox - unchecked" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 20, 20, 3);
    defer img.deinit();
    
    render(&img, .{
        .checked = false,
        .rect = .{ .x = 2, .y = 2, .w = 16, .h = 16 },
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}
