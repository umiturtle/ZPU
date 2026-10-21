//! ZAPP Separator Widget
//!
//! 区切り線（セパレーター）ウィジェット。

const std = @import("std");
const zgf = @import("zgf");
const drawing = @import("drawing.zig");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// セパレーターの方向
pub const SeparatorOrientation = enum {
    horizontal,
    vertical,
};

/// セパレーターの設定
pub const SeparatorConfig = struct {
    rect: VirtualRect,
    orientation: SeparatorOrientation = .horizontal,
    color: zgf.Color = .{ .r = 0.3, .g = 0.3, .b = 0.35, .a = 1.0 },
    thickness: f32 = 1.0,
};

/// セパレーターをレンダリングする
pub fn render(fb: *zgf.Image, config: SeparatorConfig) void {
    switch (config.orientation) {
        .horizontal => {
            const y = config.rect.y + config.rect.h / 2;
            drawing.drawHorizontalLine(fb, config.rect.x, y, config.rect.w, config.color);
        },
        .vertical => {
            const x = config.rect.x + config.rect.w / 2;
            drawing.drawVerticalLine(fb, x, config.rect.y, config.rect.h, config.color);
        },
    }
}

// テスト
test "separator - horizontal" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 100, 10, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 10, .y = 0, .w = 80, .h = 10 },
        .orientation = .horizontal,
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}

test "separator - vertical" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 10, 100, 3);
    defer img.deinit();
    
    render(&img, .{
        .rect = .{ .x = 0, .y = 10, .w = 10, .h = 80 },
        .orientation = .vertical,
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    try std.testing.expect(pixels.len > 0);
}
