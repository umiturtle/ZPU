//! ZAPP Spacer Widget
//!
//! 空のスペース（間隔調整用）ウィジェット。

const std = @import("std");
const zgf = @import("zgf");
const VirtualRect = @import("../coordinates.zig").VirtualRect;

/// スペーサーの設定
pub const SpacerConfig = struct {
    rect: VirtualRect,
};

/// スペーサーは実際には何も描画しない
/// レイアウト上の間隔を確保するだけ
pub fn render(fb: *zgf.Image, config: SpacerConfig) void {
    _ = fb;
    _ = config;
    // No-op: スペーサーは描画しない
}

// テスト
test "spacer - no rendering" {
    const alloc = std.testing.allocator;
    var img = try zgf.Image.init(alloc, 100, 100, 3);
    defer img.deinit();
    
    // 背景を塗りつぶす
    img.fill(.{ .r = 0.5, .g = 0.5, .b = 0.5 });
    
    // スペーサーをレンダリング（何もしない）
    render(&img, .{
        .rect = .{ .x = 10, .y = 10, .w = 80, .h = 80 },
    });
    
    const pixels = try img.toPixels(alloc);
    defer alloc.free(pixels);
    
    // 全ピクセルが0.5（128）のまま
    for (pixels) |p| {
        try std.testing.expectEqual(@as(u8, 128), p);
    }
}
