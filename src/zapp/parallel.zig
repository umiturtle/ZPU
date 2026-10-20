//! ZAPP Parallel Rendering (Placeholder)
//!
//! マルチスレッドによるレンダリング並列化。
//! Zig 0.17.0ではMutexが未実装のため、現在はシングルスレッド版。

const std = @import("std");
const zgf = @import("zgf");

/// レンダリングタスク
pub const RenderTask = struct {
    id: usize,
    y_start: usize,
    y_end: usize,
    render_fn: *const fn (fb: *zgf.Image, y_start: usize, y_end: usize) void,
};

/// シングル スレッド版レンダリング
pub fn renderParallel(
    fb: *zgf.Image,
    render_fn: *const fn (fb: *zgf.Image, y_start: usize, y_end: usize) void,
    worker_count: usize,
) !void {
    _ = worker_count; // 現在は未使用
    const height = fb.height();
    render_fn(fb, 0, height);
}

// テスト
test "parallel - render parallel" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();

    const render_fn = struct {
        fn render(image: *zgf.Image, y_start: usize, y_end: usize) void {
            for (y_start..y_end) |y| {
                for (0..image.width()) |x| {
                    image.setPixel(@intCast(x), @intCast(y), .{
                        .r = 1.0,
                        .g = 0.0,
                        .b = 0.0,
                    });
                }
            }
        }
    }.render;

    try renderParallel(&fb, render_fn, 4);

    const pixels = try fb.toPixels(allocator);
    defer allocator.free(pixels);

    try std.testing.expect(pixels.len > 0);
}
