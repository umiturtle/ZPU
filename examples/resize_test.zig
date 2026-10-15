//! リサイズテストプログラム
//!
//! 100x100 から 4K (3840x2160) まで、様々なサイズにリサイズして
//! レンダリングが正しく動作することを確認する。

const std = @import("std");
const zapp = @import("zapp");
const zgf = @import("zgf");
const zui = @import("zui");

const TestSize = struct {
    width: u32,
    height: u32,
    name: []const u8,
};

const test_sizes = [_]TestSize{
    .{ .width = 100, .height = 100, .name = "Minimum (100x100)" },
    .{ .width = 320, .height = 240, .name = "QVGA" },
    .{ .width = 640, .height = 480, .name = "VGA" },
    .{ .width = 800, .height = 600, .name = "SVGA" },
    .{ .width = 1024, .height = 768, .name = "XGA" },
    .{ .width = 1280, .height = 720, .name = "HD" },
    .{ .width = 1366, .height = 768, .name = "WXGA" },
    .{ .width = 1920, .height = 1080, .name = "Full HD" },
    .{ .width = 2560, .height = 1440, .name = "QHD" },
    .{ .width = 3840, .height = 2160, .name = "4K UHD" },
};

fn renderTestPattern(framebuffer: *zgf.Image, size_name: []const u8) void {
    const width = framebuffer.width();
    const height = framebuffer.height();
    
    // グラデーション背景
    for (0..height) |y| {
        for (0..width) |x| {
            const r = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(width));
            const g = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(height));
            const b = 1.0 - (r + g) / 2.0;
            framebuffer.setPixel(@intCast(x), @intCast(y), zgf.Color{ .r = r, .g = g, .b = b, .a = 1.0 });
        }
    }
    
    // チェッカーパターン（中央）
    const checker_size = @max(width / 20, 10);
    for (0..height) |y| {
        for (0..width) |x| {
            const check_x = x / checker_size;
            const check_y = y / checker_size;
            if ((check_x + check_y) % 2 == 0) {
                if (x > width / 4 and x < width * 3 / 4 and y > height / 4 and y < height * 3 / 4) {
                    framebuffer.setPixel(@intCast(x), @intCast(y), zgf.Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 0.5 });
                }
            }
        }
    }
    
    // 枠線
    const border_color = zgf.Color{ .r = 1.0, .g = 0.0, .b = 0.0, .a = 1.0 };
    for (0..width) |x| {
        framebuffer.setPixel(@intCast(x), 0, border_color);
        framebuffer.setPixel(@intCast(x), @intCast(height - 1), border_color);
    }
    for (0..height) |y| {
        framebuffer.setPixel(0, @intCast(y), border_color);
        framebuffer.setPixel(@intCast(width - 1), @intCast(y), border_color);
    }
    
    _ = size_name; // サイズ名はUIで表示（ソフトウェアバックエンドではテキスト描画なし）
}

fn saveScreenshot(framebuffer: *const zgf.Image, name: []const u8) !void {
    const ppm = try framebuffer.toPpm(std.heap.page_allocator);
    defer std.heap.page_allocator.free(ppm);
    
    const filename = try std.fmt.allocPrint(std.heap.page_allocator, "zig-out/resize_test_{s}.ppm", .{name});
    defer std.heap.page_allocator.free(filename);
    
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();
    try file.writeAll(ppm);
    
    std.debug.print("  ✓ スクリーンショット保存: {s}\n", .{filename});
}

pub fn main() !void {
    std.debug.print("\n=== ZAPP リサイズテスト ===\n\n", .{});
    
    // ソフトウェアバックエンドで初期化
    var window = zapp.Window.init(.{
        .title = "Resize Test",
        .width = 100,
        .height = 100,
        .backend = .software,
    }) catch |err| {
        std.debug.print("エラー: ウィンドウ初期化に失敗しました: {}\n", .{err});
        return err;
    };
    defer window.deinit();
    
    // 出力ディレクトリを作成
    std.fs.cwd().makePath("zig-out") catch {};
    
    // 各サイズでテスト
    for (test_sizes) |test_size| {
        std.debug.print("テスト中: {s} ({d}x{d})...\n", .{ test_size.name, test_size.width, test_size.height });
        
        // リサイズ
        window.resize(test_size.width, test_size.height) catch |err| {
            std.debug.print("  ✗ リサイズ失敗: {}\n", .{err});
            continue;
        };
        
        // サイズ確認
        const size = window.getSize();
        if (size.width != test_size.width or size.height != test_size.height) {
            std.debug.print("  ✗ サイズ不一致: 期待={d}x{d}, 実際={d}x{d}\n", .{
                test_size.width, test_size.height, size.width, size.height,
            });
            continue;
        }
        
        // フレームバッファ確認
        const fb = window.getFramebuffer();
        if (fb.width() != test_size.width or fb.height() != test_size.height) {
            std.debug.print("  ✗ フレームバッファサイズ不一致: 期待={d}x{d}, 実際={d}x{d}\n", .{
                test_size.width, test_size.height, fb.width(), fb.height(),
            });
            continue;
        }
        
        // テストパターンを描画
        renderTestPattern(fb, test_size.name);
        
        // スクリーンショットを保存
        const safe_name = try std.fmt.allocPrint(std.heap.page_allocator, "{d}x{d}", .{ test_size.width, test_size.height });
        defer std.heap.page_allocator.free(safe_name);
        
        saveScreenshot(fb, safe_name) catch |err| {
            std.debug.print("  ✗ スクリーンショット保存失敗: {}\n", .{err});
            continue;
        };
        
        std.debug.print("  ✓ 成功\n", .{});
    }
    
    std.debug.print("\n=== テスト完了 ===\n", .{});
}
