//! ZAPP テストユーティリティ
//!
//! テスト用のモックフレームバッファと検証ヘルパーを提供する。

const std = @import("std");
const zgf = @import("zgf");

/// テスト用のモックフレームバッファ
///
/// 実際のレンダリング結果を検証するための簡易イメージ。
/// ピクセル値の直接アクセスと検証機能を提供する。
pub const MockFramebuffer = struct {
    width: usize,
    height: usize,
    pixels: []zgf.Color,
    alloc: std.mem.Allocator,

    /// モックフレームバッファを初期化
    pub fn init(alloc: std.mem.Allocator, w: usize, h: usize) !MockFramebuffer {
        const pixels = try alloc.alloc(zgf.Color, w * h);
        // 全ピクセルを黒で初期化
        for (pixels) |*p| {
            p.* = .{ .r = 0, .g = 0, .b = 0, .a = 1 };
        }
        return .{
            .width = w,
            .height = h,
            .pixels = pixels,
            .alloc = alloc,
        };
    }

    /// モックフレームバッファを解放
    pub fn deinit(self: *MockFramebuffer) void {
        self.alloc.free(self.pixels);
    }

    /// ピクセルを設定
    pub fn setPixel(self: *MockFramebuffer, x: usize, y: usize, color: zgf.Color) void {
        if (x >= self.width or y >= self.height) return;
        self.pixels[y * self.width + x] = color;
    }

    /// ピクセルを取得
    pub fn getPixel(self: *const MockFramebuffer, x: usize, y: usize) zgf.Color {
        if (x >= self.width or y >= self.height) {
            return .{ .r = 0, .g = 0, .b = 0, .a = 0 };
        }
        return self.pixels[y * self.width + x];
    }

    /// 全ピクセルを指定色で塗りつぶす
    pub fn fill(self: *MockFramebuffer, color: zgf.Color) void {
        for (self.pixels) |*p| {
            p.* = color;
        }
    }

    /// 矩形領域を塗りつぶす
    pub fn fillRect(self: *MockFramebuffer, x0: usize, y0: usize, x1: usize, y1: usize, color: zgf.Color) void {
        const x_start = @min(x0, self.width);
        const y_start = @min(y0, self.height);
        const x_end = @min(x1, self.width);
        const y_end = @min(y1, self.height);

        var y = y_start;
        while (y < y_end) : (y += 1) {
            var x = x_start;
            while (x < x_end) : (x += 1) {
                self.setPixel(x, y, color);
            }
        }
    }

    /// 指定ピクセルが期待値と一致するか検証
    pub fn expectPixel(self: *const MockFramebuffer, x: usize, y: usize, expected: zgf.Color, tolerance: f32) !void {
        const actual = self.getPixel(x, y);
        const dr = @abs(actual.r - expected.r);
        const dg = @abs(actual.g - expected.g);
        const db = @abs(actual.b - expected.b);
        const da = @abs(actual.a - expected.a);

        if (dr > tolerance or dg > tolerance or db > tolerance or da > tolerance) {
            std.debug.print(
                "Pixel mismatch at ({d},{d}): expected ({d:.2},{d:.2},{d:.2},{d:.2}), got ({d:.2},{d:.2},{d:.2},{d:.2})\n",
                .{ x, y, expected.r, expected.g, expected.b, expected.a, actual.r, actual.g, actual.b, actual.a },
            );
            return error.TestFailed;
        }
    }

    /// 矩形領域が全て指定色か検証
    pub fn expectRectFilled(self: *const MockFramebuffer, x0: usize, y0: usize, x1: usize, y1: usize, color: zgf.Color, tolerance: f32) !void {
        var y = y0;
        while (y < y1) : (y += 1) {
            var x = x0;
            while (x < x1) : (x += 1) {
                try self.expectPixel(x, y, color, tolerance);
            }
        }
    }

    /// 矩形領域が全て背景色（黒）か検証
    pub fn expectRectEmpty(self: *const MockFramebuffer, x0: usize, y0: usize, x1: usize, y1: usize, tolerance: f32) !void {
        const black: zgf.Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 };
        try self.expectRectFilled(x0, y0, x1, y1, black, tolerance);
    }

    /// イメージのサイズを取得
    pub fn size(self: *const MockFramebuffer) struct { w: usize, h: usize } {
        return .{ .w = self.width, .h = self.height };
    }
};

/// テストヘルパー: 2つの色が許容誤差范围内かチェック
pub fn colorsEqual(a: zgf.Color, b: zgf.Color, tolerance: f32) bool {
    const dr = @abs(a.r - b.r);
    const dg = @abs(a.g - b.g);
    const db = @abs(a.b - b.b);
    const da = @abs(a.a - b.a);
    return dr <= tolerance and dg <= tolerance and db <= tolerance and da <= tolerance;
}
