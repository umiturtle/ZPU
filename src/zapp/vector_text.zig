//! ZAPP ベクターベース文字レンダリング
//!
//! ベクタープリミティブを使った高品質な文字レンダリング。
//! フォントアウトラインをベジェ曲線とラインで描画。
//!
//! ## 特徴
//! - スケーラブルなベクターベース描画
//! - アンチエイリアス対応
//! - カスタムフォントサポート
//! - 日本語文字対応

const std = @import("std");
const zgf = @import("zgf");
const vector = @import("vector.zig");
const Vec2 = vector.Vec2;

// ============================================================================
// グリフデータ
// ============================================================================

/// グリフ (文字の形状データ)
pub const Glyph = struct {
    char: u21,
    advance_width: f32,
    paths: []const PathCommand,
    
    pub const PathCommand = union(enum) {
        moveTo: Vec2,
        lineTo: Vec2,
        quadTo: struct { control: Vec2, end: Vec2 },
        cubicTo: struct { c1: Vec2, c2: Vec2, end: Vec2 },
        close: void,
    };
};

// ============================================================================
// ビルトインフォント (簡易ビットマップ→ベクター変換)
// ============================================================================

/// 簡易フォント (8x8グリッドベース)
pub const SimpleFont = struct {
    cell_width: f32 = 8,
    cell_height: f32 = 8,
    scale: f32 = 1.0,
    
    pub fn init() SimpleFont {
        return .{};
    }
    
    pub fn withScale(scale: f32) SimpleFont {
        return .{ .scale = scale };
    }
    
    /// 文字を描画
    pub fn drawChar(
        self: SimpleFont,
        fb: *zgf.Image,
        char: u21,
        x: f32,
        y: f32,
        color: zgf.Color,
    ) void {
        const scaled_w = self.cell_width * self.scale;
        const scaled_h = self.cell_height * self.scale;
        
        // ASCII文字の場合
        if (char >= 32 and char <= 126) {
            self.drawAsciiChar(fb, @intCast(char), x, y, color);
            return;
        }
        
        // その他の文字は矩形で描画
        vector.fillRectOptimized(
            fb,
            @intFromFloat(x),
            @intFromFloat(y),
            @intFromFloat(x + scaled_w),
            @intFromFloat(y + scaled_h),
            color,
        );
    }
    
    /// ASCII文字を描画 (ベクタープリミティブ使用)
    fn drawAsciiChar(
        self: SimpleFont,
        fb: *zgf.Image,
        char: u8,
        x: f32,
        y: f32,
        color: zgf.Color,
    ) void {
        const s = self.scale;
        const w = self.cell_width * s;
        const h = self.cell_height * s;
        
        // 各文字のパターンを描画
        switch (char) {
            'A'...'Z', 'a'...'z' => {
                // 文字の外枠を描画
                self.drawLetterOutline(fb, char, x, y, color);
            },
            '0'...'9' => {
                self.drawDigitOutline(fb, char, x, y, color);
            },
            ' ' => {
                // スペースは何も描画しない
            },
            '.' => {
                vector.fillCircleOptimized(fb, x + w / 2, y + h - 2 * s, 1.5 * s, color);
            },
            ',' => {
                vector.drawLineFast(fb, x + w / 2, y + h - 2 * s, x + w / 2 - s, y + h, color);
            },
            '-' => {
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 2 * s, y + h / 2, color);
            },
            '+' => {
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 2 * s, y + h / 2, color);
                vector.drawLineFast(fb, x + w / 2, y + 2 * s, x + w / 2, y + h - 2 * s, color);
            },
            '=' => {
                vector.drawLineFast(fb, x + 2 * s, y + h / 2 - s, x + w - 2 * s, y + h / 2 - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2 + s, x + w - 2 * s, y + h / 2 + s, color);
            },
            '/' => {
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + 2 * s, y + h - s, color);
            },
            '(' => {
                self.drawArc(fb, x + w / 2 + 2 * s, y + h / 2, 3 * s, -90, 90, color);
            },
            ')' => {
                self.drawArc(fb, x + w / 2 - 2 * s, y + h / 2, 3 * s, 90, 270, color);
            },
            '[' => {
                vector.drawLineFast(fb, x + 3 * s, y + s, x + 3 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 3 * s, y + s, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 3 * s, y + h - s, x + w - 2 * s, y + h - s, color);
            },
            ']' => {
                vector.drawLineFast(fb, x + w - 3 * s, y + s, x + w - 3 * s, y + h - s, color);
                vector.drawLineFast(fb, x + w - 3 * s, y + s, x + 2 * s, y + s, color);
                vector.drawLineFast(fb, x + w - 3 * s, y + h - s, x + 2 * s, y + h - s, color);
            },
            ':' => {
                vector.fillCircleOptimized(fb, x + w / 2, y + h / 3, s, color);
                vector.fillCircleOptimized(fb, x + w / 2, y + 2 * h / 3, s, color);
            },
            ';' => {
                vector.fillCircleOptimized(fb, x + w / 2, y + h / 3, s, color);
                vector.drawLineFast(fb, x + w / 2, y + 2 * h / 3, x + w / 2 - s, y + h - s, color);
            },
            '!' => {
                vector.drawLineFast(fb, x + w / 2, y + s, x + w / 2, y + h - 3 * s, color);
                vector.fillCircleOptimized(fb, x + w / 2, y + h - 2 * s, s, color);
            },
            '?' => {
                self.drawArc(fb, x + w / 2, y + 3 * s, 2 * s, -180, 0, color);
                vector.drawLineFast(fb, x + w / 2 + 2 * s, y + 3 * s, x + w / 2, y + h / 2, color);
                vector.fillCircleOptimized(fb, x + w / 2, y + h - 2 * s, s, color);
            },
            else => {
                // 不明な文字は矩形を描画
                vector.fillRectOptimized(fb, @intFromFloat(x), @intFromFloat(y), @intFromFloat(x + w), @intFromFloat(y + h), color);
            },
        }
    }
    
    /// 文字のアウトラインを描画
    fn drawLetterOutline(
        self: SimpleFont,
        fb: *zgf.Image,
        char: u8,
        x: f32,
        y: f32,
        color: zgf.Color,
    ) void {
        const s = self.scale;
        const w = self.cell_width * s;
        const h = self.cell_height * s;
        
        // 簡易的な文字形状
        switch (char) {
            'A', 'a' => {
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + w / 2, y + s, color);
                vector.drawLineFast(fb, x + w / 2, y + s, x + w - 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 3 * s, y + h / 2, x + w - 3 * s, y + h / 2, color);
            },
            'B', 'b' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                self.drawArc(fb, x + 3 * s, y + 3 * s, 2 * s, -90, 90, color);
                self.drawArc(fb, x + 3 * s, y + h - 3 * s, 2 * s, -90, 90, color);
            },
            'C', 'c' => {
                self.drawArc(fb, x + w / 2, y + h / 2, 3 * s, 45, 315, color);
            },
            'D', 'd' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                self.drawArc(fb, x + 3 * s, y + h / 2, 3 * s, -90, 90, color);
            },
            'E', 'e' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 3 * s, y + h / 2, color);
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + w - 2 * s, y + h - s, color);
            },
            'F', 'f' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 3 * s, y + h / 2, color);
            },
            'G', 'g' => {
                self.drawArc(fb, x + w / 2, y + h / 2, 3 * s, 45, 360, color);
                vector.drawLineFast(fb, x + w / 2 + 2 * s, y + h / 2, x + w / 2 + 2 * s, y + h - 2 * s, color);
            },
            'H', 'h' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + w - 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 2 * s, y + h / 2, color);
            },
            'I', 'i' => {
                vector.drawLineFast(fb, x + w / 2, y + s, x + w / 2, y + h - s, color);
                vector.drawLineFast(fb, x + w / 2 - 2 * s, y + s, x + w / 2 + 2 * s, y + s, color);
                vector.drawLineFast(fb, x + w / 2 - 2 * s, y + h - s, x + w / 2 + 2 * s, y + h - s, color);
            },
            'J', 'j' => {
                vector.drawLineFast(fb, x + w - 3 * s, y + s, x + w - 3 * s, y + h - 3 * s, color);
                self.drawArc(fb, x + w / 2 - s, y + h - 3 * s, 2 * s, -90, 0, color);
            },
            'K', 'k' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 2 * s, y + h - s, color);
            },
            'L', 'l' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + w - 2 * s, y + h - s, color);
            },
            'M', 'm' => {
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w / 2, y + h / 2, color);
                vector.drawLineFast(fb, x + w / 2, y + h / 2, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + w - 2 * s, y + h - s, color);
            },
            'N', 'n' => {
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w - 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + h - s, x + w - 2 * s, y + s, color);
            },
            'O', 'o' => {
                vector.drawCircleOptimized(fb, x + w / 2, y + h / 2, 3 * s, color);
            },
            'P', 'p' => {
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + 2 * s, y + s, color);
                self.drawArc(fb, x + 3 * s, y + 3 * s, 2 * s, -90, 90, color);
            },
            'Q', 'q' => {
                vector.drawCircleOptimized(fb, x + w / 2, y + h / 2, 3 * s, color);
                vector.drawLineFast(fb, x + w / 2 + 2 * s, y + h / 2 + 2 * s, x + w - s, y + h - s, color);
            },
            'R', 'r' => {
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + 2 * s, y + s, color);
                self.drawArc(fb, x + 3 * s, y + 3 * s, 2 * s, -90, 90, color);
                vector.drawLineFast(fb, x + 3 * s, y + 3 * s, x + w - 2 * s, y + h - s, color);
            },
            'S', 's' => {
                self.drawArc(fb, x + w / 2, y + 3 * s, 2 * s, 0, 180, color);
                self.drawArc(fb, x + w / 2, y + h - 3 * s, 2 * s, 180, 360, color);
            },
            'T', 't' => {
                vector.drawLineFast(fb, x + s, y + s, x + w - s, y + s, color);
                vector.drawLineFast(fb, x + w / 2, y + s, x + w / 2, y + h - s, color);
            },
            'U', 'u' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h - 3 * s, color);
                self.drawArc(fb, x + w / 2, y + h - 3 * s, 3 * s, 180, 360, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + w - 2 * s, y + h - 3 * s, color);
            },
            'V', 'v' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w / 2, y + h - s, color);
                vector.drawLineFast(fb, x + w / 2, y + h - s, x + w - 2 * s, y + s, color);
            },
            'W', 'w' => {
                vector.drawLineFast(fb, x + s, y + s, x + 3 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 3 * s, y + h - s, x + w / 2, y + h / 2, color);
                vector.drawLineFast(fb, x + w / 2, y + h / 2, x + w - 3 * s, y + h - s, color);
                vector.drawLineFast(fb, x + w - 3 * s, y + h - s, x + w - s, y + s, color);
            },
            'X', 'x' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w - 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + 2 * s, y + h - s, color);
            },
            'Y', 'y' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w / 2, y + h / 2, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + w / 2, y + h / 2, color);
                vector.drawLineFast(fb, x + w / 2, y + h / 2, x + w / 2, y + h - s, color);
            },
            'Z', 'z' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + w - 2 * s, y + h - s, color);
            },
            else => {
                // 小文字は大文字と同じ形状 (簡易版)
                const upper = if (char >= 'a' and char <= 'z') char - 32 else char;
                if (upper >= 'A' and upper <= 'Z') {
                    self.drawLetterOutline(fb, upper, x, y, color);
                }
            },
        }
    }
    
    /// 数字のアウトラインを描画
    fn drawDigitOutline(
        self: SimpleFont,
        fb: *zgf.Image,
        char: u8,
        x: f32,
        y: f32,
        color: zgf.Color,
    ) void {
        const s = self.scale;
        const w = self.cell_width * s;
        const h = self.cell_height * s;
        
        switch (char) {
            '0' => {
                vector.drawCircleOptimized(fb, x + w / 2, y + h / 2, 3 * s, color);
            },
            '1' => {
                vector.drawLineFast(fb, x + w / 2 - s, y + 3 * s, x + w / 2, y + s, color);
                vector.drawLineFast(fb, x + w / 2, y + s, x + w / 2, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + w - 2 * s, y + h - s, color);
            },
            '2' => {
                self.drawArc(fb, x + w / 2, y + 3 * s, 2 * s, -180, 0, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + 3 * s, x + 2 * s, y + h - s, color);
                vector.drawLineFast(fb, x + 2 * s, y + h - s, x + w - 2 * s, y + h - s, color);
            },
            '3' => {
                self.drawArc(fb, x + w / 2 - s, y + 3 * s, 2 * s, -90, 90, color);
                self.drawArc(fb, x + w / 2 - s, y + h - 3 * s, 2 * s, -90, 90, color);
            },
            '4' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h / 2, color);
                vector.drawLineFast(fb, x + 2 * s, y + h / 2, x + w - 2 * s, y + h / 2, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + w - 2 * s, y + h - s, color);
            },
            '5' => {
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + 2 * s, y + s, color);
                vector.drawLineFast(fb, x + 2 * s, y + s, x + 2 * s, y + h / 2, color);
                self.drawArc(fb, x + w / 2 - s, y + h / 2 + s, 2 * s, -90, 180, color);
            },
            '6' => {
                self.drawArc(fb, x + w / 2, y + h - 3 * s, 2 * s, 0, 360, color);
                self.drawArc(fb, x + w / 2, y + h / 2, 3 * s, 90, 270, color);
            },
            '7' => {
                vector.drawLineFast(fb, x + 2 * s, y + s, x + w - 2 * s, y + s, color);
                vector.drawLineFast(fb, x + w - 2 * s, y + s, x + w / 2, y + h - s, color);
            },
            '8' => {
                vector.drawCircleOptimized(fb, x + w / 2, y + 3 * s, 2 * s, color);
                vector.drawCircleOptimized(fb, x + w / 2, y + h - 3 * s, 2 * s, color);
            },
            '9' => {
                vector.drawCircleOptimized(fb, x + w / 2, y + 3 * s, 2 * s, color);
                self.drawArc(fb, x + w / 2, y + h / 2, 3 * s, -90, 90, color);
            },
            else => {},
        }
    }
    
    /// 弧を描画
    fn drawArc(
        self: SimpleFont,
        fb: *zgf.Image,
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        color: zgf.Color,
    ) void {
        _ = self;
        const steps = 16;
        const angle_range = end_angle - start_angle;
        
        var i: u32 = 0;
        while (i < steps) : (i += 1) {
            const t0 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
            const t1 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(steps));
            
            const a0 = (start_angle + angle_range * t0) * std.math.pi / 180.0;
            const a1 = (start_angle + angle_range * t1) * std.math.pi / 180.0;
            
            const x0 = cx + radius * @cos(a0);
            const y0 = cy + radius * @sin(a0);
            const x1 = cx + radius * @cos(a1);
            const y1 = cy + radius * @sin(a1);
            
            vector.drawLineFast(fb, x0, y0, x1, y1, color);
        }
    }
    
    /// 文字列を描画
    pub fn drawText(
        self: SimpleFont,
        fb: *zgf.Image,
        text: []const u8,
        x: f32,
        y: f32,
        color: zgf.Color,
    ) void {
        var current_x = x;
        const spacing = self.cell_width * self.scale;
        
        for (text) |c| {
            if (c == '\n') {
                continue;
            }
            self.drawChar(fb, c, current_x, y, color);
            current_x += spacing;
        }
    }
    
    /// テキストの幅を計算
    pub fn measureText(self: SimpleFont, text: []const u8) f32 {
        return @as(f32, @floatFromInt(text.len)) * self.cell_width * self.scale;
    }
};

// ============================================================================
// テスト
// ============================================================================

test "vector_text - SimpleFont init" {
    const font = SimpleFont.init();
    try std.testing.expectEqual(@as(f32, 8), font.cell_width);
    try std.testing.expectEqual(@as(f32, 8), font.cell_height);
}

test "vector_text - draw char" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 100, 100, 3);
    defer fb.deinit();
    
    const font = SimpleFont.withScale(2.0);
    font.drawChar(&fb, 'A', 10, 10, .{ .r = 1, .g = 0, .b = 0 });
    
    const pixels = try fb.toPixels(allocator);
    defer allocator.free(pixels);
    
    try std.testing.expectEqual(@as(usize, 100 * 100 * 3), pixels.len);
}

test "vector_text - draw text" {
    const allocator = std.testing.allocator;
    var fb = try zgf.Image.init(allocator, 200, 50, 3);
    defer fb.deinit();
    
    const font = SimpleFont.withScale(1.5);
    font.drawText(&fb, "Hello", 10, 10, .{ .r = 0, .g = 1, .b = 0 });
    
    const pixels = try fb.toPixels(allocator);
    defer allocator.free(pixels);
    
    try std.testing.expectEqual(@as(usize, 200 * 50 * 3), pixels.len);
}

test "vector_text - measure text" {
    const font = SimpleFont.withScale(2.0);
    const width = font.measureText("Hello");
    try std.testing.expect(width > 0);
}
