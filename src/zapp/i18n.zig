//! ZAPP Internationalization (i18n)
//!
//! 多言語対応、特に日本語入力サポートの基盤。

const std = @import("std");

/// Unicodeコードポイント
pub const CodePoint = u21;

/// UTF-8デコーダー
pub const Utf8Decoder = struct {
    bytes: []const u8,
    pos: usize,

    pub fn init(bytes: []const u8) Utf8Decoder {
        return .{
            .bytes = bytes,
            .pos = 0,
        };
    }

    pub fn next(self: *Utf8Decoder) ?CodePoint {
        if (self.pos >= self.bytes.len) return null;

        const byte = self.bytes[self.pos];
        self.pos += 1;

        if (byte < 0x80) {
            // 1バイト文字 (ASCII)
            return byte;
        } else if (byte < 0xC0) {
            // 不正なバイト
            return null;
        } else if (byte < 0xE0) {
            // 2バイト文字
            if (self.pos >= self.bytes.len) return null;
            const cp = @as(CodePoint, byte & 0x1F) << 6 |
                       @as(CodePoint, self.bytes[self.pos] & 0x3F);
            self.pos += 1;
            return cp;
        } else if (byte < 0xF0) {
            // 3バイト文字
            if (self.pos + 1 >= self.bytes.len) return null;
            const cp = @as(CodePoint, byte & 0x0F) << 12 |
                       @as(CodePoint, self.bytes[self.pos] & 0x3F) << 6 |
                       @as(CodePoint, self.bytes[self.pos + 1] & 0x3F);
            self.pos += 2;
            return cp;
        } else if (byte < 0xF8) {
            // 4バイト文字
            if (self.pos + 2 >= self.bytes.len) return null;
            const cp = @as(CodePoint, byte & 0x07) << 18 |
                       @as(CodePoint, self.bytes[self.pos] & 0x3F) << 12 |
                       @as(CodePoint, self.bytes[self.pos + 1] & 0x3F) << 6 |
                       @as(CodePoint, self.bytes[self.pos + 2] & 0x3F);
            self.pos += 3;
            return cp;
        }

        return null;
    }

    pub fn reset(self: *Utf8Decoder) void {
        self.pos = 0;
    }
};

/// UTF-8エンコーダー
pub fn encodeUtf8(cp: CodePoint, buf: []u8) usize {
    if (cp < 0x80) {
        buf[0] = @truncate(cp);
        return 1;
    } else if (cp < 0x800) {
        buf[0] = @truncate(0xC0 | (cp >> 6));
        buf[1] = @truncate(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = @truncate(0xE0 | (cp >> 12));
        buf[1] = @truncate(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = @truncate(0x80 | (cp & 0x3F));
        return 3;
    } else if (cp < 0x110000) {
        buf[0] = @truncate(0xF0 | (cp >> 18));
        buf[1] = @truncate(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = @truncate(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = @truncate(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

/// 日本語ひらがな判定
pub fn isHiragana(cp: CodePoint) bool {
    return cp >= 0x3040 and cp <= 0x309F;
}

/// 日本語カタカナ判定
pub fn isKatakana(cp: CodePoint) bool {
    return cp >= 0x30A0 and cp <= 0x30FF;
}

/// 日本語漢字判定（CJK統一漢字）
pub fn isKanji(cp: CodePoint) bool {
    return cp >= 0x4E00 and cp <= 0x9FFF;
}

/// 日本語文字判定
pub fn isJapanese(cp: CodePoint) bool {
    return isHiragana(cp) or isKatakana(cp) or isKanji(cp);
}

// テスト
test "i18n - utf8 decode ASCII" {
    var decoder = Utf8Decoder.init("Hello");
    
    try std.testing.expectEqual(@as(CodePoint, 'H'), decoder.next().?);
    try std.testing.expectEqual(@as(CodePoint, 'e'), decoder.next().?);
    try std.testing.expectEqual(@as(CodePoint, 'l'), decoder.next().?);
    try std.testing.expectEqual(@as(CodePoint, 'l'), decoder.next().?);
    try std.testing.expectEqual(@as(CodePoint, 'o'), decoder.next().?);
    try std.testing.expectEqual(@as(?CodePoint, null), decoder.next());
}

test "i18n - utf8 decode Japanese hiragana" {
    // あ (U+3042) = 0xE3 0x81 0x82
    var decoder = Utf8Decoder.init(&.{ 0xE3, 0x81, 0x82 });
    
    try std.testing.expectEqual(@as(CodePoint, 0x3042), decoder.next().?);
    try std.testing.expectEqual(@as(?CodePoint, null), decoder.next());
}

test "i18n - utf8 encode" {
    var buf: [4]u8 = undefined;
    
    // ASCII
    const len1 = encodeUtf8('A', &buf);
    try std.testing.expectEqual(@as(usize, 1), len1);
    try std.testing.expectEqual(@as(u8, 0x41), buf[0]);
    
    // ひらがな あ (U+3042)
    const len2 = encodeUtf8(0x3042, &buf);
    try std.testing.expectEqual(@as(usize, 3), len2);
    try std.testing.expectEqual(@as(u8, 0xE3), buf[0]);
    try std.testing.expectEqual(@as(u8, 0x81), buf[1]);
    try std.testing.expectEqual(@as(u8, 0x82), buf[2]);
}

test "i18n - Japanese character detection" {
    try std.testing.expect(isHiragana(0x3042)); // あ
    try std.testing.expect(!isHiragana(0x30A2)); // ア
    try std.testing.expect(!isHiragana(0x4E00)); // 一
    
    try std.testing.expect(isKatakana(0x30A2)); // ア
    try std.testing.expect(!isKatakana(0x3042)); // あ
    
    try std.testing.expect(isKanji(0x4E00)); // 一
    try std.testing.expect(!isKanji(0x3042)); // あ
    
    try std.testing.expect(isJapanese(0x3042)); // あ
    try std.testing.expect(isJapanese(0x30A2)); // ア
    try std.testing.expect(isJapanese(0x4E00)); // 一
    try std.testing.expect(!isJapanese(0x0041)); // A
}
