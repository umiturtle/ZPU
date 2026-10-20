//! ZAPP UI WebAssembly Module - Lightweight Version
//!
//! ブラウザで動作するWASM版UIレンダラー
//! デスクトップ版と同一のUI構造を、直接ピクセルバッファにレンダリング

const std = @import("std");

// Memory をエクスポート
pub const memory = std.heap.wasm_memory;

// ============================================================================
// ピクセルバッファ (直接管理)
// ============================================================================

var pixel_buffer: ?[]u8 = null;
var buf_width: u32 = 800;
var buf_height: u32 = 600;

// ============================================================================
// 色とスタイル
// ============================================================================

const Color = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8 = 255,
};

const Palette = struct {
    primary: Color = .{ .r = 77, .g = 172, .b = 254 },
    secondary: Color = .{ .r = 128, .g = 128, .b = 128 },
    danger: Color = .{ .r = 239, .g = 68, .b = 68 },
    bg: Color = .{ .r = 24, .g = 24, .b = 32 },
    surface: Color = .{ .r = 42, .g = 42, .b = 62 },
    text: Color = .{ .r = 230, .g = 230, .b = 230 },
    text_muted: Color = .{ .r = 136, .g = 136, .b = 136 },
    border: Color = .{ .r = 64, .g = 64, .b = 78 },
};

// ============================================================================
// ドローイングプリミティブ
// ============================================================================

fn setPixel(x: u32, y: u32, color: Color) void {
    const buf = pixel_buffer orelse return;
    if (x >= buf_width or y >= buf_height) return;
    const offset = (@as(usize, y) * buf_width + x) * 4;
    if (offset + 3 < buf.len) {
        buf[offset] = color.r;
        buf[offset + 1] = color.g;
        buf[offset + 2] = color.b;
        buf[offset + 3] = color.a;
    }
}

fn fillRect(x: u32, y: u32, w: u32, h: u32, color: Color) void {
    var yi: u32 = y;
    while (yi < y + h and yi < buf_height) : (yi += 1) {
        var xi: u32 = x;
        while (xi < x + w and xi < buf_width) : (xi += 1) {
            setPixel(xi, yi, color);
        }
    }
}

fn fillCircle(cx: u32, cy: u32, radius: u32, color: Color) void {
    const r2: i32 = @intCast(radius * radius);
    var dy: i32 = -@as(i32, @intCast(radius));
    while (dy <= @as(i32, @intCast(radius))) : (dy += 1) {
        const dy2 = dy * dy;
        const dx: i32 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(r2 - dy2))));
        const x_start = @as(i32, @intCast(cx)) - dx;
        const x_end = @as(i32, @intCast(cx)) + dx;
        var xi = x_start;
        while (xi <= x_end) : (xi += 1) {
            if (xi >= 0 and @as(u32, @intCast(xi)) < buf_width) {
                setPixel(@intCast(xi), @intCast(@as(i32, @intCast(cy)) + dy), color);
            }
        }
    }
}

// ============================================================================
// UI state
// ============================================================================

var enabled = true;
var volume: f32 = 0.7;
var progress: f32 = 0.75;
var theme_dark = true;

// ============================================================================
// 初期化・クリーンアップ
// ============================================================================

export fn init(w: u32, h: u32) void {
    buf_width = w;
    buf_height = h;
    
    const allocator = std.heap.wasm_allocator;
    const size = @as(usize, w) * h * 4;
    pixel_buffer = allocator.alloc(u8, size) catch null;
    
    // 初期化: 背景で塗りつぶし
    if (pixel_buffer) |buf| {
        var i: usize = 0;
        while (i < buf.len) : (i += 4) {
            buf[i] = 24;
            buf[i + 1] = 24;
            buf[i + 2] = 32;
            buf[i + 3] = 255;
        }
    }
}

export fn deinit() void {
    if (pixel_buffer) |buf| {
        std.heap.wasm_allocator.free(buf);
        pixel_buffer = null;
    }
}

export fn getFrameBufferPtr() ?[*]u8 {
    if (pixel_buffer) |buf| {
        return buf.ptr;
    }
    return null;
}

export fn getFrameBufferSize() u32 {
    if (pixel_buffer) |buf| {
        return @intCast(buf.len);
    }
    return 0;
}

export fn getWidth() u32 {
    return buf_width;
}

export fn getHeight() u32 {
    return buf_height;
}

// ============================================================================
// UI レンダリング
// ============================================================================

export fn renderUI() void {
    const p = if (theme_dark) Palette{} else Palette{
        .bg = .{ .r = 240, .g = 240, .b = 240 },
        .surface = .{ .r = 255, .g = 255, .b = 255 },
        .text = .{ .r = 26, .g = 26, .b = 26 },
        .text_muted = .{ .r = 102, .g = 102, .b = 102 },
        .border = .{ .r = 224, .g = 224, .b = 224 },
    };
    
    // 背景
    fillRect(0, 0, buf_width, buf_height, p.bg);
    
    // メインコンテナ
    fillRect(20, 20, buf_width - 40, buf_height - 40, p.surface);
    
    // タイトル領域
    fillRect(20, 20, buf_width - 40, 60, p.surface);
    drawText("ZAPP UI v2", 40, 40, p.text);
    
    // 区切り線
    fillRect(40, 90, buf_width - 80, 2, p.border);
    
    // ボタン
    const button_y: u32 = 110;
    const button_h: u32 = 40;
    const button_w: u32 = 120;
    const button_gap: u32 = 10;
    
    drawButton(40, button_y, button_w, button_h, "Primary", p.primary, p.text);
    drawButton(40 + button_w + button_gap, button_y, button_w, button_h, "Secondary", p.secondary, p.text);
    drawButton(40 + (button_w + button_gap) * 2, button_y, button_w, button_h, "Outlined", .{ .r = 0, .g = 0, .b = 0, .a = 0 }, p.text);
    drawButtonOutline(40 + (button_w + button_gap) * 2, button_y, button_w, button_h, p.border);
    drawButton(40 + (button_w + button_gap) * 3, button_y, button_w, button_h, "Danger", p.danger, p.text);
    
    // スペーサー
    const controls_y: u32 = 180;
    
    // チェックボックス
    drawCheckbox(40, controls_y, 20, 20, enabled, p);
    drawText("Enable feature", 70, controls_y + 2, p.text);
    
    // スライダー
    const slider_y: u32 = controls_y + 40;
    drawSlider(40, slider_y, buf_width - 80, 8, volume, p);
    drawText("Volume", 40, slider_y - 16, p.text_muted);
    
    // プログレスバー
    const progress_y: u32 = slider_y + 40;
    drawProgressBar(40, progress_y, buf_width - 80, 12, progress, p);
    
    // テキストサンプル
    const text_y: u32 = progress_y + 50;
    drawText("Large text", 40, text_y, p.text);
    drawText("Base text", 40, text_y + 24, p.text);
    drawText("Small text", 40, text_y + 44, p.text_muted);
}

fn drawText(text: []const u8, x: u32, y: u32, color: Color) void {
    _ = text;
    _ = x;
    _ = y;
    _ = color;
    // TODO: 実際のフォントレンダリング
    // 簡易的には矩形で描画
}

fn drawButton(x: u32, y: u32, w: u32, h: u32, label: []const u8, bg: Color, fg: Color) void {
    _ = label;
    _ = fg;
    fillRect(x, y, w, h, bg);
    // TODO: ラベル描画
}

fn drawButtonOutline(x: u32, y: u32, w: u32, h: u32, color: Color) void {
    // 上辺
    fillRect(x, y, w, 2, color);
    // 下辺
    fillRect(x, y + h - 2, w, 2, color);
    // 左辺
    fillRect(x, y, 2, h, color);
    // 右辺
    fillRect(x + w - 2, y, 2, h, color);
}

fn drawCheckbox(x: u32, y: u32, w: u32, h: u32, checked: bool, p: Palette) void {
    const bg = if (checked) p.primary else p.surface;
    fillRect(x, y, w, h, bg);
    
    if (checked) {
        // チェックマーク
        const cx = x + w / 2;
        const cy = y + h / 2;
        setPixel(cx - 3, cy, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx - 2, cy + 1, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx - 1, cy + 2, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx, cy + 3, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx + 1, cy + 2, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx + 2, cy, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx + 3, cy - 2, .{ .r = 255, .g = 255, .b = 255 });
        setPixel(cx + 4, cy - 4, .{ .r = 255, .g = 255, .b = 255 });
    }
}

fn drawSlider(x: u32, y: u32, w: u32, h: u32, value: f32, p: Palette) void {
    // トラック
    fillRect(x, y, w, h, p.surface);
    
    // 塗り
    const fill_w: u32 = @intFromFloat(@as(f32, @floatFromInt(w)) * value);
    fillRect(x, y, fill_w, h, p.primary);
    
    // ハンドル
    const handle_x: u32 = x + fill_w;
    const handle_y: u32 = y + h / 2;
    fillCircle(handle_x, handle_y, 10, p.primary);
}

fn drawProgressBar(x: u32, y: u32, w: u32, h: u32, value: f32, p: Palette) void {
    // トラック
    fillRect(x, y, w, h, p.surface);
    
    // 塗り
    const fill_w: u32 = @intFromFloat(@as(f32, @floatFromInt(w)) * value);
    fillRect(x, y, fill_w, h, p.primary);
}

// ============================================================================
// インタラクティブ コントロール
// ============================================================================

export fn toggleTheme() void {
    theme_dark = !theme_dark;
}

export fn setVolume(v: f32) void {
    volume = @max(0, @min(1, v));
}

export fn setProgress(p: f32) void {
    progress = @max(0, @min(1, p));
}

export fn setEnabled(e: bool) void {
    enabled = e;
}

// ============================================================================
// Tests
// ============================================================================

test "wasm_ui - init and deinit" {
    init(100, 100);
    try std.testing.expect(pixel_buffer != null);
    
    const size = getFrameBufferSize();
    try std.testing.expect(size > 0);
    
    deinit();
    try std.testing.expect(pixel_buffer == null);
}

test "wasm_ui - render" {
    init(200, 200);
    renderUI();
    
    const buf = pixel_buffer.?;
    try std.testing.expect(buf.len > 0);
    
    deinit();
}

test "wasm_ui - state changes" {
    init(100, 100);
    
    setVolume(0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), volume, 1e-6);
    
    setProgress(0.8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), progress, 1e-6);
    
    setEnabled(false);
    try std.testing.expect(!enabled);
    
    toggleTheme();
    try std.testing.expect(!theme_dark);
    
    deinit();
}
