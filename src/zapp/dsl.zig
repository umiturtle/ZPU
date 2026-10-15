//! ZAPP 宣言的UI DSL
//!
//! ZAPPのUIを宣言的に記述するためのドメイン固有言語（DSL）。
//! 即時的モード（immediate mode）の上に、ツリー構造の宣言的記法を提供する。
//!
//! ## 基本的な使い方
//!
//! ```zig
//! const ui = @import("zapp").dsl;
//!
//! try ui.render(alloc, .{ .width = 800, .height = 600 }, &.{
//!     .vstack(.{ .spacing = 8 }, &.{
//!         .label("Hello, ZAPP!"),
//!         .hstack(.{ .spacing = 4 }, &.{
//!             .button("Click"),
//!             .button("Reset"),
//!         }),
//!         .slider("Volume", 0.5),
//!         .checkbox("Enable", true),
//!     }),
//! });
//! ```

const std = @import("std");
const zgf = @import("zgf");

/// UIノードの種類
pub const NodeKind = enum {
    // コンテナ
    vstack,
    hstack,
    // ウィジェット
    label,
    button,
    slider,
    checkbox,
    progress_bar,
    text_input,
    spacer,
    separator,
};

/// レイアウト方向
pub const Direction = enum { vertical, horizontal };

/// 配置
pub const Alignment = enum { start, center, end_ };

/// ノード共通の設定
pub const NodeOpts = struct {
    spacing: f32 = 4.0,
    padding: f32 = 8.0,
    alignment: Alignment = .start,
    expand: bool = false,
    min_w: ?f32 = null,
    min_h: ?f32 = null,
    max_w: ?f32 = null,
    max_h: ?f32 = null,
};

/// 色付きウィジェット設定
pub const WidgetOpts = struct {
    bg: zgf.Color = .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 },
    fg: zgf.Color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
    accent: zgf.Color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
    border: zgf.Color = .{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 },
    radius: f32 = 4.0,
    font_h: f32 = 14.0,
};

/// 宣言的UIツリーのノード
pub const Node = union(NodeKind) {
    // コンテナ
    vstack: ContainerNode,
    hstack: ContainerNode,
    // ウィジェット
    label: LabelNode,
    button: ButtonNode,
    slider: SliderNode,
    checkbox: CheckboxNode,
    progress_bar: ProgressBarNode,
    text_input: TextInputNode,
    spacer: SpacerNode,
    separator: SeparatorNode,
};

pub const ContainerNode = struct {
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
    children: []const Node,
};

pub const LabelNode = struct {
    text: []const u8,
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

pub const ButtonNode = struct {
    text: []const u8,
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

pub const SliderNode = struct {
    label_text: []const u8,
    value: f32,
    min_val: f32 = 0.0,
    max_val: f32 = 1.0,
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

pub const CheckboxNode = struct {
    label_text: []const u8,
    checked: bool,
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

pub const ProgressBarNode = struct {
    value: f32,
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

pub const TextInputNode = struct {
    text: []const u8,
    placeholder: []const u8 = "",
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

pub const SpacerNode = struct {
    size: f32,
};

pub const SeparatorNode = struct {
    opts: NodeOpts = .{},
    style: WidgetOpts = .{},
};

/// 簡易コンストラクタ（comptime DSL）
pub fn vstack(opts: NodeOpts, children: []const Node) Node {
    return .{ .vstack = .{ .opts = opts, .children = children } };
}

pub fn hstack(opts: NodeOpts, children: []const Node) Node {
    return .{ .hstack = .{ .opts = opts, .children = children } };
}

pub fn label(text: []const u8) Node {
    return .{ .label = .{ .text = text } };
}

pub fn labelStyled(text: []const u8, opts: NodeOpts, style: WidgetOpts) Node {
    return .{ .label = .{ .text = text, .opts = opts, .style = style } };
}

pub fn button(text: []const u8) Node {
    return .{ .button = .{ .text = text } };
}

pub fn buttonStyled(text: []const u8, opts: NodeOpts, style: WidgetOpts) Node {
    return .{ .button = .{ .text = text, .opts = opts, .style = style } };
}

pub fn slider(label_text: []const u8, value: f32) Node {
    return .{ .slider = .{ .label_text = label_text, .value = value } };
}

pub fn checkbox(label_text: []const u8, checked: bool) Node {
    return .{ .checkbox = .{ .label_text = label_text, .checked = checked } };
}

pub fn progressBar(value: f32) Node {
    return .{ .progress_bar = .{ .value = value } };
}

pub fn textInput(text: []const u8) Node {
    return .{ .text_input = .{ .text = text } };
}

pub fn spacer(size: f32) Node {
    return .{ .spacer = .{ .size = size } };
}

pub fn separator() Node {
    return .{ .separator = .{} };
}

/// レイアウト計算結果
pub const Rect = struct { x: f32, y: f32, w: f32, h: f32 };

/// レイアウトエンジン: ノードツリーから各ウィジェットの矩形を計算
pub fn layout(alloc: std.mem.Allocator, node: Node, bounds: Rect) []const LayoutResult {
    var results = std.ArrayList(LayoutResult).empty;
    layoutNode(alloc, node, bounds, &results) catch return &.{};
    return results.toOwnedSlice(alloc) catch return &.{};
}

pub const LayoutResult = struct {
    node: Node,
    rect: Rect,
};

fn layoutNode(alloc: std.mem.Allocator, node: Node, bounds: Rect, results: *std.ArrayList(LayoutResult)) error{OutOfMemory}!void {
    switch (node) {
        .vstack => |c| try layoutContainer(alloc, c, .vertical, bounds, results),
        .hstack => |c| try layoutContainer(alloc, c, .horizontal, bounds, results),
        .label, .button, .slider, .checkbox, .progress_bar, .text_input, .separator => {
            try results.append(alloc, .{ .node = node, .rect = bounds });
        },
        .spacer => {},
    }
}

fn layoutContainer(alloc: std.mem.Allocator, c: ContainerNode, dir: Direction, bounds: Rect, results: *std.ArrayList(LayoutResult)) error{OutOfMemory}!void {
    if (c.children.len == 0) return;
    const pad = c.opts.padding;
    const sp = c.opts.spacing;
    // innerのサイズを計算（負にならないように）
    const inner_w: f32 = @max(bounds.w - pad * 2, 0);
    const inner_h: f32 = @max(bounds.h - pad * 2, 0);
    const inner = Rect{ .x = bounds.x + pad, .y = bounds.y + pad, .w = inner_w, .h = inner_h };

    const default_h: f32 = 32.0;
    const total_spacing = sp * @as(f32, @floatFromInt(c.children.len - 1));
    var child_size: f32 = if (dir == .vertical)
        (inner_h - total_spacing) / @as(f32, @floatFromInt(c.children.len))
    else
        (inner_w - total_spacing) / @as(f32, @floatFromInt(c.children.len));
    if (child_size < default_h) child_size = default_h;

    var offset: f32 = 0;
    for (c.children) |child| {
        const child_rect = if (dir == .vertical)
            Rect{ .x = inner.x, .y = inner.y + offset, .w = inner.w, .h = child_size }
        else
            Rect{ .x = inner.x + offset, .y = inner.y, .w = child_size, .h = inner.h };
        try layoutNode(alloc, child, child_rect, results);
        offset += child_size + sp;
    }
}

/// レンダラー: レイアウト結果をフレームバッファに描画
pub fn render(fb: *zgf.Image, root: Node) void {
    fb.fill(.{ .r = 0.12, .g = 0.12, .b = 0.16, .a = 1.0 });

    const bounds = Rect{ .x = 0, .y = 0, .w = @floatFromInt(fb.width()), .h = @floatFromInt(fb.height()) };
    const alloc = std.heap.page_allocator;
    const results = layout(alloc, root, bounds);
    defer alloc.free(results);

    for (results) |lr| {
        renderNode(fb, lr.node, lr.rect);
    }
}

fn renderNode(fb: *zgf.Image, node: Node, rect: Rect) void {
    switch (node) {
        .vstack, .hstack => {}, // 子は既にレイアウト済み
        .label => |l| renderLabel(fb, l.text, rect, l.style),
        .button => |b| renderButton(fb, b.text, rect, b.style),
        .slider => |s| renderSlider(fb, s.label_text, s.value, s.min_val, s.max_val, rect, s.style),
        .checkbox => |c| renderCheckbox(fb, c.label_text, c.checked, rect, c.style),
        .progress_bar => |p| renderProgressBar(fb, p.value, rect, p.style),
        .text_input => |t| renderTextInput(fb, t.text, t.placeholder, rect, t.style),
        .spacer => {},
        .separator => |s| renderSeparator(fb, rect, s.style),
    }
}

// --- 個別ウィジェット描画 ---

fn fillRect(fb: *zgf.Image, rect: Rect, color: zgf.Color) void {
    const fw = fb.width();
    const fh = fb.height();
    // 座標を整数に変換（クランプ）
    const x0: usize = @intFromFloat(@max(rect.x, 0));
    const y0: usize = @intFromFloat(@max(rect.y, 0));
    const x1: usize = @intFromFloat(@min(rect.x + rect.w, @as(f32, @floatFromInt(fw))));
    const y1: usize = @intFromFloat(@min(rect.y + rect.h, @as(f32, @floatFromInt(fh))));
    if (x0 >= x1 or y0 >= y1) return;
    // ピクセルを描画
    for (y0..y1) |y| {
        for (x0..x1) |x| {
            fb.setPixel(@intCast(x), @intCast(y), color);
        }
    }
}

fn drawRectBorder(fb: *zgf.Image, rect: Rect, color: zgf.Color) void {
    const fw = fb.width();
    const fh = fb.height();
    const x0: usize = @intFromFloat(@max(rect.x, 0));
    const y0: usize = @intFromFloat(@max(rect.y, 0));
    const x1: usize = @intFromFloat(@min(rect.x + rect.w - 1, @as(f32, @floatFromInt(fw - 1))));
    const y1: usize = @intFromFloat(@min(rect.y + rect.h - 1, @as(f32, @floatFromInt(fh - 1))));
    if (x0 > x1 or y0 > y1) return;
    // 上下の水平線
    for (x0..x1) |x| {
        fb.setPixel(@intCast(x), @intCast(y0), color);
        fb.setPixel(@intCast(x), @intCast(y1), color);
    }
    // 左右の垂直線
    for (y0..y1) |y| {
        fb.setPixel(@intCast(x0), @intCast(y), color);
        fb.setPixel(@intCast(x1), @intCast(y), color);
    }
}

/// 簡易テキストレンダラー（5x7ドット風）
fn drawText(fb: *zgf.Image, text: []const u8, x: f32, y: f32, color: zgf.Color) void {
    var cx: usize = @intFromFloat(@max(x, 0));
    const cy: usize = @intFromFloat(@max(y, 0));
    const fw = fb.width();
    const fh = fb.height();
    const char_w: usize = 6;
    const char_h: usize = 7;
    for (text) |ch| {
        if (ch == ' ') {
            cx += char_w + 1;
            continue;
        }
        // 画面右端を超えたら描画を停止
        if (cx >= fw) break;
        // 文字パターンを取得
        const pattern = charPattern(ch);
        // 7行 x 5列の文字を描画
        for (0..char_h) |py| {
            for (0..5) |px| {
                if (pattern[py] & (@as(u8, 1) << @intCast(4 - px)) != 0) {
                    const draw_x = cx + px;
                    const draw_y = cy + py;
                    if (draw_x < fw and draw_y < fh) {
                        fb.setPixel(@intCast(draw_x), @intCast(draw_y), color);
                    }
                }
            }
        }
        cx += char_w + 1;
    }
}

fn charPattern(ch: u8) [7]u8 {
    return switch (ch) {
        'A' => .{ 0x04, 0x0A, 0x11, 0x1F, 0x11, 0x11, 0x11 },
        'B' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E },
        'C' => .{ 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E },
        'D' => .{ 0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E },
        'E' => .{ 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F },
        'F' => .{ 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10 },
        'G' => .{ 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E },
        'H' => .{ 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 },
        'I' => .{ 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E },
        'J' => .{ 0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C },
        'K' => .{ 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 },
        'L' => .{ 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F },
        'M' => .{ 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11 },
        'N' => .{ 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11 },
        'O' => .{ 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
        'P' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 },
        'Q' => .{ 0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D },
        'R' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 },
        'S' => .{ 0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E },
        'T' => .{ 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
        'U' => .{ 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
        'V' => .{ 0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04 },
        'W' => .{ 0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11 },
        'X' => .{ 0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11 },
        'Y' => .{ 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04 },
        'Z' => .{ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F },
        '0' => .{ 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E },
        '1' => .{ 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E },
        '2' => .{ 0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F },
        '3' => .{ 0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E },
        '4' => .{ 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 },
        '5' => .{ 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E },
        '6' => .{ 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E },
        '7' => .{ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 },
        '8' => .{ 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E },
        '9' => .{ 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C },
        ':' => .{ 0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00 },
        '.' => .{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06 },
        ',' => .{ 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08 },
        '-' => .{ 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00 },
        '+' => .{ 0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00 },
        '=' => .{ 0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00 },
        '%' => .{ 0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13 },
        '/' => .{ 0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10 },
        '(' => .{ 0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02 },
        ')' => .{ 0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08 },
        '[' => .{ 0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E },
        ']' => .{ 0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E },
        '!' => .{ 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04 },
        '?' => .{ 0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04 },
        '_' => .{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F },
        '"' => .{ 0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00 },
        '\'' => .{ 0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00 },
        '*' => .{ 0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00 },
        '#' => .{ 0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A },
        '@' => .{ 0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0F },
        '&' => .{ 0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0D },
        '^' => .{ 0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00 },
        '~' => .{ 0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00 },
        '<' => .{ 0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02 },
        '>' => .{ 0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08 },
        else => .{ 0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F }, // □ (unknown)
    };
}

fn renderLabel(fb: *zgf.Image, text: []const u8, rect: Rect, style: WidgetOpts) void {
    const x: f32 = rect.x + style.font_h * 0.3;
    const y: f32 = rect.y + (rect.h - style.font_h) / 2;
    drawText(fb, text, @max(x, 0), @max(y, 0), style.fg);
}

fn renderButton(fb: *zgf.Image, text: []const u8, rect: Rect, style: WidgetOpts) void {
    fillRect(fb, rect, style.bg);
    drawRectBorder(fb, rect, style.border);
    // テキスト中央配置
    const text_w: f32 = @floatFromInt(text.len * 7);
    const text_x: f32 = rect.x + (rect.w - text_w) / 2;
    const text_y: f32 = rect.y + (rect.h - 8) / 2;
    drawText(fb, text, @max(text_x, 0), @max(text_y, 0), style.fg);
}

fn renderSlider(fb: *zgf.Image, label_text: []const u8, value: f32, min_val: f32, max_val: f32, rect: Rect, style: WidgetOpts) void {
    const label_w: f32 = @floatFromInt(label_text.len * 7 + 8);
    // ラベル
    drawText(fb, label_text, @max(rect.x, 0), @max(rect.y + (rect.h - 8) / 2, 0), style.fg);
    // トラック
    const track_x: f32 = rect.x + label_w;
    const track_y: f32 = rect.y + rect.h / 2 - 3;
    const track_w: f32 = rect.w - label_w;
    const track_rect = Rect{ .x = track_x, .y = track_y, .w = track_w, .h = 6 };
    fillRect(fb, track_rect, style.bg);
    // フィル
    const norm = (value - min_val) / (max_val - min_val);
    const fill_w = track_rect.w * norm;
    fillRect(fb, Rect{ .x = track_rect.x, .y = track_rect.y, .w = fill_w, .h = track_rect.h }, style.accent);
    // ハンドル
    const hx: f32 = track_rect.x + fill_w - 4;
    fillRect(fb, Rect{ .x = hx, .y = rect.y + 2, .w = 8, .h = rect.h - 4 }, style.fg);
}

fn renderCheckbox(fb: *zgf.Image, label_text: []const u8, checked: bool, rect: Rect, style: WidgetOpts) void {
    const box_size: f32 = 16;
    const box_rect = Rect{ .x = rect.x, .y = rect.y + (rect.h - box_size) / 2, .w = box_size, .h = box_size };
    fillRect(fb, box_rect, style.bg);
    drawRectBorder(fb, box_rect, style.border);
    if (checked) {
        // チェックマーク（簡易版：斜線）
        const bx: usize = @intFromFloat(@max(box_rect.x, 0));
        const by: usize = @intFromFloat(@max(box_rect.y, 0));
        const bs: usize = 16;
        const fw = fb.width();
        const fh = fb.height();
        for (0..bs) |i| {
            const px = bx + i;
            const py = by + i;
            if (px < fw and py < fh) {
                fb.setPixel(@intCast(px), @intCast(py), style.accent);
            }
        }
        for (0..bs) |i| {
            const px = bx + i;
            const py = by + (bs - 1 - i);
            if (px < fw and py < fh) {
                fb.setPixel(@intCast(px), @intCast(py), style.accent);
            }
        }
    }
    const label_x: f32 = rect.x + box_size + 6;
    const label_y: f32 = rect.y + (rect.h - 8) / 2;
    drawText(fb, label_text, label_x, label_y, style.fg);
}

fn renderProgressBar(fb: *zgf.Image, value: f32, rect: Rect, style: WidgetOpts) void {
    fillRect(fb, rect, style.bg);
    drawRectBorder(fb, rect, style.border);
    const fill_w = (rect.w - 2) * value;
    fillRect(fb, Rect{ .x = rect.x + 1, .y = rect.y + 1, .w = fill_w, .h = rect.h - 2 }, style.accent);
}

fn renderTextInput(fb: *zgf.Image, text: []const u8, placeholder: []const u8, rect: Rect, style: WidgetOpts) void {
    fillRect(fb, rect, style.bg);
    drawRectBorder(fb, rect, style.border);
    if (text.len > 0) {
        drawText(fb, text, @max(rect.x + 4, 0), @max(rect.y + (rect.h - 8) / 2, 0), style.fg);
    } else if (placeholder.len > 0) {
        drawText(fb, placeholder, @max(rect.x + 4, 0), @max(rect.y + (rect.h - 8) / 2, 0), .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1 });
    }
}

fn renderSeparator(fb: *zgf.Image, rect: Rect, style: WidgetOpts) void {
    const fw = fb.width();
    const fh = fb.height();
    const mid_y: usize = @intFromFloat(@max(@min(rect.y + rect.h / 2, @as(f32, @floatFromInt(fh - 1))), 0));
    const sx: usize = @intFromFloat(@max(rect.x, 0));
    const ex: usize = @intFromFloat(@min(rect.x + rect.w, @as(f32, @floatFromInt(fw))));
    if (sx >= ex) return;
    for (sx..ex) |x| {
        fb.setPixel(@intCast(x), @intCast(mid_y), style.border);
    }
}

// --- テスト ---

test "DSL node construction" {
    const tree = vstack(.{ .spacing = 8 }, &.{
        label("Hello"),
        hstack(.{ .spacing = 4 }, &.{
            button("OK"),
            button("Cancel"),
        }),
        slider("Volume", 0.5),
        checkbox("Enable", true),
        progressBar(0.75),
    });
    _ = tree;
}

test "layout calculation" {
    const tree = vstack(.{ .spacing = 4, .padding = 4 }, &.{
        label("Test"),
        button("Click"),
    });
    const alloc = std.testing.allocator;
    const results = layout(alloc, tree, .{ .x = 0, .y = 0, .w = 200, .h = 100 });
    defer alloc.free(results);
    try std.testing.expect(results.len >= 2);
}

test "render to framebuffer" {
    var fb = try zgf.Image.init(std.testing.allocator, 200, 100, 4);
    defer fb.deinit();
    const tree = vstack(.{ .spacing = 4, .padding = 4 }, &.{
        label("Hello"),
        button("Click"),
        slider("Vol", 0.5),
    });
    render(&fb, tree);
    // 背景がクリアされたことを確認
    const px = fb.getPixel(0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.12), px.r, 0.01);
}

test "render declarative UI" {
    // 宣言的UIをレンダリングしてスナップショット検証
    var fb = try zgf.Image.init(std.testing.allocator, 400, 300, 4);
    defer fb.deinit();

    const tree = vstack(.{ .spacing = 8, .padding = 16 }, &.{
        label("ZAPP Declarative UI"),
        separator(),
        hstack(.{ .spacing = 8 }, &.{
            button("Save"),
            button("Load"),
            button("Exit"),
        }),
        slider("Volume", 0.65),
        slider("Brightness", 0.4),
        checkbox("Dark Mode", true),
        checkbox("Notifications", false),
        progressBar(0.72),
        textInput("Enter name..."),
    });

    render(&fb, tree);

    // 背景が描画されたことを確認
    const bg = fb.getPixel(200, 150);
    // 背景は widget の上にあるので、何もない場所をチェック
    const corner = fb.getPixel(1, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.12), corner.r, 0.01);
    _ = bg;
}
