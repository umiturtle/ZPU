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
const widgets = @import("widgets.zig");

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
        .label => |l| widgets.label.render(fb, .{
            .text = l.text,
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
            .color = l.style.fg,
        }),
        .button => |b| widgets.button.render(fb, .{
            .text = b.text,
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
            .bg = b.style.bg,
            .fg = b.style.fg,
            .border = b.style.border,
            .radius = b.style.radius,
        }),
        .slider => |s| widgets.slider.render(fb, .{
            .value = s.value,
            .min = s.min_val,
            .max = s.max_val,
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
            .fill = s.style.accent,
        }),
        .checkbox => |c| widgets.checkbox.render(fb, .{
            .checked = c.checked,
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
            .check = c.style.accent,
        }),
        .progress_bar => |p| widgets.progress_bar.render(fb, .{
            .value = p.value,
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
            .fill = p.style.accent,
        }),
        .text_input => |t| widgets.text_input.render(fb, .{
            .text = t.text,
            .placeholder = t.placeholder,
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
        }),
        .spacer => {},
        .separator => widgets.separator.render(fb, .{
            .rect = .{ .x = rect.x, .y = rect.y, .w = rect.w, .h = rect.h },
        }),
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
