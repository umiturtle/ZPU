//! ZAPP 宣言的UI DSL v2
//!
//! 読みやすく、書きやすく、カスタマイズ性が高く、ロジックとの親和度が高いUI記法。
//!
//! ## 特徴
//! - チェーン可能なビルダーパターン
//! - 統合されたスタイルシステム
//! - イベントハンドラの統合
//! - テーマサポート
//! - 型安全なプロパティ
//!
//! ## 使用例
//!
//! ```zig
//! const ui = @import("zapp").ui;
//!
//! // シンプルな例
//! const view = ui.vstack()
//!     .spacing(12)
//!     .padding(16)
//!     .children(&.{
//!         ui.text("Hello, ZAPP!")
//!             .font(.large)
//!             .color(.primary),
//!         
//!         ui.button("Click me")
//!             .onClick(handleClick)
//!             .style(.{ .bg = .blue, .radius = 8 }),
//!         
//!         ui.hstack()
//!             .gap(8)
//!             .children(&.{
//!                 ui.checkbox("Enable", &state.enabled),
//!                 ui.slider("Volume", &state.volume, .{ .min = 0, .max = 100 }),
//!             }),
//!     });
//!
//! // テーマ付き
//! const themed = ui.theme(.dark)
//!     .build(ui.vstack()
//!         .children(&.{
//!             ui.text("Themed UI"),
//!         }));
//! ```

const std = @import("std");
const zgf = @import("zgf");
const vector = @import("vector.zig");
const vector_text = @import("vector_text.zig");

// ============================================================================
// スタイルシステム
// ============================================================================

/// カラーパレット
pub const Palette = struct {
    primary: zgf.Color = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 },
    secondary: zgf.Color = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 },
    success: zgf.Color = .{ .r = 0.2, .g = 0.8, .b = 0.4, .a = 1.0 },
    warning: zgf.Color = .{ .r = 0.9, .g = 0.7, .b = 0.2, .a = 1.0 },
    danger: zgf.Color = .{ .r = 0.9, .g = 0.3, .b = 0.3, .a = 1.0 },
    bg: zgf.Color = .{ .r = 0.12, .g = 0.12, .b = 0.16, .a = 1.0 },
    surface: zgf.Color = .{ .r = 0.2, .g = 0.2, .b = 0.25, .a = 1.0 },
    text: zgf.Color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1.0 },
    text_muted: zgf.Color = .{ .r = 0.6, .g = 0.6, .b = 0.6, .a = 1.0 },
    border: zgf.Color = .{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 },
    
    pub const light: Palette = .{
        .bg = .{ .r = 0.95, .g = 0.95, .b = 0.95, .a = 1.0 },
        .surface = .{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 },
        .text = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
        .text_muted = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 },
        .border = .{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1.0 },
    };
};

/// フォントサイズ
pub const FontSize = enum {
    xs, sm, base, lg, xl, @"2xl", @"3xl",
    
    pub fn size(self: FontSize) f32 {
        return switch (self) {
            .xs => 10,
            .sm => 12,
            .base => 14,
            .lg => 16,
            .xl => 20,
            .@"2xl" => 24,
            .@"3xl" => 32,
        };
    }
};

/// スペーシングスケール
pub const Spacing = enum {
    none, xs, sm, base, lg, xl, @"2xl",
    
    pub fn value(self: Spacing) f32 {
        return switch (self) {
            .none => 0,
            .xs => 4,
            .sm => 8,
            .base => 12,
            .lg => 16,
            .xl => 24,
            .@"2xl" => 32,
        };
    }
};

/// ボーダースタイル
pub const BorderRadius = enum {
    none, sm, base, lg, xl, full,
    
    pub fn radius(self: BorderRadius) f32 {
        return switch (self) {
            .none => 0,
            .sm => 2,
            .base => 4,
            .lg => 8,
            .xl => 12,
            .full => 9999,
        };
    }
};

/// コンポーネントスタイル
pub const Style = struct {
    bg: ?zgf.Color = null,
    fg: ?zgf.Color = null,
    border_color: ?zgf.Color = null,
    border_width: f32 = 0,
    radius: ?f32 = null,
    font_size: ?FontSize = null,
    padding: ?Spacing = null,
    margin: ?Spacing = null,
    gap: ?Spacing = null,
    opacity: f32 = 1.0,
    
    // プリセット
    pub const primary: Style = .{ .bg = .{ .r = 0.3, .g = 0.6, .b = 0.9, .a = 1.0 }, .fg = .{ .r = 1, .g = 1, .b = 1, .a = 1 }, .radius = BorderRadius.base.radius() };
    pub const secondary: Style = .{ .bg = .{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 }, .fg = .{ .r = 1, .g = 1, .b = 1, .a = 1 }, .radius = BorderRadius.base.radius() };
    pub const ghost: Style = .{ .bg = .{ .r = 0, .g = 0, .b = 0, .a = 0 }, .fg = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1 } };
    pub const outlined: Style = .{ .bg = .{ .r = 0, .g = 0, .b = 0, .a = 0 }, .fg = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1 }, .border_color = .{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1 }, .border_width = 1, .radius = BorderRadius.base.radius() };
};

// ============================================================================
// イベントシステム
// ============================================================================

/// イベントハンドラ
pub const EventHandler = union(enum) {
    click: *const fn () void,
    change_float: *const fn (f32) void,
    change_bool: *const fn (bool) void,
    change_string: *const fn ([]const u8) void,
    none: void,
};

/// インタラクション状態
pub const InteractionState = enum {
    normal,
    hover,
    pressed,
    focused,
    disabled,
};

// ============================================================================
// UIコンポーネント
// ============================================================================

/// 方向
pub const Direction = enum { horizontal, vertical };

/// アライメント
pub const Align = enum { start, center, end_, stretch };

/// UIノード
pub const Node = struct {
    kind: NodeKind,
    props: Props,
    style: Style = .{},
    children: []const Node = &.{},
    event: EventHandler = .{ .none = {} },
    
    pub const NodeKind = enum {
        stack,
        text,
        button,
        input,
        checkbox,
        slider,
        progress,
        spacer,
        divider,
        image,
        scroll,
    };
    
    pub const Props = union(NodeKind) {
        stack: StackProps,
        text: TextProps,
        button: ButtonProps,
        input: InputProps,
        checkbox: CheckboxProps,
        slider: SliderProps,
        progress: ProgressProps,
        spacer: SpacerProps,
        divider: void,
        image: ImageProps,
        scroll: ScrollProps,
    };
    
    pub const StackProps = struct {
        direction: Direction = .vertical,
        alignment: Align = .start,
        justification: Align = .start,
        wrap: bool = false,
    };
    
    pub const TextProps = struct {
        content: []const u8,
        selectable: bool = false,
    };
    
    pub const ButtonProps = struct {
        label: []const u8,
        variant: ButtonVariant = .primary,
        size: ButtonSize = .medium,
    };
    
    pub const ButtonVariant = enum { primary, secondary, ghost, outlined, danger };
    pub const ButtonSize = enum { small, medium, large };
    
    pub const InputProps = struct {
        value: *[]const u8,
        placeholder: []const u8 = "",
        input_type: InputType = .text,
    };
    
    pub const InputType = enum { text, password, number, email };
    
    pub const CheckboxProps = struct {
        checked: *bool,
        label: []const u8 = "",
    };
    
    pub const SliderProps = struct {
        value: *f32,
        min: f32 = 0,
        max: f32 = 1,
        step: f32 = 0.01,
        label: []const u8 = "",
    };
    
    pub const ProgressProps = struct {
        value: f32,
        show_label: bool = false,
    };
    
    pub const SpacerProps = struct {
        size: f32 = 0,
        flex: f32 = 1,
    };
    
    pub const ImageProps = struct {
        data: []const u8,
        width: ?u32 = null,
        height: ?u32 = null,
    };
    
    pub const ScrollProps = struct {
        horizontal: bool = false,
        vertical: bool = true,
    };
};

// ============================================================================
// ビルダーAPI
// ============================================================================

/// スタックビルダー
pub const StackBuilder = struct {
    node: Node,
    
    pub fn direction(self: StackBuilder, dir: Direction) StackBuilder {
        var n = self; n.node.props.stack.direction = dir;
        return n;
    }
    
    pub fn alignment(self: StackBuilder, a: Align) StackBuilder {
        var n = self; n.node.props.stack.alignment = a;
        return n;
    }
    
    pub fn justify(self: StackBuilder, j: Align) StackBuilder {
        var n = self; n.node.props.stack.justify = j;
        return n;
    }
    
    pub fn wrap(self: StackBuilder, w: bool) StackBuilder {
        var n = self; n.node.props.stack.wrap = w;
        return n;
    }
    
    pub fn withSpacing(self: StackBuilder, s: Spacing) StackBuilder {
        var n = self; n.node.style.gap = s;
        return n;
    }
    
    pub fn withPadding(self: StackBuilder, p: Spacing) StackBuilder {
        var n = self; n.node.style.padding = p;
        return n;
    }
    
    pub fn bg(self: StackBuilder, color: zgf.Color) StackBuilder {
        var n = self; n.node.style.bg = color;
        return n;
    }
    
    pub fn radius(self: StackBuilder, r: BorderRadius) StackBuilder {
        var n = self; n.node.style.radius = r.radius();
        return n;
    }
    
    pub fn children(self: StackBuilder, c: []const Node) StackBuilder {
        var n = self; n.node.children = c;
        return n;
    }
    
    pub fn style(self: StackBuilder, s: Style) StackBuilder {
        var n = self; n.node.style = s;
        return n;
    }
    
    pub fn build(self: StackBuilder) Node {
        return self.node;
    }
};

/// テキストビルダー
pub const TextBuilder = struct {
    node: Node,
    
    pub fn font(self: TextBuilder, size: FontSize) TextBuilder {
        var n = self; n.node.style.font_size = size;
        return n;
    }
    
    pub fn color(self: TextBuilder, c: zgf.Color) TextBuilder {
        var n = self; n.node.style.fg = c;
        return n;
    }
    
    pub fn selectable(self: TextBuilder, s: bool) TextBuilder {
        var n = self; n.node.props.text.selectable = s;
        return n;
    }
    
    pub fn style(self: TextBuilder, s: Style) TextBuilder {
        var n = self; n.node.style = s;
        return n;
    }
    
    pub fn build(self: TextBuilder) Node {
        return self.node;
    }
};

/// ボタンビルダー
pub const ButtonBuilder = struct {
    node: Node,
    
    pub fn variant(self: ButtonBuilder, v: Node.ButtonVariant) ButtonBuilder {
        var n = self; n.node.props.button.variant = v;
        return n;
    }
    
    pub fn size(self: ButtonBuilder, s: Node.ButtonSize) ButtonBuilder {
        var n = self; n.node.props.button.size = s;
        return n;
    }
    
    pub fn onClick(self: ButtonBuilder, handler: *const fn () void) ButtonBuilder {
        var n = self; n.node.event = .{ .click = handler };
        return n;
    }
    
    pub fn disabled(self: ButtonBuilder, d: bool) ButtonBuilder {
        var n = self;
        if (d) {
            n.node.style.opacity = 0.5;
        }
        return n;
    }
    
    pub fn style(self: ButtonBuilder, s: Style) ButtonBuilder {
        var n = self; n.node.style = s;
        return n;
    }
    
    pub fn build(self: ButtonBuilder) Node {
        return self.node;
    }
};

// ============================================================================
// コンストラクタ関数
// ============================================================================

/// スタックコンテナを作成
pub fn stack(dir: Direction) StackBuilder {
    return .{
        .node = .{
            .kind = .stack,
            .props = .{ .stack = .{ .direction = dir } },
        },
    };
}

/// 垂直スタック
pub fn vstack() StackBuilder {
    return stack(.vertical);
}

/// 水平スタック
pub fn hstack() StackBuilder {
    return stack(.horizontal);
}

/// テキストを作成
pub fn text(content: []const u8) TextBuilder {
    return .{
        .node = .{
            .kind = .text,
            .props = .{ .text = .{ .content = content } },
        },
    };
}

/// ボタンを作成
pub fn button(label: []const u8) ButtonBuilder {
    return .{
        .node = .{
            .kind = .button,
            .props = .{ .button = .{ .label = label } },
        },
    };
}

/// チェックボックスを作成
pub fn checkbox(label: []const u8, checked: *bool) Node {
    return .{
        .kind = .checkbox,
        .props = .{ .checkbox = .{ .label = label, .checked = checked } },
    };
}

pub const SliderOpts = struct { min: f32 = 0, max: f32 = 1 };

/// スライダーを作成
pub fn slider(label: []const u8, value: *f32, opts: SliderOpts) Node {
    return .{
        .kind = .slider,
        .props = .{ .slider = .{ .value = value, .min = opts.min, .max = opts.max, .label = label } },
    };
}

/// プログレスバーを作成
pub fn progress(value: f32) Node {
    return .{
        .kind = .progress,
        .props = .{ .progress = .{ .value = value } },
    };
}

/// スペーサーを作成
pub fn spacer(size: f32) Node {
    return .{
        .kind = .spacer,
        .props = .{ .spacer = .{ .size = size } },
    };
}

/// 区切り線を作成
pub fn divider() Node {
    return .{
        .kind = .divider,
        .props = .{ .divider = {} },
    };
}

// ============================================================================
// テーマシステム
// ============================================================================

pub const Theme = enum {
    light,
    dark,
    
    pub fn palette(self: Theme) Palette {
        return switch (self) {
            .light => Palette.light,
            .dark => .{},
        };
    }
};

// ============================================================================
// レンダラー
// ============================================================================

/// UIをフレームバッファにレンダリング
pub fn render(fb: *zgf.Image, root: Node, theme: Theme) void {
    const p = theme.palette();
    
    // 背景を描画
    fb.fill(p.bg);
    
    // 再帰的にレンダリング
    renderNode(fb, root, .{ .x = 0, .y = 0, .w = @floatFromInt(fb.width()), .h = @floatFromInt(fb.height()) }, p);
}

const RenderRect = struct { x: f32, y: f32, w: f32, h: f32 };

fn renderNode(fb: *zgf.Image, node: Node, rect: RenderRect, palette: Palette) void {
    const style = node.style;
    const bg: zgf.Color = if (style.bg) |b| b else .{ .r = 0, .g = 0, .b = 0, .a = 0 };
    const fg: zgf.Color = if (style.fg) |f| f else palette.text;
    const radius: f32 = if (style.radius) |r| r else 0;
    
    switch (node.kind) {
        .stack => {
            // 背景を描画
            if (bg.a > 0) {
                drawRoundedRect(fb, rect, bg, radius);
            }
            
            // 子要素をレンダリング
            const padding_val = if (style.padding) |p| p.value() else 0;
            const gap_val = if (style.gap) |g| g.value() else 8;
            
            const inner = RenderRect{
                .x = rect.x + padding_val,
                .y = rect.y + padding_val,
                .w = rect.w - padding_val * 2,
                .h = rect.h - padding_val * 2,
            };
            
            const dir = node.props.stack.direction;
            const child_count = node.children.len;
            if (child_count == 0) return;
            
            const total_gap = gap_val * @as(f32, @floatFromInt(child_count - 1));
            const child_size = if (dir == .vertical)
                (inner.h - total_gap) / @as(f32, @floatFromInt(child_count))
            else
                (inner.w - total_gap) / @as(f32, @floatFromInt(child_count));
            
            var offset: f32 = 0;
            for (node.children) |child| {
                const child_rect = if (dir == .vertical)
                    RenderRect{ .x = inner.x, .y = inner.y + offset, .w = inner.w, .h = child_size }
                else
                    RenderRect{ .x = inner.x + offset, .y = inner.y, .w = child_size, .h = inner.h };
                
                renderNode(fb, child, child_rect, palette);
                offset += child_size + gap_val;
            }
        },
        .text => {
            const font = vector_text.SimpleFont.withScale((if (style.font_size) |f| f.size() else FontSize.base.size()));
            const content = node.props.text.content;
            font.drawText(fb, content, rect.x + 4, rect.y + 4, fg);
        },
        .button => {
            const btn_bg: zgf.Color = switch (node.props.button.variant) {
                .primary => palette.primary,
                .secondary => palette.secondary,
                .ghost => .{ .r = 0, .g = 0, .b = 0, .a = 0 },
                .outlined => .{ .r = 0, .g = 0, .b = 0, .a = 0 },
                .danger => palette.danger,
            };
            const btn_fg: zgf.Color = switch (node.props.button.variant) {
                .primary, .danger => .{ .r = 1, .g = 1, .b = 1, .a = 1 },
                else => fg,
            };
            
            drawRoundedRect(fb, rect, btn_bg, radius);
            
            if (node.props.button.variant == .outlined) {
                // 枠線を描画
                drawRoundedRectOutline(fb, rect, palette.border, radius, 1);
            }
            
            // ラベルを描画
            const font = vector_text.SimpleFont.withScale(FontSize.base.size());
            font.drawText(fb, node.props.button.label, rect.x + 8, rect.y + rect.h / 2 - 6, btn_fg);
        },
        .checkbox => {
            const checked = node.props.checkbox.checked.*;
            const box_size: f32 = 16;
            const box_rect = RenderRect{ .x = rect.x + 4, .y = rect.y + rect.h / 2 - box_size / 2, .w = box_size, .h = box_size };
            
            drawRoundedRect(fb, box_rect, if (checked) palette.primary else palette.surface, 2);
            
            if (checked) {
                // チェックマーク
                vector.drawLineFast(fb, box_rect.x + 3, box_rect.y + 8, box_rect.x + 7, box_rect.y + 12, .{ .r = 1, .g = 1, .b = 1 });
                vector.drawLineFast(fb, box_rect.x + 7, box_rect.y + 12, box_rect.x + 13, box_rect.y + 4, .{ .r = 1, .g = 1, .b = 1 });
            }
            
            // ラベル
            const font = vector_text.SimpleFont.withScale(FontSize.base.size());
            font.drawText(fb, node.props.checkbox.label, rect.x + 24, rect.y + rect.h / 2 - 6, fg);
        },
        .slider => {
            const val = node.props.slider.value.*;
            const min_val = node.props.slider.min;
            const max_val = node.props.slider.max;
            const ratio = (val - min_val) / (max_val - min_val);
            
            const track_rect = RenderRect{ .x = rect.x + 4, .y = rect.y + rect.h / 2 - 2, .w = rect.w - 8, .h = 4 };
            const fill_width = track_rect.w * ratio;
            const fill_rect = RenderRect{ .x = track_rect.x, .y = track_rect.y, .w = fill_width, .h = track_rect.h };
            
            drawRoundedRect(fb, track_rect, palette.surface, 2);
            drawRoundedRect(fb, fill_rect, palette.primary, 2);
            
            // ハンドル
            const handle_x = track_rect.x + fill_width;
            vector.fillCircleOptimized(fb, handle_x, rect.y + rect.h / 2, 8, palette.primary);
            
            // ラベル
            if (node.props.slider.label.len > 0) {
                const font = vector_text.SimpleFont.withScale(FontSize.sm.size());
                font.drawText(fb, node.props.slider.label, rect.x + 4, rect.y + 2, palette.text_muted);
            }
        },
        .progress => {
            const val = node.props.progress.value;
            const track_rect = RenderRect{ .x = rect.x + 4, .y = rect.y + rect.h / 2 - 4, .w = rect.w - 8, .h = 8 };
            const fill_width = track_rect.w * val;
            const fill_rect = RenderRect{ .x = track_rect.x, .y = track_rect.y, .w = fill_width, .h = track_rect.h };
            
            drawRoundedRect(fb, track_rect, palette.surface, 4);
            drawRoundedRect(fb, fill_rect, palette.primary, 4);
        },
        .spacer => {},
        .divider => {
            const y = rect.y + rect.h / 2;
            vector.drawLineFast(fb, rect.x + 8, y, rect.x + rect.w - 8, y, palette.border);
        },
        .image => {},
        .scroll => {},
        .input => {},
    }
}

fn drawRoundedRect(fb: *zgf.Image, rect: RenderRect, color: zgf.Color, radius: f32) void {
    if (radius <= 0) {
        // 角丸なし
        vector.fillRectOptimized(
            fb,
            @intFromFloat(rect.x),
            @intFromFloat(rect.y),
            @intFromFloat(rect.x + rect.w),
            @intFromFloat(rect.y + rect.h),
            color,
        );
    } else {
        // 角丸付き (簡易実装)
        vector.fillRectOptimized(fb, @intFromFloat(rect.x + radius), @intFromFloat(rect.y), @intFromFloat(rect.x + rect.w - radius), @intFromFloat(rect.y + rect.h), color);
        vector.fillRectOptimized(fb, @intFromFloat(rect.x), @intFromFloat(rect.y + radius), @intFromFloat(rect.x + rect.w), @intFromFloat(rect.y + rect.h - radius), color);
        vector.fillCircleOptimized(fb, rect.x + radius, rect.y + radius, radius, color);
        vector.fillCircleOptimized(fb, rect.x + rect.w - radius, rect.y + radius, radius, color);
        vector.fillCircleOptimized(fb, rect.x + radius, rect.y + rect.h - radius, radius, color);
        vector.fillCircleOptimized(fb, rect.x + rect.w - radius, rect.y + rect.h - radius, radius, color);
    }
}

fn drawRoundedRectOutline(fb: *zgf.Image, rect: RenderRect, color: zgf.Color, radius: f32, thickness: f32) void {
    _ = radius;
    _ = thickness;
    vector.drawLineFast(fb, rect.x, rect.y, rect.x + rect.w, rect.y, color);
    vector.drawLineFast(fb, rect.x, rect.y + rect.h, rect.x + rect.w, rect.y + rect.h, color);
    vector.drawLineFast(fb, rect.x, rect.y, rect.x, rect.y + rect.h, color);
    vector.drawLineFast(fb, rect.x + rect.w, rect.y, rect.x + rect.w, rect.y + rect.h, color);
}

// ============================================================================
// テスト
// ============================================================================

test "ui - stack builder" {
    const node = vstack()
        .withSpacing(.base)
        .withPadding(.lg)
        .children(&.{
            text("Hello").build(),
            text("World").build(),
        })
        .build();
    
    try std.testing.expectEqual(Node.NodeKind.stack, node.kind);
    try std.testing.expectEqual(@as(usize, 2), node.children.len);
}

test "ui - button builder" {
    const node = button("Click me")
        .variant(.primary)
        .size(.medium)
        .build();
    
    try std.testing.expectEqual(Node.NodeKind.button, node.kind);
    try std.testing.expectEqualStrings("Click me", node.props.button.label);
}

test "ui - text builder" {
    const node = text("Hello")
        .font(.lg)
        .color(.{ .r = 1, .g = 0, .b = 0 })
        .build();
    
    try std.testing.expectEqual(Node.NodeKind.text, node.kind);
    try std.testing.expectEqualStrings("Hello", node.props.text.content);
}

test "ui - theme palette" {
    const dark = Theme.dark.palette();
    const light = Theme.light.palette();
    
    try std.testing.expect(dark.bg.r < light.bg.r);
}
