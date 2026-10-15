//! イベント処理システム
//!
//! ウィンドウイベント、マウスイベント、キーボードイベントを統一的に扱う。
//!
//! ```zig
//! while (window.isOpen()) {
//!     while (events.poll()) |event| {
//!         switch (event) {
//!             .key_press => |key| handleKey(key),
//!             .mouse_move => |pos| handleMouse(pos),
//!             .resize => |size| handleResize(size),
//!             else => {},
//!         }
//!     }
//!     // 描画...
//! }
//! ```

const std = @import("std");
const zui = @import("zui");

/// イベントの種類
pub const Event = union(enum) {
    // ウィンドウイベント
    close: void,
    resize: struct { width: u32, height: u32 },
    focus: bool, // true = focused, false = unfocused
    
    // キーボードイベント
    key_press: KeyEvent,
    key_release: KeyEvent,
    text_input: []const u8,
    
    // マウスイベント
    mouse_move: struct { x: f32, y: f32 },
    mouse_button_press: MouseButtonEvent,
    mouse_button_release: MouseButtonEvent,
    mouse_scroll: struct { dx: f32, dy: f32 },
    
    // タッチイベント
    touch_start: TouchEvent,
    touch_move: TouchEvent,
    touch_end: TouchEvent,
    
    // その他
    none: void,
};

pub const KeyEvent = struct {
    key: Key,
    modifiers: Modifiers,
};

pub const MouseButtonEvent = struct {
    button: MouseButton,
    x: f32,
    y: f32,
    modifiers: Modifiers,
};

pub const TouchEvent = struct {
    id: u32,
    x: f32,
    y: f32,
};

pub const Key = enum(u32) {
    // 文字キー
    a, b, c, d, e, f, g, h, i, j, k, l, m,
    n, o, p, q, r, s, t, u, v, w, x, y, z,
    
    // 数字キー
    num0, num1, num2, num3, num4, num5, num6, num7, num8, num9,
    
    // 関数キー
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
    
    // 特殊キー
    escape, tab, caps_lock,
    left_shift, right_shift,
    left_control, right_control,
    left_alt, right_alt,
    left_super, right_super, // Windows/Cmd key
    space, enter, backspace, delete,
    
    // 矢印キー
    left, right, up, down,
    
    // ホーム/エンド
    home, end, page_up, page_down,
    
    // その他
    insert, print_screen, num_lock, scroll_lock, pause,
    
    // テンキー
    kp0, kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9,
    kp_add, kp_subtract, kp_multiply, kp_divide, kp_decimal, kp_enter,
    
    // 記号キー
    semicolon, comma, period, slash, backslash,
    left_bracket, right_bracket, apostrophe, grave_accent,
    minus, equal,
    
    unknown,
};

pub const MouseButton = enum(u8) {
    left,
    right,
    middle,
    button4,
    button5,
};

pub const Modifiers = packed struct {
    shift: bool = false,
    control: bool = false,
    alt: bool = false,
    super: bool = false,
    caps_lock: bool = false,
    num_lock: bool = false,
    _padding: u2 = 0,
};

/// イベントキュー
pub const EventQueue = struct {
    events: std.ArrayList(Event),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) EventQueue {
        return .{
            .events = std.ArrayList(Event).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *EventQueue) void {
        self.events.deinit();
    }
    
    pub fn push(self: *EventQueue, event: Event) !void {
        try self.events.append(event);
    }
    
    pub fn poll(self: *EventQueue) ?Event {
        if (self.events.items.len == 0) return null;
        return self.events.orderedRemove(0);
    }
    
    pub fn clear(self: *EventQueue) void {
        self.events.clearRetainingCapacity();
    }
    
    pub fn hasEvents(self: *const EventQueue) bool {
        return self.events.items.len > 0;
    }
};

/// ZUIの入力に変換
pub fn toZuiInput(queue: *EventQueue) zui.Input {
    var input = zui.Input{};
    
    // TODO: イベントキューからZUIの入力状態を構築
    // - マウス位置
    // - マウスボタン状態
    // - キーボード状態
    // - スクロール
    
    _ = queue;
    return input;
}

// テスト
test "EventQueue basic" {
    var queue = EventQueue.init(std.testing.allocator);
    defer queue.deinit();
    
    try std.testing.expect(!queue.hasEvents());
    try std.testing.expect(queue.poll() == null);
    
    try queue.push(.{ .close = {} });
    try std.testing.expect(queue.hasEvents());
    
    const event = queue.poll();
    try std.testing.expect(event != null);
    try std.testing.expect(event.? == .close);
    
    try std.testing.expect(!queue.hasEvents());
}

test "Modifiers" {
    const mods = Modifiers{ .shift = true, .control = true };
    try std.testing.expect(mods.shift);
    try std.testing.expect(mods.control);
    try std.testing.expect(!mods.alt);
}
