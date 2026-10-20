//! ZAPP Input Handling
//!
//! キーボード、マウス、タッチ入力を処理するモジュール。

const std = @import("std");

/// キーコード
pub const KeyCode = enum(u16) {
    // 文字キー
    a = 0x0061,
    b = 0x0062,
    c = 0x0063,
    d = 0x0064,
    e = 0x0065,
    f = 0x0066,
    g = 0x0067,
    h = 0x0068,
    i = 0x0069,
    j = 0x006A,
    k = 0x006B,
    l = 0x006C,
    m = 0x006D,
    n = 0x006E,
    o = 0x006F,
    p = 0x0070,
    q = 0x0071,
    r = 0x0072,
    s = 0x0073,
    t = 0x0074,
    u = 0x0075,
    v = 0x0076,
    w = 0x0077,
    x = 0x0078,
    y = 0x0079,
    z = 0x007A,
    
    // 数字キー
    @"0" = 0x0030,
    @"1" = 0x0031,
    @"2" = 0x0032,
    @"3" = 0x0033,
    @"4" = 0x0034,
    @"5" = 0x0035,
    @"6" = 0x0036,
    @"7" = 0x0037,
    @"8" = 0x0038,
    @"9" = 0x0039,
    
    // 特殊キー
    space = 0x0020,
    enter = 0x000D,
    escape = 0x001B,
    tab = 0x0009,
    backspace = 0x0008,
    delete = 0x007F,
    
    // 矢印キー
    up = 0xFF52,
    down = 0xFF54,
    left = 0xFF51,
    right = 0xFF53,
    
    // 修飾キー
    shift = 0xFFE1,
    control = 0xFFE3,
    alt = 0xFFE9,
    
    // ファンクションキー
    f1 = 0xFFBE,
    f2 = 0xFFBF,
    f3 = 0xFFC0,
    f4 = 0xFFC1,
    f5 = 0xFFC2,
    f6 = 0xFFC3,
    f7 = 0xFFC4,
    f8 = 0xFFC5,
    f9 = 0xFFC6,
    f10 = 0xFFC7,
    f11 = 0xFFC8,
    f12 = 0xFFC9,
    
    _,
};

/// キーイベント
pub const KeyEvent = struct {
    key: KeyCode,
    pressed: bool,
    modifiers: ModifierState,
};

/// 修飾キー状態
pub const ModifierState = struct {
    shift: bool = false,
    control: bool = false,
    alt: bool = false,
};

/// マウスボタン
pub const MouseButton = enum(u8) {
    left = 0,
    right = 1,
    middle = 2,
};

/// マウスイベント
pub const MouseEvent = struct {
    x: f32,
    y: f32,
    button: ?MouseButton,
    pressed: bool,
};

/// タッチイベント
pub const TouchEvent = struct {
    id: usize,
    x: f32,
    y: f32,
    phase: TouchPhase,
};

/// タッチフェーズ
pub const TouchPhase = enum {
    began,
    moved,
    ended,
    cancelled,
};

/// 入力状態
pub const InputState = struct {
    keys: std.AutoHashMap(KeyCode, bool),
    mouse_x: f32,
    mouse_y: f32,
    mouse_buttons: [3]bool,
    
    pub fn init(allocator: std.mem.Allocator) InputState {
        return .{
            .keys = std.AutoHashMap(KeyCode, bool).init(allocator),
            .mouse_x = 0,
            .mouse_y = 0,
            .mouse_buttons = .{ false, false, false },
        };
    }
    
    pub fn deinit(self: *InputState) void {
        self.keys.deinit();
    }
    
    pub fn handleKeyEvent(self: *InputState, event: KeyEvent) void {
        self.keys.put(event.key, event.pressed) catch {};
    }
    
    pub fn handleMouseEvent(self: *InputState, event: MouseEvent) void {
        self.mouse_x = event.x;
        self.mouse_y = event.y;
        if (event.button) |btn| {
            self.mouse_buttons[@intFromEnum(btn)] = event.pressed;
        }
    }
    
    pub fn isKeyPressed(self: *InputState, key: KeyCode) bool {
        return self.keys.get(key) orelse false;
    }
};

// テスト
test "input - key event" {
    const allocator = std.testing.allocator;
    var state = InputState.init(allocator);
    defer state.deinit();
    
    state.handleKeyEvent(.{
        .key = .a,
        .pressed = true,
        .modifiers = .{},
    });
    
    try std.testing.expect(state.isKeyPressed(.a));
}

test "input - mouse event" {
    const allocator = std.testing.allocator;
    var state = InputState.init(allocator);
    defer state.deinit();
    
    state.handleMouseEvent(.{
        .x = 100,
        .y = 200,
        .button = .left,
        .pressed = true,
    });
    
    try std.testing.expectEqual(@as(f32, 100), state.mouse_x);
    try std.testing.expectEqual(@as(f32, 200), state.mouse_y);
    try std.testing.expect(state.mouse_buttons[0]);
}
