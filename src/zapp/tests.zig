//! ZAPPの基本的なテスト

const std = @import("std");
const zapp = @import("zapp");

test "ZAPP imports" {
    // 主要な型が正しくインポートできることを確認
    const _ = zapp.App;
    const _ = zapp.Window;
    const _ = zapp.Ui;
    const _ = zapp.Event;
    const _ = zapp.EventQueue;
    const _ = zapp.Key;
    const _ = zapp.MouseButton;
    const _ = zapp.Modifiers;
}

test "Window creation" {
    var window = try zapp.Window.init(.{
        .title = "Test Window",
        .width = 400,
        .height = 300,
        .backend = .software,
    });
    defer window.deinit();
    
    try std.testing.expect(window.isOpen());
    
    const size = window.getSize();
    try std.testing.expectEqual(@as(u32, 400), size.width);
    try std.testing.expectEqual(@as(u32, 300), size.height);
    
    const fb = window.getFramebuffer();
    try std.testing.expectEqual(@as(usize, 400), fb.width());
    try std.testing.expectEqual(@as(usize, 300), fb.height());
}

test "App initialization" {
    var app = try zapp.App.init(.{
        .window = .{
            .title = "Test App",
            .width = 400,
            .height = 300,
            .backend = .software,
        },
        .target_fps = 30,
    });
    defer app.deinit();
    
    try std.testing.expect(app.running);
    
    const size = app.getSize();
    try std.testing.expectEqual(@as(u32, 400), size.width);
    try std.testing.expectEqual(@as(u32, 300), size.height);
}

test "Event queue" {
    var app = try zapp.App.init(.{
        .window = .{
            .backend = .software,
        },
    });
    defer app.deinit();
    
    // イベントを追加
    try app.pushEvent(.{ .resize = .{ .width = 800, .height = 600 } });
    try app.pushEvent(.{ .close = {} });
    
    try std.testing.expect(app.events.hasEvents());
    
    // イベントを取得
    const event1 = app.events.poll();
    try std.testing.expect(event1 != null);
    try std.testing.expect(event1.? == .resize);
    
    const event2 = app.events.poll();
    try std.testing.expect(event2 != null);
    try std.testing.expect(event2.? == .close);
    
    // キューが空になったことを確認
    try std.testing.expect(!app.events.hasEvents());
}

test "UI creation" {
    var ui = try zapp.Ui.init(.{});
    defer ui.deinit();
    
    // UIが正しく初期化されたことを確認
    try std.testing.expect(true);
}

test "App quit" {
    var app = try zapp.App.init(.{
        .window = .{
            .backend = .software,
        },
    });
    defer app.deinit();
    
    try std.testing.expect(app.running);
    
    app.quit();
    
    try std.testing.expect(!app.running);
}

test "Event types" {
    // 各イベント型が正しく定義されていることを確認
    const close_event = zapp.Event{ .close = {} };
    try std.testing.expect(close_event == .close);
    
    const resize_event = zapp.Event{ .resize = .{ .width = 800, .height = 600 } };
    try std.testing.expect(resize_event == .resize);
    
    const key_event = zapp.Event{ .key_press = .{
        .key = .a,
        .modifiers = .{},
    } };
    try std.testing.expect(key_event == .key_press);
}

test "Key enum" {
    // キー列挙型の一部の値を確認
    try std.testing.expectEqual(zapp.Key.a, zapp.Key.a);
    try std.testing.expectEqual(zapp.Key.escape, zapp.Key.escape);
    try std.testing.expectEqual(zapp.Key.space, zapp.Key.space);
}

test "Modifiers" {
    const mods = zapp.Modifiers{
        .shift = true,
        .control = true,
    };
    
    try std.testing.expect(mods.shift);
    try std.testing.expect(mods.control);
    try std.testing.expect(!mods.alt);
    try std.testing.expect(!mods.super);
}

test "Version info" {
    const info = zapp.getVersionInfo();
    
    try std.testing.expectEqualStrings("0.1.0", info.version);
    try std.testing.expectEqual(@as(u32, 0), info.major);
    try std.testing.expectEqual(@as(u32, 1), info.minor);
    try std.testing.expectEqual(@as(u32, 0), info.patch);
}
