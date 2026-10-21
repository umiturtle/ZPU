//! Tests for ui.zig

const std = @import("std");
const ui_mod = @import("ui.zig");
const Style = @import("style.zig").Style;
const Ui = ui_mod.Ui;

test "style size resolution" {
    const s = Style{ .w = .{ .px = 40 }, .h = .full };
    try std.testing.expectEqual(@as(f32, 40), s.sizeAlong(.x).resolve(10, 100));
    try std.testing.expectEqual(@as(f32, 100), s.sizeAlong(.y).resolve(10, 100));
    try std.testing.expectEqual(@as(f32, 10), (Style{}).sizeAlong(.x).resolve(10, 100));
}

test "ids are scoped by pushId" {
    var ui = Ui.init(std.testing.allocator, .{});
    defer ui.deinit();
    const a = ui.idOf("ok");
    ui.pushId("panel");
    const b = ui.idOf("ok");
    ui.pushId("inner");
    const c = ui.idOf("ok");
    ui.popId();
    const d = ui.idOf("ok");
    try std.testing.expect(!a.eql(b));
    try std.testing.expect(!b.eql(c));
    try std.testing.expect(b.eql(d));
}
