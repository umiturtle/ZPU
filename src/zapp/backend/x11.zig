//! Linux/X11 Backend for ZAPP (stub - no @cImport)
const std = @import("std");
const builtin = @import("builtin");
const zgf = @import("zgf");
const zpu = @import("zpu");

pub const Config = struct { title: []const u8, width: u32, height: u32, resizable: bool };

pub const X11Window = struct {
    framebuffer: zgf.Image,
    config: Config,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: Config) !X11Window {
        const framebuffer = try zgf.Image.init(allocator, config.width, config.height, 4);
        return .{ .framebuffer = framebuffer, .config = config, .allocator = allocator };
    }
    pub fn deinit(self: *X11Window) void { self.framebuffer.deinit(); }
    pub fn isOpen(_: *const X11Window) bool { return true; }
    pub fn pollEvents(_: *X11Window) void {}
    pub fn present(_: *X11Window) void {}
    pub fn getSize(self: *const X11Window) struct { width: u32, height: u32 } {
        return .{ .width = self.config.width, .height = self.config.height };
    }
    pub fn setTitle(_: *X11Window, _: []const u8) void {}
    pub fn getFramebuffer(self: *X11Window) *zgf.Image { return &self.framebuffer; }
    pub fn resize(self: *X11Window, width: u32, height: u32) !void {
        if (width < 100 or width > 3840) return error.InvalidWidth;
        if (height < 100 or height > 2160) return error.InvalidHeight;
        self.framebuffer.deinit();
        self.framebuffer = try zgf.Image.init(self.allocator, width, height, 4);
        self.config.width = width;
        self.config.height = height;
    }
};

test "X11 Window basic" {
    var w = try X11Window.init(std.testing.allocator, .{ .title = "T", .width = 400, .height = 300, .resizable = true });
    defer w.deinit();
    try std.testing.expect(w.isOpen());
    try std.testing.expectEqual(@as(u32, 400), w.getSize().width);
}
