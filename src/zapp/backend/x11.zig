//! Linux/X11 Backend for ZAPP
//! 
//! Xlibを使用してネイティブウィンドウを管理する。
//! 外部依存なしで動作する。

const std = @import("std");
const builtin = @import("builtin");
const zgf = @import("zgf");
const zpu = @import("zpu");
const c = @cImport({
    @cInclude("X11/Xlib.h");
    @cInclude("X11/Xutil.h");
    @cInclude("X11/keysym.h");
});

pub const X11Window = struct {
    display: *c.Display,
    window: c.Window,
    gc: c.GC,
    ximage: ?*c.XImage,
    framebuffer: zgf.Image,
    config: Config,
    should_close: bool,
    allocator: std.mem.Allocator,
    
    const Config = struct {
        title: []const u8,
        width: u32,
        height: u32,
        resizable: bool,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: Config) !X11Window {
        // X11ディスプレイを開く
        const display = c.XOpenDisplay(null) orelse {
            return error.DisplayNotFound;
        };
        
        const screen = c.DefaultScreen(display);
        const root = c.RootWindow(display, screen);
        
        // ウィンドウを作成
        const window = c.XCreateSimpleWindow(
            display,
            root,
            0, 0,
            @intCast(config.width),
            @intCast(config.height),
            0,
            c.BlackPixel(display, screen),
            c.WhitePixel(display, screen),
        );
        
        // ウィンドウタイトルを設定
        _ = c.XStoreName(display, window, @ptrCast(config.title.ptr));
        
        // ウィンドウサイズ変更を許可/禁止
        if (!config.resizable) {
            var size_hints = c.XAllocSizeHints();
            if (size_hints) |hints| {
                hints.flags = c.PMinSize | c.PMaxSize;
                hints.min_width = @intCast(config.width);
                hints.max_width = @intCast(config.width);
                hints.min_height = @intCast(config.height);
                hints.max_height = @intCast(config.height);
                _ = c.XSetWMNormalHints(display, window, hints);
                c.XFree(hints);
            }
        }
        
        // ウィンドウを表示
        _ = c.XMapWindow(display, window);
        _ = c.XFlush(display);
        
        // Graphics Contextを作成
        const gc = c.XCreateGC(display, window, 0, null);
        
        // XImageを作成（ピクセルデータ用）
        const ximage = c.XCreateImage(
            display,
            c.DefaultVisual(display, screen),
            @intCast(c.DefaultDepth(display, screen)),
            c.ZPixmap,
            0,
            null,
            @intCast(config.width),
            @intCast(config.height),
            32,
            0,
        );
        
        // フレームバッファを作成
        var framebuffer = try zgf.Image.init(allocator, config.width, config.height, 4);
        
        // XImageのデータをフレームバッファに割り当て
        if (ximage) |img| {
            img.data = @ptrCast(framebuffer.pixels.flatSlice(f32).?);
        }
        
        // ウィンドウ_CLOSEイベントを有効化
        const wm_delete_window = c.XInternAtom(display, "WM_DELETE_WINDOW", c.False);
        _ = c.XSetWMProtocols(display, window, @ptrCast(&wm_delete_window), 1);
        
        return X11Window{
            .display = display,
            .window = window,
            .gc = gc,
            .ximage = ximage,
            .framebuffer = framebuffer,
            .config = config,
            .should_close = false,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *X11Window) void {
        if (self.ximage) |img| {
            img.data = null;
            _ = c.XDestroyImage(img);
        }
        _ = c.XFreeGC(self.display, self.gc);
        _ = c.XDestroyWindow(self.display, self.window);
        _ = c.XCloseDisplay(self.display);
        self.framebuffer.deinit();
    }
    
    pub fn isOpen(self: *const X11Window) bool {
        return !self.should_close;
    }
    
    pub fn pollEvents(self: *X11Window) void {
        while (c.XPending(self.display) > 0) {
            var event: c.XEvent = undefined;
            _ = c.XNextEvent(self.display, &event);
            
            switch (event.type) {
                c.ClientMessage => {
                    // ウィンドウを閉じるリクエスト
                    self.should_close = true;
                },
                c.KeyPress => {
                    // キー入力処理
                    const ke: *c.XKeyEvent = @ptrCast(&event);
                    var keysym: c.KeySym = undefined;
                    _ = c.XLookupString(@ptrCast(ke), null, 0, &keysym, null);
                    
                    // ESCキーで閉じる
                    if (keysym == c.XK_Escape) {
                        self.should_close = true;
                    }
                },
                c.ConfigureNotify => {
                    // ウィンドウリサイズ処理
                    const ce: *c.XConfigureEvent = @ptrCast(&event);
                    if (ce.width != self.config.width or ce.height != self.config.height) {
                        self.config.width = @intCast(ce.width);
                        self.config.height = @intCast(ce.height);
                        // TODO: フレームバッファをリサイズ
                    }
                },
                else => {},
            }
        }
    }
    
    pub fn present(self: *X11Window) void {
        // フレームバッファをウィンドウに描画
        if (self.ximage) |img| {
            _ = c.XPutImage(
                self.display,
                self.window,
                self.gc,
                img,
                0, 0,
                0, 0,
                @intCast(self.config.width),
                @intCast(self.config.height),
            );
            _ = c.XFlush(self.display);
        }
    }
    
    pub fn getSize(self: *const X11Window) struct { width: u32, height: u32 } {
        return .{
            .width = self.config.width,
            .height = self.config.height,
        };
    }
    
    pub fn setTitle(self: *X11Window, title: []const u8) void {
        _ = c.XStoreName(self.display, self.window, @ptrCast(title.ptr));
        _ = c.XFlush(self.display);
    }
    
    pub fn getFramebuffer(self: *X11Window) *zgf.Image {
        return &self.framebuffer;
    }
};

test "X11 Window basic" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    
    var window = try X11Window.init(std.testing.allocator, .{
        .title = "Test Window",
        .width = 400,
        .height = 300,
        .resizable = true,
    });
    defer window.deinit();
    
    try std.testing.expect(window.isOpen());
    const size = window.getSize();
    try std.testing.expectEqual(@as(u32, 400), size.width);
    try std.testing.expectEqual(@as(u32, 300), size.height);
}
