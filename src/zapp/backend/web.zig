//! Web Backend for ZAPP
//! 
//! WebAssembly環境でCanvas APIを使用して描画する。

const std = @import("std");
const builtin = @import("builtin");
const zgf = @import("zgf");
const zpu = @import("zpu");

// WebAssembly用のJavaScript関数をインポート
const js = if (builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64) 
    struct {
        extern fn canvas_init(width: i32, height: i32) void;
        extern fn canvas_clear(r: f32, g: f32, b: f32, a: f32) void;
        extern fn canvas_put_image(data: [*]const u8, len: i32, x: i32, y: i32, width: i32, height: i32) void;
        extern fn canvas_set_title(title: [*]const u8, len: i32) void;
        extern fn window_should_close() i32;
        extern fn get_timestamp() f64;
    }
else
    struct {
        pub fn canvas_init(_: i32, _: i32) void {}
        pub fn canvas_clear(_: f32, _: f32, _: f32, _: f32) void {}
        pub fn canvas_put_image(_: [*]const u8, _: i32, _: i32, _: i32, _: i32, _: i32) void {}
        pub fn canvas_set_title(_: [*]const u8, _: i32) void {}
        pub fn window_should_close() i32 { return 0; }
        pub fn get_timestamp() f64 { return 0.0; }
    };

pub const WebWindow = struct {
    framebuffer: zgf.Image,
    config: Config,
    should_close: bool,
    allocator: std.mem.Allocator,
    
    const Config = struct {
        title: []const u8,
        width: u32,
        height: u32,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: Config) !WebWindow {
        // Canvasを初期化
        js.canvas_init(@intCast(config.width), @intCast(config.height));
        
        // タイトルを設定
        js.canvas_set_title(config.title.ptr, @intCast(config.title.len));
        
        // フレームバッファを作成
        var framebuffer = try zgf.Image.init(allocator, config.width, config.height, 4);
        
        return WebWindow{
            .framebuffer = framebuffer,
            .config = config,
            .should_close = false,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *WebWindow) void {
        self.framebuffer.deinit();
    }
    
    pub fn isOpen(self: *const WebWindow) bool {
        return !self.should_close and js.window_should_close() == 0;
    }
    
    pub fn pollEvents(self: *WebWindow) void {
        // Web環境ではブラウザがイベントを処理するため、ここでは何もしない
        _ = self;
    }
    
    pub fn present(self: *WebWindow) void {
        // フレームバッファをCanvasに転送
        const pixels = self.framebuffer.pixels.flatSlice(f32).?;
        const len = pixels.len;
        
        // f32 RGBAデータをu8 RGBAデータに変換
        var buffer: [4096 * 4]u8 = undefined;
        const buffer_len = @min(len, buffer.len);
        
        var i: usize = 0;
        while (i < buffer_len) : (i += 1) {
            const value = pixels[i];
            buffer[i] = @intFromFloat(std.math.clamp(value * 255.0, 0.0, 255.0));
        }
        
        js.canvas_put_image(&buffer, @intCast(buffer_len), 0, 0, @intCast(self.config.width), @intCast(self.config.height));
    }
    
    pub fn getSize(self: *const WebWindow) struct { width: u32, height: u32 } {
        return .{
            .width = self.config.width,
            .height = self.config.height,
        };
    }
    
    pub fn setTitle(self: *WebWindow, title: []const u8) void {
        js.canvas_set_title(title.ptr, @intCast(title.len));
    }
    
    pub fn getFramebuffer(self: *WebWindow) *zgf.Image {
        return &self.framebuffer;
    }
};

// WebAssembly用のエントリポイント
pub export fn main() void {
    if (builtin.target.cpu.arch != .wasm32 and builtin.target.cpu.arch != .wasm64) return;
    
    const allocator = std.heap.wasm_allocator;
    
    var window = WebWindow.init(allocator, .{
        .title = "ZAPP Web Application",
        .width = 800,
        .height = 600,
    }) catch return;
    defer window.deinit();
    
    // メインループ
    while (window.isOpen()) {
        window.pollEvents();
        
        // 描画処理
        const fb = window.getFramebuffer();
        _ = fb;
        // ここでZUIのUIを描画
        
        window.present();
    }
}

test "Web Window basic" {
    if (builtin.target.cpu.arch != .wasm32 and builtin.target.cpu.arch != .wasm64) return error.SkipZigTest;
    
    var window = try WebWindow.init(std.testing.allocator, .{
        .title = "Test Window",
        .width = 400,
        .height = 300,
    });
    defer window.deinit();
    
    try std.testing.expect(window.isOpen());
    const size = window.getSize();
    try std.testing.expectEqual(@as(u32, 400), size.width);
    try std.testing.expectEqual(@as(u32, 300), size.height);
}
