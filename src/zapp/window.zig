//! ウィンドウ抽象化レイヤー
//!
//! ネイティブウィンドウ、Webキャンバス、ヘッドレスモードを統一APIで扱う。
//! バックエンドはプラットフォームに応じて自動的に選択される。
//!
//! ```zig
//! const zapp = @import("zapp");
//!
//! var window = try zapp.Window.init(.{
//!     .title = "My App",
//!     .width = 800,
//!     .height = 600,
//!     .backend = .auto,  // auto, software, vulkan, webgpu, web
//! });
//! defer window.deinit();
//!
//! while (window.isOpen()) {
//!     window.pollEvents();
//!     
//!     // ZUIでUIを描画
//!     try ui.draw(&window.renderer);
//!     
//!     window.present();
//! }
//! ```

const std = @import("std");
const builtin = @import("builtin");
const zui = @import("zui");
const zgf = @import("zgf");
const zpu = @import("zpu");

// バックエンドモジュール
const x11_backend = @import("backend/x11.zig");
const web_backend = @import("backend/web.zig");

pub const Backend = enum {
    auto,       // プラットフォームに応じて自動選択
    software,   // ソフトウェアレンダリング（ヘッドレス）
    x11,        // X11（Linux）
    vulkan,     // Vulkan（デスクトップ）
    webgpu,     // WebGPU（Web/デスクトップ）
    web,        // WebCanvas（WebAssembly）
};

pub const WindowConfig = struct {
    title: []const u8 = "ZPU Application",
    width: u32 = 800,
    height: u32 = 600,
    backend: Backend = .auto,
    resizable: bool = true,
    vsync: bool = true,
    scale: f32 = 1.0, // HiDPI対応
};

pub const WindowError = error{
    InitializationFailed,
    BackendNotAvailable,
    ContextCreationFailed,
    SurfaceCreationFailed,
    SwapchainCreationFailed,
    PlatformNotSupported,
    InvalidWidth,
    InvalidHeight,
    OutOfMemory,
};

/// ウィンドウの統一インターフェース
pub const Window = struct {
    impl: *anyopaque,
    vtable: *const VTable,
    config: WindowConfig,
    
    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        isOpen: *const fn (*const anyopaque) bool,
        pollEvents: *const fn (*anyopaque) void,
        present: *const fn (*anyopaque) void,
        getSize: *const fn (*const anyopaque) struct { width: u32, height: u32 },
        setTitle: *const fn (*anyopaque, []const u8) void,
        getFramebuffer: *const fn (*anyopaque) *zgf.Image,
        resize: *const fn (*anyopaque, u32, u32) anyerror!void,
    };
    
    /// ウィンドウを初期化
    pub fn init(config: WindowConfig) WindowError!Window {
        const backend = resolveBackend(config.backend);
        
        return switch (backend) {
            .software => SoftwareWindow.init(config),
            .vulkan => VulkanWindow.init(config),
            .webgpu => WebGPUWindow.init(config),
            .web => WebWindow.init(config),
            .x11 => X11Window.init(config),
            else => WindowError.BackendNotAvailable,
        };
    }
    
    pub fn deinit(self: *Window) void {
        self.vtable.deinit(self.impl);
    }
    
    pub fn isOpen(self: *const Window) bool {
        return self.vtable.isOpen(self.impl);
    }
    
    pub fn pollEvents(self: *Window) void {
        self.vtable.pollEvents(self.impl);
    }
    
    pub fn present(self: *Window) void {
        self.vtable.present(self.impl);
    }
    
    pub fn getSize(self: *const Window) struct { width: u32, height: u32 } {
        return self.vtable.getSize(self.impl);
    }
    
    pub fn setTitle(self: *Window, title: []const u8) void {
        self.vtable.setTitle(self.impl, title);
    }
    
    /// レンダリング先の画像を取得
    pub fn getFramebuffer(self: *Window) *zgf.Image {
        return self.vtable.getFramebuffer(self.impl);
    }
    
    /// ウィンドウサイズを変更
    pub fn resize(self: *Window, width: u32, height: u32) !void {
        // サイズの検証（100x100 から 3840x2160 (4K) まで）
        if (width < 100 or width > 3840) {
            return error.InvalidWidth;
        }
        if (height < 100 or height > 2160) {
            return error.InvalidHeight;
        }
        
        try self.vtable.resize(self.impl, width, height);
        self.config.width = width;
        self.config.height = height;
    }
    
    /// ZUIのUIをウィンドウに描画
    pub fn drawUi(self: *Window, ui_instance: *zui.Ui) !void {
        const framebuffer = self.getFramebuffer();
        const size = self.getSize();
        
        // ZUIのフレームを実行
        const input = zui.Input{}; // TODO: イベントから入力を作成
        _ = try ui_instance.run(input, struct {
            pub fn run(_: @This(), u: *zui.Ui) zui.Ui.Error!void {
                _ = u;
            }
        }{});
        
        // レンダリング
        // TODO: DisplayListをframebufferにレンダリング
        _ = framebuffer;
        _ = size;
    }
    
    fn resolveBackend(requested: Backend) Backend {
        if (requested != .auto) return requested;
        
        // プラットフォームに応じて自動選択
        const arch = builtin.target.cpu.arch;
        const os = builtin.target.os.tag;
        
        // WebAssembly環境
        if (arch == .wasm32 or arch == .wasm64) {
            return .web;
        }
        
        // Linux環境
        if (os == .linux) {
            // X11が利用可能であればX11を使用
            // それ以外はソフトウェアレンダリング
            return .x11;
        }
        
        // Windows/macOSはソフトウェアレンダリング（Vulkan/WebGPUは未実装）
        return .software;
    }
};

/// ソフトウェアレンダリングウィンドウ（ヘッドレス/テスト用）
const SoftwareWindow = struct {
    framebuffer: zgf.Image,
    open: bool,
    config: WindowConfig,
    
    pub fn init(config: WindowConfig) WindowError!Window {
        var self = std.heap.page_allocator.create(SoftwareWindow) catch return WindowError.InitializationFailed;
        errdefer std.heap.page_allocator.destroy(self);
        
        self.framebuffer = zgf.Image.init(std.heap.page_allocator, config.width, config.height, 4) catch return WindowError.InitializationFailed;
        self.open = true;
        self.config = config;
        
        const vtable = std.heap.page_allocator.create(Window.VTable) catch return WindowError.InitializationFailed;
        vtable.* = .{
            .deinit = deinit,
            .isOpen = isOpen,
            .pollEvents = pollEvents,
            .present = present,
            .getSize = getSize,
            .setTitle = setTitle,
            .getFramebuffer = getFramebuffer,
            .resize = resize,
        };
        
        return .{
            .impl = self,
            .vtable = vtable,
            .config = config,
        };
    }
    
    fn deinit(ptr: *anyopaque) void {
        const self: *SoftwareWindow = @ptrCast(@alignCast(ptr));
        self.framebuffer.deinit();
        std.heap.page_allocator.destroy(self);
    }
    
    fn isOpen(ptr: *const anyopaque) bool {
        const self: *const SoftwareWindow = @ptrCast(@alignCast(ptr));
        return self.open;
    }
    
    fn pollEvents(_: *anyopaque) void {
        // ソフトウェアモードではイベントなし
    }
    
    fn present(_: *anyopaque) void {
        // ソフトウェアモードでは何もしない（またはPPM出力）
    }
    
    fn getSize(ptr: *const anyopaque) struct { width: u32, height: u32 } {
        const self: *const SoftwareWindow = @ptrCast(@alignCast(ptr));
        return .{ .width = self.config.width, .height = self.config.height };
    }
    
    fn setTitle(_: *anyopaque, _: []const u8) void {
        // ソフトウェアモードではタイトルなし
    }
    
    fn getFramebuffer(ptr: *anyopaque) *zgf.Image {
        const self: *SoftwareWindow = @ptrCast(@alignCast(ptr));
        return &self.framebuffer;
    }
    
    fn resize(ptr: *anyopaque, width: u32, height: u32) anyerror!void {
        const self: *SoftwareWindow = @ptrCast(@alignCast(ptr));
        
        // 古いフレームバッファを解放
        self.framebuffer.deinit();
        
        // 新しいサイズのフレームバッファを作成
        self.framebuffer = zgf.Image.init(std.heap.page_allocator, width, height, 4) catch return error.OutOfMemory;
        self.config.width = width;
        self.config.height = height;
    }
};

/// Vulkanウィンドウ（デスクトップ）
const VulkanWindow = struct {
    // TODO: Vulkan実装
    pub fn init(_: WindowConfig) WindowError!Window {
        return WindowError.BackendNotAvailable;
    }
};

/// WebGPUウィンドウ
const WebGPUWindow = struct {
    // TODO: WebGPU実装
    pub fn init(_: WindowConfig) WindowError!Window {
        return WindowError.BackendNotAvailable;
    }
};

/// Webウィンドウ（WebAssembly + Canvas）
const WebWindow = struct {
    impl: web_backend.WebWindow,
    
    pub fn init(config: WindowConfig) WindowError!Window {
        var self = std.heap.page_allocator.create(WebWindow) catch return WindowError.InitializationFailed;
        errdefer std.heap.page_allocator.destroy(self);
        
        self.impl = web_backend.WebWindow.init(std.heap.page_allocator, .{
            .title = config.title,
            .width = config.width,
            .height = config.height,
        }) catch return WindowError.InitializationFailed;
        
        const vtable = std.heap.page_allocator.create(Window.VTable) catch return WindowError.InitializationFailed;
        vtable.* = .{
            .deinit = deinit,
            .isOpen = isOpen,
            .pollEvents = pollEvents,
            .present = present,
            .getSize = getSize,
            .setTitle = setTitle,
            .getFramebuffer = getFramebuffer,
            .resize = resize,
        };
        
        return .{
            .impl = self,
            .vtable = vtable,
            .config = config,
        };
    }
    
    fn deinit(ptr: *anyopaque) void {
        const self: *WebWindow = @ptrCast(@alignCast(ptr));
        self.impl.deinit();
        std.heap.page_allocator.destroy(self);
    }
    
    fn isOpen(ptr: *const anyopaque) bool {
        const self: *const WebWindow = @ptrCast(@alignCast(ptr));
        return self.impl.isOpen();
    }
    
    fn pollEvents(ptr: *anyopaque) void {
        const self: *WebWindow = @ptrCast(@alignCast(ptr));
        self.impl.pollEvents();
    }
    
    fn present(ptr: *anyopaque) void {
        const self: *WebWindow = @ptrCast(@alignCast(ptr));
        self.impl.present();
    }
    
    fn getSize(ptr: *const anyopaque) struct { width: u32, height: u32 } {
        const self: *const WebWindow = @ptrCast(@alignCast(ptr));
        return self.impl.getSize();
    }
    
    fn setTitle(ptr: *anyopaque, title: []const u8) void {
        const self: *WebWindow = @ptrCast(@alignCast(ptr));
        self.impl.setTitle(title);
    }
    
    fn getFramebuffer(ptr: *anyopaque) *zgf.Image {
        const self: *WebWindow = @ptrCast(@alignCast(ptr));
        return self.impl.getFramebuffer();
    }
    
    fn resize(ptr: *anyopaque, width: u32, height: u32) anyerror!void {
        const self: *WebWindow = @ptrCast(@alignCast(ptr));
        try self.impl.resize(width, height);
    }
};

/// X11ウィンドウ（Linux）
const X11Window = struct {
    impl: x11_backend.X11Window,
    
    pub fn init(config: WindowConfig) WindowError!Window {
        var self = std.heap.page_allocator.create(X11Window) catch return WindowError.InitializationFailed;
        errdefer std.heap.page_allocator.destroy(self);
        
        self.impl = x11_backend.X11Window.init(std.heap.page_allocator, .{
            .title = config.title,
            .width = config.width,
            .height = config.height,
            .resizable = config.resizable,
        }) catch return WindowError.InitializationFailed;
        
        const vtable = std.heap.page_allocator.create(Window.VTable) catch return WindowError.InitializationFailed;
        vtable.* = .{
            .deinit = deinit,
            .isOpen = isOpen,
            .pollEvents = pollEvents,
            .present = present,
            .getSize = getSize,
            .setTitle = setTitle,
            .getFramebuffer = getFramebuffer,
            .resize = resize,
        };
        
        return .{
            .impl = self,
            .vtable = vtable,
            .config = config,
        };
    }
    
    fn deinit(ptr: *anyopaque) void {
        const self: *X11Window = @ptrCast(@alignCast(ptr));
        self.impl.deinit();
        std.heap.page_allocator.destroy(self);
    }
    
    fn isOpen(ptr: *const anyopaque) bool {
        const self: *const X11Window = @ptrCast(@alignCast(ptr));
        return self.impl.isOpen();
    }
    
    fn pollEvents(ptr: *anyopaque) void {
        const self: *X11Window = @ptrCast(@alignCast(ptr));
        self.impl.pollEvents();
    }
    
    fn present(ptr: *anyopaque) void {
        const self: *X11Window = @ptrCast(@alignCast(ptr));
        self.impl.present();
    }
    
    fn getSize(ptr: *const anyopaque) struct { width: u32, height: u32 } {
        const self: *const X11Window = @ptrCast(@alignCast(ptr));
        return self.impl.getSize();
    }
    
    fn setTitle(ptr: *anyopaque, title: []const u8) void {
        const self: *X11Window = @ptrCast(@alignCast(ptr));
        self.impl.setTitle(title);
    }
    
    fn getFramebuffer(ptr: *anyopaque) *zgf.Image {
        const self: *X11Window = @ptrCast(@alignCast(ptr));
        return self.impl.getFramebuffer();
    }
    
    fn resize(ptr: *anyopaque, width: u32, height: u32) anyerror!void {
        const self: *X11Window = @ptrCast(@alignCast(ptr));
        try self.impl.resize(width, height);
    }
};

// テスト
test "SoftwareWindow basic" {
    var window = try Window.init(.{
        .title = "Test",
        .width = 100,
        .height = 100,
        .backend = .software,
    });
    defer window.deinit();
    
    try std.testing.expect(window.isOpen());
    const size = window.getSize();
    try std.testing.expectEqual(@as(u32, 100), size.width);
    try std.testing.expectEqual(@as(u32, 100), size.height);
    
    const fb = window.getFramebuffer();
    try std.testing.expectEqual(@as(usize, 100), fb.width());
    try std.testing.expectEqual(@as(usize, 100), fb.height());
}

test "SoftwareWindow resize from 100x100 to 4K" {
    var window = try Window.init(.{
        .title = "Resize Test",
        .width = 100,
        .height = 100,
        .backend = .software,
    });
    defer window.deinit();
    
    // 初期サイズを確認
    {
        const size = window.getSize();
        try std.testing.expectEqual(@as(u32, 100), size.width);
        try std.testing.expectEqual(@as(u32, 100), size.height);
    }
    
    // 小さいサイズにリサイズ
    try window.resize(200, 150);
    {
        const size = window.getSize();
        try std.testing.expectEqual(@as(u32, 200), size.width);
        try std.testing.expectEqual(@as(u32, 150), size.height);
        const fb = window.getFramebuffer();
        try std.testing.expectEqual(@as(usize, 200), fb.width());
        try std.testing.expectEqual(@as(usize, 150), fb.height());
    }
    
    // HDサイズにリサイズ
    try window.resize(1280, 720);
    {
        const size = window.getSize();
        try std.testing.expectEqual(@as(u32, 1280), size.width);
        try std.testing.expectEqual(@as(u32, 720), size.height);
        const fb = window.getFramebuffer();
        try std.testing.expectEqual(@as(usize, 1280), fb.width());
        try std.testing.expectEqual(@as(usize, 720), fb.height());
    }
    
    // Full HDサイズにリサイズ
    try window.resize(1920, 1080);
    {
        const size = window.getSize();
        try std.testing.expectEqual(@as(u32, 1920), size.width);
        try std.testing.expectEqual(@as(u32, 1080), size.height);
        const fb = window.getFramebuffer();
        try std.testing.expectEqual(@as(usize, 1920), fb.width());
        try std.testing.expectEqual(@as(usize, 1080), fb.height());
    }
    
    // 4Kサイズにリサイズ
    try window.resize(3840, 2160);
    {
        const size = window.getSize();
        try std.testing.expectEqual(@as(u32, 3840), size.width);
        try std.testing.expectEqual(@as(u32, 2160), size.height);
        const fb = window.getFramebuffer();
        try std.testing.expectEqual(@as(usize, 3840), fb.width());
        try std.testing.expectEqual(@as(usize, 2160), fb.height());
    }
    
    // 再度小さいサイズにリサイズ
    try window.resize(100, 100);
    {
        const size = window.getSize();
        try std.testing.expectEqual(@as(u32, 100), size.width);
        try std.testing.expectEqual(@as(u32, 100), size.height);
    }
}

test "Window resize validation - too small" {
    var window = try Window.init(.{
        .title = "Validation Test",
        .width = 200,
        .height = 200,
        .backend = .software,
    });
    defer window.deinit();
    
    // 小さすぎるサイズはエラー
    try std.testing.expectError(error.InvalidWidth, window.resize(50, 200));
    try std.testing.expectError(error.InvalidHeight, window.resize(200, 50));
    try std.testing.expectError(error.InvalidWidth, window.resize(50, 50));
    
    // サイズが変わっていないことを確認
    const size = window.getSize();
    try std.testing.expectEqual(@as(u32, 200), size.width);
    try std.testing.expectEqual(@as(u32, 200), size.height);
}

test "Window resize validation - too large" {
    var window = try Window.init(.{
        .title = "Validation Test",
        .width = 200,
        .height = 200,
        .backend = .software,
    });
    defer window.deinit();
    
    // 大きすぎるサイズはエラー
    try std.testing.expectError(error.InvalidWidth, window.resize(4000, 200));
    try std.testing.expectError(error.InvalidHeight, window.resize(200, 2200));
    try std.testing.expectError(error.InvalidWidth, window.resize(4000, 2200));
    
    // サイズが変わっていないことを確認
    const size = window.getSize();
    try std.testing.expectEqual(@as(u32, 200), size.width);
    try std.testing.expectEqual(@as(u32, 200), size.height);
}
