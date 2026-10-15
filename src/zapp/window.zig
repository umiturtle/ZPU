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

pub const Backend = enum {
    auto,       // プラットフォームに応じて自動選択
    software,   // ソフトウェアレンダリング（ヘッドレス）
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
    };
    
    /// ウィンドウを初期化
    pub fn init(config: WindowConfig) WindowError!Window {
        const backend = resolveBackend(config.backend);
        
        return switch (backend) {
            .software => SoftwareWindow.init(config),
            .vulkan => VulkanWindow.init(config),
            .webgpu => WebGPUWindow.init(config),
            .web => WebWindow.init(config),
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
        if (builtin.target.os.tag == .emscripten or builtin.target.cpu.arch == .wasm32) {
            return .web;
        }
        
        // デスクトップではWebGPUを優先（なければVulkan、最後にソフトウェア）
        return .software; // TODO: WebGPU/Vulkanの検出
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
    // TODO: Web実装
    pub fn init(_: WindowConfig) WindowError!Window {
        return WindowError.BackendNotAvailable;
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
