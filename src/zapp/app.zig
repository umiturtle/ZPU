//! アプリケーションライフサイクル管理
//!
//! ウィンドウ、イベント、UI、レンダリングを統合管理する。
//!
//! ```zig
//! const zapp = @import("zapp");
//!
//! const MyApp = struct {
//!     ui: zapp.Ui,
//!     state: AppState,
//!     
//!     pub fn init() !MyApp {
//!         return .{
//!             .ui = try zapp.Ui.init(.{}),
//!             .state = .{},
//!         };
//!     }
//!     
//!     pub fn deinit(self: *MyApp) void {
//!         self.ui.deinit();
//!     }
//!     
//!     pub fn update(self: *MyApp, app: *zapp.App) !void {
//!         // UI構築
//!         try self.ui.frame(.{
//!             .width = app.window.getSize().width,
//!             .height = app.window.getSize().height,
//!         }, struct {
//!             fn frame(state: *AppState, ui: *zapp.Ui) !void {
//!                 try ui.label("Hello, World!", .{});
//!             }
//!         }.frame, &self.state);
//!     }
//! };
//!
//! pub fn main() !void {
//!     var app = try zapp.App.init(.{
//!         .title = "My App",
//!         .width = 800,
//!         .height = 600,
//!     });
//!     defer app.deinit();
//!     
//!     var my_app = try MyApp.init();
//!     defer my_app.deinit();
//!     
//!     try app.run(&my_app);
//! }
//! ```

const std = @import("std");
const zui = @import("zui");
const zgf = @import("zgf");
const zpu = @import("zpu");

pub const Window = @import("window.zig").Window;
pub const WindowConfig = @import("window.zig").WindowConfig;
pub const Event = @import("event.zig").Event;
pub const EventQueue = @import("event.zig").EventQueue;

pub const AppConfig = struct {
    window: WindowConfig = .{},
    
    /// フレームレート制限（0 = 制限なし）
    target_fps: u32 = 60,
    
    /// デバッグモード
    debug: bool = false,
};

pub const App = struct {
    window: Window,
    events: EventQueue,
    config: AppConfig,
    running: bool,
    frame_count: u64,
    last_frame_time: i128,
    allocator: std.mem.Allocator,
    
    pub fn init(config: AppConfig) !App {
        const allocator = std.heap.page_allocator;
        
        // ZPUを初期化
        try zpu.init(allocator, .{});
        errdefer zpu.deinit();
        
        // ウィンドウを初期化
        var window = try Window.init(config.window);
        errdefer window.deinit();
        
        return .{
            .window = window,
            .events = EventQueue.init(allocator),
            .config = config,
            .running = true,
            .frame_count = 0,
            .last_frame_time = std.time.nanoTimestamp(),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *App) void {
        self.events.deinit();
        self.window.deinit();
        zpu.deinit();
    }
    
    /// アプリケーションのメインループ
    pub fn run(self: *App, handler: anytype) !void {
        while (self.running and self.window.isOpen()) {
            // イベント処理
            self.window.pollEvents();
            
            // イベントキューを処理
            while (self.events.poll()) |event| {
                switch (event) {
                    .close => {
                        self.running = false;
                        break;
                    },
                    else => {
                        if (@hasDecl(@TypeOf(handler), "onEvent")) {
                            try handler.onEvent(self, event);
                        }
                    },
                }
            }
            
            if (!self.running) break;
            
            // 更新
            if (@hasDecl(@TypeOf(handler), "update")) {
                try handler.update(self);
            }
            
            // 描画
            if (@hasDecl(@TypeOf(handler), "render")) {
                try handler.render(self);
            }
            
            // 表示
            self.window.present();
            
            // フレームレート制限
            self.limitFrameRate();
            
            self.frame_count += 1;
        }
    }
    
    /// アプリケーションを終了
    pub fn quit(self: *App) void {
        self.running = false;
    }
    
    /// ウィンドウタイトルを変更
    pub fn setTitle(self: *App, title: []const u8) void {
        self.window.setTitle(title);
    }
    
    /// ウィンドウサイズを取得
    pub fn getSize(self: *const App) struct { width: u32, height: u32 } {
        return self.window.getSize();
    }
    
    /// フレームバッファを取得
    pub fn getFramebuffer(self: *App) *zgf.Image {
        return self.window.getFramebuffer();
    }
    
    /// イベントを送信
    pub fn pushEvent(self: *App, event: Event) !void {
        try self.events.push(event);
    }
    
    fn limitFrameRate(self: *App) void {
        if (self.config.target_fps == 0) return;
        
        const target_frame_time = @divTrunc(1_000_000_000, self.config.target_fps);
        const current_time = std.time.nanoTimestamp();
        const elapsed = current_time - self.last_frame_time;
        
        if (elapsed < target_frame_time) {
            const sleep_time = target_frame_time - elapsed;
            std.time.sleep(@intCast(sleep_time));
        }
        
        self.last_frame_time = std.time.nanoTimestamp();
    }
};

/// UIヘルパー（ZUIのラッパー）
pub const Ui = struct {
    ui: zui.Ui,
    
    pub fn init(config: zui.Ui.Options) !Ui {
        return .{
            .ui = try zui.Ui.init(std.heap.page_allocator, config),
        };
    }
    
    pub fn deinit(self: *Ui) void {
        self.ui.deinit();
    }
    
    /// UIフレームを描画
    pub fn frame(self: *Ui, size: struct { width: u32, height: u32 }, comptime body_fn: anytype, ctx: anytype) !void {
        const input = zui.Input{}; // TODO: イベントから生成
        
        _ = try self.ui.run(input, struct {
            const Body = @TypeOf(body_fn);
            
            pub fn run(_: @This(), u: *zui.Ui) zui.Ui.Error!void {
                try u.container("root", .{
                    .w = .{ .px = @floatFromInt(size.width) },
                    .h = .{ .px = @floatFromInt(size.height) },
                    .bg = u.theme.bg,
                }, struct {
                    ctx_ptr: @TypeOf(ctx),
                    
                    pub fn run(inner_s: @This(), inner_u: *zui.Ui) zui.Ui.Error!void {
                        try body_fn(inner_s.ctx_ptr, inner_u);
                    }
                }{ .ctx_ptr = ctx });
            }
        }{});
    }
};

// テスト
test "App initialization" {
    var app = try App.init(.{
        .window = .{
            .title = "Test App",
            .width = 400,
            .height = 300,
            .backend = .software,
        },
        .target_fps = 30,
    });
    defer app.deinit();
    
    const size = app.getSize();
    try std.testing.expectEqual(@as(u32, 400), size.width);
    try std.testing.expectEqual(@as(u32, 300), size.height);
    
    try std.testing.expect(app.running);
    app.quit();
    try std.testing.expect(!app.running);
}

test "Event handling" {
    var app = try App.init(.{
        .window = .{
            .backend = .software,
        },
    });
    defer app.deinit();
    
    try app.pushEvent(.{ .resize = .{ .width = 800, .height = 600 } });
    try std.testing.expect(app.events.hasEvents());
    
    const event = app.events.poll();
    try std.testing.expect(event != null);
    try std.testing.expect(event.? == .resize);
}
