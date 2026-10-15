//! ZAPP - ZPU Application Framework
//!
//! ウィンドウ管理、イベント処理、UI、レンダリングを統合するアプリケーションフレームワーク。
//! ネイティブ（Vulkan/WebGPU）、Web（Canvas）、ヘッドレス（ソフトウェア）の
//! 複数プラットフォームに対応。
//!
//! ## 基本的な使い方
//!
//! ```zig
//! const std = @import("std");
//! const zapp = @import("zapp");
//!
//! const MyApp = struct {
//!     ui: zapp.Ui,
//!     count: u32 = 0,
//!     
//!     pub fn init() !MyApp {
//!         return .{
//!             .ui = try zapp.Ui.init(.{}),
//!         };
//!     }
//!     
//!     pub fn deinit(self: *MyApp) void {
//!         self.ui.deinit();
//!     }
//!     
//!     pub fn update(self: *MyApp, app: *zapp.App) !void {
//!         const size = app.getSize();
//!         try self.ui.frame(.{
//!             .width = size.width,
//!             .height = size.height,
//!         }, struct {
//!             fn frame(state: *MyApp, ui: *zapp.Ui) !void {
//!                 try ui.label("ZAPP Application", .{});
//!                 
//!                 if (try ui.button("Click me", .{})) {
//!                     state.count += 1;
//!                 }
//!                 
//!                 var buf: [32]u8 = undefined;
//!                 const text = std.fmt.bufPrint(&buf, "Count: {d}", .{state.count}) catch unreachable;
//!                 try ui.label(text, .{});
//!             }
//!         }.frame, self);
//!     }
//! };
//!
//! pub fn main() !void {
//!     var app = try zapp.App.init(.{
//!         .window = .{
//!             .title = "My ZAPP App",
//!             .width = 800,
//!             .height = 600,
//!         },
//!     });
//!     defer app.deinit();
//!     
//!     var my_app = try MyApp.init();
//!     defer my_app.deinit();
//!     
//!     try app.run(&my_app);
//! }
//! ```
//!
//! ## バックエンド
//!
//! - **software**: ソフトウェアレンダリング（テスト/ヘッドレス）
//! - **vulkan**: Vulkan（デスクトップ、高性能）
//! - **webgpu**: WebGPU（Web/デスクトップ、次世代）
//! - **web**: WebCanvas（WebAssembly）
//!
//! ## 機能
//!
//! - ウィンドウ管理（ネイティブ/Web）
//! - イベント処理（マウス、キーボード、タッチ）
//! - ZUI統合（即時モードUI）
//! - 複数バックエンド対応
//! - HiDPI対応
//! - フレームレート制限

const std = @import("std");

// コアモジュール
pub const window = @import("zapp/window.zig");
pub const event = @import("zapp/event.zig");
pub const app = @import("zapp/app.zig");
pub const dsl = @import("zapp/dsl.zig");

// 主要な型の再エクスポート
pub const Window = window.Window;
pub const WindowConfig = window.WindowConfig;
pub const Backend = window.Backend;
pub const WindowError = window.WindowError;

pub const Event = event.Event;
pub const EventQueue = event.EventQueue;
pub const Key = event.Key;
pub const MouseButton = event.MouseButton;
pub const Modifiers = event.Modifiers;

pub const App = app.App;
pub const AppConfig = app.AppConfig;
pub const Ui = app.Ui;

// バージョン情報
pub const version = "0.1.0";
pub const version_major = 0;
pub const version_minor = 1;
pub const version_patch = 0;

/// フレームワークの初期化情報を取得
pub fn getVersionInfo() struct {
    version: []const u8,
    major: u32,
    minor: u32,
    patch: u32,
} {
    return .{
        .version = version,
        .major = version_major,
        .minor = version_minor,
        .patch = version_patch,
    };
}

// テスト
test "ZAPP version" {
    const info = getVersionInfo();
    try std.testing.expectEqualStrings("0.1.0", info.version);
    try std.testing.expectEqual(@as(u32, 0), info.major);
    try std.testing.expectEqual(@as(u32, 1), info.minor);
    try std.testing.expectEqual(@as(u32, 0), info.patch);
}

test "ZAPP imports" {
    _ = Window;
    _ = WindowConfig;
    _ = Backend;
    _ = Event;
    _ = EventQueue;
    _ = App;
    _ = AppConfig;
    _ = Ui;
}
