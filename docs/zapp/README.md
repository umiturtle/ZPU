# ZAPP - ZPU Application Framework

ZAPP（ザップ）は、ZPUエコシステムのための統合アプリケーションフレームワークです。ウィンドウ管理、イベント処理、UI、レンダリングを統一APIで扱い、複数プラットフォーム（ネイティブ、Web）に対応します。

## 特徴

- **統合API**: ウィンドウ、イベント、UI、レンダリングを一つのフレームワークで管理
- **マルチプラットフォーム**: ネイティブ（Vulkan/WebGPU）、Web（Canvas）、ヘッドレス（ソフトウェア）
- **ZUI統合**: 即時モードUIライブラリZUIをネイティブに統合
- **イベント駆動**: マウス、キーボード、タッチイベントを統一的に処理
- **HiDPI対応**: スケーラブルなUIレンダリング
- **テスト容易**: ヘッドレスモードでウィンドウなしテストが可能

## アーキテクチャ

```
┌─────────────────────────────────────┐
│        Application Code             │
│    (zapp.App, zapp.Ui handlers)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│           ZAPP Framework            │
│  - App lifecycle management         │
│  - Event handling                   │
│  - UI integration                   │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼─────┐   ┌──────▼─────┐
│  Window    │   │    ZUI     │
│ Management │   │    (UI)    │
└──────┬─────┘   └──────┬─────┘
       │                │
       │         ┌──────▼─────┐
       │         │  Display   │
       │         │    List    │
       │         └──────┬─────┘
       │                │
┌──────▼────────────────▼──────┐
│      Rendering Backend       │
│  - Software (zgf.Image)      │
│  - Vulkan                    │
│  - WebGPU                    │
│  - Web (Canvas)              │
└──────────────────────────────┘
```

## 基本的な使い方

### 最小限のアプリケーション

```zig
const std = @import("std");
const zapp = @import("zapp");

const MyApp = struct {
    ui: zapp.Ui,
    count: u32 = 0,
    
    pub fn init() !MyApp {
        return .{
            .ui = try zapp.Ui.init(.{}),
        };
    }
    
    pub fn deinit(self: *MyApp) void {
        self.ui.deinit();
    }
    
    pub fn update(self: *MyApp, app: *zapp.App) !void {
        const size = app.getSize();
        try self.ui.frame(.{
            .width = size.width,
            .height = size.height,
        }, struct {
            fn frame(state: *MyApp, ui: *zapp.Ui) !void {
                try ui.label("Hello, ZAPP!", .{});
                
                if (try ui.button("Click me", .{})) {
                    state.count += 1;
                }
                
                var buf: [32]u8 = undefined;
                const text = std.fmt.bufPrint(&buf, "Count: {d}", .{state.count}) catch unreachable;
                try ui.label(text, .{});
            }
        }.frame, self);
    }
};

pub fn main() !void {
    var app = try zapp.App.init(.{
        .window = .{
            .title = "My App",
            .width = 800,
            .height = 600,
        },
    });
    defer app.deinit();
    
    var my_app = try MyApp.init();
    defer my_app.deinit();
    
    try app.run(&my_app);
}
```

### イベント処理

```zig
fn onEvent(state: *MyState, app: *zapp.App, event: zapp.Event) !void {
    switch (event) {
        .key_press => |key| {
            if (key.key == .escape) {
                app.quit();
            }
        },
        .mouse_move => |pos| {
            std.debug.print("Mouse at ({d}, {d})\n", .{ pos.x, pos.y });
        },
        .resize => |size| {
            std.debug.print("Resized to {}x{}\n", .{ size.width, size.height });
        },
        else => {},
    }
}
```

## バックエンド

### Software（ソフトウェア）

- **用途**: テスト、ヘッドレスレンダリング、CI/CD
- **特徴**: GPU不要、PPM出力可能
- **設定**: `.backend = .software`

```zig
var app = try zapp.App.init(.{
    .window = .{
        .backend = .software,
    },
});
```

### Vulkan

- **用途**: デスクトップアプリケーション（高性能）
- **特徴**: ハードウェアアクセラレーション
- **設定**: `.backend = .vulkan`

**状態**: 実装中

### WebGPU

- **用途**: Web/デスクトップ（次世代API）
- **特徴**: モダンなGPU API、クロスプラットフォーム
- **設定**: `.backend = .webgpu`

**状態**: 実装中

### Web

- **用途**: WebAssembly（ブラウザ）
- **特徴**: HTML5 Canvas出力
- **設定**: `.backend = .web`

**状態**: 実装中

## API リファレンス

### zapp.App

アプリケーションのライフサイクルを管理。

```zig
const app = try zapp.App.init(.{
    .window = .{
        .title = "My App",
        .width = 800,
        .height = 600,
        .backend = .auto,
    },
    .target_fps = 60,
    .debug = false,
});
defer app.deinit();

// メインループ
try app.run(&my_handler);

// 手動制御
while (app.running) {
    app.window.pollEvents();
    // 更新...
    app.window.present();
}

// アプリケーション終了
app.quit();
```

### zapp.Window

ウィンドウを管理。

```zig
const window = try zapp.Window.init(.{
    .title = "Window",
    .width = 800,
    .height = 600,
    .backend = .software,
    .resizable = true,
});
defer window.deinit();

while (window.isOpen()) {
    window.pollEvents();
    
    const size = window.getSize();
    const framebuffer = window.getFramebuffer();
    
    // レンダリング...
    
    window.present();
}
```

### zapp.Ui

ZUIのラッパー。UIフレームを管理。

```zig
const ui = try zapp.Ui.init(.{});
defer ui.deinit();

try ui.frame(.{
    .width = 800,
    .height = 600,
}, struct {
    fn frame(state: *MyState, ui: *zapp.Ui) !void {
        try ui.label("Hello!", .{});
        if (try ui.button("Click", .{})) {
            // ボタンが押された
        }
    }
}.frame, &state);
```

### zapp.Event

イベントの種類。

```zig
const Event = union(enum) {
    close: void,
    resize: struct { width: u32, height: u32 },
    focus: bool,
    key_press: KeyEvent,
    key_release: KeyEvent,
    text_input: []const u8,
    mouse_move: struct { x: f32, y: f32 },
    mouse_button_press: MouseButtonEvent,
    mouse_button_release: MouseButtonEvent,
    mouse_scroll: struct { dx: f32, dy: f32 },
    touch_start: TouchEvent,
    touch_move: TouchEvent,
    touch_end: TouchEvent,
    none: void,
};
```

## 例

### カウンターアプリケーション

```zig
const CounterApp = struct {
    ui: zapp.Ui,
    count: u32 = 0,
    
    pub fn init() !CounterApp {
        return .{ .ui = try zapp.Ui.init(.{}) };
    }
    
    pub fn deinit(self: *CounterApp) void {
        self.ui.deinit();
    }
    
    pub fn update(self: *CounterApp, app: *zapp.App) !void {
        const size = app.getSize();
        try self.ui.frame(.{
            .width = size.width,
            .height = size.height,
        }, struct {
            fn frame(state: *CounterApp, ui: *zapp.Ui) !void {
                try ui.label("Counter", .{});
                
                if (try ui.button("+", .{})) {
                    state.count += 1;
                }
                
                if (try ui.button("-", .{})) {
                    if (state.count > 0) state.count -= 1;
                }
                
                var buf: [32]u8 = undefined;
                const text = std.fmt.bufPrint(&buf, "{d}", .{state.count}) catch unreachable;
                try ui.label(text, .{});
            }
        }.frame, self);
    }
};
```

### テキストエディタ

```zig
const EditorApp = struct {
    ui: zapp.Ui,
    text: zui.TextField,
    buffer: [1024]u8 = undefined,
    
    pub fn init() !EditorApp {
        var app = EditorApp{
            .ui = try zapp.Ui.init(.{}),
            .text = undefined,
        };
        app.text = zui.TextField.init(&app.buffer);
        return app;
    }
    
    pub fn deinit(self: *EditorApp) void {
        self.ui.deinit();
    }
    
    pub fn update(self: *EditorApp, app: *zapp.App) !void {
        const size = app.getSize();
        try self.ui.frame(.{
            .width = size.width,
            .height = size.height,
        }, struct {
            fn frame(state: *EditorApp, ui: *zapp.Ui) !void {
                try ui.label("Text Editor", .{});
                _ = try ui.textEdit("editor", &state.text, .{
                    .placeholder = "Type here...",
                });
            }
        }.frame, self);
    }
};
```

## ビルドと実行

```bash
# 例をビルド
zig build run-zapp-demo

# テストを実行
zig build test-zapp

# 特定のテスト
zig build test -- -Dtest-filter=App
```

## 現在の状態

### 実装済み ✅

- [x] ウィンドウ抽象化（software backend）
- [x] イベントシステム
- [x] ZUI統合
- [x] アプリケーションライフサイクル
- [x] ヘッドレスレンダリング
- [x] テストフレームワーク

### 実装中 🔧

- [ ] Vulkan バックエンド
- [ ] WebGPU バックエンド
- [ ] Web (Canvas) バックエンド
- [ ] ネイティブウィンドウ（GLFW/SDL）
- [ ] IME統合
- [ ] クリップボード
- [ ] ドラッグ&ドロップ
- [ ] ファイルダイアログ

### 予定 📋

- [ ] モバイル対応（iOS/Android）
- [ ] ゲームパッド入力
- [ ] 音声入力
- [ ] ビデオ出力
- [ ] プラグインシステム

## ライセンス

このプロジェクトはZPUプロジェクトの一部です。

## 関連プロジェクト

- [ZPU](../README.md) - 数値計算ライブラリ
- [ZGF](../src/zgf.zig) - 画像処理ライブラリ
- [ZUI](../src/zui.zig) - UIライブラリ

## 貢献

バグ報告、機能リクエスト、プルリクエストを歓迎します。
