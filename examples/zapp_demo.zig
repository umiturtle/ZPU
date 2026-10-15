//! ZAPP使用例 - 完全なアプリケーション
//!
//! この例は、ZAPPフレームワークを使用して、ウィンドウ、イベント、
//! UI、レンダリングを統合する方法を示します。
//!
//! 実行方法:
//! ```bash
//! zig build run-zapp-demo
//! ```

const std = @import("std");
const zapp = @import("zapp");
const zui = @import("zui");
const zgf = @import("zgf");
const zpu = @import("zpu");

/// アプリケーションの状態
const AppState = struct {
    ui: zapp.Ui,
    count: u32 = 0,
    slider_value: f32 = 0.5,
    checkbox_value: bool = false,
    text_buffer: [64]u8 = undefined,
    text_field: zui.TextField = undefined,
    show_dialog: bool = false,
    
    pub fn init() !AppState {
        var state = AppState{
            .ui = try zapp.Ui.init(.{}),
        };
        state.text_field = zui.TextField.init(&state.text_buffer);
        return state;
    }
    
    pub fn deinit(self: *AppState) void {
        self.ui.deinit();
    }
};

/// アプリケーションの更新ロジック
fn update(state: *AppState, app: *zapp.App) !void {
    const size = app.getSize();
    
    // UIフレームを描画
    try state.ui.frame(.{
        .width = size.width,
        .height = size.height,
    }, struct {
        fn frame(s: *AppState, ui: *zui.Ui) !void {
            // ルートコンテナ
            try ui.container("root", .{
                .w = .full,
                .h = .full,
                .pad = .all(20),
                .gap = 15,
                .bg = ui.theme.bg,
            }, struct {
                s: *AppState,
                
                pub fn run(self: @This(), ui: *zui.Ui) !void {
                    // タイトル
                    try ui.label("ZAPP Application Demo", .{
                        .color = ui.theme.text,
                    });
                    
                    try ui.separator();
                    
                    // カウンターセクション
                    try ui.label("Counter", .{ .color = ui.theme.text_dim });
                    try ui.container("counter", .{ .dir = .x, .gap = 10 }, struct {
                        s: *AppState,
                        
                        pub fn run(self: @This(), ui: *zui.Ui) !void {
                            if (try ui.button("Increment", .{})) {
                                self.s.count += 1;
                            }
                            
                            if (try ui.button("Decrement", .{})) {
                                if (self.s.count > 0) {
                                    self.s.count -= 1;
                                }
                            }
                            
                            if (try ui.button("Reset", .{})) {
                                self.s.count = 0;
                            }
                            
                            var buf: [32]u8 = undefined;
                            const text = std.fmt.bufPrint(&buf, "Count: {d}", .{self.s.count}) catch unreachable;
                            try ui.label(text, .{});
                        }
                    }{ .s = self.s });
                    
                    try ui.separator();
                    
                    // スライダーセクション
                    try ui.label("Slider", .{ .color = ui.theme.text_dim });
                    _ = try ui.slider("Value", &self.s.slider_value, .{
                        .min = 0,
                        .max = 1,
                    });
                    
                    // チェックボックス
                    _ = try ui.checkbox("Enable feature", &self.s.checkbox_value, .{});
                    
                    try ui.separator();
                    
                    // テキスト入力
                    try ui.label("Text Input", .{ .color = ui.theme.text_dim });
                    _ = try ui.textEdit("text", &self.s.text_field, .{
                        .placeholder = "Type here...",
                    });
                    
                    try ui.separator();
                    
                    // ダイアログ
                    if (try ui.button("Show Dialog", .{})) {
                        self.s.show_dialog = true;
                    }
                    
                    if (self.s.show_dialog) {
                        try ui.container("dialog", .{
                            .pad = .all(20),
                            .bg = ui.theme.surface,
                            .border = 2,
                            .border_color = ui.theme.border,
                            .radius = 8,
                        }, struct {
                            s: *AppState,
                            
                            pub fn run(self: @This(), ui: *zui.Ui) !void {
                                try ui.label("This is a dialog!", .{});
                                try ui.label("Click the button to close.", .{});
                                
                                if (try ui.button("Close", .{})) {
                                    self.s.show_dialog = false;
                                }
                            }
                        }{ .s = self.s });
                    }
                    
                    // 状態表示
                    try ui.separator();
                    try ui.label("State:", .{ .color = ui.theme.text_dim });
                    
                    var state_buf: [128]u8 = undefined;
                    const state_text = std.fmt.bufPrint(&state_buf, 
                        "Slider: {d:.2}, Checkbox: {}, Dialog: {}", 
                        .{ self.s.slider_value, self.s.checkbox_value, self.s.show_dialog }) catch unreachable;
                    try ui.label(state_text, .{ .color = ui.theme.text_dim });
                }
            }{ .s = self });
        }
    }.frame, state);
}

/// イベントハンドラ
fn onEvent(state: *AppState, app: *zapp.App, event: zapp.Event) !void {
    switch (event) {
        .key_press => |key| {
            if (key.key == .escape) {
                app.quit();
            }
        },
        .resize => |size| {
            std.debug.print("Window resized to {}x{}\n", .{ size.width, size.height });
        },
        else => {},
    }
    _ = state;
}

pub fn main() !void {
    std.debug.print("Starting ZAPP Demo Application...\n", .{});
    
    // アプリケーションを初期化
    var app = try zapp.App.init(.{
        .window = .{
            .title = "ZAPP Demo",
            .width = 800,
            .height = 600,
            .backend = .software, // ソフトウェアレンダリング
            .resizable = true,
        },
        .target_fps = 60,
        .debug = true,
    });
    defer app.deinit();
    
    // アプリケーション状態を初期化
    var state = try AppState.init();
    defer state.deinit();
    
    std.debug.print("Application initialized. Running main loop...\n", .{});
    
    // メインループを実行
    // 注意: ソフトウェアバックエンドはisOpen()が常にtrueを返すため、
    // 実際には無限ループになります。テスト用の有限ループに変更します。
    var frame_count: u32 = 0;
    const max_frames = 10; // デモ用に10フレームで終了
    
    while (app.running and frame_count < max_frames) {
        // イベント処理
        app.window.pollEvents();
        
        while (app.events.poll()) |event| {
            switch (event) {
                .close => {
                    app.quit();
                    break;
                },
                else => {
                    try onEvent(&state, &app, event);
                },
            }
        }
        
        if (!app.running) break;
        
        // 更新
        try update(&state, &app);
        
        // 表示
        app.window.present();
        
        frame_count += 1;
        
        if (frame_count % 100 == 0) {
            std.debug.print("Frame {d}\n", .{frame_count});
        }
    }
    
    std.debug.print("Application finished after {d} frames\n", .{frame_count});
    
    // フレームバッファをPPMとして保存（デモ用）
    const framebuffer = app.getFramebuffer();
    const ppm = try framebuffer.toPpm(std.heap.page_allocator);
    defer std.heap.page_allocator.free(ppm);
    
    const file = try std.fs.cwd().createFile("zig-out/zapp-demo.ppm", .{});
    defer file.close();
    try file.writeAll(ppm);
    
    std.debug.print("Saved framebuffer to zig-out/zapp-demo.ppm\n", .{});
}
