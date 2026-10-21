// 開発者向けツール: インスペクタ、プロファイラ、デバッグユーティリティ
const std = @import("std");
const zgf = @import("zgf");

/// UIインスペクタ
/// ウィジェットのプロパティ、状態、レイアウトを可視化
pub const Inspector = struct {
    enabled: bool = false,
    selected_widget: ?usize = null,
    overlay_color: zgf.Color = .{ .r = 0.0, .g = 1.0, .b = 0.0, .a = 0.3 },
    
    /// ウィジェット情報を収集
    pub const WidgetInfo = struct {
        id: usize,
        name: []const u8,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        visible: bool,
        properties: []const Property,
    };
    
    pub const Property = struct {
        name: []const u8,
        value_type: ValueType,
        value: Value,
    };
    
    pub const ValueType = enum {
        string,
        number,
        boolean,
    };
    
    pub const Value = union {
        string: []const u8,
        number: f64,
        boolean: bool,
    };
    
    pub fn init() Inspector {
        return .{};
    }
    
    /// インスペクタを有効化
    pub fn enable(self: *Inspector) void {
        self.enabled = true;
    }
    
    /// インスペクタを無効化
    pub fn disable(self: *Inspector) void {
        self.enabled = false;
    }
    
    /// ウィジェットを選択
    pub fn selectWidget(self: *Inspector, widget_id: usize) void {
        self.selected_widget = widget_id;
    }
    
    /// 選択を解除
    pub fn deselectWidget(self: *Inspector) void {
        self.selected_widget = null;
    }
    
    /// ウィジェットをオーバーレイ描画
    pub fn drawOverlay(
        self: *Inspector,
        fb: *zgf.Image,
        widgets: []const WidgetInfo,
    ) void {
        if (!self.enabled) return;
        
        for (widgets) |widget| {
            // 境界を描画
            self.drawWidgetBounds(fb, widget);
            
            // 選択されている場合は追加情報を表示
            if (self.selected_widget) |selected| {
                if (selected == widget.id) {
                    self.drawSelectionHighlight(fb, widget);
                }
            }
        }
    }
    
    fn drawWidgetBounds(self: *Inspector, fb: *zgf.Image, widget: WidgetInfo) void {
        const color = self.overlay_color;
        
        // 四隅にマーカーを描画
        const marker_size: f32 = 5.0;
        
        // 左上
        fb.drawLine( widget.x, widget.y, widget.x + marker_size, widget.y, color);
        fb.drawLine( widget.x, widget.y, widget.x, widget.y + marker_size, color);
        
        // 右上
        fb.drawLine( widget.x + widget.width - marker_size, widget.y, widget.x + widget.width, widget.y, color);
        fb.drawLine( widget.x + widget.width, widget.y, widget.x + widget.width, widget.y + marker_size, color);
        
        // 左下
        fb.drawLine( widget.x, widget.y + widget.height, widget.x + marker_size, widget.y + widget.height, color);
        fb.drawLine( widget.x, widget.y + widget.height - marker_size, widget.x, widget.y + widget.height, color);
        
        // 右下
        fb.drawLine( widget.x + widget.width - marker_size, widget.y + widget.height, widget.x + widget.width, widget.y + widget.height, color);
        fb.drawLine( widget.x + widget.width, widget.y + widget.height - marker_size, widget.x + widget.width, widget.y + widget.height, color);
    }
    
    fn drawSelectionHighlight(self: *Inspector, fb: *zgf.Image, widget: WidgetInfo) void {
        _ = self;
        const highlight_color: zgf.Color = .{ .r = 1.0, .g = 1.0, .b = 0.0, .a = 0.5 };
        
        // 選択されたウィジェットの境界を強調
        const x: i32 = @intFromFloat(widget.x);
        const y: i32 = @intFromFloat(widget.y);
        const w: i32 = @intFromFloat(widget.width);
        const h: i32 = @intFromFloat(widget.height);
        
        // 上下の線
        var i: i32 = 0;
        while (i < w) : (i += 1) {
            fb.setPixel(x + i, y, highlight_color);
            fb.setPixel(x + i, y + h - 1, highlight_color);
        }
        
        // 左右の線
        i = 0;
        while (i < h) : (i += 1) {
            fb.setPixel(x, y + i, highlight_color);
            fb.setPixel(x + w - 1, y + i, highlight_color);
        }
    }
};

/// パフォーマンスプロファイラ
/// フレーム時間、描画呼び出し、メモリ使用量を計測
pub const Profiler = struct {
    frame_times: []f64,
    frame_count: usize = 0,
    draw_calls: u32 = 0,
    memory_allocated: usize = 0,
    max_samples: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, max_samples: usize) Profiler {
        return .{
            .frame_times = allocator.alloc(f64, max_samples) catch &[_]f64{},
            .frame_count = 0,
            .draw_calls = 0,
            .max_samples = max_samples,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Profiler) void {
        if (self.frame_times.len > 0) {
            self.allocator.free(self.frame_times);
        }
    }
    
    /// フレーム時間を記録
    pub fn recordFrameTime(self: *Profiler, time_ms: f64) void {
        if (self.frame_count < self.max_samples) {
            self.frame_times[self.frame_count] = time_ms;
            self.frame_count += 1;
        }
    }
    
    /// 描画呼び出しをインクリメント
    pub fn incrementDrawCalls(self: *Profiler) void {
        self.draw_calls += 1;
    }
    
    /// フレーム終了時に呼び出し
    pub fn endFrame(self: *Profiler) void {
        self.draw_calls = 0;
    }
    
    /// 平均FPSを取得
    pub fn getAverageFPS(self: *Profiler) f64 {
        if (self.frame_count == 0) return 0;
        
        var sum: f64 = 0;
        for (self.frame_times[0..self.frame_count]) |time| {
            sum += time;
        }
        
        const avg_frame_time = sum / @as(f64, @floatFromInt(self.frame_count));
        return 1000.0 / avg_frame_time;
    }
    
    /// 平均フレーム時間を取得
    pub fn getAverageFrameTime(self: *Profiler) f64 {
        if (self.frame_count == 0) return 0;
        
        var sum: f64 = 0;
        for (self.frame_times[0..self.frame_count]) |time| {
            sum += time;
        }
        
        return sum / @as(f64, @floatFromInt(self.frame_count));
    }
    
    /// プロファイラ情報を文字列で取得
    pub fn getStats(self: *Profiler, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "FPS: {d:.1}, Frame: {d:.2}ms, DrawCalls: {d}", .{
            self.getAverageFPS(),
            self.getAverageFrameTime(),
            self.draw_calls,
        });
    }
};

/// デバッグレンダラー
/// デバッグ情報をフレームバッファに描画
pub const DebugRenderer = struct {
    show_bounds: bool = false,
    show_fps: bool = false,
    show_grid: bool = false,
    grid_size: f32 = 50.0,
    grid_color: zgf.Color = .{ .r = 0.3, .g = 0.3, .b = 0.3, .a = 0.5 },
    
    pub fn init() DebugRenderer {
        return .{};
    }
    
    /// グリッドを描画
    pub fn drawGrid(self: *DebugRenderer, fb: *zgf.Image) void {
        if (!self.show_grid) return;
        
        const width = fb.width();
        const height = fb.height();
        
        // 垂直線
        var x: f32 = 0;
        while (x < @as(f32, @floatFromInt(width))) : (x += self.grid_size) {
            fb.drawLine( x, 0, x, @floatFromInt(height), self.grid_color);
        }
        
        // 水平線
        var y: f32 = 0;
        while (y < @as(f32, @floatFromInt(height))) : (y += self.grid_size) {
            fb.drawLine( 0, y, @floatFromInt(width), y, self.grid_color);
        }
    }
    
    /// トグル表示
    pub fn toggleBounds(self: *DebugRenderer) void {
        self.show_bounds = !self.show_bounds;
    }
    
    pub fn toggleFPS(self: *DebugRenderer) void {
        self.show_fps = !self.show_fps;
    }
    
    pub fn toggleGrid(self: *DebugRenderer) void {
        self.show_grid = !self.show_grid;
    }
};

// ============================================================================
// テスト
// ============================================================================

test "dev_tools - Inspector init" {
    const inspector = Inspector.init();
    try std.testing.expect(!inspector.enabled);
    try std.testing.expect(inspector.selected_widget == null);
}

test "dev_tools - Inspector enable/disable" {
    var inspector = Inspector.init();
    
    inspector.enable();
    try std.testing.expect(inspector.enabled);
    
    inspector.disable();
    try std.testing.expect(!inspector.enabled);
}

test "dev_tools - Inspector select widget" {
    var inspector = Inspector.init();
    
    inspector.selectWidget(42);
    try std.testing.expectEqual(@as(usize, 42), inspector.selected_widget.?);
    
    inspector.deselectWidget();
    try std.testing.expect(inspector.selected_widget == null);
}

test "dev_tools - Profiler init and deinit" {
    const allocator = std.testing.allocator;
    var profiler = Profiler.init(allocator, 100);
    defer profiler.deinit();
    
    try std.testing.expectEqual(@as(u32, 0), profiler.draw_calls);
}

test "dev_tools - Profiler record frame time" {
    const allocator = std.testing.allocator;
    var profiler = Profiler.init(allocator, 100);
    defer profiler.deinit();
    
    profiler.recordFrameTime(16.67);
    profiler.recordFrameTime(16.67);
    profiler.recordFrameTime(16.67);
    
    try std.testing.expectApproxEqAbs(@as(f64, 60.0), profiler.getAverageFPS(), 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 16.67), profiler.getAverageFrameTime(), 0.01);
}

test "dev_tools - Profiler increment draw calls" {
    const allocator = std.testing.allocator;
    var profiler = Profiler.init(allocator, 100);
    defer profiler.deinit();
    
    profiler.incrementDrawCalls();
    profiler.incrementDrawCalls();
    profiler.incrementDrawCalls();
    
    try std.testing.expectEqual(@as(u32, 3), profiler.draw_calls);
    
    profiler.endFrame();
    try std.testing.expectEqual(@as(u32, 0), profiler.draw_calls);
}

test "dev_tools - DebugRenderer init" {
    const renderer = DebugRenderer.init();
    try std.testing.expect(!renderer.show_bounds);
    try std.testing.expect(!renderer.show_fps);
    try std.testing.expect(!renderer.show_grid);
}

test "dev_tools - DebugRenderer toggle" {
    var renderer = DebugRenderer.init();
    
    renderer.toggleBounds();
    try std.testing.expect(renderer.show_bounds);
    
    renderer.toggleFPS();
    try std.testing.expect(renderer.show_fps);
    
    renderer.toggleGrid();
    try std.testing.expect(renderer.show_grid);
}
