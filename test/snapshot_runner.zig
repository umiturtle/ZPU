//! スナップショットテストランナー
//! 
//! 両環境（Linux X11, Web）でZAPPを実行し、スクリーンショットを比較して
//! 一貫性を検証する。

const std = @import("std");
const zapp = @import("zapp");
const zgf = @import("zgf");
const zpu = @import("zpu");

const Screenshot = struct {
    image: zgf.Image,
    platform: []const u8,
    version: []const u8,
    timestamp: i64,
    
    pub fn save(self: *const Screenshot, path: []const u8) !void {
        const ppm = try self.image.toPpm(std.heap.page_allocator);
        defer std.heap.page_allocator.free(ppm);
        
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(ppm);
    }
};

const ComparisonResult = struct {
    pixel_diff_count: usize,
    max_pixel_diff: f32,
    mean_pixel_diff: f32,
    structural_similarity: f32,
    passed: bool,
    
    pub fn print(self: *const ComparisonResult) void {
        std.debug.print("=== 比較結果 ===\n", .{});
        std.debug.print("  差分ピクセル数: {d}\n", .{self.pixel_diff_count});
        std.debug.print("  最大ピクセル差分: {d:.4}\n", .{self.max_pixel_diff});
        std.debug.print("  平均ピクセル差分: {d:.4}\n", .{self.mean_pixel_diff});
        std.debug.print("  構造的類似度: {d:.4}\n", .{self.structural_similarity});
        std.debug.print("  結果: {}\n", .{if (self.passed) "✓ PASS" else "✗ FAIL"});
    }
};

/// 2つのスクリーンショットを比較
pub fn compareScreenshots(a: *const Screenshot, b: *const Screenshot) ComparisonResult {
    if (a.image.width() != b.image.width() or a.image.height() != b.image.height()) {
        return .{
            .pixel_diff_count = 0,
            .max_pixel_diff = 1.0,
            .mean_pixel_diff = 1.0,
            .structural_similarity = 0.0,
            .passed = false,
        };
    }
    
    const width = a.image.width();
    const height = a.image.height();
    
    var pixel_diff_count: usize = 0;
    var max_pixel_diff: f32 = 0.0;
    var total_diff: f64 = 0.0;
    var matching_pixels: usize = 0;
    
    const tolerance: f32 = 0.01; // 1%の差分は許容
    
    for (0..height) |y| {
        for (0..width) |x| {
            const pixel_a = a.image.getPixel(x, y);
            const pixel_b = b.image.getPixel(x, y);
            
            const diff_r = @abs(pixel_a.r - pixel_b.r);
            const diff_g = @abs(pixel_a.g - pixel_b.g);
            const diff_b = @abs(pixel_a.b - pixel_b.b);
            const diff_a = @abs(pixel_a.a - pixel_b.a);
            
            const max_diff = @max(@max(diff_r, diff_g), @max(diff_b, diff_a));
            const avg_diff = (diff_r + diff_g + diff_b + diff_a) / 4.0;
            
            if (max_diff > tolerance) {
                pixel_diff_count += 1;
            } else {
                matching_pixels += 1;
            }
            
            if (max_diff > max_pixel_diff) {
                max_pixel_diff = max_diff;
            }
            
            total_diff += avg_diff;
        }
    }
    
    const total_pixels = width * height;
    const mean_pixel_diff = @as(f32, @floatCast(total_diff / @as(f64, @floatFromInt(total_pixels))));
    const structural_similarity = @as(f32, @floatCast(@as(f64, @floatFromInt(matching_pixels)) / @as(f64, @floatFromInt(total_pixels))));
    
    // 合格判定: 95%以上のピクセルが一致し、平均差分が1%未満
    const passed = structural_similarity > 0.95 and mean_pixel_diff < 0.01;
    
    return .{
        .pixel_diff_count = pixel_diff_count,
        .max_pixel_diff = max_pixel_diff,
        .mean_pixel_diff = mean_pixel_diff,
        .structural_similarity = structural_similarity,
        .passed = passed,
    };
}

/// スナップショットディレクトリを管理
pub const SnapshotManager = struct {
    base_dir: []const u8,
    
    pub fn init(base_dir: []const u8) SnapshotManager {
        return .{ .base_dir = base_dir };
    }
    
    pub fn getSnapshotPath(self: *const SnapshotManager, platform: []const u8, version: []const u8, name: []const u8) ![]const u8 {
        const allocator = std.heap.page_allocator;
        return std.fmt.allocPrint(allocator, "{s}/{s}/{s}/{s}.ppm", .{
            self.base_dir, platform, version, name,
        });
    }
    
    pub fn saveScreenshot(self: *const SnapshotManager, platform: []const u8, version: []const u8, name: []const u8, image: *const zgf.Image) !void {
        const path = try self.getSnapshotPath(platform, version, name);
        defer std.heap.page_allocator.free(path);
        
        // ディレクトリを作成
        const dir_path = std.fs.path.dirname(path) orelse return error.InvalidPath;
        try std.fs.cwd().makePath(dir_path);
        
        // PPM形式で保存
        const ppm = try image.toPpm(std.heap.page_allocator);
        defer std.heap.page_allocator.free(ppm);
        
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(ppm);
        
        std.debug.print("✓ スナップショット保存: {s}\n", .{path});
    }
    
    pub fn loadScreenshot(self: *const SnapshotManager, platform: []const u8, version: []const u8, name: []const u8) !zgf.Image {
        const path = try self.getSnapshotPath(platform, version, name);
        defer std.heap.page_allocator.free(path);
        
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const stat = try file.stat();
        const buffer = try std.heap.page_allocator.alloc(u8, stat.size);
        defer std.heap.page_allocator.free(buffer);
        
        _ = try file.readAll(buffer);
        
        // PPM形式から読み込み
        return zgf.Image.fromPpm(std.heap.page_allocator, buffer) catch |err| {
            std.debug.print("✗ スナップショット読み込み失敗: {s} (エラー: {})\n", .{ path, err });
            return err;
        };
    }
};

/// テストシナリオ
pub const TestScenario = struct {
    name: []const u8,
    description: []const u8,
    setup_fn: *const fn (*zapp.App) anyerror!void,
    render_fn: *const fn (*zapp.App) anyerror!void,
    duration_frames: u32,
};

/// スナップショットテストスイート
pub fn runSnapshotTests() !void {
    const allocator = std.heap.page_allocator;
    const manager = SnapshotManager.init("snapshots");
    
    std.debug.print("\n=== ZAPP スナップショットテスト ===\n\n", .{});
    
    // プラットフォーム情報
    const platform = if (@import("builtin").os.tag == .linux) "linux_x11" else "web";
    const version = "0.2.0";
    
    std.debug.print("プラットフォーム: {s}\n", .{platform});
    std.debug.print("バージョン: {s}\n\n", .{version});
    
    // テストシナリオ
    const scenarios = [_]TestScenario{
        .{
            .name = "basic_ui",
            .description = "基本的なUI要素の描画",
            .setup_fn = setupBasicUI,
            .render_fn = renderBasicUI,
            .duration_frames = 10,
        },
        .{
            .name = "interactive",
            .description = "インタラクティブ要素",
            .setup_fn = setupInteractive,
            .render_fn = renderInteractive,
            .duration_frames = 30,
        },
        .{
            .name = "animation",
            .description = "アニメーション効果",
            .setup_fn = setupAnimation,
            .render_fn = renderAnimation,
            .duration_frames = 60,
        },
    };
    
    var all_passed = true;
    
    for (scenarios) |scenario| {
        std.debug.print("\n--- シナリオ: {s} ---\n", .{scenario.name});
        std.debug.print("説明: {s}\n", .{scenario.description});
        
        // アプリケーション初期化
        var app = try zapp.App.init(.{
            .window = .{
                .title = scenario.name,
                .width = 800,
                .height = 600,
                .backend = .auto,
            },
        });
        defer app.deinit();
        
        // セットアップ
        try scenario.setup_fn(&app);
        
        // 指定フレーム数レンダリング
        var frame_count: u32 = 0;
        while (frame_count < scenario.duration_frames and app.running) {
            app.window.pollEvents();
            try scenario.render_fn(&app);
            app.window.present();
            frame_count += 1;
        }
        
        // 最終フレームのスクリーンショットを取得
        const framebuffer = app.getFramebuffer();
        
        // 参照スナップショットが存在するか確認
        const ref_path = try manager.getSnapshotPath("reference", version, scenario.name);
        defer allocator.free(ref_path);
        
        if (std.fs.cwd().access(ref_path, .{})) |_| {
            // 参照スナップショットが存在する場合、比較
            std.debug.print("\n参照スナップショットと比較中...\n", .{});
            
            var ref_image = try manager.loadScreenshot("reference", version, scenario.name);
            defer ref_image.deinit();
            
            const ref_screenshot = Screenshot{
                .image = ref_image,
                .platform = "reference",
                .version = version,
                .timestamp = 0,
            };
            
            const current_screenshot = Screenshot{
                .image = framebuffer.*,
                .platform = platform,
                .version = version,
                .timestamp = std.time.timestamp(),
            };
            
            const result = compareScreenshots(&ref_screenshot, &current_screenshot);
            result.print();
            
            if (!result.passed) {
                all_passed = false;
                
                // 差分画像を保存
                const diff_path = try std.fmt.allocPrint(allocator, "snapshots/diffs/{s}_{s}_diff.ppm", .{ scenario.name, platform });
                defer allocator.free(diff_path);
                
                // TODO: 差分画像を生成して保存
                std.debug.print("✗ 差分画像: {s}\n", .{diff_path});
            }
        } else |err| {
            // 参照スナップショットが存在しない場合、新規保存
            if (err == error.FileNotFound) {
                std.debug.print("\n参照スナップショットが存在しません。新規保存します...\n", .{});
                try manager.saveScreenshot(platform, version, scenario.name, framebuffer);
            } else {
                return err;
            }
        }
    }
    
    std.debug.print("\n=== テスト結果 ===\n", .{});
    if (all_passed) {
        std.debug.print("✓ すべてのテストがパスしました\n", .{});
    } else {
        std.debug.print("✗ 一部のテストが失敗しました\n", .{});
        return error.TestFailed;
    }
}

// テストシナリオの実装
fn setupBasicUI(app: *zapp.App) !void {
    _ = app;
    // 基本的なUI要素のセットアップ
}

fn renderBasicUI(app: *zapp.App) !void {
    const ui = &app.ui;
    const size = app.window.getSize();
    
    try ui.frame(.{
        .width = size.width,
        .height = size.height,
    }, struct {
        fn frame(_: void, ui: *zapp.Ui) !void {
            try ui.label("Basic UI Test", .{});
            
            _ = try ui.button("Button 1", .{});
            _ = try ui.button("Button 2", .{});
            
            var value: f32 = 0.5;
            _ = try ui.slider("Slider", &value, .{ .min = 0, .max = 1 });
            
            var checked = true;
            _ = try ui.checkbox("Checkbox", &checked, .{});
        }
    }.frame, {});
}

fn setupInteractive(app: *zapp.App) !void {
    _ = app;
    // インタラクティブ要素のセットアップ
}

fn renderInteractive(app: *zapp.App) !void {
    const ui = &app.ui;
    const size = app.window.getSize();
    
    try ui.frame(.{
        .width = size.width,
        .height = size.height,
    }, struct {
        fn frame(_: void, ui: *zapp.Ui) !void {
            try ui.label("Interactive Test", .{});
            
            // テキスト入力
            var text_buffer: [256]u8 = undefined;
            var text_field = zui.TextField.init(&text_buffer);
            _ = try ui.textInput("Text Input", &text_field, .{});
            
            // ドロップダウン
            _ = try ui.dropdown("Dropdown", &[_][]const u8{ "Option 1", "Option 2", "Option 3" }, &@as(u32, 0), .{});
        }
    }.frame, {});
}

fn setupAnimation(app: *zapp.App) !void {
    _ = app;
    // アニメーションのセットアップ
}

fn renderAnimation(app: *zapp.App) !void {
    const ui = &app.ui;
    const size = app.window.getSize();
    
    try ui.frame(.{
        .width = size.width,
        .height = size.height,
    }, struct {
        fn frame(_: void, ui: *zapp.Ui) !void {
            try ui.label("Animation Test", .{});
            
            // プログレスバー
            const progress = @as(f32, @floatFromInt(ui.frame_count % 100)) / 100.0;
            try ui.progressBar(progress, .{});
        }
    }.frame, {});
}

pub fn main() !void {
    try runSnapshotTests();
}
