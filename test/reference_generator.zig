//! リファレンス画像ジェネレーター
//! 
//! 既知の正しいレンダリング結果を生成し、スナップショットテストの基準として使用する。

const std = @import("std");
const zapp = @import("zapp");
const zgf = @import("zgf");
const zpu = @import("zpu");

/// リファレンス画像を生成
pub fn generateReferenceImages() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("\n=== リファレンス画像生成 ===\n\n", .{});
    
    // バージョン情報
    const version = "0.2.0";
    const output_dir = try std.fmt.allocPrint(allocator, "snapshots/reference/{s}", .{version});
    defer allocator.free(output_dir);
    
    // ディレクトリ作成
    try std.fs.cwd().makePath(output_dir);
    
    // シナリオごとにリファレンス画像を生成
    const scenarios = [_]struct {
        name: []const u8,
        render_fn: *const fn (*zgf.Image) void,
    }{
        .{ .name = "basic_ui", .render_fn = &renderBasicUIReference },
        .{ .name = "interactive", .render_fn = &renderInteractiveReference },
        .{ .name = "animation", .render_fn = &renderAnimationReference },
    };
    
    for (scenarios) |scenario| {
        std.debug.print("シナリオ: {s}\n", .{scenario.name});
        
        // 画像作成
        var image = try zgf.Image.init(allocator, 800, 600, 4);
        defer image.deinit();
        
        // リファレンス画像をレンダリング
        scenario.render_fn(&image);
        
        // 保存
        const output_path = try std.fmt.allocPrint(allocator, "{s}/{s}.ppm", .{ output_dir, scenario.name });
        defer allocator.free(output_path);
        
        const ppm = try image.toPpm(allocator);
        defer allocator.free(ppm);
        
        const file = try std.fs.cwd().createFile(output_path, .{});
        defer file.close();
        try file.writeAll(ppm);
        
        std.debug.print("  ✓ 保存: {s}\n\n", .{output_path});
    }
    
    std.debug.print("=== リファレンス画像生成完了 ===\n", .{});
}

/// 基本的なUIのリファレンス画像
fn renderBasicUIReference(image: *zgf.Image) void {
    // 背景
    image.fill(zgf.Color{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 });
    
    // タイトル
    drawText(image, 20, 20, "Basic UI Test", zgf.Color{ .r = 1, .g = 1, .b = 1, .a = 1 });
    
    // ボタン1
    drawButton(image, 20, 60, 120, 40, "Button 1", zgf.Color{ .r = 0.3, .g = 0.5, .b = 0.8, .a = 1 });
    
    // ボタン2
    drawButton(image, 160, 60, 120, 40, "Button 2", zgf.Color{ .r = 0.3, .g = 0.5, .b = 0.8, .a = 1 });
    
    // スライダー
    drawSlider(image, 20, 120, 300, 20, 0.5);
    drawText(image, 20, 150, "Slider", zgf.Color{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1 });
    
    // チェックボックス
    drawCheckbox(image, 20, 180, true);
    drawText(image, 50, 185, "Checkbox", zgf.Color{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1 });
}

/// インタラクティブ要素のリファレンス画像
fn renderInteractiveReference(image: *zgf.Image) void {
    // 背景
    image.fill(zgf.Color{ .r = 0.1, .g = 0.1, .b = 0.15, .a = 1.0 });
    
    // タイトル
    drawText(image, 20, 20, "Interactive Test", zgf.Color{ .r = 1, .g = 1, .b = 1, .a = 1 });
    
    // テキスト入力フィールド
    drawTextInput(image, 20, 60, 300, 30, "Enter text...");
    drawText(image, 20, 100, "Text Input", zgf.Color{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1 });
    
    // ドロップダウン
    drawDropdown(image, 20, 130, 200, 30, "Option 1");
    drawText(image, 20, 170, "Dropdown", zgf.Color{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1 });
}

/// アニメーションのリファレンス画像
fn renderAnimationReference(image: *zgf.Image) void {
    // 背景
    image.fill(zgf.Color{ .r = 0.05, .g = 0.05, .b = 0.1, .a = 1.0 });
    
    // タイトル
    drawText(image, 20, 20, "Animation Test", zgf.Color{ .r = 1, .g = 1, .b = 1, .a = 1 });
    
    // プログレスバー（50%）
    drawProgressBar(image, 20, 60, 400, 30, 0.5);
    drawText(image, 20, 100, "Progress: 50%", zgf.Color{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1 });
}

// 描画ヘルパー関数

fn drawText(image: *zgf.Image, x: i32, y: i32, text: []const u8, color: zgf.Color) void {
    // 簡単なテキスト描画（実際のフォントレンダリングはZUIに任せる）
    var current_x = x;
    for (text) |char| {
        // 各文字を小さな矩形で表現
        if (char != ' ') {
            const char_width = 8;
            const char_height = 12;
            
            // 文字の簡単なビットマップパターン
            for (0..char_height) |dy| {
                for (0..char_width) |dx| {
                    // 簡易的な文字パターン（実際の実装ではフォントデータを使用）
                    if ((dx + dy) % 3 == 0) {
                        image.setPixel(current_x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), color);
                    }
                }
            }
            
            current_x += char_width + 2;
        } else {
            current_x += 6;
        }
    }
}

fn drawButton(image: *zgf.Image, x: i32, y: i32, width: i32, height: i32, label: []const u8, color: zgf.Color) void {
    // ボタン背景
    for (0..@as(usize, @intCast(height))) |dy| {
        for (0..@as(usize, @intCast(width))) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), color);
        }
    }
    
    // ボーダー
    const border_color = zgf.Color{ .r = color.r * 0.7, .g = color.g * 0.7, .b = color.b * 0.7, .a = 1.0 };
    for (0..@as(usize, @intCast(width))) |dx| {
        image.setPixel(x + @as(i32, @intCast(dx)), y, border_color);
        image.setPixel(x + @as(i32, @intCast(dx)), y + height - 1, border_color);
    }
    for (0..@as(usize, @intCast(height))) |dy| {
        image.setPixel(x, y + @as(i32, @intCast(dy)), border_color);
        image.setPixel(x + width - 1, y + @as(i32, @intCast(dy)), border_color);
    }
    
    // ラベル
    const text_x = x + (width - @as(i32, @intCast(label.len * 10))) / 2;
    const text_y = y + (height - 12) / 2;
    drawText(image, text_x, text_y, label, zgf.Color{ .r = 1, .g = 1, .b = 1, .a = 1 });
}

fn drawSlider(image: *zgf.Image, x: i32, y: i32, width: i32, height: i32, value: f32) void {
    // トラック
    const track_color = zgf.Color{ .r = 0.3, .g = 0.3, .b = 0.3, .a = 1.0 };
    for (0..@as(usize, @intCast(height))) |dy| {
        for (0..@as(usize, @intCast(width))) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), track_color);
        }
    }
    
    // ハンドル
    const handle_x = x + @as(i32, @intFromFloat(value * @as(f32, @floatFromInt(width - 20))));
    const handle_color = zgf.Color{ .r = 0.5, .g = 0.7, .b = 1.0, .a = 1.0 };
    for (0..20) |dy| {
        for (0..20) |dx| {
            image.setPixel(handle_x + @as(i32, @intCast(dx)), y - 5 + @as(i32, @intCast(dy)), handle_color);
        }
    }
}

fn drawCheckbox(image: *zgf.Image, x: i32, y: i32, checked: bool) void {
    const size = 20;
    const box_color = zgf.Color{ .r = 0.3, .g = 0.3, .b = 0.3, .a = 1.0 };
    
    // ボックス
    for (0..size) |dy| {
        for (0..size) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), box_color);
        }
    }
    
    // チェックマーク
    if (checked) {
        const check_color = zgf.Color{ .r = 0.5, .g = 0.8, .b = 0.5, .a = 1.0 };
        for (0..size) |dy| {
            for (0..size) |dx| {
                // 簡単なチェックマークパターン
                if ((dx == dy and dx >= 5 and dx <= 15) or
                    (dx + dy == size - 1 and dx >= 5 and dx <= 15)) {
                    image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), check_color);
                }
            }
        }
    }
}

fn drawTextInput(image: *zgf.Image, x: i32, y: i32, width: i32, height: i32, placeholder: []const u8) void {
    // 背景
    const bg_color = zgf.Color{ .r = 0.2, .g = 0.2, .b = 0.2, .a = 1.0 };
    for (0..@as(usize, @intCast(height))) |dy| {
        for (0..@as(usize, @intCast(width))) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), bg_color);
        }
    }
    
    // ボーダー
    const border_color = zgf.Color{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 };
    for (0..@as(usize, @intCast(width))) |dx| {
        image.setPixel(x + @as(i32, @intCast(dx)), y, border_color);
        image.setPixel(x + @as(i32, @intCast(dx)), y + height - 1, border_color);
    }
    for (0..@as(usize, @intCast(height))) |dy| {
        image.setPixel(x, y + @as(i32, @intCast(dy)), border_color);
        image.setPixel(x + width - 1, y + @as(i32, @intCast(dy)), border_color);
    }
    
    // プレースホルダー
    drawText(image, x + 5, y + (height - 12) / 2, placeholder, zgf.Color{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1 });
}

fn drawDropdown(image: *zgf.Image, x: i32, y: i32, width: i32, height: i32, selected: []const u8) void {
    // 背景
    const bg_color = zgf.Color{ .r = 0.2, .g = 0.2, .b = 0.2, .a = 1.0 };
    for (0..@as(usize, @intCast(height))) |dy| {
        for (0..@as(usize, @intCast(width))) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), bg_color);
        }
    }
    
    // ボーダー
    const border_color = zgf.Color{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 };
    for (0..@as(usize, @intCast(width))) |dx| {
        image.setPixel(x + @as(i32, @intCast(dx)), y, border_color);
        image.setPixel(x + @as(i32, @intCast(dx)), y + height - 1, border_color);
    }
    for (0..@as(usize, @intCast(height))) |dy| {
        image.setPixel(x, y + @as(i32, @intCast(dy)), border_color);
        image.setPixel(x + width - 1, y + @as(i32, @intCast(dy)), border_color);
    }
    
    // 選択されたテキスト
    drawText(image, x + 5, y + (height - 12) / 2, selected, zgf.Color{ .r = 1, .g = 1, .b = 1, .a = 1 });
    
    // ドロップダウン矢印
    const arrow_x = x + width - 20;
    const arrow_y = y + height / 2;
    const arrow_color = zgf.Color{ .r = 0.8, .g = 0.8, .b = 0.8, .a = 1 };
    for (0..5) |dy| {
        for (-dy..dy + 1) |dx| {
            image.setPixel(arrow_x + @as(i32, @intCast(dx)), arrow_y + @as(i32, @intCast(dy)), arrow_color);
        }
    }
}

fn drawProgressBar(image: *zgf.Image, x: i32, y: i32, width: i32, height: i32, progress: f32) void {
    // 背景
    const bg_color = zgf.Color{ .r = 0.2, .g = 0.2, .b = 0.2, .a = 1.0 };
    for (0..@as(usize, @intCast(height))) |dy| {
        for (0..@as(usize, @intCast(width))) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), bg_color);
        }
    }
    
    // プログレス
    const progress_width = @as(i32, @intFromFloat(progress * @as(f32, @floatFromInt(width))));
    const progress_color = zgf.Color{ .r = 0.3, .g = 0.7, .b = 0.3, .a = 1.0 };
    for (0..@as(usize, @intCast(height))) |dy| {
        for (0..@as(usize, @intCast(progress_width))) |dx| {
            image.setPixel(x + @as(i32, @intCast(dx)), y + @as(i32, @intCast(dy)), progress_color);
        }
    }
    
    // ボーダー
    const border_color = zgf.Color{ .r = 0.4, .g = 0.4, .b = 0.4, .a = 1.0 };
    for (0..@as(usize, @intCast(width))) |dx| {
        image.setPixel(x + @as(i32, @intCast(dx)), y, border_color);
        image.setPixel(x + @as(i32, @intCast(dx)), y + height - 1, border_color);
    }
    for (0..@as(usize, @intCast(height))) |dy| {
        image.setPixel(x, y + @as(i32, @intCast(dy)), border_color);
        image.setPixel(x + width - 1, y + @as(i32, @intCast(dy)), border_color);
    }
}

pub fn main() !void {
    try generateReferenceImages();
}
