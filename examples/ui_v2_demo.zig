//! ZAPP 宣言的UI v2 デモ
//!
//! 新しいビルダーパターンのDSLを使ったUIデモ

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");
const ui = zapp.ui;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    
    var fb = try zgf.Image.init(alloc, 800, 600, 4);
    defer fb.deinit();
    
    // 状態
    var enabled = true;
    var volume: f32 = 0.7;
    
    // 新しい宣言的UI v2 でUIを構築
    const root = ui.vstack()
        .withSpacing(.base)
        .withPadding(.xl)
        .bg(.{ .r = 0.15, .g = 0.15, .b = 0.2, .a = 1.0 })
        .children(&.{
            // タイトル
            ui.text("ZAPP UI v2")
                .font(.@"2xl")
                .color(.{ .r = 0.9, .g = 0.9, .b = 1.0 })
                .build(),
            
            ui.divider(),
            
            // ボタン群
            ui.hstack()
                .withSpacing(.sm)
                .children(&.{
                    ui.button("Primary")
                        .variant(.primary)
                        .build(),
                    ui.button("Secondary")
                        .variant(.secondary)
                        .build(),
                    ui.button("Outlined")
                        .variant(.outlined)
                        .build(),
                    ui.button("Danger")
                        .variant(.danger)
                        .build(),
                })
                .build(),
            
            ui.spacer(16),
            
            // コントロール
            ui.vstack()
                .withSpacing(.sm)
                .children(&.{
                    ui.checkbox("Enable feature", &enabled),
                    ui.slider("Volume", &volume, .{ .min = 0, .max = 1 }),
                    ui.progress(0.75),
                })
                .build(),
            
            ui.spacer(16),
            
            // テキスト
            ui.vstack()
                .withSpacing(.xs)
                .children(&.{
                    ui.text("Large text").font(.lg).build(),
                    ui.text("Base text").font(.base).build(),
                    ui.text("Small text").font(.sm).color(.{ .r = 0.6, .g = 0.6, .b = 0.6 }).build(),
                })
                .build(),
        })
        .build();
    
    // ダークテーマでレンダリング
    ui.render(&fb, root, .dark);
    
    std.debug.print("\n=== UI v2 Demo Complete ===\n", .{});
    std.debug.print("✓ Rendered 800x600 UI with builder pattern DSL\n", .{});
}
