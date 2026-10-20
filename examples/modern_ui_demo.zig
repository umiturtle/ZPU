//! Modern UI Demo
//!
//! Material Design品質のモダンUIコンポーネントデモ
//! ベクトルベースで高品質、リッチなビジュアル

const std = @import("std");
const zgf = @import("zgf");
const zapp = @import("zapp");
const rich = zapp.rich_render;
const modern = zapp.modern_ui;
const Dir = std.Io.Dir;
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    std.debug.print("\n=== Modern UI Components Demo ===\n\n", .{});

    Dir.cwd().createDirPath(io, "snapshots/zapp/modern") catch {};

    // Material Design品質のデモ
    try demoModernComponents(io, alloc);

    // カード + イレベーションデモ
    try demoElevationSystem(io, alloc);

    // ボタンバリアント
    try demoButtonVariants(io, alloc);

    // フォームコンポーネント
    try demoFormComponents(io, alloc);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

fn demoModernComponents(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Creating: modern_components.ppm\n", .{});

    var fb = try zgf.Image.init(alloc, 800, 600, 4);
    defer fb.deinit();

    // 背景
    fb.fill(modern.MaterialColors.background);

    const render_config = rich.RenderConfig{
        .global_quality = .high,
        .shadows = true,
        .gradients = true,
        .antialiasing = true,
    };

    // ヘッダー
    const font = zapp.vector_text.SimpleFont.withScale(1.8);
    font.drawText(&fb, "Modern UI Components", 40, 30, modern.MaterialColors.on_background);

    // カード1: Primary
    modern.drawCard(&fb, .{
        .rect = .{ .x = 40, .y = 80, .w = 220, .h = 160 },
        .elevation = .level2,
        .radius = 16,
    }, render_config);
    font.drawText(&fb, "Primary Card", 60, 100, modern.MaterialColors.on_surface);
    font.drawText(&fb, "Elevation Level 2", 60, 130, modern.MaterialColors.on_surface_variant);

    // カード2: Secondary
    modern.drawCard(&fb, .{
        .rect = .{ .x = 290, .y = 80, .w = 220, .h = 160 },
        .elevation = .level3,
        .radius = 16,
        .bg = modern.MaterialColors.primary,
    }, render_config);
    font.drawText(&fb, "Primary Color", 310, 100, modern.MaterialColors.on_primary);
    font.drawText(&fb, "Elevation Level 3", 310, 130, .{ .r = 0.9, .g = 0.9, .b = 1.0, .a = 1.0 });

    // カード3: Outlined
    modern.drawCard(&fb, .{
        .rect = .{ .x = 540, .y = 80, .w = 220, .h = 160 },
        .elevation = .none,
        .radius = 16,
        .border_color = modern.MaterialColors.on_surface_variant,
        .border_width = 1.5,
    }, render_config);
    font.drawText(&fb, "Outlined Card", 560, 100, modern.MaterialColors.on_surface);
    font.drawText(&fb, "No Elevation", 560, 130, modern.MaterialColors.on_surface_variant);

    // スイッチ
    const switch_y: f32 = 300;
    font.drawText(&fb, "Switches:", 40, switch_y - 30, modern.MaterialColors.on_background);

    modern.drawSwitch(&fb, .{
        .x = 40,
        .y = switch_y,
        .is_on = true,
    }, render_config);

    modern.drawSwitch(&fb, .{
        .x = 120,
        .y = switch_y,
        .is_on = false,
    }, render_config);

    modern.drawSwitch(&fb, .{
        .x = 200,
        .y = switch_y,
        .is_on = true,
        .track_color_on = modern.MaterialColors.success,
    }, render_config);

    // プログレス
    const progress_y: f32 = 400;
    font.drawText(&fb, "Progress Indicators:", 40, progress_y - 30, modern.MaterialColors.on_background);

    zapp.widgets.progress_bar.renderRich(&fb, .{
        .rect = .{ .x = 40, .y = progress_y, .w = 300, .h = 8 },
        .value = 0.75,
        .radius = 4,
    }, render_config);

    zapp.widgets.progress_bar.renderRich(&fb, .{
        .rect = .{ .x = 40, .y = progress_y + 40, .w = 300, .h = 8 },
        .value = 0.45,
        .fg = modern.MaterialColors.secondary,
        .radius = 4,
    }, render_config);

    zapp.widgets.progress_bar.renderRich(&fb, .{
        .rect = .{ .x = 40, .y = progress_y + 80, .w = 300, .h = 8 },
        .value = 0.9,
        .fg = modern.MaterialColors.success,
        .radius = 4,
    }, render_config);

    // 保存
    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const f = try Dir.createFile(Dir.cwd(), io, "snapshots/zapp/modern/modern_components.ppm", .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ Saved modern_components.ppm\n\n", .{});
}

fn demoElevationSystem(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Creating: elevation_system.ppm\n", .{});

    var fb = try zgf.Image.init(alloc, 800, 400, 4);
    defer fb.deinit();

    fb.fill(modern.MaterialColors.background);

    const render_config = rich.RenderConfig{
        .global_quality = .ultra,
        .shadows = true,
        .gradients = false,
        .antialiasing = true,
    };

    const font = zapp.vector_text.SimpleFont.withScale(1.5);
    font.drawText(&fb, "Elevation System", 40, 30, modern.MaterialColors.on_background);

    // 5段階のイレベーション
    const levels = [_]modern.Elevation.Level{ .level1, .level2, .level3, .level4, .level5 };
    const names = [_][]const u8{ "Level 1", "Level 2", "Level 3", "Level 4", "Level 5" };

    for (levels, names, 0..) |level, name, i| {
        const x: f32 = 40 + @as(f32, @floatFromInt(i)) * 150;
        const y: f32 = 100;

        modern.drawCard(&fb, .{
            .rect = .{ .x = x, .y = y, .w = 120, .h = 120 },
            .elevation = level,
            .radius = 12,
        }, render_config);

        font.drawText(&fb, name, x + 20, y + 50, modern.MaterialColors.on_surface);
    }

    // 説明
    font.drawText(&fb, "Higher elevation = more prominent = more attention", 40, 280, modern.MaterialColors.on_surface_variant);
    font.drawText(&fb, "Use elevation to establish visual hierarchy", 40, 310, modern.MaterialColors.on_surface_variant);

    // 保存
    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const f = try Dir.createFile(Dir.cwd(), io, "snapshots/zapp/modern/elevation_system.ppm", .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ Saved elevation_system.ppm\n\n", .{});
}

fn demoButtonVariants(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Creating: button_variants.ppm\n", .{});

    var fb = try zgf.Image.init(alloc, 800, 500, 4);
    defer fb.deinit();

    fb.fill(modern.MaterialColors.background);

    const render_config = rich.RenderConfig{
        .global_quality = .high,
        .shadows = true,
        .gradients = true,
        .antialiasing = true,
    };

    const font = zapp.vector_text.SimpleFont.withScale(1.5);
    font.drawText(&fb, "Button Variants", 40, 30, modern.MaterialColors.on_background);

    const variants = [_]modern.ButtonVariant{ .filled, .elevated, .outlined, .text, .tonal };
    const names = [_][]const u8{ "Filled", "Elevated", "Outlined", "Text", "Tonal" };

    // 各バリアントの通常状態
    for (variants, names, 0..) |variant, name, i| {
        const y: f32 = 100 + @as(f32, @floatFromInt(i)) * 70;

        modern.drawModernButton(&fb, .{
            .rect = .{ .x = 40, .y = y, .w = 140, .h = 48 },
            .text = name,
            .variant = variant,
            .color = modern.MaterialColors.primary,
        }, render_config);

        font.drawText(&fb, name, 200, y + 15, modern.MaterialColors.on_background);
    }

    // 色のバリエーション
    font.drawText(&fb, "Color Variations:", 400, 80, modern.MaterialColors.on_background);

    const colors = [_]zgf.Color{
        modern.MaterialColors.primary,
        modern.MaterialColors.secondary,
        modern.MaterialColors.success,
        modern.MaterialColors.error_color,
        modern.MaterialColors.warning,
    };
    const color_names = [_][]const u8{ "Primary", "Secondary", "Success", "Error", "Warning" };

    for (colors, color_names, 0..) |color, name, i| {
        const x: f32 = 400;
        const y: f32 = 120 + @as(f32, @floatFromInt(i)) * 70;

        modern.drawModernButton(&fb, .{
            .rect = .{ .x = x, .y = y, .w = 140, .h = 48 },
            .text = name,
            .variant = .filled,
            .color = color,
        }, render_config);
    }

    // 保存
    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const f = try Dir.createFile(Dir.cwd(), io, "snapshots/zapp/modern/button_variants.ppm", .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ Saved button_variants.ppm\n\n", .{});
}

fn demoFormComponents(io: Io, alloc: std.mem.Allocator) !void {
    std.debug.print("Creating: form_components.ppm\n", .{});

    var fb = try zgf.Image.init(alloc, 800, 600, 4);
    defer fb.deinit();

    fb.fill(modern.MaterialColors.background);

    const render_config = rich.RenderConfig{
        .global_quality = .high,
        .shadows = true,
        .gradients = true,
        .antialiasing = true,
    };

    const font = zapp.vector_text.SimpleFont.withScale(1.5);
    font.drawText(&fb, "Form Components", 40, 30, modern.MaterialColors.on_background);

    // テキストフィールド
    modern.drawTextField(&fb, .{
        .rect = .{ .x = 40, .y = 80, .w = 350, .h = 56 },
        .label = "Email",
        .value = "user@example.com",
        .state = .normal,
    }, render_config);
    font.drawText(&fb, "Normal State", 410, 100, modern.MaterialColors.on_surface_variant);

    modern.drawTextField(&fb, .{
        .rect = .{ .x = 40, .y = 160, .w = 350, .h = 56 },
        .label = "Password",
        .value = "••••••••",
        .state = .focused,
    }, render_config);
    font.drawText(&fb, "Focused State", 410, 180, modern.MaterialColors.on_surface_variant);

    modern.drawTextField(&fb, .{
        .rect = .{ .x = 40, .y = 240, .w = 350, .h = 56 },
        .label = "Invalid Email",
        .value = "invalid-email",
        .state = .error_state,
        .has_error = true,
        .helper_text = "Please enter a valid email address",
    }, render_config);
    font.drawText(&fb, "Error State", 410, 260, modern.MaterialColors.error_color);

    // スライダー
    font.drawText(&fb, "Sliders:", 40, 340, modern.MaterialColors.on_background);

    zapp.widgets.slider.renderRich(&fb, .{
        .rect = .{ .x = 40, .y = 370, .w = 300, .h = 40 },
        .value = 0.65,
    }, render_config);
    font.drawText(&fb, "65%", 360, 385, modern.MaterialColors.on_surface_variant);

    zapp.widgets.slider.renderRich(&fb, .{
        .rect = .{ .x = 40, .y = 430, .w = 300, .h = 40 },
        .value = 0.3,
        .fill_color = modern.MaterialColors.secondary,
        .handle_color = modern.MaterialColors.secondary,
    }, render_config);
    font.drawText(&fb, "30%", 360, 445, modern.MaterialColors.on_surface_variant);

    // 保存
    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const f = try Dir.createFile(Dir.cwd(), io, "snapshots/zapp/modern/form_components.ppm", .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ Saved form_components.ppm\n\n", .{});
}
