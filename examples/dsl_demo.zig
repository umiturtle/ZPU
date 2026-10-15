//! 宣言的UI DSLのデモ
//! `zig build run-dsl-demo` で実行し、snapshots/zapp/dsl/ にスクリーンショットを保存する。

const std = @import("std");
const zapp = @import("zapp");
const zgf = @import("zgf");
const dsl = zapp.dsl;

const Dir = std.Io.Dir;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    std.debug.print("\n=== ZAPP 宣言的UI DSL デモ ===\n\n", .{});

    Dir.cwd().createDirPath(io, "snapshots/zapp/dsl") catch {};

    // シナリオ1: 基本的なフォーム
    try renderAndSave(io, alloc, "form", 400, 300, dsl.vstack(.{ .spacing = 8, .padding = 16 }, &.{
        dsl.label("User Registration"),
        dsl.separator(),
        dsl.textInput("Name"),
        dsl.textInput("Email"),
        dsl.slider("Age", 0.35),
        dsl.checkbox("Agree to Terms", true),
        dsl.button("Submit"),
    }));

    // シナリオ2: ダッシュボード
    try renderAndSave(io, alloc, "dashboard", 600, 400, dsl.vstack(.{ .spacing = 8, .padding = 16 }, &.{
        dsl.label("Dashboard"),
        dsl.separator(),
        dsl.hstack(.{ .spacing = 8 }, &.{
            dsl.button("Overview"),
            dsl.button("Analytics"),
            dsl.button("Settings"),
        }),
        dsl.label("CPU Usage"),
        dsl.progressBar(0.72),
        dsl.label("Memory"),
        dsl.progressBar(0.45),
        dsl.label("Disk"),
        dsl.progressBar(0.88),
        dsl.separator(),
        dsl.hstack(.{ .spacing = 8 }, &.{
            dsl.checkbox("Auto-refresh", true),
            dsl.checkbox("Dark Mode", true),
            dsl.checkbox("Notifications", false),
        }),
        dsl.slider("Update Interval", 0.5),
    }));

    // シナリオ3: 各種ウィジェット
    try renderAndSave(io, alloc, "widgets", 400, 400, dsl.vstack(.{ .spacing = 6, .padding = 12 }, &.{
        dsl.label("Widget Gallery"),
        dsl.separator(),
        dsl.label("Labels and Text"),
        dsl.textInput("Search..."),
        dsl.label("Buttons"),
        dsl.hstack(.{ .spacing = 6 }, &.{
            dsl.button("OK"),
            dsl.button("Cancel"),
            dsl.button("Apply"),
        }),
        dsl.label("Sliders"),
        dsl.slider("Red", 0.8),
        dsl.slider("Green", 0.4),
        dsl.slider("Blue", 0.6),
        dsl.label("Checkboxes"),
        dsl.checkbox("Feature A", true),
        dsl.checkbox("Feature B", false),
        dsl.checkbox("Feature C", true),
        dsl.label("Progress"),
        dsl.progressBar(0.33),
        dsl.progressBar(0.66),
        dsl.progressBar(1.0),
    }));

    std.debug.print("\n=== 完了: snapshots/zapp/dsl/ に保存しました ===\n", .{});
}

fn renderAndSave(io: std.Io, alloc: std.mem.Allocator, name: []const u8, width: u32, height: u32, tree: dsl.Node) !void {
    var fb = try zgf.Image.init(alloc, width, height, 4);
    defer fb.deinit();

    dsl.render(&fb, tree);

    const ppm = try fb.toPpm(alloc);
    defer alloc.free(ppm);

    const path = try std.fmt.allocPrint(alloc, "snapshots/zapp/dsl/{s}.ppm", .{name});
    defer alloc.free(path);

    const f = try Dir.createFile(Dir.cwd(), io, path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, ppm);

    std.debug.print("  ✓ {s} ({d}x{d})\n", .{ path, width, height });
}
