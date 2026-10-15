//! ZGFのレンダリングテスト - 単純なグラデーションを生成して確認
const std = @import("std");
const zgf = @import("zgf");

const Dir = std.Io.Dir;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    std.debug.print("\n=== ZGF レンダリングテスト ===\n\n", .{});

    Dir.cwd().createDirPath(io, "snapshots/zgf") catch {};

    // テスト1: 単純なグラデーション（左から右へ赤→青）
    {
        var img = try zgf.Image.init(alloc, 100, 100, 3);
        defer img.deinit();

        for (0..100) |y| {
            for (0..100) |x| {
                const r: f32 = @as(f32, @floatFromInt(x)) / 100.0;
                const g: f32 = 0.0;
                const b: f32 = @as(f32, @floatFromInt(y)) / 100.0;
                img.setPixel(@intCast(x), @intCast(y), .{ .r = r, .g = g, .b = b });
            }
        }

        const ppm = try img.toPpm(alloc);
        defer alloc.free(ppm);
        const path = "snapshots/zgf/gradient_test.ppm";
        const f = try Dir.createFile(Dir.cwd(), io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, ppm);
        std.debug.print("  ✓ {s}\n", .{path});
    }

    // テスト2: 単色ベタ塗り
    {
        var img = try zgf.Image.init(alloc, 50, 50, 3);
        defer img.deinit();

        img.fill(.{ .r = 0.5, .g = 0.5, .b = 0.5 });

        const ppm = try img.toPpm(alloc);
        defer alloc.free(ppm);
        const path = "snapshots/zgf/solid_test.ppm";
        const f = try Dir.createFile(Dir.cwd(), io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, ppm);
        std.debug.print("  ✓ {s}\n", .{path});
    }

    // テスト3: 赤い矩形
    {
        var img = try zgf.Image.init(alloc, 100, 100, 3);
        defer img.deinit();

        img.fill(.{ .r = 1.0, .g = 0.0, .b = 0.0 });

        const ppm = try img.toPpm(alloc);
        defer alloc.free(ppm);
        const path = "snapshots/zgf/red_rect_test.ppm";
        const f = try Dir.createFile(Dir.cwd(), io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, ppm);
        std.debug.print("  ✓ {s}\n", .{path});
    }

    std.debug.print("\n=== テスト完了 ===\n", .{});
}
