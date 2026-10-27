//! Renderer - comptime-specialized rendering backend abstraction.
//!
//! The Renderer is parameterized by a Backend type at compile time, allowing
//! different rendering strategies (software, test/mock, null/benchmark) to be
//! selected with zero runtime overhead.
//!
//! Usage:
//! ```zig
//! const SoftwareRenderer = Renderer(SoftwareBackend);
//! var renderer = SoftwareRenderer.init(alloc);
//! renderer.draw(display_list, target);
//! ```

const std = @import("std");
const types = @import("types.zig");
const frame_mod = @import("frame.zig");
const config_mod = @import("config.zig");

const DisplayList = frame_mod.DisplayList;
const Rect = types.Rect;
const Vec2 = types.Vec2;
const Color = types.Color;

/// Backend interface - all backends must implement these functions
pub const BackendInterface = struct {
    /// Draw a display list to a target
    pub fn draw(
        comptime Backend: type,
        list: *const DisplayList,
        target: Backend.Target,
        opts: Backend.Options,
    ) void {
        Backend.draw(list, target, opts);
    }
};

/// Generic Renderer parameterized by Backend type
pub fn Renderer(comptime Backend: type) type {
    return struct {
        pub const backend = Backend;
        pub const Options = Backend.Options;
        pub const Target = Backend.Target;

        /// Draw a display list using this backend
        pub fn draw(list: *const DisplayList, target: Target, opts: Options) void {
            Backend.draw(list, target, opts);
        }

        /// Check if this is a software backend (comptime-known)
        pub fn isSoftware() bool {
            return Backend == config_mod.SoftwareBackend;
        }

        /// Check if this is a test/mock backend (comptime-known)
        pub fn isTest() bool {
            return Backend == TestBackend;
        }

        /// Check if this is a null backend (comptime-known)
        pub fn isNull() bool {
            return Backend == NullBackend;
        }
    };
}

/// Test/Mock backend - records draw calls for verification
pub const TestBackend = struct {
    pub const Options = struct {
        record: bool = true,
    };

    pub const Target = struct {
        draw_calls: *std.ArrayListUnmanaged(DrawCall),
        allocator: std.mem.Allocator,
    };

    pub const DrawCall = struct {
        command_type: CommandKind,
        rect: ?Rect = null,
    };

    pub const CommandKind = enum {
        rect,
        rounded,
        triangle,
        text,
        line,
        push_clip,
        pop_clip,
    };

    pub fn draw(list: *const DisplayList, target: Target, opts: Options) void {
        if (!opts.record) return;

        for (list.commands.items) |cmd| {
            const draw_call = switch (cmd) {
                .rect => |r| DrawCall{ .command_type = .rect, .rect = r.r },
                .rounded => |r| DrawCall{ .command_type = .rounded, .rect = r.r },
                .triangle => DrawCall{ .command_type = .triangle },
                .text => |t| DrawCall{ .command_type = .text, .rect = Rect.init(t.pos.x, t.pos.y, 1, 1) },
                .line => DrawCall{ .command_type = .line },
                .push_clip => |r| DrawCall{ .command_type = .push_clip, .rect = r },
                .pop_clip => DrawCall{ .command_type = .pop_clip },
            };
            target.draw_calls.append(target.allocator, draw_call) catch {};
        }
    }
};

/// Null backend - does nothing, used for benchmarking
pub const NullBackend = struct {
    pub const Options = struct {};
    pub const Target = struct {};

    pub fn draw(list: *const DisplayList, target: Target, opts: Options) void {
        _ = list;
        _ = target;
        _ = opts;
        // Do nothing - used for benchmarking the display list construction
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Renderer comptime specialization" {
    const SoftwareR = Renderer(config_mod.SoftwareBackend);
    const TestR = Renderer(TestBackend);
    const NullR = Renderer(NullBackend);

    // Different backends produce different renderer types
    try std.testing.expect(SoftwareR != TestR);
    try std.testing.expect(TestR != NullR);

    // Comptime backend checks
    try std.testing.expect(SoftwareR.isSoftware());
    try std.testing.expect(!SoftwareR.isTest());
    try std.testing.expect(!SoftwareR.isNull());

    try std.testing.expect(!TestR.isSoftware());
    try std.testing.expect(TestR.isTest());
    try std.testing.expect(!TestR.isNull());

    try std.testing.expect(!NullR.isSoftware());
    try std.testing.expect(!NullR.isTest());
    try std.testing.expect(NullR.isNull());
}

test "NullBackend does nothing" {
    var list: DisplayList = .{};
    defer list.commands.deinit(std.testing.allocator);

    const white = Color{ .r = 1, .g = 1, .b = 1, .a = 1 };
    try list.push(std.testing.allocator, .{ .rect = .{ .r = Rect{ .x = 0, .y = 0, .w = 10, .h = 10 }, .color = white } });

    // NullBackend should not crash and should do nothing
    const NullR = Renderer(NullBackend);
    NullR.draw(&list, .{}, .{});
}

test "TestBackend records draw calls" {
    var list: DisplayList = .{};
    defer list.commands.deinit(std.testing.allocator);

    const white = Color{ .r = 1, .g = 1, .b = 1, .a = 1 };
    try list.push(std.testing.allocator, .{ .rect = .{ .r = Rect{ .x = 0, .y = 0, .w = 10, .h = 10 }, .color = white } });
    try list.push(std.testing.allocator, .{ .push_clip = Rect{ .x = 5, .y = 5, .w = 20, .h = 20 } });
    try list.push(std.testing.allocator, .{ .pop_clip = {} });

    var draw_calls: std.ArrayListUnmanaged(TestBackend.DrawCall) = .empty;
    defer draw_calls.deinit(std.testing.allocator);

    const TestR = Renderer(TestBackend);
    TestR.draw(&list, .{ .draw_calls = &draw_calls, .allocator = std.testing.allocator }, .{ .record = true });

    try std.testing.expectEqual(@as(usize, 3), draw_calls.items.len);
    try std.testing.expectEqual(TestBackend.CommandKind.rect, draw_calls.items[0].command_type);
    try std.testing.expectEqual(TestBackend.CommandKind.push_clip, draw_calls.items[1].command_type);
    try std.testing.expectEqual(TestBackend.CommandKind.pop_clip, draw_calls.items[2].command_type);
}
