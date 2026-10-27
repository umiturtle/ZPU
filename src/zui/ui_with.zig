//! UiWith - comptime-configurable UI wrapper.

const std = @import("std");
const ui_mod = @import("ui.zig");
const Options = @import("options.zig").Options;

const Ui = ui_mod.Ui;
const Config = ui_mod.Config;
const Quality = ui_mod.Quality;

/// Ui with comptime configuration for specialization.
/// Different configs generate different code at compile time.
///
/// Example:
/// ```zig
/// const MyUi = UiWith(.{ .quality = .high, .animations = false });
/// var ui = MyUi.init(alloc, .{});
/// 
/// // With custom backend:
/// const TestUi = UiWith(.{ .backend = MockBackend });
/// var test_ui = TestUi.init(alloc, .{});
/// ```
pub fn UiWith(comptime cfg: Config) type {
    return struct {
        pub const config = cfg;
        pub const Backend = cfg.backend;
        pub const Inner = Ui;

        inner: Ui,

        pub fn init(alloc: std.mem.Allocator, opts: Options) @This() {
            return .{
                .inner = Ui.init(alloc, opts),
            };
        }

        pub fn deinit(self: *@This()) void {
            self.inner.deinit();
        }

        /// Check if animations are enabled for this config (comptime-known)
        pub fn animationsEnabled() bool {
            return cfg.animations;
        }

        /// Check if tooltips are enabled for this config (comptime-known)
        pub fn tooltipsEnabled() bool {
            return cfg.tooltips;
        }

        /// Check if keyboard navigation is enabled for this config (comptime-known)
        pub fn keyboardNavEnabled() bool {
            return cfg.keyboard_nav;
        }

        /// Get the quality level for this config (comptime-known)
        pub fn qualityLevel() Quality {
            return cfg.quality;
        }

        /// Get the antialiasing level based on quality (comptime-known)
        pub fn antialiasLevel() u2 {
            return cfg.quality.antialiasLevel();
        }

        /// Get the backend type (comptime-known)
        pub fn backendType() type {
            return cfg.backend;
        }

        /// Check if using software backend (comptime-known)
        pub fn isSoftwareBackend() bool {
            return cfg.backend == @import("config.zig").SoftwareBackend;
        }

        /// Check if using mock backend (comptime-known)
        pub fn isMockBackend() bool {
            return cfg.backend == @import("config.zig").MockBackend;
        }

        /// Access the underlying Ui for full API
        pub fn ui(self: *@This()) *Ui {
            return &self.inner;
        }

        pub const ui_ptr = ui;
    };
}

// ============================================================================
// Tests
// ============================================================================

test "UiWith comptime specialization" {
    // Test default config
    const DefaultUi = UiWith(.{});
    try std.testing.expect(DefaultUi.animationsEnabled());
    try std.testing.expect(DefaultUi.tooltipsEnabled());
    try std.testing.expect(DefaultUi.keyboardNavEnabled());
    try std.testing.expectEqual(Quality.medium, DefaultUi.qualityLevel());

    // Test custom config with animations disabled
    const NoAnimUi = UiWith(.{ .animations = false });
    try std.testing.expect(!NoAnimUi.animationsEnabled());
    try std.testing.expect(NoAnimUi.tooltipsEnabled()); // default

    // Test custom config with low quality
    const LowQualityUi = UiWith(.{ .quality = .low });
    try std.testing.expectEqual(Quality.low, LowQualityUi.qualityLevel());
    try std.testing.expectEqual(@as(u2, 0), LowQualityUi.antialiasLevel());

    // Test high quality
    const HighQualityUi = UiWith(.{ .quality = .high });
    try std.testing.expectEqual(Quality.high, HighQualityUi.qualityLevel());
    try std.testing.expectEqual(@as(u2, 2), HighQualityUi.antialiasLevel());
}

test "UiWith with different configs generates different types" {
    const Ui1 = UiWith(.{ .quality = .low });
    const Ui2 = UiWith(.{ .quality = .high });

    // Different configs should produce different types
    try std.testing.expect(Ui1 != Ui2);

    // But they should have the same inner type
    try std.testing.expect(Ui1.Inner == Ui2.Inner);
    try std.testing.expect(Ui1.Inner == Ui);
}

test "UiWith runtime usage" {
    var ui_wrapper = UiWith(.{ .quality = .high, .animations = false }).init(std.testing.allocator, .{});
    defer ui_wrapper.deinit();

    // Access underlying Ui
    const u = ui_wrapper.ui();
    try std.testing.expect(u.viewport.x > 0 or u.viewport.x == 0); // Just verify it works

    // Verify comptime config is accessible
    try std.testing.expect(!@TypeOf(ui_wrapper).animationsEnabled());
    try std.testing.expectEqual(Quality.high, @TypeOf(ui_wrapper).qualityLevel());
}

test "UiWith with different backends generates different types" {
    const SoftwareUi = UiWith(.{ .backend = @import("config.zig").SoftwareBackend });
    const MockUi = UiWith(.{ .backend = @import("config.zig").MockBackend });

    // Different backends should produce different types
    try std.testing.expect(SoftwareUi != MockUi);

    // Verify backend type is accessible at comptime
    try std.testing.expectEqual(@import("config.zig").SoftwareBackend, SoftwareUi.Backend);
    try std.testing.expectEqual(@import("config.zig").MockBackend, MockUi.Backend);

    // Verify comptime backend checks
    try std.testing.expect(SoftwareUi.isSoftwareBackend());
    try std.testing.expect(!SoftwareUi.isMockBackend());
    try std.testing.expect(!MockUi.isSoftwareBackend());
    try std.testing.expect(MockUi.isMockBackend());
}

test "UiWith backendType comptime function" {
    const DefaultUi = UiWith(.{});
    try std.testing.expectEqual(@import("config.zig").SoftwareBackend, DefaultUi.backendType());

    const CustomUi = UiWith(.{ .backend = @import("config.zig").MockBackend });
    try std.testing.expectEqual(@import("config.zig").MockBackend, CustomUi.backendType());
}
