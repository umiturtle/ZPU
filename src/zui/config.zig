//! Configuration types for ZUI comptime specialization.
//!
//! This module defines the types used to configure `Ui` at compile time,
//! allowing different backends and quality levels to generate specialized code.
//!
//! Usage:
//! ```zig
//! const MyUi = zui.UiWith(.{ .quality = .high });
//! var ui = MyUi.init(alloc, .{});
//! ```

const std = @import("std");
const types = @import("types.zig");

/// Quality level for rendering optimizations
pub const Quality = enum {
    /// Fastest rendering, suitable for low-power devices
    low,
    /// Balanced quality and performance (default)
    medium,
    /// Highest quality, suitable for high-DPI displays
    high,

    /// Get the default quality level
    pub fn default() Quality {
        return .medium;
    }

    /// Get antialiasing level for this quality
    pub fn antialiasLevel(self: Quality) u2 {
        return switch (self) {
            .low => 0,
            .medium => 1,
            .high => 2,
        };
    }
};

/// Backend interface marker - types that implement rendering
pub const BackendInterface = struct {
    /// Check if a type implements the Backend interface
    pub fn implements(comptime T: type) bool {
        return @hasDecl(T, "draw");
    }
};

/// Software backend (render.zig) - the default
pub const SoftwareBackend = struct {
    pub const is_software = true;

    /// Draw options for software backend
    pub const DrawOptions = struct {
        origin: types.Vec2 = .{},
        scale: f32 = 1,
        clear: ?types.Color = null,
    };

    /// Draw function (stub for backend interface)
    pub fn draw(_: DrawOptions) void {
        // Software backend draws to an image
        // Actual implementation is in render.zig
    }
};

/// Mock backend for testing - no actual rendering
pub const MockBackend = struct {
    pub const is_mock = true;

    pub const DrawOptions = struct {
        record: bool = true,
    };

    /// Draw function (stub for backend interface)
    pub fn draw(_: DrawOptions) void {
        // Mock backend does nothing
    }
};

/// Configuration for Ui comptime specialization
pub const Config = struct {
    /// Quality level for rendering optimizations
    quality: Quality = .medium,

    /// Backend type for rendering (comptime-specialized)
    backend: type = SoftwareBackend,

    /// Enable animations (comptime-eliminable)
    animations: bool = true,

    /// Enable tooltips (comptime-eliminable)
    tooltips: bool = true,

    /// Enable keyboard navigation (comptime-eliminable)
    keyboard_nav: bool = true,

    /// Enable text editing (comptime-eliminable)
    text_editing: bool = true,
};

/// Default configuration
pub const default_config = Config{};

// ============================================================================
// Tests
// ============================================================================

test "Quality default" {
    try std.testing.expectEqual(Quality.medium, Quality.default());
}

test "Quality antialiasLevel" {
    try std.testing.expectEqual(@as(u2, 0), Quality.low.antialiasLevel());
    try std.testing.expectEqual(@as(u2, 1), Quality.medium.antialiasLevel());
    try std.testing.expectEqual(@as(u2, 2), Quality.high.antialiasLevel());
}

test "BackendInterface check" {
    try std.testing.expect(BackendInterface.implements(SoftwareBackend));
    try std.testing.expect(BackendInterface.implements(MockBackend));
}

test "Config default values" {
    const config = Config{};
    try std.testing.expectEqual(Quality.medium, config.quality);
    try std.testing.expectEqual(SoftwareBackend, config.backend);
    try std.testing.expect(config.animations);
    try std.testing.expect(config.tooltips);
    try std.testing.expect(config.keyboard_nav);
    try std.testing.expect(config.text_editing);
}

test "Config custom values" {
    const config = Config{
        .quality = .low,
        .backend = MockBackend,
        .animations = false,
        .tooltips = false,
    };
    try std.testing.expectEqual(Quality.low, config.quality);
    try std.testing.expectEqual(MockBackend, config.backend);
    try std.testing.expect(!config.animations);
    try std.testing.expect(!config.tooltips);
    try std.testing.expect(config.keyboard_nav); // default
}

test "Config with different backends" {
    const software_config = Config{ .backend = SoftwareBackend };
    const mock_config = Config{ .backend = MockBackend };
    
    // Different backends should be different types
    try std.testing.expect(software_config.backend != mock_config.backend);
    
    // Both should implement BackendInterface
    try std.testing.expect(BackendInterface.implements(software_config.backend));
    try std.testing.expect(BackendInterface.implements(mock_config.backend));
}

test "default_config" {
    try std.testing.expectEqual(Quality.medium, default_config.quality);
}
