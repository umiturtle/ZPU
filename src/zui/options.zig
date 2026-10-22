//! Options - UI configuration options.

const Theme = @import("theme.zig").Theme;

pub const Options = struct {
    width: f32 = 320,
    height: f32 = 200,
    theme: Theme = Theme.dark,
    /// Cap on the lines one label may wrap to: the arena is not a bin.
    max_lines: usize = 64,
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "options default values" {
    const options = Options{};
    try std.testing.expect(options.width == 320);
    try std.testing.expect(options.height == 200);
    try std.testing.expect(options.max_lines == 64);
}
