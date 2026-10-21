//! FrameResult - the result of a frame.

pub const FrameResult = struct {
    rebuilt: bool,
    /// Nothing could have changed and the body was not called: the display list
    /// is last frame's, command for command.
    skipped: bool,
    commands: usize,
    revision: u64,
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "frame result default values" {
    const result = FrameResult{
        .rebuilt = false,
        .skipped = true,
        .commands = 10,
        .revision = 5,
    };
    try std.testing.expect(!result.rebuilt);
    try std.testing.expect(result.skipped);
    try std.testing.expect(result.commands == 10);
    try std.testing.expect(result.revision == 5);
}
