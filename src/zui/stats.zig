//! Stats - frame statistics.

pub const Stats = struct {
    frames: u64,
    rebuilt: u64,
    skipped: u64,
    commands: usize,
    revision: u64,
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "stats struct" {
    const stats = Stats{
        .frames = 100,
        .rebuilt = 10,
        .skipped = 5,
        .commands = 50,
        .revision = 3,
    };
    try std.testing.expect(stats.frames == 100);
    try std.testing.expect(stats.rebuilt == 10);
    try std.testing.expect(stats.skipped == 5);
    try std.testing.expect(stats.commands == 50);
    try std.testing.expect(stats.revision == 3);
}
