//! Unique identifier for ZUI widgets
//!
//! FNV-1a hash based ID generation.
//! Dependency-free, fully testable.

const std = @import("std");

/// Unique identifier for widgets (FNV-1a hash)
pub const Id = struct {
    h: u64 = 0,

    pub const none: Id = .{};

    pub fn eql(a: Id, b: Id) bool {
        return a.h == b.h;
    }

    pub fn isNone(self: Id) bool {
        return self.h == 0;
    }

    /// Hashes a label. `seed` is 0 at the root and the parent's hash below it,
    /// so the same label in two containers gives two ids (Wyhash, 64 bit: the
    /// collision chance is not something a UI will ever meet).
    pub fn hash(seed: u64, label: []const u8) Id {
        return .{ .h = std.hash.Wyhash.hash(seed, label) };
    }

    /// Integer labels cost nothing at all: no hashing, no string, and a `for`
    /// loop can name its rows with them.
    pub fn index(seed: u64, i: usize) Id {
        return .{ .h = seed ^ (0x9E3779B97F4A7C15 *% (@as(u64, @intCast(i)) +% 1)) };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "id - fromString" {
    const id1 = Id.hash(0, "test");
    const id2 = Id.hash(0, "test");
    const id3 = Id.hash(0, "other");

    try std.testing.expect(id1.eql(id2));
    try std.testing.expect(!id1.eql(id3));
    try std.testing.expect(!id1.isNone());
}

test "id - none" {
    try std.testing.expect(Id.none.isNone());
    try std.testing.expect(Id.none.eql(.{}));
}

test "id - hash" {
    const parent: u64 = 12345;
    const child = Id.hash(parent, "widget");

    try std.testing.expect(!child.isNone());
    try std.testing.expect(!child.eql(Id.hash(parent, "other")));

    // Same input produces same output
    const child2 = Id.hash(parent, "widget");
    try std.testing.expect(child.eql(child2));
}

test "id - index" {
    const seed: u64 = 12345;
    const id0 = Id.index(seed, 0);
    const id1 = Id.index(seed, 1);
    const id2 = Id.index(seed, 2);

    try std.testing.expect(!id0.isNone());
    try std.testing.expect(!id0.eql(id1));
    try std.testing.expect(!id1.eql(id2));

    // Same index produces same ID
    const id0_again = Id.index(seed, 0);
    try std.testing.expect(id0.eql(id0_again));
}
