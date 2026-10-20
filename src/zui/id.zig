//! Unique identifier for ZUI widgets
//!
//! FNV-1a hash based ID generation.
//! Dependency-free, fully testable.

const std = @import("std");

/// Unique identifier for widgets (FNV-1a hash)
pub const Id = struct {
    value: u64,

    pub fn fromString(s: []const u8) Id {
        const fnv_offset: u64 = 14695981039346656037;
        const fnv_prime: u64 = 1099511628211;

        var hash: u64 = fnv_offset;
        for (s) |byte| {
            hash ^= byte;
            hash *%= fnv_prime;
        }
        return .{ .value = hash };
    }

    pub fn combine(self: Id, other: Id) Id {
        const fnv_prime: u64 = 1099511628211;
        return .{ .value = (self.value ^ other.value) *% fnv_prime };
    }

    pub fn combineString(self: Id, s: []const u8) Id {
        return self.combine(Id.fromString(s));
    }

    pub fn eql(a: Id, b: Id) bool {
        return a.value == b.value;
    }
};

/// Build an ID at compile time from a comptime string
pub fn idFromComptime(comptime s: []const u8) Id {
    const fnv_offset: u64 = 14695981039346656037;
    const fnv_prime: u64 = 1099511628211;

    comptime var hash: u64 = fnv_offset;
    inline for (s) |byte| {
        hash ^= byte;
        hash *%= fnv_prime;
    }
    return .{ .value = hash };
}

// ============================================================================
// Tests
// ============================================================================

test "id - fromString" {
    const a = Id.fromString("hello");
    const b = Id.fromString("hello");
    const c = Id.fromString("world");

    try std.testing.expect(a.eql(b));
    try std.testing.expect(!a.eql(c));
}

test "id - different strings produce different ids" {
    const a = Id.fromString("button");
    const b = Id.fromString("label");
    const c = Id.fromString("panel");

    try std.testing.expect(!a.eql(b));
    try std.testing.expect(!b.eql(c));
    try std.testing.expect(!a.eql(c));
}

test "id - combine" {
    const a = Id.fromString("parent");
    const b = Id.fromString("child");
    const c = a.combine(b);

    try std.testing.expect(!c.eql(a));
    try std.testing.expect(!c.eql(b));
}

test "id - combineString" {
    const base = Id.fromString("parent");
    const a = base.combineString("child1");
    const b = base.combineString("child2");

    try std.testing.expect(!a.eql(b));
}

test "id - comptime generation" {
    const a = idFromComptime("hello");
    const b = Id.fromString("hello");

    try std.testing.expect(a.eql(b));
}

test "id - empty string" {
    const empty = Id.fromString("");
    try std.testing.expect(empty.value > 0);
}

test "id - unicode" {
    const a = Id.fromString("日本語");
    const b = Id.fromString("日本語");
    const c = Id.fromString("English");

    try std.testing.expect(a.eql(b));
    try std.testing.expect(!a.eql(c));
}
