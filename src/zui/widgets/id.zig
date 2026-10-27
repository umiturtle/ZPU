//! ID management functions - scopeHash, idOf, pushIdIndex, pushId, popId.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Id = types.Id;

/// Returns the hash of the current ID scope.
pub fn scopeHash(self: *const Ui) u64 {
    return if (self.id_top == 0) 0 else self.id_stack[self.id_top - 1];
}

/// Creates an ID from a name, scoped by the current ID stack.
pub fn idOf(self: *const Ui, name: []const u8) Id {
    return Id.hash(self.scopeHash(), name);
}

/// Scopes an id by index instead of by name: row `i` of a list is a widget
/// id like any other, and formatting a label for each of a million rows
/// would be work for nothing.
pub fn pushIdIndex(self: *Ui, i: usize) void {
    self.id_stack[self.id_top] = Id.index(self.scopeHash(), i).h;
    self.id_top += 1;
}

/// Pushes a named ID onto the ID stack.
pub fn pushId(self: *Ui, name: []const u8) void {
    self.id_stack[self.id_top] = Id.hash(self.scopeHash(), name).h;
    self.id_top += 1;
}

/// Pops the most recent ID from the ID stack.
pub fn popId(self: *Ui) void {
    self.id_top -= 1;
}

// ============================================================================
// Tests
// ============================================================================

test "id functions exist" {
    // Just verify the functions exist and have correct signature
    const ScopeHashFunc = @TypeOf(scopeHash);
    const IdOfFunc = @TypeOf(idOf);
    const PushIdIndexFunc = @TypeOf(pushIdIndex);
    const PushIdFunc = @TypeOf(pushId);
    const PopIdFunc = @TypeOf(popId);
    
    const scope_hash_info = @typeInfo(ScopeHashFunc);
    const id_of_info = @typeInfo(IdOfFunc);
    const push_id_index_info = @typeInfo(PushIdIndexFunc);
    const push_id_info = @typeInfo(PushIdFunc);
    const pop_id_info = @typeInfo(PopIdFunc);
    
    try std.testing.expect(scope_hash_info == .@"fn");
    try std.testing.expect(id_of_info == .@"fn");
    try std.testing.expect(push_id_index_info == .@"fn");
    try std.testing.expect(push_id_info == .@"fn");
    try std.testing.expect(pop_id_info == .@"fn");
}
