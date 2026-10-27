//! Focus management functions.

const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Id = types.Id;

/// Requests focus for a specific widget ID.
pub fn requestFocus(self: *Ui, id: Id) void {
    if (!self.measuring) self.ctx.focus = id;
}

/// Clears focus from all widgets.
pub fn clearFocus(self: *Ui) void {
    if (!self.measuring) self.ctx.focus = Id.none;
}

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "focus functions exist" {
    const requestFunc = @TypeOf(requestFocus);
    const clearFunc = @TypeOf(clearFocus);

    const request_info = @typeInfo(requestFunc);
    const clear_info = @typeInfo(clearFunc);

    try std.testing.expect(request_info == .@"fn");
    try std.testing.expect(clear_info == .@"fn");
}
