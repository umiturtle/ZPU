//! Frame lifecycle helpers - beginFrame and endFrame.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");
const input_mod = @import("../input.zig");

const Ui = ui_mod.Ui;
const Id = types.Id;
const Input = input_mod.Input;

/// Begins a new frame: resets state, checks for idle skipping, prepares layout.
pub fn beginFrame(self: *Ui, input: Input) void {
    self.ctx.frames += 1;

    // The one retained decision, and the reason this library is a hybrid:
    // when nothing arrived, nothing is animating and the program did not mark
    // anything dirty, the picture cannot differ from last frame's, so the
    // body is not run and the display list is reused as-is.
    //
    // The pointer position is part of "nothing arrived": a moved pointer can
    // light up a different widget, so the body has to see it. `isIdle` alone
    // cannot know that — it is a property of one frame, and a position needs
    // two — which is why the previous position is retained here.
    const pointer_moved = !input.pointer.eql(self.last_pointer);
    self.skipping = self.can_skip and input.isIdle() and !pointer_moved;

    self.input = input;
    self.last_pointer = input.pointer;
    self.ctx.animating = false;
    self.dirty = false;
    _ = self.arena.reset(.retain_capacity);
    self.id_top = 0;
    self.depth = 0;
    self.clip = .{ .w = std.math.floatMax(f32), .h = std.math.floatMax(f32) };
    // Hotness is "under the pointer *this* frame", so it is recomputed from
    // scratch; `active`, `focus` and the animations are what persist.
    self.ctx.hot = Id.none;
    std.mem.swap(std.ArrayListUnmanaged(Id), &self.focusables, &self.focusables_prev);
    self.focusables.clearRetainingCapacity();

    if (self.skipping) {
        self.ctx.skipped += 1;
        return;
    }
    self.handleFocusKeys();
    self.list.reset();
    self.pushRootLayout();
}

/// Ends the current frame: finalizes layout, paints tooltips, updates state.
pub fn endFrame(self: *Ui) void {
    if (self.skipping) {
        self.list.reused = true;
        return;
    }
    self.list.reused = false;
    self.list.revision += 1;
    while (self.depth > 0) self.popLayout();
    // Last, so nothing in the frame can cover it.
    if (self.tip) |tip| self.paintTooltip(tip) catch {};
    self.tip = null;
    self.ctx.rebuilt += 1;
    self.can_skip = self.input.isIdle() and !self.dirty and !self.ctx.animating;
    self.ctx.focusables.clearRetainingCapacity();
    self.ctx.focusables.appendSlice(self.ctx.alloc, self.focusables.items) catch {};
}

// ============================================================================
// Tests
// ============================================================================

test "frame functions exist" {
    // Just verify the functions exist and have correct signature
    const BeginFunc = @TypeOf(beginFrame);
    const EndFunc = @TypeOf(endFrame);
    const begin_info = @typeInfo(BeginFunc);
    const end_info = @typeInfo(EndFunc);
    try std.testing.expect(begin_info == .@"fn");
    try std.testing.expect(end_info == .@"fn");
}
