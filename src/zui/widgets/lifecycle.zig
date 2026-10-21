//! Ui lifecycle functions - init, deinit, resize, isMeasuring, markDirty, stats, run, result.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");
const input_mod = @import("../input.zig");
const context_mod = @import("../context.zig");

const Ui = ui_mod.Ui;
const Options = ui_mod.Options;
const Stats = ui_mod.Stats;
const FrameResult = ui_mod.FrameResult;
const Input = input_mod.Input;

/// Creates a new Ui instance with the given options.
pub fn init(alloc: std.mem.Allocator, opts: Options) Ui {
    return .{
        .alloc = alloc,
        .ctx = context_mod.Context.init(alloc),
        .list = .{},
        .theme = opts.theme,
        .viewport = .{ .x = opts.width, .y = opts.height },
        .arena = std.heap.ArenaAllocator.init(alloc),
        .max_lines = opts.max_lines,
    };
}

/// Cleans up all resources held by the Ui.
pub fn deinit(self: *Ui) void {
    self.arena.deinit();
    self.focusables.deinit(self.alloc);
    self.focusables_prev.deinit(self.alloc);
    self.list.deinit(self.alloc);
    self.ctx.deinit();
}

/// Resizes the viewport. Marks the UI dirty so the next frame rebuilds.
pub fn resize(self: *Ui, width: f32, height: f32) void {
    if (self.viewport.x == width and self.viewport.y == height) return;
    self.viewport = .{ .x = width, .y = height };
    self.markDirty();
}

/// Returns true if the UI is currently in measure mode.
pub fn isMeasuring(self: *const Ui) bool {
    return self.measuring;
}

/// Marks the UI as dirty, forcing a rebuild on the next frame.
pub fn markDirty(self: *Ui) void {
    self.dirty = true;
}

/// Returns statistics about the UI state.
pub fn stats(self: *const Ui) Stats {
    return .{
        .frames = self.ctx.frames,
        .rebuilt = self.ctx.rebuilt,
        .skipped = self.ctx.skipped,
        .commands = self.list.len(),
        .revision = self.list.revision,
    };
}

/// Runs one frame. `body` is any value with `run(self, ui) Error!void`: a
/// hand-rolled closure, which is how Zig spells one without allocating.
pub fn run(self: *Ui, input: Input, body: anytype) Ui.Error!FrameResult {
    self.beginFrame(input);
    // `endFrame` must have run before the result is taken (it is what bumps
    // the revision) and must still run if the body fails half way through,
    // or the next frame would start with a half-open layout stack.
    errdefer self.endFrame();
    if (!self.skipping) try body.run(self);
    self.endFrame();
    return self.result();
}

/// Returns the frame result after a frame has completed.
pub fn result(self: *const Ui) FrameResult {
    return .{
        .rebuilt = !self.skipping,
        .skipped = self.skipping,
        .commands = self.list.len(),
        .revision = self.list.revision,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "lifecycle functions exist" {
    // Just verify the functions exist and have correct signature
    const InitFunc = @TypeOf(init);
    const DeinitFunc = @TypeOf(deinit);
    const ResizeFunc = @TypeOf(resize);
    const IsMeasuringFunc = @TypeOf(isMeasuring);
    const MarkDirtyFunc = @TypeOf(markDirty);
    const StatsFunc = @TypeOf(stats);
    const RunFunc = @TypeOf(run);
    const ResultFunc = @TypeOf(result);
    
    const init_info = @typeInfo(InitFunc);
    const deinit_info = @typeInfo(DeinitFunc);
    const resize_info = @typeInfo(ResizeFunc);
    const is_measuring_info = @typeInfo(IsMeasuringFunc);
    const mark_dirty_info = @typeInfo(MarkDirtyFunc);
    const stats_info = @typeInfo(StatsFunc);
    const run_info = @typeInfo(RunFunc);
    const result_info = @typeInfo(ResultFunc);
    
    try std.testing.expect(init_info == .@"fn");
    try std.testing.expect(deinit_info == .@"fn");
    try std.testing.expect(resize_info == .@"fn");
    try std.testing.expect(is_measuring_info == .@"fn");
    try std.testing.expect(mark_dirty_info == .@"fn");
    try std.testing.expect(stats_info == .@"fn");
    try std.testing.expect(run_info == .@"fn");
    try std.testing.expect(result_info == .@"fn");
}
