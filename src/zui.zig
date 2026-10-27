//! ZUI — a 2-D UI library that is a hybrid on purpose.
//!
//! The widget API is **immediate mode**: the picture is a function of the
//! program's state, written as ordinary Zig (`if`, `for`, a call per widget), so
//! there is no widget tree to keep in sync, no callbacks to register and no
//! `*anyopaque` context to thread through by hand — the things Zig's lack of
//! closures and lack of reflection make expensive.
//!
//! The frame is **retained**: widget calls append to a display list, which is
//! what makes an idle frame free (nothing changed, so nothing is rebuilt) and
//! what a GPU backend can consume (instances, not callbacks).
//!
//! The frame is *also* an **instance stream**: the same display list converts,
//! with one walk, into fixed-size records (`zui/instances.zig`) that a backend
//! can draw or upload — and `zui/pipeline.zig` draws exactly that stream
//! headlessly, so "the GPU frame and the software frame are the same picture" is
//! a test rather than a hope.
//!
//! The state is **retained, but tiny**: hot/active/focus/scroll/animation and
//! text-edit state, keyed by widget id and listed in one struct.
//!
//! ```zig
//! var ui = zui.Ui.init(alloc, .{ .width = 320, .height = 240 });
//! defer ui.deinit();
//!
//! const Body = struct {
//!     app: *App,
//!     fn run(self: @This(), u: *zui.Ui) !void {
//!         try u.label("Hello", .{});
//!         if (try u.button("count", .{})) self.app.count += 1;
//!     }
//! };
//!
//! const input = zui.Input{ .pointer = .{ .x = 10, .y = 10 }, .pointer_down = true, .pointer_pressed = true };
//! _ = try ui.run(input, Body{ .app = &app });
//! ```
const std = @import("std");

pub const types = @import("zui/types.zig");
pub const config = @import("zui/config.zig");
pub const input = @import("zui/input.zig");
pub const font = @import("zui/font.zig");
pub const text = @import("zui/text.zig");
pub const frame = @import("zui/frame.zig");
pub const context = @import("zui/context.zig");
pub const ui_mod = @import("zui/ui.zig");
pub const render = @import("zui/render.zig");
pub const raster = @import("zui/raster.zig");
pub const instances = @import("zui/instances.zig");
pub const pipeline = @import("zui/pipeline.zig");
pub const atlas = @import("zui/atlas.zig");

// New loosely-coupled modules (reintegrated with existing API)
pub const color = @import("zui/color.zig");
pub const vec2 = @import("zui/vec2.zig");
pub const rect = @import("zui/rect.zig");
pub const id = @import("zui/id.zig");
pub const theme = @import("zui/theme.zig");
pub const style = @import("zui/style.zig");
pub const layout = @import("zui/layout.zig");
pub const animation = @import("zui/animation.zig");
pub const event_bus = @import("zui/event_bus.zig");

pub const Vec2 = types.Vec2;
pub const Axis = types.Axis;
pub const Rect = types.Rect;
pub const Edges = types.Edges;
pub const Color = types.Color;
pub const Alignment = types.Alignment;
pub const Size = types.Size;
pub const Id = types.Id;

pub const Input = input.Input;
pub const Key = input.Key;
pub const KeyCode = input.KeyCode;
pub const Script = input.Script;

pub const DisplayList = frame.DisplayList;
pub const Command = frame.Command;
pub const Instance = frame.Instance;

pub const Context = context.Context;
pub const EditState = context.EditState;

pub const Ui = ui_mod.Ui;
pub const UiWith = ui_mod.UiWith;
pub const Config = config.Config;
pub const Quality = config.Quality;
pub const SoftwareBackend = config.SoftwareBackend;
pub const MockBackend = config.MockBackend;
pub const Style = ui_mod.Style;
pub const Theme = ui_mod.Theme;
pub const Response = ui_mod.Response;
pub const FrameResult = ui_mod.FrameResult;
pub const Options = ui_mod.Options;
pub const TextField = text.TextField;

test {
    _ = @import("zui/atlas.zig");
    _ = @import("zui/context.zig");
    _ = @import("zui/instances.zig");
    _ = @import("zui/pipeline.zig");
    _ = @import("zui/raster.zig");
    _ = @import("zui/render.zig");
    _ = @import("zui/tests.zig");
    _ = @import("zui/ui_tests.zig");
    _ = @import("zui/renderer.zig");
    _ = @import("zui/font.zig");
    _ = @import("zui/frame.zig");
    _ = @import("zui/input.zig");
    _ = @import("zui/text.zig");
    _ = @import("zui/types.zig");
    _ = @import("zui/ui.zig");
    // New loosely-coupled modules (reintegrated)
    _ = @import("zui/color.zig");
    _ = @import("zui/vec2.zig");
    _ = @import("zui/rect.zig");
    _ = @import("zui/id.zig");
    _ = @import("zui/theme.zig");
    _ = @import("zui/style.zig");
    _ = @import("zui/layout.zig");
    _ = @import("zui/animation.zig");
    _ = @import("zui/event_bus.zig");
}
