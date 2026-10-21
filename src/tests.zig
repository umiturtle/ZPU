const std = @import("std");
const zpu = @import("zpu");
const zgf = @import("zgf");
const zui = @import("zui");
const zbackend = @import("zbackend");

test {
    std.testing.refAllDecls(zpu);
    std.testing.refAllDecls(zgf);
    std.testing.refAllDecls(zui);
    std.testing.refAllDecls(zbackend);
    std.testing.refAllDecls(@import("zapp/coordinates.zig"));
    std.testing.refAllDecls(@import("zapp/widgets.zig"));
    std.testing.refAllDecls(@import("zapp/parallel.zig"));
    std.testing.refAllDecls(@import("zapp/vector.zig"));
    std.testing.refAllDecls(@import("zapp/input.zig"));
    std.testing.refAllDecls(@import("zapp/i18n.zig"));
    std.testing.refAllDecls(@import("zapp/hot_reload.zig"));
    std.testing.refAllDecls(@import("zapp/dev_tools.zig"));
    std.testing.refAllDecls(@import("zapp/layout.zig"));
    std.testing.refAllDecls(@import("zapp/state.zig"));
    std.testing.refAllDecls(@import("zapp/ecs.zig"));
    std.testing.refAllDecls(@import("zapp/vector_text.zig"));
    std.testing.refAllDecls(@import("zapp/rich_render.zig"));
    std.testing.refAllDecls(@import("zapp/geometry.zig"));
    std.testing.refAllDecls(@import("zapp/primitives.zig"));
    std.testing.refAllDecls(@import("zapp/comptime_vector.zig"));
    std.testing.refAllDecls(@import("zapp/adaptive_render.zig"));
    std.testing.refAllDecls(@import("zapp/ui.zig"));
}

test "library integration" {
    _ = @import("zpu/tests.zig");
    _ = zbackend;
}
