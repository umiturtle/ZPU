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
}

test "library integration" {
    _ = @import("zpu/tests.zig");
    _ = @import("zgf/tests.zig");
    _ = zbackend;
}
