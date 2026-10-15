//! Snapshot tests, all four libraries.
//!
//! Each block below answers "what does this produce today", for something where
//! the answer is a *reference*: a tensor of numbers, a gradient, an image, a
//! frame's command list. The references live in `snapshots/` and are compared by
//! `tests/snapshot.zig` — numbers with a tolerance, structure and text exactly.
//!
//! The four files are imported in order of dependency: `zpu` is the numerical
//! core, `zbackend` sits under `zgf`, and `zui` is built on both.
const std = @import("std");

test {
    _ = @import("snap/zpu.zig");
    _ = @import("snap/zgf.zig");
    _ = @import("snap/zui.zig");
    _ = @import("snap/backend.zig");
}
