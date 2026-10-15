//! ZAPP configuration - set at build time
//! This module is generated/provided by build.zig with platform-specific settings.

/// Whether X11 backend is available (requires Linux + libc)
pub const has_x11: bool = @import("builtin").os.tag == .linux;

/// Whether Web backend is available (requires wasm target)
pub const has_web: bool = switch (@import("builtin").target.cpu.arch) {
    .wasm32, .wasm64 => true,
    else => false,
};
