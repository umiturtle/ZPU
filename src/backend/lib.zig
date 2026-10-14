//! Dynamic library loading, on every platform this project ships to.
//!
//! Zig's `std.DynLib` covers ELF and Mach-O. On Windows the same type is a
//! `@compileError` ("unsupported platform", `std/dynamic_library.zig`) and the
//! standard library declares no `LoadLibrary` at all, so this file is the small
//! piece of platform glue the backend layer needs: `std.DynLib` where it exists,
//! `kernel32` on Windows.
//!
//! Both sides give the same three calls — `open`, `lookup`, `close` — and a
//! `lookup` that answers `null` for a symbol a library does not export, which is
//! exactly what probing a graphics driver needs.
const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

/// An open library. `open` takes a name the platform's loader can resolve (a
/// bare file name, searched along the platform's path) or an absolute path.
pub const Library = if (builtin.os.tag == .windows) Windows else std.DynLib;

/// Windows: `LoadLibraryA` + `GetProcAddress` + `FreeLibrary`.
///
/// * A bare name such as `vulkan-1.dll` is searched along the DLL search order;
///   an absolute path is opened directly. That is what the candidate list in
///   `src/backend.zig` expects.
/// * `GetProcAddress` answers `null` for a missing symbol, exactly like `dlsym`.
/// * The names are passed through as given (the ANSI entry point: driver paths
///   are ASCII in practice, and an unusually long one is reported, not guessed).
pub const Windows = struct {
    handle: *anyopaque,

    pub const Error = error{ NotFound, Unexpected, NameTooLong };

    /// A driver path is short; `MAX_PATH` with room to spare is plenty.
    const max_name = 4096;

    pub fn open(name: []const u8) Error!Windows {
        if (name.len == 0) return error.NotFound;
        if (name.len >= max_name) return error.NameTooLong;
        var buffer: [max_name]u8 = undefined;
        @memcpy(buffer[0..name.len], name);
        buffer[name.len] = 0;
        const handle = LoadLibraryA(buffer[0..name.len :0]) orelse return lastError();
        return .{ .handle = handle };
    }

    pub fn close(self: *Windows) void {
        _ = FreeLibrary(self.handle);
        self.handle = undefined;
    }

    pub fn lookup(self: *Windows, comptime T: type, symbol: [:0]const u8) ?T {
        const raw = GetProcAddress(self.handle, symbol.ptr) orelse return null;
        return @ptrCast(@alignCast(raw));
    }

    fn lastError() Error {
        return switch (windows.GetLastError()) {
            .FILE_NOT_FOUND, .PATH_NOT_FOUND, .MOD_NOT_FOUND => error.NotFound,
            else => error.Unexpected,
        };
    }

    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(handle: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(handle: *anyopaque) callconv(.winapi) windows.BOOL;
};

test "a library that is not there fails, and a library that is there opens" {
    // The assertion that matters on every platform: a missing driver is a clean
    // error, never a crash — `tryLoad` and `probe` rely on that to answer "no
    // Vulkan on this machine". The two implementations name it differently
    // (`FileNotFound` from the ELF loader, `NotFound` from the Windows mapping).
    if (Library.open("/nonexistent/zpu-does-not-exist.so")) |lib| {
        var l = lib;
        l.close();
        return error.ExpectedMissingLibrary;
    } else |e| {
        try std.testing.expect(e == error.FileNotFound or e == error.NotFound);
    }

    if (builtin.os.tag == .windows) {
        const kernel = try Library.open("kernel32.dll");
        var lib = kernel;
        defer lib.close();
        try std.testing.expect(lib.lookup(*const fn () callconv(.winapi) u32, "GetCurrentProcessId") != null);
        try std.testing.expect(lib.lookup(*const fn () callconv(.winapi) u32, "NoSuchSymbolInKernel32") == null);
    } else if (builtin.os.tag == .linux) {
        const libc = Library.open("libc.so.6") catch return; // no libc: nothing to check
        var lib = libc;
        defer lib.close();
        try std.testing.expect(lib.lookup(*const fn (?*anyopaque) callconv(.c) c_int, "getpid") != null);
        try std.testing.expect(lib.lookup(*const fn () callconv(.c) void, "NoSuchSymbolInLibc") == null);
    }
}
