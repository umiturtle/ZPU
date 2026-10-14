//! Element types.
const std = @import("std");

pub const DType = enum(u8) {
    f64,
    f32,
    f16,
    i64,
    i32,
    i16,
    i8,
    u8,
    bool,

    /// Backing Zig type. `dt` must be comptime known: element types are used to
    /// pick the typed code path inside kernels (`switch (dtype) { inline else => |d| ... }`).
    pub fn Elt(comptime dt: DType) type {
        return switch (dt) {
            .f64 => f64,
            .f32 => f32,
            .f16 => f16,
            .i64 => i64,
            .i32 => i32,
            .i16 => i16,
            .i8 => i8,
            .u8 => u8,
            .bool => bool,
        };
    }

    pub fn of(comptime T: type) DType {
        return switch (T) {
            f64 => .f64,
            f32 => .f32,
            f16 => .f16,
            i64 => .i64,
            i32 => .i32,
            i16 => .i16,
            i8 => .i8,
            u8 => .u8,
            bool => .bool,
            else => @compileError("unsupported dtype " ++ @typeName(T)),
        };
    }

    pub fn parse(s: []const u8) ?DType {
        // Zig 0.17: `std.meta.fields` is a compile error, and `Type.Enum` exposes
        // `field_names` / `field_values` instead of an array of fields.
        const info = @typeInfo(DType).@"enum";
        inline for (info.field_names, info.field_values) |fname, fvalue| {
            if (std.mem.eql(u8, s, fname)) return @fromBackingInt(@intCast(fvalue));
        }
        if (std.mem.eql(u8, s, "float32")) return .f32;
        if (std.mem.eql(u8, s, "float64")) return .f64;
        if (std.mem.eql(u8, s, "int64")) return .i64;
        if (std.mem.eql(u8, s, "int32")) return .i32;
        if (std.mem.eql(u8, s, "uint8")) return .u8;
        return null;
    }

    pub fn itemsize(self: DType) usize {
        return switch (self) {
            inline else => |d| @sizeOf(DType.Elt(d)),
        };
    }

    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f64, .f32, .f16 => true,
            else => false,
        };
    }

    pub fn isInt(self: DType) bool {
        return switch (self) {
            .i64, .i32, .i16, .i8, .u8 => true,
            else => false,
        };
    }

    pub fn isSigned(self: DType) bool {
        return switch (self) {
            .i64, .i32, .i16, .i8 => true,
            else => false,
        };
    }

    /// Common type of `a` and `b` (NumPy semantics, simplified: floats win).
    pub fn promote(a: DType, b: DType) DType {
        if (a == b) return a;
        if (a.isFloat() and !b.isFloat()) return a;
        if (b.isFloat() and !a.isFloat()) return b;
        if (a.isFloat() and b.isFloat()) {
            return if (a.itemsize() >= b.itemsize()) a else b;
        }
        // mixed integers
        if (a.isSigned() != b.isSigned()) {
            const ia = a.itemsize();
            const ib = b.itemsize();
            return if (ia > ib) a else b;
        }
        return if (a.itemsize() >= b.itemsize()) a else b;
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .f64 => "f64",
            .f32 => "f32",
            .f16 => "f16",
            .i64 => "i64",
            .i32 => "i32",
            .i16 => "i16",
            .i8 => "i8",
            .u8 => "u8",
            .bool => "bool",
        };
    }
};

test "dtype basics" {
    try std.testing.expect(DType.f32.itemsize() == 4);
    try std.testing.expect(DType.of(f32) == .f32);
    try std.testing.expect(DType.promote(.f32, .i64) == .f32);
    try std.testing.expect(DType.parse("f64").? == .f64);
    try std.testing.expect(DType.bool.isFloat() == false);
}
