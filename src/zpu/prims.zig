//! Primitive operations: the *only* things the tracing, autodiff and JIT layers
//! need to know about.
const std = @import("std");

pub const MAX_RANK: usize = 8;

/// Every primitive ZPU knows. The set is deliberately small: higher level
/// routines (linalg, nn) are expressed as compositions of these.
pub const Prim = enum(u16) {
    // element-wise unary
    neg,
    abs,
    exp,
    expm1,
    log,
    log1p,
    sqrt,
    rsqrt,
    square,
    recip,
    cbrt,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    erf,
    sigmoid,
    softplus,
    relu,
    floor,
    ceil,
    round,
    trunc,
    sign,
    isnan,
    isinf,
    isfinite,
    // element-wise binary
    add,
    sub,
    mul,
    div,
    pow,
    maximum,
    minimum,
    atan2,
    hypot,
    mod_op,
    // comparisons / logic (boolean results)
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    logical_and,
    logical_or,
    logical_xor,
    logical_not,
    // selection
    where,
    clamp,
    lerp,
    // reductions
    sum,
    prod_op,
    max_reduce,
    min_reduce,
    any_reduce,
    all_reduce,
    argmax,
    argmin,
    // shape / movement
    reshape,
    transpose,
    broadcast,
    expand_dims,
    squeeze,
    slice,
    concat_op,
    stack_op,
    take,
    scatter_add,
    pad,
    flip,
    cumsum,
    reverse,
    // linear algebra
    matmul,
    solve,
    inv,
    det,
    cholesky,
    triangular_solve,
    conv,
    // constant (no inputs)
    constant,
    // misc
    copy,
    cast_op,
};

pub const Error = error{ InvalidAxis, ShapeMismatch, RankTooLarge, IndexOutOfRange };

/// Attributes of a primitive. ZPU stores payload-relative axes for the shape
/// ops (see `ops.zig`), which is what makes `vmap` composable with them.
pub const Attrs = struct {
    axis: i32 = AXIS_NONE,
    axis2: i32 = AXIS_NONE,
    /// explicit permutation (transpose), length in `nsl`
    perm: [MAX_RANK]u8 = @splat(0),
    /// target shape (reshape/broadcast)
    shape: [MAX_RANK]usize = @splat(0),
    starts: [MAX_RANK]i64 = @splat(0),
    lens: [MAX_RANK]usize = @splat(0),
    steps: [MAX_RANK]isize = @splat(1),
    pads_lo: [MAX_RANK]usize = @splat(0),
    pads_hi: [MAX_RANK]usize = @splat(0),
    /// number of significant entries in `perm` / `starts` / `lens` / `steps`
    nsl: u8 = 0,
    shape_len: u8 = 0,
    /// scalar payload for prims that take one (pad value, clamp bounds...)
    scalar: f64 = 0,
    /// dtype payload for `cast` (a `DType` tag)
    dt: u8 = 0,
    flags: u16 = 0,

    pub const KEEP_DIMS: u16 = 1 << 0;
    pub const UPPER: u16 = 1 << 1;
    pub const LEFT: u16 = 1 << 2;
    pub const RIGHT: u16 = 1 << 3;
    pub const MASK: u16 = 1 << 4;
    pub const REVERSE: u16 = 1 << 5;

    pub fn keepDims(self: Attrs) bool {
        return self.flags & KEEP_DIMS != 0;
    }
    pub fn withKeepDims(self: Attrs, v: bool) Attrs {
        var a = self;
        if (v) a.flags |= KEEP_DIMS else a.flags &= ~KEEP_DIMS;
        return a;
    }
    pub fn upper(self: Attrs) bool {
        return self.flags & UPPER != 0;
    }
    pub fn withUpper(self: Attrs, v: bool) Attrs {
        var a = self;
        if (v) a.flags |= UPPER else a.flags &= ~UPPER;
        return a;
    }
    pub fn reverse(self: Attrs) bool {
        return self.flags & REVERSE != 0;
    }
    pub fn withReverse(self: Attrs, v: bool) Attrs {
        var a = self;
        if (v) a.flags |= REVERSE else a.flags &= ~REVERSE;
        return a;
    }
};

pub const AXIS_NONE: i32 = -2;
pub const AXIS_LAST: i32 = -1;

pub fn arity(p: Prim) u8 {
    return switch (p) {
        .constant => 0,
        .where => 3,
        .add, .sub, .mul, .div, .pow, .maximum, .minimum, .atan2, .hypot, .mod_op, .eq, .ne, .lt, .le, .gt, .ge, .logical_and, .logical_or, .logical_xor, .matmul, .concat_op, .stack_op, .take, .scatter_add, .lerp, .clamp, .conv, .solve, .triangular_solve => 2,
        else => 1,
    };
}

/// Whether the primitive's result is only valid for well formed matrices.
pub fn isLinalg(p: Prim) bool {
    return switch (p) {
        .matmul, .solve, .inv, .det, .cholesky, .triangular_solve => true,
        else => false,
    };
}

/// Whether the primitive produces a boolean result.
pub fn isPredicate(p: Prim) bool {
    return switch (p) {
        .eq, .ne, .lt, .le, .gt, .ge, .logical_and, .logical_or, .logical_xor, .logical_not, .isnan, .isinf, .isfinite, .any_reduce, .all_reduce => true,
        else => false,
    };
}

/// Broadcastable element-wise primitives (no shape change beyond broadcasting).
pub fn isElementwise(p: Prim) bool {
    return switch (p) {
        .neg, .abs, .exp, .expm1, .log, .log1p, .sqrt, .rsqrt, .square, .recip, .cbrt, .sin, .cos, .tan, .asin, .acos, .atan, .sinh, .cosh, .tanh, .asinh, .acosh, .atanh, .erf, .sigmoid, .softplus, .relu, .floor, .ceil, .round, .trunc, .sign, .isnan, .isinf, .isfinite, .add, .sub, .mul, .div, .pow, .maximum, .minimum, .atan2, .hypot, .mod_op, .eq, .ne, .lt, .le, .gt, .ge, .logical_and, .logical_or, .logical_xor, .logical_not, .where, .clamp, .lerp, .copy, .cast_op => true,
        else => false,
    };
}

/// Primitives that produce a *view* of their input (aliasing storage).
pub fn isView(p: Prim) bool {
    return switch (p) {
        .reshape, .broadcast, .expand_dims, .squeeze, .slice, .transpose, .flip, .reverse, .copy, .cast_op => true,
        else => false,
    };
}

test "prim arity" {
    try std.testing.expectEqual(@as(u8, 2), arity(.matmul));
    try std.testing.expectEqual(@as(u8, 3), arity(.where));
    try std.testing.expect(isPredicate(.lt));
    try std.testing.expect(!isPredicate(.add));
}

/// True when the primitive has a meaningful derivative.
pub fn differentiable(p: Prim) bool {
    return switch (p) {
        .floor, .ceil, .round, .trunc, .sign, .isnan, .isinf, .isfinite, .argmax, .argmin, .all_reduce, .any_reduce, .constant, .eq, .ne, .lt, .le, .gt, .ge, .logical_and, .logical_or, .logical_xor, .logical_not => false,
        else => true,
    };
}

/// `true` when the primitive only accepts floating point operands.
pub fn isFloatOnly(p: Prim) bool {
    return switch (p) {
        .exp, .expm1, .log, .log1p, .sqrt, .rsqrt, .sin, .cos, .tan, .asin, .acos, .atan, .sinh, .cosh, .tanh, .asinh, .acosh, .atanh, .erf, .sigmoid, .softplus, .cbrt, .atan2, .hypot => true,
        else => false,
    };
}
