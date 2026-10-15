//! The gradient table: every differentiable operation, checked against the
//! *definition* of the derivative rather than against another hand-written
//! formula.
//!
//! Each case is `d f / d x_i ≈ (f(x + h) − f(x − h)) / 2h`, computed by
//! `zpu.gradcheck`. The scalar function is the sum of squares of the op's output,
//! so a mistake cannot cancel element against element within a projection: a rule
//! that is wrong by a factor, a sign, a transpose or a missing term shows up in
//! the numbers.
//!
//! The table is `comptime`: each row names a Zig function, and the harness
//! instantiates a closure per row (`B.unary(op)`), because a Zig function pointer
//! is not a closure and a test table should not need one.
//!
//! Both floating point widths run for the elementwise rows. `f32` is not a
//! smaller `f64` for this purpose: the step and the tolerance both come from the
//! dtype, and a gradient that only survives in `f64` is a gradient users cannot
//! train with.
const std = @import("std");
const zpu = @import("../zpu.zig");

const alloc = std.testing.allocator;
const Tensor = zpu.Tensor;
const Shape = zpu.Shape;
const DType = zpu.DType;
const Error = zpu.tensor.Error;
const Fn1 = fn (Tensor) Error!Tensor;
const Fn2 = fn (Tensor, Tensor) Error!Tensor;

// ---------------------------------------------------------------- the harness

/// Where the sample values of a row live: a row that needs `log` cannot use the
/// same inputs as one that needs `acos`.
const Domain = enum {
    /// |v| >= 0.07 — away from the kinks of `abs`, `relu` and `sign`.
    any,
    /// Strictly positive, bounded away from zero: `log`, `sqrt`, `rsqrt`, `prod`.
    positive,
    /// |v| <= 0.9: `asin`, `acos`, `atanh`, `tan`.
    unit,
    /// v >= 1.3: `acosh`.
    above_one,
};

fn value(seed: f64, i: usize, domain: Domain) f64 {
    const t = seed + @as(f64, @floatFromInt(i)) * 1.7;
    const raw = std.math.sin(t) * 1.2 + 0.05 * std.math.cos(t * 2.3);
    return switch (domain) {
        .any => if (@abs(raw) < 0.07) raw + 0.19 else raw,
        .positive => 0.45 + @abs(raw),
        .unit => std.math.clamp(raw * 0.5, -0.9, 0.9),
        .above_one => 1.3 + @abs(raw),
    };
}

/// A deterministic `dims`-shaped sample. Everything in the table is built from
/// it, so a failure is reproducible without a seed.
fn sample(dt: DType, comptime dims: []const usize, domain: Domain, seed: f64) !Tensor {
    var buf: [256]f64 = undefined;
    var n: usize = 1;
    for (dims) |d| n *= d;
    std.debug.assert(n <= buf.len);
    for (0..n) |i| buf[i] = value(seed, i, domain);
    const base = try Tensor.fromArray(f64, buf[0..n], try Shape.init(dims));
    return if (dt == .f64) base else zpu.cast(base, dt);
}

/// `sum(y²)`: monotone in every element, so nothing cancels.
fn energy(y: Tensor) Error!Tensor {
    return zpu.sumAll(try zpu.square(y));
}

const B = struct {
    /// `x ↦ sum(op(x)²)`, as a `transforms.Fn`.
    fn unary(comptime op: Fn1) zpu.transforms.Fn {
        return struct {
            fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
                _ = ud;
                return energy(try op(args[0]));
            }
        }.call;
    }
    /// `(a, b) ↦ sum(op(a, b)²)`.
    fn binary(comptime op: Fn2) zpu.transforms.Fn {
        return struct {
            fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
                _ = ud;
                return energy(try op(args[0], args[1]));
            }
        }.call;
    }
    /// `(x, y) ↦ sum(op(x, y)²)` for ops with two arguments and an axis.
    fn pair(comptime op: fn (Tensor, Tensor, i32, bool) Error!Tensor, comptime axis: i32) zpu.transforms.Fn {
        return struct {
            fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
                _ = ud;
                return energy(try op(args[0], args[1], axis, false));
            }
        }.call;
    }
    /// `x ↦ sum(op(x, axis)²)` for the reductions.
    fn reduce(comptime op: fn (Tensor, i32, bool) Error!Tensor, comptime axis: i32) zpu.transforms.Fn {
        return struct {
            fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
                _ = ud;
                return energy(try op(args[0], axis, false));
            }
        }.call;
    }
};

fn optsFor(dt: DType) zpu.gradcheck.Options {
    return switch (dt) {
        .f32 => .{ .atol = 3e-3, .rtol = 2e-2 },
        .f16 => .{ .atol = 5e-2, .rtol = 5e-2 },
        else => .{ .atol = 1e-6, .rtol = 1e-5 },
    };
}

/// Runs the rows and reports every failure, not only the first: a table with one
/// broken row is more useful than a table that stops at it.
const Harness = struct {
    checked: usize = 0,
    failed: usize = 0,

    fn one(self: *Harness, name: []const u8, dt: DType, f: zpu.transforms.Fn, args: []Tensor, wrt: []const usize) !void {
        const report = zpu.gradcheck.check(alloc, f, null, args, wrt, optsFor(dt)) catch |err| {
            std.debug.print("gradcheck {s} ({s}): error {s}\n", .{ name, @tagName(dt), @errorName(err) });
            self.failed += 1;
            return;
        };
        self.checked += report.compared;
        if (report.mismatched > 0 or report.compared == 0) {
            std.debug.print("gradcheck {s} ({s}): {f}", .{ name, @tagName(dt), report });
            self.failed += 1;
        }
    }

    fn unary(self: *Harness, comptime name: []const u8, comptime op: Fn1, domain: Domain, dt: DType) !void {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{try sample(dt, &.{ 6, 2 }, domain, 0.31)};
        try self.one(name, dt, B.unary(op), &args, &.{0});
    }

    fn binary(self: *Harness, comptime name: []const u8, comptime op: Fn2, a: Domain, b: Domain, dt: DType) !void {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{
            try sample(dt, &.{ 6, 2 }, a, 0.31),
            try sample(dt, &.{ 6, 2 }, b, 3.77),
        };
        try self.one(name, dt, B.binary(op), &args, &.{ 0, 1 });
    }

    fn finish(self: *Harness, comptime table: []const u8) !void {
        if (self.failed > 0) {
            std.debug.print("gradient table {s}: {d} rows failed, {d} derivatives checked\n", .{
                table, self.failed, self.checked,
            });
            return error.GradientTableFailed;
        }
        std.debug.assert(self.checked > 0);
    }
};

// ------------------------------------------------------------------- the table

test "gradient table: elementwise unary, f64 and f32" {
    const cases = .{
        .{ "neg", zpu.neg, Domain.any },
        .{ "abs", zpu.abs, Domain.any },
        .{ "exp", zpu.exp, Domain.any },
        .{ "expm1", zpu.expm1, Domain.any },
        .{ "log", zpu.log, Domain.positive },
        .{ "log1p", zpu.log1p, Domain.positive },
        .{ "sqrt", zpu.sqrt, Domain.positive },
        .{ "rsqrt", zpu.rsqrt, Domain.positive },
        .{ "square", zpu.square, Domain.any },
        .{ "recip", zpu.recip, Domain.any },
        .{ "cbrt", zpu.cbrt, Domain.positive },
        .{ "sin", zpu.sin, Domain.any },
        .{ "cos", zpu.cos, Domain.any },
        .{ "tan", zpu.tan, Domain.unit },
        .{ "asin", zpu.asin, Domain.unit },
        .{ "acos", zpu.acos, Domain.unit },
        .{ "atan", zpu.atan, Domain.any },
        .{ "sinh", zpu.sinh, Domain.any },
        .{ "cosh", zpu.cosh, Domain.any },
        .{ "tanh", zpu.tanh, Domain.any },
        .{ "asinh", zpu.asinh, Domain.any },
        .{ "acosh", zpu.acosh, Domain.above_one },
        .{ "atanh", zpu.atanh, Domain.unit },
        .{ "erf", zpu.erf, Domain.any },
        .{ "sigmoid", zpu.sigmoid, Domain.any },
        .{ "softplus", zpu.softplus, Domain.any },
        .{ "relu", zpu.relu, Domain.any },
        .{ "leaky-relu", Leaky, Domain.any },
        .{ "silu", zpu.nn.silu, Domain.any },
        .{ "gelu", zpu.nn.gelu, Domain.any },
        .{ "clamp", Clamp, Domain.any },
        .{ "cast-up", CastUp, Domain.any },
    };
    var h = Harness{};
    inline for (cases) |c| {
        try h.unary(c[0], c[1], c[2], .f64);
        try h.unary(c[0], c[1], c[2], .f32);
    }
    try h.finish("elementwise unary");
}

fn Leaky(x: Tensor) Error!Tensor {
    return zpu.nn.leakyRelu(x, 0.1);
}

fn Clamp(x: Tensor) Error!Tensor {
    return zpu.clamp(x, -0.8, 0.9);
}

fn CastUp(x: Tensor) Error!Tensor {
    return zpu.cast(x, .f64);
}

test "gradient table: elementwise binary, f64 and f32" {
    const cases = .{
        .{ "add", zpu.add, Domain.any, Domain.any },
        .{ "sub", zpu.sub, Domain.any, Domain.any },
        .{ "mul", zpu.mul, Domain.any, Domain.any },
        .{ "div", zpu.div, Domain.any, Domain.any },
        .{ "pow", zpu.pow, Domain.positive, Domain.any },
        .{ "maximum", zpu.maximum, Domain.any, Domain.any },
        .{ "minimum", zpu.minimum, Domain.any, Domain.any },
        .{ "atan2", zpu.atan2, Domain.any, Domain.any },
        .{ "hypot", zpu.hypot, Domain.any, Domain.any },
        .{ "lerp", Lerp, Domain.any, Domain.any },
        .{ "mse", zpu.nn.mse, Domain.any, Domain.any },
        .{ "mae", zpu.nn.mae, Domain.any, Domain.any },
    };
    var h = Harness{};
    inline for (cases) |c| {
        try h.binary(c[0], c[1], c[2], c[3], .f64);
        try h.binary(c[0], c[1], c[2], c[3], .f32);
    }
    try h.finish("elementwise binary");
}

fn Lerp(a: Tensor, b: Tensor) Error!Tensor {
    return zpu.lerp(a, b, 0.3);
}

test "gradient table: reductions and normalisations, f64 and f32" {
    const cases = .{
        .{ "sum-axis1", B.reduce(zpu.sum, 1), &.{0} },
        .{ "sum-axis0", B.reduce(zpu.sum, 0), &.{0} },
        .{ "sum-all", SumAll, &.{0} },
        .{ "mean-axis1", B.reduce(zpu.mean, 1), &.{0} },
        .{ "mean-all", MeanAll, &.{0} },
        .{ "max-axis1", B.reduce(zpu.max, 1), &.{0} },
        .{ "min-axis1", B.reduce(zpu.min, 1), &.{0} },
        .{ "variance-axis1", B.reduce(zpu.variance, 1), &.{0} },
        .{ "stddev-axis1", B.reduce(zpu.stddev, 1), &.{0} },
        .{ "cumsum-axis1", Cumsum, &.{0} },
        .{ "cumsum-reverse-axis1", CumsumReverse, &.{0} },
        .{ "softmax-axis1", Softmax, &.{0} },
        .{ "log-softmax-axis1", LogSoftmax, &.{0} },
        .{ "log-sum-exp", LogSumExp, &.{0} },
    };
    var h = Harness{};
    inline for (cases) |c| {
        inline for ([_]DType{ .f64, .f32 }) |dt| {
            var scope = try zpu.Scope.init(alloc);
            defer scope.deinit();
            var args = [_]Tensor{try sample(dt, &.{ 3, 4 }, .positive, 0.31)};
            try h.one(c[0], dt, c[1], &args, c[2]);
        }
    }
    try h.finish("reductions");
}

fn SumAll(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.sumAll(args[0]));
}

fn MeanAll(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.meanAll(args[0]));
}

fn Cumsum(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.cumsum(args[0], 1, false));
}

fn CumsumReverse(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.cumsum(args[0], 1, true));
}

fn Softmax(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.softmax(args[0], 1));
}

fn LogSoftmax(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.logSoftmax(args[0], 1));
}

fn LogSumExp(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.nn.logSumExp(args[0], 1, false));
}

fn Prod(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    return energy(try zpu.prod(args[0], 1, false));
}

test "gradient table: shape, indexing and layout" {
    var h = Harness{};

    const Cases = struct {
        fn transposeFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.transpose(args[0], &.{ 1, 0 }));
        }
        fn reshapeFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.reshape(args[0], try Shape.init(&.{ 3, 4 })));
        }
        fn flipFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.flip(args[0], 1));
        }
        fn padFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.pad(args[0], &.{ .{ 1, 2 }, .{ 0, 1 } }, 0.25));
        }
        fn narrowFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.narrow(args[0], 1, 1, 2));
        }
        fn broadcastFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.broadcastTo(args[0], try Shape.init(&.{ 5, 5 })));
        }
        fn concatFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.concat(args[0], args[1], 1));
        }
        fn stackFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.stack(args[0], args[1], 0));
        }
        fn expandFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.squeeze(try zpu.expandDims(args[0], 1), 1));
        }
        fn whereFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const mask = try zpu.gt(args[0], try zpu.scalarLike(args[0], 0));
            return energy(try zpu.where(mask, args[0], args[1]));
        }
        fn takeFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const idx = try Tensor.fromArray(i32, &.{ 3, 0, 3 }, try Shape.init(&.{3}));
            return energy(try zpu.take(args[0], idx, 1));
        }
        fn scatterFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const idx = try Tensor.fromArray(i32, &.{ 1, 0 }, try Shape.init(&.{2}));
            const src = try Tensor.fromArray(f64, &.{ 0.5, -0.25, 0.3, 0.1, -0.4, 0.7 }, try Shape.init(&.{ 3, 2 }));
            return energy(try zpu.scatterAdd(args[0], idx, src, 1));
        }
        fn embeddingFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const idx = try Tensor.fromArray(i32, &.{ 2, 0 }, try Shape.init(&.{2}));
            return energy(try zpu.nn.embedding(args[0], idx));
        }
    };

    const rows = .{
        .{ "transpose", Cases.transposeFn, &[_]usize{0} },
        .{ "reshape", Cases.reshapeFn, &[_]usize{0} },
        .{ "flip", Cases.flipFn, &[_]usize{0} },
        .{ "pad", Cases.padFn, &[_]usize{0} },
        .{ "narrow", Cases.narrowFn, &[_]usize{0} },
        .{ "concat", Cases.concatFn, &[_]usize{ 0, 1 } },
        .{ "stack", Cases.stackFn, &[_]usize{ 0, 1 } },
        .{ "expand-squeeze", Cases.expandFn, &[_]usize{0} },
        .{ "where", Cases.whereFn, &[_]usize{ 0, 1 } },
        .{ "take", Cases.takeFn, &[_]usize{0} },
        .{ "embedding", Cases.embeddingFn, &[_]usize{0} },
    };

    inline for (rows) |row| {
        inline for ([_]DType{ .f64, .f32 }) |dt| {
            var scope = try zpu.Scope.init(alloc);
            defer scope.deinit();
            var args = [_]Tensor{
                try sample(dt, &.{ 3, 4 }, .any, 0.31),
                try sample(dt, &.{ 3, 4 }, .any, 3.77),
            };
            try h.one(row[0], dt, row[1], &args, row[2]);
        }
    }
    try h.finish("shape and indexing");
}

test "gradient table: linear algebra" {
    var h = Harness{};

    const Cases = struct {
        fn matmulFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.matmul(args[0], args[1]));
        }
        fn matVecFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.matmul(args[0], args[1]));
        }
        fn batchedFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.matmul(args[0], args[1]));
        }
        fn solveFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.solve(args[0], args[1]));
        }
        fn invFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.inv(args[0]));
        }
        fn detFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.det(args[0]));
        }
        fn diagFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.diag(args[0]));
        }
        fn traceFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.trace(args[0]));
        }
        fn normFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.normAll(args[0]));
        }
        fn outerFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.linalg.outer(args[0], args[1]));
        }
    };

    // A fixed symmetric positive definite matrix: `solve`, `inv` and `det` all
    // want a well conditioned input, so a failure cannot be a numerical accident.
    const spd = [_]f64{
        4.0, 1.0, 0.5,
        1.0, 3.0, 0.25,
        0.5, 0.25, 2.0,
    };

    inline for ([_]DType{ .f64, .f32 }) |dt| {
        inline for (.{ "matmul", "batched" }) |name| {
            var scope = try zpu.Scope.init(alloc);
            defer scope.deinit();
            if (comptime std.mem.eql(u8, name, "batched")) {
                var args = [_]Tensor{
                    try sample(dt, &.{ 2, 3, 3 }, .any, 0.31),
                    try sample(dt, &.{ 2, 3, 3 }, .any, 3.77),
                };
                try h.one(name, dt, Cases.batchedFn, &args, &.{ 0, 1 });
            } else {
                var args = [_]Tensor{
                    try sample(dt, &.{ 3, 3 }, .any, 0.31),
                    try sample(dt, &.{ 3, 3 }, .any, 3.77),
                };
                try h.one(name, dt, Cases.matmulFn, &args, &.{ 0, 1 });
            }
        }
    }

    inline for (.{
        .{ "inv", Cases.invFn, false },
        .{ "det", Cases.detFn, false },
        .{ "norm", Cases.normFn, false },
    }) |row| {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        const a = try Tensor.fromArray(f64, &spd, try Shape.init(&.{ 3, 3 }));
        if (row[2]) {
            var args = [_]Tensor{ a, try sample(.f64, &.{3}, .any, 0.9) };
            try h.one(row[0], .f64, row[1], &args, &.{ 0, 1 });
        } else {
            var args = [_]Tensor{a};
            try h.one(row[0], .f64, row[1], &args, &.{0});
        }
    }

    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{
            try sample(.f64, &.{3}, .any, 0.4),
            try sample(.f64, &.{4}, .any, 2.1),
        };
        try h.one("outer", .f64, Cases.outerFn, &args, &.{ 0, 1 });
    }
    try h.finish("linear algebra");
}

test "gradient table: neural network layers and losses" {
    var h = Harness{};

    const Cases = struct {
        fn layerNormFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.layerNorm(args[0], args[1], args[2], 1e-5, -1));
        }
        fn rmsNormFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.rmsNorm(args[0], args[1], 1e-5, -1));
        }
        fn convFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.conv2d(args[0], args[1], 1, 1));
        }
        fn convStrideFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.conv2d(args[0], args[1], 2, 0));
        }
        fn denseFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.dense(args[0], args[1], args[2]));
        }
        fn maxPoolFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.maxPool2d(args[0], 2));
        }
        fn avgPoolFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.nn.avgPool2d(args[0], 2));
        }
        fn crossEntropyFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const labels = try zpu.softmax(args[1], 1);
            return zpu.nn.softmaxCrossEntropy(args[0], labels);
        }
        fn bceFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const pred = try zpu.sigmoid(args[0]);
            return zpu.nn.binaryCrossEntropy(pred, args[1], 1e-7);
        }
        fn bceLogitsFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return zpu.nn.binaryCrossEntropyWithLogits(args[0], args[1]);
        }
        fn attentionFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            // Scaled dot-product attention: `softmax(x xᵀ / s) x`, the shape a
            // transformer is assembled from.
            const scores = try zpu.divScalar(try zpu.matmul(args[0], try zpu.transpose(args[0], &.{ 1, 0 })), 2.0);
            return energy(try zpu.matmul(try zpu.softmax(scores, -1), args[0]));
        }
    };

    inline for ([_]DType{ .f64, .f32 }) |dt| {
        inline for (.{ "layer-norm", "rms-norm" }) |name| {
            var scope = try zpu.Scope.init(alloc);
            defer scope.deinit();
            const f: zpu.transforms.Fn = if (comptime std.mem.eql(u8, name, "layer-norm")) Cases.layerNormFn else Cases.rmsNormFn;
            var args = if (comptime std.mem.eql(u8, name, "layer-norm"))
                [_]Tensor{
                    try sample(dt, &.{ 4, 5 }, .any, 0.31),
                    try sample(dt, &.{5}, .any, 1.1),
                    try sample(dt, &.{5}, .any, 2.2),
                }
            else
                [_]Tensor{
                    try sample(dt, &.{ 4, 5 }, .any, 0.31),
                    try sample(dt, &.{5}, .any, 1.1),
                };
            const wrt: []const usize = if (comptime std.mem.eql(u8, name, "layer-norm")) &.{ 0, 1, 2 } else &.{ 0, 1 };
            try h.one(name, dt, f, &args, wrt);
        }
    }

    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        // 1 batch, 1 channel, 6x6 image, 2 filters of 3x3.
        var args = [_]Tensor{
            try sample(.f64, &.{ 1, 1, 6, 6 }, .any, 0.31),
            try sample(.f64, &.{ 2, 1, 3, 3 }, .any, 3.3),
        };
        try h.one("conv2d", .f64, Cases.convFn, &args, &.{ 0, 1 });
    }
    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{
            try sample(.f64, &.{ 1, 1, 4, 4 }, .any, 0.31),
        };
        try h.one("max-pool2d", .f64, Cases.maxPoolFn, &args, &.{0});
        try h.one("avg-pool2d", .f64, Cases.avgPoolFn, &args, &.{0});
    }
    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{
            try sample(.f64, &.{ 3, 4 }, .any, 0.31),
            try sample(.f64, &.{ 4, 2 }, .any, 1.7),
            try sample(.f64, &.{2}, .any, 2.4),
        };
        try h.one("dense", .f64, Cases.denseFn, &args, &.{ 0, 1, 2 });
    }
    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{
            try sample(.f64, &.{ 4, 3 }, .any, 0.31),
            try sample(.f64, &.{ 4, 3 }, .unit, 0.98),
        };
        try h.one("softmax-cross-entropy", .f64, Cases.crossEntropyFn, &args, &.{ 0, 1 });
    }

    {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        var args = [_]Tensor{try sample(.f64, &.{ 4, 3 }, .any, 0.31)};
        try h.one("attention", .f64, Cases.attentionFn, &args, &.{0});
    }
    try h.finish("neural network layers");
}

test "gradient table: discrete ops really do have a zero gradient" {
    // Not every op needs a rule: a floor or a comparison is locally constant, and
    // the honest answer is zero. This checks that the *numeric* derivative agrees
    // — the zero is a property of the function, not a classifier giving up.
    var h = Harness{};
    const Cases = struct {
        fn floorFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.floor(args[0]));
        }
        fn roundFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.round(args[0]));
        }
        fn truncFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.trunc(args[0]));
        }
        fn signFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            return energy(try zpu.sign(args[0]));
        }
        fn comparisonFn(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const m = try zpu.gt(args[0], try zpu.scalarLike(args[0], 0));
            return energy(try zpu.cast(m, .f64));
        }
    };
    inline for (.{
        .{ "floor", Cases.floorFn },
        .{ "round", Cases.roundFn },
        .{ "trunc", Cases.truncFn },
        .{ "sign", Cases.signFn },
        .{ "gt-then-cast", Cases.comparisonFn },
    }) |row| {
        var scope = try zpu.Scope.init(alloc);
        defer scope.deinit();
        // Sample values are away from the integers, so a step of `h` cannot
        // cross a boundary and turn a constant into a slope.
        var args = [_]Tensor{try sample(.f64, &.{ 4, 2 }, .any, 0.31)};
        try h.one(row[0], .f64, row[1], &args, &.{0});
    }
    try h.finish("discrete ops");
}


test "gradient table: forward and reverse mode agree on the same number" {
    // `jvp` is reverse-over-reverse internally, so this is also a check that the
    // nested tape lands on the gradient of the gradient rather than on something
    // that merely has the right shape.
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    const Loss = struct {
        fn call(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
            _ = ud;
            const s = try zpu.sin(args[0]);
            const e = try zpu.exp(try zpu.mulScalar(args[0], 0.25));
            return zpu.sumAll(try zpu.mul(s, e));
        }
    };

    const x = try sample(.f64, &.{5}, .unit, 0.31);
    var args = [_]Tensor{x};
    const tangent = try sample(.f64, &.{5}, .unit, 1.7);
    var tangents = [_]Tensor{tangent};

    const primal = try zpu.jvp(alloc, Loss.call, null, &args, &tangents);
    defer primal.value.deinit();
    defer primal.tangent.deinit();

    const grads = try zpu.grad(alloc, Loss.call, null, &args, &.{0});
    defer {
        for (grads) |g| {
            var gg = g;
            gg.deinit();
        }
        alloc.free(grads);
    }
    const contracted = (try zpu.sumAll(try zpu.mul(grads[0], tangent))).item();
    try std.testing.expectApproxEqAbs(contracted, primal.tangent.item(), 1e-9 * @max(1, @abs(contracted)));

    // And the second derivative, which is the reason `jvp` exists at all.
    const dir = try sample(.f64, &.{5}, .unit, 2.4);
    const hvp = try zpu.hvp(alloc, Loss.call, null, &args, 0, dir);
    defer hvp.deinit();
    try std.testing.expectEqual(@as(usize, 5), hvp.numel());
}
