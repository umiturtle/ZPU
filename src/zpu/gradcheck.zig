//! Finite-difference gradient checking.
//!
//! A backward rule is code, and code is wrong sometimes — quietly. The test that
//! catches a wrong rule is not another hand-derived derivative (that is the same
//! code twice); it is the *definition* of the derivative, evaluated numerically:
//!
//!     d f / d x_i  ≈  (f(x + h e_i) − f(x − h e_i)) / 2h
//!
//! This module is that comparison, over every element of every argument you ask
//! about. It is shipped rather than test-only because it is the first thing to
//! reach for when *your* function's gradient looks wrong: it says which argument,
//! which element, what the analytic value was and what the number says.
//!
//! ```zig
//! const report = try zpu.gradcheck.check(alloc, loss, null, &args, &.{0});
//! if (!report.ok(.{})) {
//!     std.debug.print("{f}\n", .{report});
//!     return error.GradientMismatch;
//! }
//! ```
//!
//! Choices that matter:
//!
//! * **Central differences, not forward ones.** Two evaluations per element,
//!   but the truncation error is `O(h²)` instead of `O(h)`, which is what makes
//!   the default tolerances tight enough to be worth checking.
//! * **The step comes from the dtype.** `h` is chosen near `cbrt(eps)` for the
//!   *element type* (`~6e-6` for f64, `~5e-3` for f32), because a step that is
//!   tiny relative to the arithmetic's own noise measures rounding rather than
//!   slope. Override it when you know better.
//! * **Elements are compared with `atol + rtol·scale`.** A gradient of 1e-12
//!   and a gradient of 1 are not held to the same absolute standard.
const std = @import("std");
const tmod = @import("tensor.zig");
const transforms = @import("transforms.zig");

const Tensor = tmod.Tensor;
const Error = tmod.Error;

pub const Options = struct {
    /// The finite-difference step. `null` picks one from the element type.
    eps: ?f64 = null,
    /// Absolute tolerance, for gradients near zero.
    atol: f64 = 1e-4,
    /// Relative tolerance, for gradients far from zero.
    rtol: f64 = 1e-2,
};

/// What the comparison found. `worst_*` describe the single largest offender,
/// which is the one a person usually wants to look at first.
pub const Report = struct {
    /// Number of partial derivatives compared.
    compared: usize = 0,
    /// Number that exceeded the tolerance.
    mismatched: usize = 0,
    max_abs_error: f64 = 0,
    max_rel_error: f64 = 0,
    /// Argument index (into the arguments passed to `check`), or 0 when nothing
    /// was compared.
    worst_arg: usize = 0,
    /// Flat element index inside that argument.
    worst_index: usize = 0,
    analytic: f64 = 0,
    numeric: f64 = 0,

    pub fn ok(self: Report, opts: Options) bool {
        return self.mismatched == 0 and self.compared > 0 and self.max_abs_error <= opts.atol;
    }

    pub fn format(self: Report, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (self.compared == 0) {
            try w.writeAll("gradcheck: nothing compared\n");
            return;
        }
        if (self.mismatched == 0) {
            try w.print("gradcheck: {d} derivatives agree (max abs {e}, max rel {e})\n", .{
                self.compared, self.max_abs_error, self.max_rel_error,
            });
            return;
        }
        try w.print("gradcheck: {d} of {d} derivatives disagree; worst is arg {d}, element {d}: analytic {e}, numeric {e} (abs {e})\n", .{
            self.mismatched, self.compared, self.worst_arg, self.worst_index,
            self.analytic,        self.numeric,   self.max_abs_error,
        });
    }
};

/// The step that balances truncation against rounding for a dtype.
pub fn defaultStep(dt: @import("dtype.zig").DType) f64 {
    return switch (dt) {
        .f16 => 5e-2,
        .f32 => 5e-3,
        .f64 => 6e-6,
        else => 1e-3,
    };
}

/// Compares the reverse-mode gradient of `f` at `args` against central
/// differences, over the arguments listed in `wrt` (indices into `args`).
///
/// `f` must return a single number: a gradient is only defined for a scalar, and
/// a checker that guessed what to do with a vector output would hide the
/// mistake. The arguments are perturbed in place and restored, so they must be
/// writable, and `f` must not keep its own copy of them.
pub fn check(
    alloc: std.mem.Allocator,
    f: transforms.Fn,
    ud: ?*anyopaque,
    args: []Tensor,
    wrt: []const usize,
    opts: Options,
) !Report {
    var report: Report = .{};
    if (wrt.len == 0) return report;

    const base = try evaluate(alloc, f, ud, args);
    for (wrt) |j| if (j >= args.len) return Error.InvalidShape;

    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    const grads = try transforms.grad(alloc, f, ud, args, wrt);
    defer {
        for (grads) |g| {
            var gg = g;
            gg.deinit();
        }
        alloc.free(grads);
    }

    for (wrt, 0..) |j, gi| {
        const a = args[j];
        const g = grads[gi];
        if (g.numel() != a.numel()) return Error.ShapeMismatch;
        const h = opts.eps orelse defaultStep(a.dtype());
        for (0..a.numel()) |i| {
            const keep = a.readLogical(a.numel(), i);
            a.writeF64(i, keep + h);
            const plus = try evaluate(alloc, f, ud, args);
            a.writeF64(i, keep - h);
            const minus = try evaluate(alloc, f, ud, args);
            a.writeF64(i, keep);

            const numeric = (plus - minus) / (2 * h);
            const analytic = g.readLogical(g.numel(), i);
            const err = @abs(numeric - analytic);
            const scale = @max(@as(f64, 1), @max(@abs(numeric), @abs(analytic)));
            const rel = err / scale;

            report.compared += 1;
            if (!(err <= opts.atol + opts.rtol * scale)) report.mismatched += 1;
            if (err > report.max_abs_error or report.compared == 1) {
                report.max_abs_error = err;
                report.worst_arg = j;
                report.worst_index = i;
                report.analytic = analytic;
                report.numeric = numeric;
            }
            report.max_rel_error = @max(report.max_rel_error, rel);
        }
    }
    // Keep the primal value alive across the loop so the comparison is against
    // the same evaluation the gradient was taken at.
    _ = base;
    return report;
}

/// Runs `f` in its own scope and reads the (scalar) result.
fn evaluate(alloc: std.mem.Allocator, f: transforms.Fn, ud: ?*anyopaque, args: []Tensor) !f64 {
    var scope = try tmod.Scope.init(alloc);
    defer scope.deinit();
    const y = try f(ud, args);
    if (y.numel() != 1) return Error.InvalidShape;
    return y.item();
}

/// `check`, but it returns the number rather than the report when everything
/// agrees and fails with `error.GradientMismatch` when it does not.
pub fn expectAll(
    alloc: std.mem.Allocator,
    f: transforms.Fn,
    ud: ?*anyopaque,
    args: []Tensor,
    wrt: []const usize,
    opts: Options,
) !Report {
    const report = try check(alloc, f, ud, args, wrt, opts);
    if (!report.ok(opts) or report.mismatched > 0) {
        std.debug.print("{f}", .{report});
        return error.GradientMismatch;
    }
    return report;
}

// ---------------------------------------------------------------- self-tests

const testing = std.testing;

fn scalarSum(ud: ?*anyopaque, args: []const Tensor) anyerror!Tensor {
    _ = ud;
    const a = try @import("ops.zig").mul(args[0], args[0]);
    const b = try @import("ops.zig").sin(args[1]);
    return @import("ops.zig").sumAll(try @import("ops.zig").add(a, b));
}

test "the checker agrees with a gradient it is supposed to agree with" {
    const alloc = testing.allocator;
    try @import("context.zig").init(alloc, .{});
    defer @import("context.zig").deinit();

    var x = try Tensor.fromArray(f64, &.{ 0.3, -1.2, 2.5 }, try @import("shape.zig").Shape.init(&.{3}));
    defer x.deinit();
    var y = try Tensor.fromArray(f64, &.{ 0.7, -0.4, 1.1 }, try @import("shape.zig").Shape.init(&.{3}));
    defer y.deinit();

    var args = [_]Tensor{ x, y };
    const report = try check(alloc, scalarSum, null, &args, &.{ 0, 1 }, .{});
    try testing.expectEqual(@as(usize, 6), report.compared);
    try testing.expectEqual(@as(usize, 0), report.mismatched);
    try testing.expect(report.max_abs_error < 1e-8);
    try testing.expect(report.ok(.{}));
}

test "the checker notices a gradient that is off by a little" {
    const alloc = testing.allocator;
    try @import("context.zig").init(alloc, .{});
    defer @import("context.zig").deinit();

    // `abs` has a kink at zero: finite differences straddling it see a slope the
    // derivative does not have. A checker that cannot fail is not a checker.
    var x = try Tensor.fromArray(f64, &.{ 1e-9, 1.0, 2.0 }, try @import("shape.zig").Shape.init(&.{3}));
    defer x.deinit();
    var args = [_]Tensor{x};

    const Abs = struct {
        fn f(ud: ?*anyopaque, a: []const Tensor) anyerror!Tensor {
            _ = ud;
            return @import("ops.zig").sumAll(try @import("ops.zig").abs(a[0]));
        }
    };
    const report = try check(alloc, Abs.f, null, &args, &.{0}, .{});
    try testing.expectEqual(@as(usize, 3), report.compared);
    try testing.expectEqual(@as(usize, 1), report.mismatched);
    try testing.expect(!report.ok(.{}));

    // ... and the same function away from the kink agrees.
    x.writeF64(0, 5.0);
    const clean = try check(alloc, Abs.f, null, &args, &.{0}, .{});
    try testing.expectEqual(@as(usize, 0), clean.mismatched);
}

test "the step comes from the dtype" {
    try testing.expectEqual(@as(f64, 6e-6), defaultStep(.f64));
    try testing.expect(defaultStep(.f32) > defaultStep(.f64));
}
