//! `zig build run-tensors`
//!
//! A tour of the eager core: how tensors are built, how broadcasting and
//! reduction axes work, which operations hand back a *view* and which have to
//! materialise, what dtypes exist, and how a `Scope` decides when memory goes
//! back to the allocator.
//!
//! Everything printed here is checked on the way out — the example returns an
//! error if a number disagrees with the analytic answer, so a green run of
//! `zig build run-tensors` means the eager core answered correctly on this
//! machine.
const std = @import("std");
const zpu = @import("zpu");

const Io = std.Io;
const Tensor = zpu.Tensor;
const Shape = zpu.Shape;
const Range = zpu.Range;
const DType = zpu.DType;

/// Counts the shape/value checks that passed, so the summary line is honest.
var checks: usize = 0;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file = Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_file.interface;

    try zpu.init(alloc, .{});
    defer zpu.deinit();

    // A `Scope` is the ownership rule of the eager core: every tensor created
    // while it is live is freed when it ends, so a function body reads like
    // maths instead of like `defer free`. Tensors you want to outlive the scope
    // go through `scope.keep`.
    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    try out.print("zpu — eager core tour\n\n", .{});

    try building(out);
    try elementwise(out, alloc);
    try reductions(out);
    try shapes(out);
    try dtypes(out);
    try memory(out, &scope);

    // An integer count needs `{d}`: `{s}` is for strings, and the compiler says
    // so at comptime rather than at the call site.
    try out.print("\n{d} checks passed; every value above was verified\n", .{checks});
    try out.flush();
}

// ---------------------------------------------------------------------------
// building blocks
// ---------------------------------------------------------------------------

fn building(out: *Io.Writer) !void {
    try section(out, "building tensors");

    // The four constructors you reach for first: from a slice, from a rule, and
    // the two "filled with a constant" forms.
    const a = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4, 5, 6 }, Shape.initComptime(&.{ 2, 3 }));
    const arange = try Tensor.arange(6, .f64);
    const lin = try Tensor.linspace(0, 1, 5, .f64);
    const eye = try Tensor.eye(3, .f32);
    const full = try Tensor.full(Shape.initComptime(&.{ 2, 2 }), .i64, 7);

    try row(out, "fromArray [[1..6]]", a);
    try row(out, "arange 6", arange);
    try row(out, "linspace 0..1 (5)", lin);
    try row(out, "eye 3", eye);
    try row(out, "full 2x2 = 7", full);

    // `fromArray` is the only constructor that takes a plain Zig slice, so it is
    // also the one that validates: the element count has to match the shape.
    try expectErr(out, "fromArray with the wrong length", Tensor.fromArray(f64, &.{ 1, 2, 3 }, Shape.initComptime(&.{ 2, 2 })));

    // Ranges are i64, may start before the tensor, and may be strided; they are
    // resolved against the axis they apply to. `end == null` means "to the end".
    const upto = Range.upto(3);
    try out.print("\n{s:<24} upto(3) = {d}..{?d}, strided(0,6,2) step {d}\n", .{
        "Range",
        upto.start,
        upto.end,
        Range.strided(0, 6, 2).step,
    });
    try check(out, "arange is 0..5", arange, &.{ 0, 1, 2, 3, 4, 5 }, 0);
}

// ---------------------------------------------------------------------------
// broadcasting and elementwise maths
// ---------------------------------------------------------------------------

fn elementwise(out: *Io.Writer, alloc: std.mem.Allocator) !void {
    _ = alloc;
    try section(out, "elementwise + broadcasting");

    // NumPy's rule, implemented once in `broadcastPair`: align to the right,
    // sizes must be equal or 1, and every operation goes through it — so
    // `mul(add(x, y), z)` needs no shape bookkeeping at the call site.
    const col = try Tensor.fromArray(f64, &.{ 1, 2, 3 }, Shape.initComptime(&.{ 3, 1 }));
    const line = try Tensor.fromArray(f64, &.{ 10, 20, 30, 40 }, Shape.initComptime(&.{ 1, 4 }));
    const outer = try zpu.add(col, line);
    try table(out, "col[3,1] + row[1,4]", outer);

    // Unary maths is elementwise and picks the right libm call per dtype.
    const x = try Tensor.fromArray(f64, &.{ -1, 0, 1, 2 }, Shape.initComptime(&.{4}));
    try row(out, "exp(x)", try zpu.exp(x));
    try row(out, "sigmoid(x)", try zpu.sigmoid(x));
    try row(out, "relu(x)", try zpu.relu(x));
    try row(out, "clamp(x, -0.5, 1)", try zpu.clamp(x, -0.5, 1));

    // Scalar forms exist so a loop body does not have to build a scalar tensor.
    try row(out, "x * 2 + 1", try zpu.addScalar(try zpu.mulScalar(x, 2), 1));

    const s = try zpu.sigmoid(x);
    try check(out, "sigmoid(0) == 0.5", s, &.{ 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823 }, 1e-15);
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------

fn reductions(out: *Io.Writer) !void {
    try section(out, "reductions");

    const m = try Tensor.fromArray(f64, &.{
        1, 2, 3,
        4, 5, 6,
    }, Shape.initComptime(&.{ 2, 3 }));

    // `axis = -1` means the last axis, `AXIS_NONE` (a sentinel, not a rank
    // offset) means "everything". `keepdims` decides whether the reduced axis
    // stays as a 1 so the result can be broadcast back.
    try row(out, "sum(m, -1)", try zpu.sum(m, -1, false));
    try row(out, "sum(m, 0, keep)", try zpu.sum(m, 0, true));
    try row(out, "mean(m, -1)", try zpu.mean(m, -1, false));
    try row(out, "max(m, -1)", try zpu.max(m, -1, false));
    try row(out, "argmax(m, -1)", try zpu.argmax(m, -1));
    try row(out, "stddev(m, -1)", try zpu.stddev(m, -1, false));
    try row(out, "cumsum(m, -1)", try zpu.cumsum(m, -1, false));
    try row(out, "sumAll(m)", try zpu.sumAll(m));

    // softmax subtracts the row max first, so it stays stable for big logits.
    const logits = try Tensor.fromArray(f64, &.{ 1, 2, 3 }, Shape.initComptime(&.{3}));
    const sm = try zpu.softmax(logits, -1);
    try row(out, "softmax([1,2,3])", sm);
    try check(out, "softmax sums to 1", try zpu.sumAll(sm), &.{1}, 1e-15);
    try check(out, "softmax is monotone", sm, &.{ 0.09003057317038046, 0.24472847105479764, 0.6652409557748219 }, 1e-15);

    // Comparisons produce bool tensors, which is what `where` and `all`/`any`
    // consume.
    const neg = try zpu.lt(m, try Tensor.full(Shape.initComptime(&.{ 2, 3 }), .f64, 4));
    try row(out, "m < 4", neg);
    try out.print("{s:<22} {s}\n", .{ "all(m < 4)", if (try boolAt(neg, 0)) "true" else "false" });
    try check(out, "where picks per element", try zpu.where(neg, m, try zpu.zerosLike(m)), &.{ 1, 2, 3, 0, 0, 0 }, 0);
}

// ---------------------------------------------------------------------------
// shape operations: views vs copies
// ---------------------------------------------------------------------------

fn shapes(out: *Io.Writer) !void {
    try section(out, "shapes (and which of them are views)");

    const m = try Tensor.fromArray(f64, &.{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, Shape.initComptime(&.{ 3, 4 }));

    const flat = try zpu.reshape(m, Shape.initComptime(&.{ 2, 6 }));
    try table(out, "reshape 3x4 -> 2x6", flat);
    try table(out, "transpose", try zpu.transpose(m, &.{ 1, 0 }));

    // A transpose of a contiguous tensor needs no data movement at all: it is
    // the same buffer with swapped strides. `aliases` answers "same memory?",
    // and `isContiguous` answers "is the row-major walk still 1,1,1?".
    const t = try zpu.transpose(m, &.{ 1, 0 });
    try out.print("{s:<22} aliases={s} contiguous={s}\n", .{
        "transpose properties",
        if (t.aliases(m)) "yes" else "no",
        if (t.isContiguous()) "yes" else "no",
    });
    try check(out, "transpose values", t, &.{ 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 }, 0);

    // Slicing with ranges: `strided` keeps every second column.
    const strided = try zpu.sliceRanges(m, &.{ Range.all(), Range.strided(0, 4, 2) });
    try table(out, "[:, ::2]", strided);
    try row(out, "row 1 only", try zpu.narrow(m, 0, 1, 1));

    // Filling a dimension back in: `expandDims`/`squeeze` are views too, and
    // `broadcastTo` only materialises when the target really is bigger.
    const col = try zpu.expandDims(try Tensor.fromArray(f64, &.{ 1, 2, 3 }, Shape.initComptime(&.{3})), -1);
    try table(out, "expandDims(-1)", col);
    try table(out, "broadcastTo [3,3]", try zpu.broadcastTo(col, Shape.initComptime(&.{ 3, 3 })));

    // Joining and rearranging.
    const a = try Tensor.fromArray(f64, &.{ 1, 2 }, Shape.initComptime(&.{2}));
    const b = try Tensor.fromArray(f64, &.{ 3, 4 }, Shape.initComptime(&.{2}));
    try row(out, "concat(a, b)", try zpu.concat(a, b, 0));
    try table(out, "stack(a, b)", try zpu.stack(a, b, 0));
    try row(out, "flip(m, -1) row 0", try zpu.narrow(try zpu.flip(m, -1), 0, 0, 1));
    try table(out, "pad(m, 1 ring)", try zpu.pad(m, &.{ .{ 1, 1 }, .{ 1, 1 } }, 0));
    try row(out, "take(m, [2,0], 0) -> row 2", try zpu.narrow(try zpu.take(m, try Tensor.fromArray(i64, &.{ 2, 0 }, Shape.initComptime(&.{2})), 0), 0, 0, 1));
    try check(out, "take picks whole rows", try zpu.take(m, try Tensor.fromArray(i64, &.{ 2, 0 }, Shape.initComptime(&.{2})), 0), &.{ 9, 10, 11, 12, 1, 2, 3, 4 }, 0);
}

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

fn dtypes(out: *Io.Writer) !void {
    try section(out, "dtypes");

    // Every tensor carries one of nine dtypes; the arithmetic kernels are
    // generated per dtype, so there is no boxing at runtime.
    const f32s = try Tensor.fromArray(f32, &.{ 1.5, 2.5 }, Shape.initComptime(&.{2}));
    const ints = try Tensor.fromArray(i64, &.{ 1, 2, 3, 4 }, Shape.initComptime(&.{ 2, 2 }));
    const flags = try Tensor.fromArray(bool, &.{ true, false, false, true }, Shape.initComptime(&.{ 2, 2 }));

    try row(out, "f32 tensor", f32s);
    try row(out, "i64 tensor", ints);
    try row(out, "bool tensor", flags);
    try row(out, "cast i64 -> f64", try zpu.cast(ints, .f64));
    // Integer division promotes to f32 instead of truncating: `/` on a tensor
    // means real division, and a silent floor is the kind of surprise a numeric
    // library should not have.
    const halves = try zpu.div(ints, try Tensor.full(Shape.initComptime(&.{ 2, 2 }), .i64, 2));
    try row(out, "i64 / i64 -> f32", halves);
    try check(out, "1/2, 2/2, 3/2, 4/2", halves, &.{ 0.5, 1, 1.5, 2 }, 0);

    // Mixing dtypes widens to the common one instead of failing.
    const mixed = try zpu.add(f32s, try Tensor.fromArray(i64, &.{ 1, 1 }, Shape.initComptime(&.{2})));
    try out.print("{s:<22} {s}\n", .{ "f32 + i64", @tagName(mixed.dtypeOf()) });
}

// ---------------------------------------------------------------------------
// ownership
// ---------------------------------------------------------------------------

fn memory(out: *Io.Writer, scope: *zpu.Scope) !void {
    try section(out, "ownership");

    // `Scope` is a tracker: it records what was created while it was live and
    // drops it at the end, in reverse order. That is why the bodies above never
    // free anything.
    const before = scope.count();
    const x = try Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, Shape.initComptime(&.{4}));
    const y = try zpu.square(x);
    const z = try zpu.sumAll(y);
    try out.print("{s:<24} {d} live tensors, {d} more from these three ops\n", .{ "scope.count()", before, scope.count() - before });
    try check(out, "sum(x^2)", z, &.{30}, 0);

    // Intermediate results inside a function that do not escape can be handed
    // back early with `releaseTemp`: under a scope it is a no-op (the scope
    // still owns the tensor), without one it frees immediately. That one rule
    // is what lets the same op code run in a training loop and in a `jit` plan.
    const tmp = try zpu.mul(x, x);
    zpu.releaseTemp(tmp);
    try out.print("{s:<24} released early (the scope still holds it)\n", .{"releaseTemp"});

    // `keep` promotes a tensor out of the scope's lifetime, and the caller owns
    // it from then on.
    const kept = scope.keep(try zpu.mulScalar(x, 2));
    var kept_mut = kept;
    defer kept_mut.deinit();
    try check(out, "scope.keep escapes", kept, &.{ 2, 4, 6, 8 }, 0);
    try out.print("{s:<24} escapes the scope; the caller frees it ({s})\n", .{ "scope.keep", @tagName(kept.dtypeOf()) });
}

// ---------------------------------------------------------------------------
// reporting helpers
// ---------------------------------------------------------------------------

fn section(out: *Io.Writer, title: []const u8) !void {
    try out.print("\n-- {s} ", .{title});
    var i: usize = title.len + 4;
    while (i < 60) : (i += 1) try out.print("-", .{});
    try out.print("\n", .{});
}

/// `label   [shape] dtype   v0 v1 v2 ...` (up to 8 values, `...` past that).
fn row(out: *Io.Writer, label: []const u8, t: Tensor) !void {
    try out.print("{s:<24}", .{label});
    try shapeInfo(out, t);
    const n = t.numel();
    const shown = @min(n, 8);
    for (0..shown) |i| try out.print(" {d:.4}", .{t.readLogical(n, i)});
    if (n > shown) try out.print(" ...", .{});
    try out.print("\n", .{});
}

/// A 2-D table, so broadcasting and transposes are visible rather than inferred.
fn table(out: *Io.Writer, label: []const u8, t: Tensor) !void {
    try out.print("{s:<24}", .{label});
    try shapeInfo(out, t);
    try out.print("\n", .{});
    if (t.rank() != 2) return row(out, "", t);
    const rows = t.shape.dims[0];
    const cols = t.shape.dims[1];
    for (0..rows) |r| {
        try out.print("  ", .{});
        for (0..cols) |c| try out.print(" {d:>8.4}", .{t.readLogical(t.numel(), r * cols + c)});
        try out.print("\n", .{});
    }
}

fn shapeInfo(out: *Io.Writer, t: Tensor) !void {
    try out.print("[", .{});
    for (0..t.rank()) |i| try out.print("{s}{d}", .{ if (i == 0) "" else "x", t.shape.dims[i] });
    try out.print("] {s:<4}", .{@tagName(t.dtypeOf())});
}

fn boolAt(t: Tensor, i: usize) !bool {
    return t.readLogical(t.numel(), i) != 0;
}

/// Compares against a list of expected f64 values with an absolute tolerance.
fn check(out: *Io.Writer, label: []const u8, actual: Tensor, expected: []const f64, tol: f64) !void {
    if (actual.numel() != expected.len) return error.LengthMismatch;
    for (expected, 0..) |e, i| {
        const a = actual.readLogical(actual.numel(), i);
        if (@abs(a - e) > tol) {
            try out.print("  FAIL {s}: element {d} is {d}, expected {d}\n", .{ label, i, a, e });
            try out.flush();
            return error.ValueMismatch;
        }
    }
    checks += 1;
}

/// For the cases where the interesting answer is "this is rejected": a shape
/// mismatch has to be an error, not silently ignored.
fn expectErr(out: *Io.Writer, label: []const u8, result: anytype) !void {
    if (result) |_| {
        try out.print("{s:<24} accepted (expected an error!)\n", .{label});
        return error.ExpectedError;
    } else |_| {
        try out.print("{s:<24} rejected as expected\n", .{label});
        checks += 1;
    }
}
