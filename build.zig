//! ZPU / ZGF build script (Zig 0.17).
//!
//! Steps:
//!
//! * `zig build test`       — unit, numerical and property tests (`-Dtest-filter`, `-Dtest-trace`)
//! * `zig build bench`      — benchmark suite (`-Dbench-filter=…`, `-Dbench-json=…`)
//! * `zig build doc`        — regenerate the HTML feature reference into `zig-out/doc`
//! * `zig build examples`   — build every example
//! * `zig build run-<name>` — run a single example
//! * `zig build serve`      — live rendering demo server
//! * `zig build check`      — `zig fmt --check` plus the test suite
//!
//! Everything the repository needs is expressed here: no Python, no Makefile and
//! no shell script is part of the build path.
const std = @import("std");
const builtin = @import("builtin");

/// True when `rel` exists in the source tree. Optional steps (examples, bench,
/// docs) are wired up only when their sources are present, so a partial checkout
/// still builds.
fn exists(b: *std.Build, rel: []const u8) bool {
    const cwd = std.Io.Dir.cwd();
    cwd.access(b.graph.io, rel, .{}) catch {
        // Fall back to the build root when the caller built from elsewhere.
        _ = b.root;
        return false;
    };
    return true;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ------------------------------------------------------------------ modules
    // `zpu`: the JAX-flavoured numerical core. `zgf`: the graphics library.
    // A single shared backend module backs both (CPU / WebGPU / Metal / Vulkan / D3D12).
    const backend_mod = b.addModule("zbackend", .{
        .root_source_file = b.path("src/backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    const zpu_mod = b.addModule("zpu", .{
        .root_source_file = b.path("src/zpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    zpu_mod.addImport("zbackend", backend_mod);
    const zgf_mod = b.addModule("zgf", .{
        .root_source_file = b.path("src/zgf.zig"),
        .target = target,
        .optimize = optimize,
    });
    zgf_mod.addImport("zpu", zpu_mod);
    zgf_mod.addImport("zbackend", backend_mod);

    // -------------------------------------------------------------------- tests
    const test_step = b.step("test", "Run the ZPU/ZGF test suite");
    const options = b.addOptions();
    options.addOption(bool, "trace_alloc", b.option(bool, "test-trace", "Dump every allocation made by the test suite") orelse false);
    options.addOption(bool, "std_alloc", if (b.option([]const u8, "test-alloc", "Allocator used by the test suite: 'detector' (default) or 'std'")) |v| std.mem.eql(u8, v, "std") else false);
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this string");
    const options_mod = options.createModule();

    const imports = [_]std.Build.Module.Import{
        .{ .name = "zpu", .module = zpu_mod },
        .{ .name = "zgf", .module = zgf_mod },
        .{ .name = "zbackend", .module = backend_mod },
        .{ .name = "zpu_options", .module = options_mod },
    };
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &imports,
    });
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    // --------------------------------------------------------------- benchmarks
    if (exists(b, "bench/bench.zig")) {
        const bench_opts = b.addOptions();
        bench_opts.addOption(?[]const u8, "bench_filter", b.option([]const u8, "bench-filter", "Only run benchmarks whose name contains this string"));
        bench_opts.addOption(?[]const u8, "bench_json", b.option([]const u8, "bench-json", "Write results as JSON to this path"));
        const bench_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zpu", .module = zpu_mod },
                .{ .name = "zgf", .module = zgf_mod },
                .{ .name = "zbackend", .module = backend_mod },
                .{ .name = "bench_options", .module = bench_opts.createModule() },
            },
        });
        const bench_exe = b.addExecutable(.{ .name = "zpu-bench", .root_module = bench_mod });
        const run_bench = b.addRunArtifact(bench_exe);
        run_bench.addPassthruArgs();
        b.step("bench", "Run the benchmark suite").dependOn(&run_bench.step);
        b.installArtifact(bench_exe);
    }

    // --------------------------------------------------------------------- docs
    // The HTML reference is a Zig program (`docs/build_docs.zig`), so the
    // toolchain stays self contained — no Python, no scripts.
    if (exists(b, "docs/build_docs.zig")) {
        const docs_mod = b.createModule(.{
            .root_source_file = b.path("docs/build_docs.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        });
        const docs_exe = b.addExecutable(.{ .name = "build-docs", .root_module = docs_mod });
        const run_docs = b.addRunArtifact(docs_exe);
        run_docs.addPassthruArgs();
        b.step("doc", "Generate the HTML feature reference into zig-out/doc").dependOn(&run_docs.step);
    }

    // ----------------------------------------------------------------- examples
    const Example = struct { name: []const u8, file: []const u8, desc: []const u8 };
    const examples = [_]Example{
        .{ .name = "tensors", .file = "examples/tensors.zig", .desc = "Tensor + ops tour" },
        .{ .name = "autodiff", .file = "examples/autodiff.zig", .desc = "grad / jvp / hvp / vmap / jit" },
        .{ .name = "train-mlp", .file = "examples/train_mlp.zig", .desc = "Train an MLP with Adam + jit" },
        .{ .name = "images", .file = "examples/images.zig", .desc = "ZGF image codecs, filters, drawing" },
        .{ .name = "vector-art", .file = "examples/vector_art.zig", .desc = "ZGF 2D vector renderer poster" },
        .{ .name = "render3d", .file = "examples/render3d.zig", .desc = "ZGF software 3D renderer" },
        .{ .name = "diff-render", .file = "examples/diff_render.zig", .desc = "Differentiable rendering (zpu x zgf)" },
        .{ .name = "gpu-probe", .file = "examples/gpu_probe.zig", .desc = "List the graphics backends available on this machine" },
    };

    const examples_step = b.step("examples", "Build every example");
    inline for (examples) |ex| {
        if (exists(b, ex.file)) {
            const mod = b.createModule(.{
                .root_source_file = b.path(ex.file),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zpu", .module = zpu_mod },
                    .{ .name = "zgf", .module = zgf_mod },
                    .{ .name = "zbackend", .module = backend_mod },
                },
            });
            const exe = b.addExecutable(.{ .name = ex.name, .root_module = mod });
            const install = b.addInstallArtifact(exe, .{});
            examples_step.dependOn(&install.step);
            const run = b.addRunArtifact(exe);
            run.addPassthruArgs();
            const run_step = b.step(b.fmt("run-{s}", .{ex.name}), ex.desc);
            run_step.dependOn(&run.step);
        }
    }

    // ------------------------------------------------------------ demo web server
    if (exists(b, "examples/serve_demo.zig")) {
        const server_mod = b.createModule(.{
            .root_source_file = b.path("examples/serve_demo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpu", .module = zpu_mod },
                .{ .name = "zgf", .module = zgf_mod },
                .{ .name = "zbackend", .module = backend_mod },
            },
        });
        const server = b.addExecutable(.{ .name = "serve-demo", .root_module = server_mod });
        const run_server = b.addRunArtifact(server);
        b.step("serve", "Start the live rendering demo web server").dependOn(&run_server.step);
    }

    // -------------------------------------------------------------------- checks
    const check_step = b.step("check", "Formatting check plus the full test suite");
    const fmt_check = b.addFmt(.{ .paths = &.{ b.path("src"), b.path("bench"), b.path("docs"), b.path("examples"), b.path("build.zig") }, .check = true });
    check_step.dependOn(&fmt_check.step);
    check_step.dependOn(&run_unit_tests.step);
    _ = builtin;
}
