//! ZPU / ZGF build script (Zig 0.17).
//!
//! Steps:
//!
//! * `zig build test`       — unit, numerical and property tests (`-Dtest-filter`)
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
    // Resolved against the build root, not the process cwd, so `zig build` works
    // from any directory.
    //
    // Note for whoever adds a file: the build runner caches the *result* of this
    // probe (keyed by `build.zig`), so a newly added example or a fresh checkout
    // that reuses a build cache may not see it until `build.zig` is touched or
    // the cache is dropped (`rm -rf .zig-cache`). CI always starts clean.
    b.root.access(b.graph.io, rel, .{}) catch return false;
    return true;
}

/// The library modules, at a chosen optimisation level.
///
/// This is a function rather than a block inside `build` because the benchmark
/// needs the *same* module graph compiled for speed: a benchmark that measures
/// `Debug` code and prints "GB/s" is measuring the wrong thing, and since each
/// module is its own compilation unit, passing `-Doptimize` to the top level
/// does not reach the library modules a program imports.
const Library = struct {
    backend: *std.Build.Module,
    zpu: *std.Build.Module,
    zgf: *std.Build.Module,
    zui: *std.Build.Module,
    zapp: *std.Build.Module,
};

fn createLibrary(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, public: bool) Library {
    const add = struct {
        fn one(bb: *std.Build, name: []const u8, path: []const u8, t: std.Build.ResolvedTarget, opt: std.builtin.OptimizeMode, is_public: bool) *std.Build.Module {
            return if (is_public)
                bb.addModule(name, .{ .root_source_file = bb.path(path), .target = t, .optimize = opt })
            else
                bb.createModule(.{ .root_source_file = bb.path(path), .target = t, .optimize = opt });
        }
    }.one;

    const backend = add(b, "zbackend", "src/backend.zig", target, optimize, public);
    const zpu = add(b, "zpu", "src/zpu.zig", target, optimize, public);
    const zgf = add(b, "zgf", "src/zgf.zig", target, optimize, public);
    zgf.addImport("zpu", zpu);
    const zui = add(b, "zui", "src/zui.zig", target, optimize, public);
    zui.addImport("zpu", zpu);
    zui.addImport("zgf", zgf);
    zui.addImport("zbackend", backend);
    const zapp = add(b, "zapp", "src/zapp.zig", target, optimize, public);
    zapp.addImport("zpu", zpu);
    zapp.addImport("zgf", zgf);
    zapp.addImport("zui", zui);
    return .{ .backend = backend, .zpu = zpu, .zgf = zgf, .zui = zui, .zapp = zapp };
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ------------------------------------------------------------------ modules
    // `zpu`: the JAX-flavoured numerical core. `zgf`: the graphics library.
    // A single shared backend module backs both (CPU / WebGPU / Metal / Vulkan / D3D12).
    // `zpu`: the JAX-flavoured numerical core. `zgf`: the graphics library.
    // `zui`: the 2-D UI library, a user of both — layout and widgets never touch
    // a device; only `zui/render.zig` turns a display list into pixels. The
    // module graph is flat, so all three are peers. A single shared backend
    // module backs all of them (CPU / WebGPU / Metal / Vulkan / D3D12).
    const lib = createLibrary(b, target, optimize, true);
    const backend_mod = lib.backend;
    const zpu_mod = lib.zpu;
    const zgf_mod = lib.zgf;
    const zui_mod = lib.zui;

    // -------------------------------------------------------------------- tests
    const test_step = b.step("test", "Run the ZPU/ZGF test suite");
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this string");

    const zapp_mod = lib.zapp;

    const imports = [_]std.Build.Module.Import{
        .{ .name = "zpu", .module = zpu_mod },
        .{ .name = "zgf", .module = zgf_mod },
        .{ .name = "zui", .module = zui_mod },
        .{ .name = "zbackend", .module = backend_mod },
        .{ .name = "zapp", .module = zapp_mod },
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

    // Every module also carries unit tests next to the code it covers. They are
    // collected by the module roots themselves (see the `test` block at the end
    // of src/zpu.zig and src/backend.zig), so each module gets its own runner.
    const module_test_mods = [_]*std.Build.Module{ zpu_mod, zgf_mod, backend_mod, zui_mod };
    for (module_test_mods) |mod| {
        const t = b.addTest(.{ .root_module = mod });
        const run = b.addRunArtifact(t);
        test_step.dependOn(&run.step);
    }

    // ----------------------------------------------------------- snapshots
    // The snapshot suite is its own root because it is the only test that reads
    // and writes files: everything else is a pure function of its inputs. Its
    // references live in `snapshots/` and are rewritten only when asked
    // (`-Dupdate-snapshots`), so a normal `zig build test` can never move them.
    if (exists(b, "tests/snapshots.zig")) {
        const snap_opts = b.addOptions();
        snap_opts.addOption(
            bool,
            "snapshot_update",
            b.option(bool, "update-snapshots", "Rewrite snapshots/ instead of comparing against it") orelse false,
        );
        const snap_mod = b.createModule(.{
            .root_source_file = b.path("tests/snapshots.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zpu", .module = zpu_mod },
                .{ .name = "zgf", .module = zgf_mod },
                .{ .name = "zui", .module = zui_mod },
                .{ .name = "zbackend", .module = backend_mod },
                .{ .name = "build_options", .module = snap_opts.createModule() },
            },
        });
        const snap_tests = b.addTest(.{ .root_module = snap_mod });
        const run_snap = b.addRunArtifact(snap_tests);
        run_snap.setCwd(b.path("."));
        test_step.dependOn(&run_snap.step);
    }

    // --------------------------------------------------------------- benchmarks
    if (exists(b, "bench/bench.zig")) {
        const bench_opts = b.addOptions();
        bench_opts.addOption(?[]const u8, "bench_filter", b.option([]const u8, "bench-filter", "Only run benchmarks whose name contains this string"));
        bench_opts.addOption(?[]const u8, "bench_json", b.option([]const u8, "bench-json", "Write results as JSON to this path"));
        // The library itself is compiled for speed here, not for debuggability:
        // a benchmark of a `Debug` build is a benchmark of the debugger.
        const fast = createLibrary(b, target, .ReleaseFast, false);
        const bench_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zpu", .module = fast.zpu },
                .{ .name = "zgf", .module = fast.zgf },
                .{ .name = "zui", .module = fast.zui },
                .{ .name = "zbackend", .module = fast.backend },
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
        .{ .name = "diff-render", .file = "examples/diff_render.zig", .desc = "Differentiable rendering: fit a scene to a drawn target (zpu x zgf)" },
        .{ .name = "ui-demo", .file = "examples/ui_demo.zig", .desc = "ZUI: a scripted UI, rendered headless into a filmstrip" },
        .{ .name = "gpu-probe", .file = "examples/gpu_probe.zig", .desc = "List the graphics backends available on this machine" },
        .{ .name = "zapp-demo", .file = "examples/zapp_demo.zig", .desc = "ZAPP: Application framework demo" },
        .{ .name = "resize-test", .file = "examples/resize_test.zig", .desc = "ZAPP: Window resize test (100x100 to 4K)" },
    };

    const examples_step = b.step("examples", "Build every example");
    inline for (examples) |ex| {
        if (exists(b, ex.file)) {
            const mod = b.createModule(.{
                .root_source_file = b.path(ex.file),
                .target = target,
                .optimize = optimize,
                // A graphics driver is a C++ shared object on most machines, so
                // its constructors have to run. The platform loader does that
                // when a program links libc; Zig's built-in ELF loader (what the
                // library itself uses) does not. Programs that talk to real
                // drivers want libc; the library never needs it.
                .link_libc = std.mem.eql(u8, ex.name, "gpu-probe") or std.mem.eql(u8, ex.name, "zapp-demo") or std.mem.eql(u8, ex.name, "resize-test"),
                .imports = &.{
                    .{ .name = "zpu", .module = zpu_mod },
                    .{ .name = "zgf", .module = zgf_mod },
                    .{ .name = "zui", .module = zui_mod },
                    .{ .name = "zbackend", .module = backend_mod },
                    .{ .name = "zapp", .module = zapp_mod },
                },
            });
            const exe = b.addExecutable(.{ .name = ex.name, .root_module = mod });
            
            // Link X11 for zapp-demo on Linux
            if ((std.mem.eql(u8, ex.name, "zapp-demo") or std.mem.eql(u8, ex.name, "resize-test")) and target.result.os.tag == .linux) {
                exe.linkSystemLibrary("X11");
            }
            
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
    // only format what is actually present, so a partial checkout still checks
    var fmt_paths: [5]std.Build.LazyPath = undefined;
    var fmt_count: usize = 0;
    for ([_][]const u8{ "src", "bench", "docs", "examples", "build.zig" }) |rel| {
        if (!exists(b, rel)) continue;
        fmt_paths[fmt_count] = b.path(rel);
        fmt_count += 1;
    }
    const fmt_check = b.addFmt(.{ .paths = fmt_paths[0..fmt_count], .check = true });
    check_step.dependOn(&fmt_check.step);
    check_step.dependOn(&run_unit_tests.step);
    _ = builtin;
}
