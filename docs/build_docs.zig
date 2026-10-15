//! ZPU documentation generator.
//!
//! `zig build doc` runs this program; it reads the *source of truth* (the
//! Markdown files under `docs/` and the doc comments in `src/`) and writes a
//! static HTML site into `zig-out/doc/`. Nothing else is in the path — no
//! Python, no Node, no template engine, no network.
//!
//! What it understands:
//!
//! * Markdown subset: `#`-headings, paragraphs, `-`/`*` and `1.` lists, tables,
//!   fenced code blocks, `` `code` ``, `**bold**`, `*italic*`,
//!   `[text](target)` links (`*.md` targets are rewritten to `.html`).
//! * Zig doc comments: the `//!` module header and every `///` block in front of
//!   a `pub` declaration become one API page per source file.
//!
//! The output is deliberately boring: one CSS file, no JavaScript, predictable
//! file names, so the site can be opened straight from disk.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const out_root = "zig-out/doc";

/// The site is written twice: Japanese at the root, English under `en/`, with a
/// switcher between them. Both are generated from Markdown sources that live
/// side by side (`docs/…` and `docs/en/…`), so the two versions are always the
/// same shape.
const Lang = enum { ja, en };

const Spec = struct {
    lang: Lang,
    /// directory prefix inside the output, "" for Japanese, "en/" for English
    out_prefix: []const u8,
    /// where the Markdown lives
    doc_dir: []const u8,
    /// Japanese pages that are working documents: MILESTONES.md and README.md
    include_repo_docs: bool,

    // labels
    toc: []const u8,
    declarations: []const u8,
    declarations_hint: []const u8,
    source_line: []const u8,
    group_intro: []const u8,
    group_features: []const u8,
    title_index: []const u8,
    title_guide: []const u8,
    title_comparison: []const u8,
    title_milestones: []const u8,
    title_readme: []const u8,
    switcher_here: []const u8,
    switcher_other: []const u8,
};

fn specFor(lang: Lang) Spec {
    return switch (lang) {
        .ja => .{
            .lang = .ja,
            .out_prefix = "",
            .doc_dir = "docs",
            .include_repo_docs = true,
            .toc = "目次",
            .declarations = "宣言",
            .declarations_hint = "ソース内の <code>///</code> コメントから生成。",
            .source_line = "元ファイル: ",
            .group_intro = "はじめに",
            .group_features = "機能",
            .title_index = "ZPU",
            .title_guide = "ガイド",
            .title_comparison = "他ライブラリとの違い",
            .title_milestones = "マイルストーン",
            .title_readme = "README",
            .switcher_here = "日本語",
            .switcher_other = "English",
        },
        .en => .{
            .lang = .en,
            .out_prefix = "en/",
            .doc_dir = "docs/en",
            .include_repo_docs = false,
            .toc = "Contents",
            .declarations = "Declarations",
            .declarations_hint = "Generated from the <code>///</code> comments in the source.",
            .source_line = "Source: ",
            .group_intro = "Getting started",
            .group_features = "Features",
            .title_index = "ZPU",
            .title_guide = "Guide",
            .title_comparison = "How it differs",
            .title_milestones = "Milestones",
            .title_readme = "README",
            .switcher_here = "English",
            .switcher_other = "日本語",
        },
    };
}

/// Everything the renderer needs to know about the page it is writing: the
/// arena, the language, and how to walk back up to the documentation root (so
/// `*.md` links resolve from any subdirectory).
const Ctx = struct {
    arena: Allocator,
    spec: Spec,
    up: []const u8,

    fn prefixed(ctx: Ctx, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(ctx.arena, "{s}{s}", .{ ctx.up, name });
    }
};

/// One navigation entry.
const Page = struct {
    /// file name inside the output directory, e.g. "features/ops.html"
    out_name: []const u8,
    /// link title
    title: []const u8,
    /// source file, relative to the repository root
    src_name: []const u8,
    /// sidebar section
    group: []const u8,
    /// rendered body (without the surrounding HTML skeleton)
    body: []const u8 = "",
};

const areas = [_][]const u8{ "features", "api" };

const Site = struct {
    io: Io,
    alloc: Allocator,
    spec: Spec,
    pages: std.ArrayListUnmanaged(Page) = .empty,

    /// Page names are stored relative to `out_root`, so the English site is
    /// simply the same names with an `en/` prefix.
    fn addPage(self: *Site, out_rel: []const u8, title: []const u8, src_name: []const u8, group: []const u8) !void {
        return self.add(.{
            .out_name = try std.fmt.allocPrint(self.alloc, "{s}{s}", .{ self.spec.out_prefix, out_rel }),
            .title = title,
            .src_name = src_name,
            .group = group,
        });
    }

    /// The same page in the other language, when that page exists.
    fn counterpart(self: *Site, other: *Site, out_name: []const u8) ?[]const u8 {
        var alt: []const u8 = undefined;
        if (self.spec.lang == .ja) {
            alt = std.fmt.allocPrint(other.alloc, "{s}{s}", .{ other.spec.out_prefix, out_name }) catch return null;
        } else {
            if (!std.mem.startsWith(u8, out_name, self.spec.out_prefix)) return null;
            alt = out_name[self.spec.out_prefix.len..];
        }
        return if (other.pageByOut(alt) != null) alt else null;
    }

    fn add(self: *Site, page: Page) !void {
        try self.pages.append(self.alloc, page);
    }

    fn pageByOut(self: *Site, name: []const u8) ?*Page {
        for (self.pages.items) |*p| {
            if (std.mem.eql(u8, p.out_name, name)) return p;
        }
        return null;
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    const cwd = Io.Dir.cwd();
    for ([_]Spec{ specFor(.ja), specFor(.en) }) |spec| {
        try cwd.createDirPath(io, try mk(arena, "{s}/{s}features", .{ out_root, spec.out_prefix }));
        try cwd.createDirPath(io, try mk(arena, "{s}/{s}api", .{ out_root, spec.out_prefix }));
    }

    // Both languages are collected first: the header of every page links to the
    // same page in the other language, which is only known once both page lists
    // exist.
    var sites = [2]Site{
        .{ .io = io, .alloc = arena, .spec = specFor(.ja) },
        .{ .io = io, .alloc = arena, .spec = specFor(.en) },
    };
    for (&sites) |*site| try collectPages(io, arena, site);
    for (&sites, 0..) |*site, i| try renderSite(io, arena, site, &sites[1 - i]);
}

fn mk(arena: Allocator, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.allocPrint(arena, fmt, args);
}

/// Fills `site` with the page list for its language.
fn collectPages(io: Io, arena: Allocator, site: *Site) !void {
    const spec = site.spec;
    const doc = spec.doc_dir;

    try site.addPage("index.html", spec.title_index, try mk(arena, "{s}/index.md", .{doc}), "");
    try site.addPage("guide.html", spec.title_guide, try mk(arena, "{s}/guide.md", .{doc}), spec.group_intro);
    try site.addPage("comparison.html", spec.title_comparison, try mk(arena, "{s}/comparison.md", .{doc}), spec.group_intro);
    if (spec.include_repo_docs) {
        try site.addPage("milestones.html", spec.title_milestones, "MILESTONES.md", spec.group_intro);
        try site.addPage("readme.html", spec.title_readme, "README.md", spec.group_intro);
    }

    // `<doc>/features/NN-name.md` in name order: the numeric prefix is the order
    // of the tour, and both languages use the same names.
    const features_dir = try mk(arena, "{s}/features", .{doc});
    const feature_names = try listMarkdown(io, arena, features_dir);
    std.mem.sort([]const u8, feature_names, {}, lessThanName);
    for (feature_names) |name| {
        try site.addPage(
            try mk(arena, "features/{s}.html", .{stripMd(name)}),
            try humanTitle(arena, name),
            try mk(arena, "{s}/{s}", .{ features_dir, name }),
            spec.group_features,
        );
    }

    // ---- API pages from the sources ----------------------------------------
    // The doc comments are written in English, so the same page serves both
    // languages; it is listed in each site's navigation.
    var source_names: std.ArrayListUnmanaged([]const u8) = .empty;
    try source_names.append(arena, "src/zpu.zig");
    try source_names.append(arena, "src/zgf.zig");
    try source_names.append(arena, "src/backend.zig");
    const mod_names = try listZig(io, arena, "src/zpu");
    std.mem.sort([]const u8, mod_names, {}, lessThanName);
    for (mod_names) |name| {
        try source_names.append(arena, try mk(arena, "src/zpu/{s}", .{name}));
    }
    for (source_names.items) |src| {
        const base = std.fs.path.basename(src);
        try site.addPage(
            try mk(arena, "api/{s}.html", .{stripZig(base)}),
            stripZig(base),
            src,
            "API",
        );
    }
    // The `zui` sources get their own pages, prefixed: several of them share a
    // basename with a `zpu` module (`tests.zig`, `types.zig`), and a page per
    // file is only useful if the URL says which library it came from.
    if (fileExists(io, "src/zui.zig")) {
        try source_names.append(arena, "src/zui.zig");
        const zui_names = listZig(io, arena, "src/zui") catch arena.alloc([]const u8, 0) catch unreachable;
        std.mem.sort([]const u8, zui_names, {}, lessThanName);
        for (zui_names) |name| try source_names.append(arena, try mk(arena, "src/zui/{s}", .{name}));
        for (source_names.items) |src| {
            if (std.mem.startsWith(u8, src, "src/zui") == false) continue;
            const base = stripZig(std.fs.path.basename(src));
            try site.addPage(
                try mk(arena, "api/zui-{s}.html", .{base}),
                try mk(arena, "zui:{s}", .{base}),
                src,
                "API",
            );
        }
    }
}

/// Renders every page of `site` and writes it to disk. `other` is the site of
/// the other language, used for the language switcher.
fn renderSite(io: Io, arena: Allocator, site: *Site, other: *Site) !void {
    const cwd = Io.Dir.cwd();
    for (site.pages.items) |*page| {
        const ctx = Ctx{
            .arena = arena,
            .spec = site.spec,
            .up = try depthPrefix(arena, page.out_name),
        };
        const text = readWhole(io, arena, page.src_name) catch |err| {
            std.debug.print("doc: cannot read {s}: {s}\n", .{ page.src_name, @errorName(err) });
            page.body = try mk(arena, "<p class=\"missing\">{s} が読めません ({s})</p>\n", .{ page.src_name, @errorName(err) });
            continue;
        };
        var buf = Io.Writer.Allocating.init(arena);
        if (std.mem.indexOf(u8, page.out_name, "api/") != null) {
            try renderZigApi(&buf.writer, ctx, text);
        } else {
            const is_index = std.mem.endsWith(u8, page.out_name, "index.html");
            try renderMarkdown(&buf.writer, ctx, text, !is_index);
        }
        page.body = buf.written();
    }

    for (site.pages.items) |*page| {
        const path = try mk(arena, "{s}/{s}", .{ out_root, page.out_name });
        const file = try cwd.createFile(io, path, .{});
        defer file.close(io);
        var html = Io.Writer.Allocating.init(arena);
        try writeSkeleton(&html.writer, arena, site, other, page);
        try file.writeStreamingAll(io, html.written());
        std.debug.print("doc: {s}\n", .{path});
    }
    std.debug.print("doc: {d} {s} pages\n", .{ site.pages.items.len, @tagName(site.spec.lang) });
}

// ---------------------------------------------------------------- file helpers

fn lessThanName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn stripMd(name: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, name, ".md")) name[0 .. name.len - 3] else name;
}

/// Whether a path exists, for the optional parts of the site (this generator
/// runs against a checkout that may not have every tree).
fn fileExists(io: Io, path: []const u8) bool {
    const f = Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

fn stripZig(name: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, name, ".zig")) name[0 .. name.len - 4] else name;
}

fn humanTitle(arena: Allocator, md_name: []const u8) ![]const u8 {
    const base = stripMd(md_name);
    // "03-transforms" -> "transforms"
    var name = base;
    if (std.mem.indexOfScalar(u8, base, '-')) |i| name = base[i + 1 ..];
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    var up = true;
    for (name) |c| {
        if (c == '-') {
            try buf.append(arena, ' ');
            up = true;
            continue;
        }
        try buf.append(arena, if (up) std.ascii.toUpper(c) else c);
        up = false;
    }
    return buf.items;
}

fn listMarkdown(io: Io, arena: Allocator, dir_path: []const u8) ![][]const u8 {
    return listDir(io, arena, dir_path, ".md");
}

fn listZig(io: Io, arena: Allocator, dir_path: []const u8) ![][]const u8 {
    return listDir(io, arena, dir_path, ".zig");
}

fn listDir(io: Io, arena: Allocator, dir_path: []const u8, suffix: []const u8) ![][]const u8 {
    var list: std.ArrayListUnmanaged([]const u8) = .empty;
    const dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return list.items;
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, suffix)) continue;
        try list.append(arena, try arena.dupe(u8, entry.name));
    }
    return list.items;
}

fn readWhole(io: Io, arena: Allocator, path: []const u8) ![]const u8 {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const buf = try arena.alloc(u8, @intCast(stat.size));
    var read: usize = 0;
    while (read < buf.len) {
        const n = try file.readPositional(io, &.{buf[read..]}, read);
        if (n == 0) break;
        read += n;
    }
    return buf[0..read];
}

// ------------------------------------------------------------------ markdown

/// Renders the Markdown subset described in the file comment.
fn renderMarkdown(w: *Io.Writer, ctx: Ctx, text: []const u8, skip_first_heading: bool) !void {
    const arena = ctx.arena;
    var lines = std.mem.splitScalar(u8, text, '\n');

    // first pass: table of contents from the headings
    var toc: std.ArrayListUnmanaged(TocEntry) = .empty;
    var in_code = false;
    var idx: usize = 0;
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code) continue;
        if (headingLevel(line)) |level| {
            if (level == 1 and skip_first_heading) continue;
            if (level >= 2 and level <= 3) {
                try toc.append(arena, .{ .level = level, .title = std.mem.trim(u8, line[level..], " \r"), .anchor = try anchorFor(arena, idx) });
            }
            idx += 1;
        }
    }
    if (toc.items.len > 1) {
        try w.print("<nav class=\"toc\"><strong>{s}</strong><ul>", .{ctx.spec.toc});
        for (toc.items) |e| {
            try w.print("<li class=\"l{d}\"><a href=\"#{s}\">{s}</a></li>", .{ e.level, e.anchor, e.title });
        }
        try w.writeAll("</ul></nav>\n");
    }

    // second pass: the body
    lines = std.mem.splitScalar(u8, text, '\n');
    in_code = false;
    var in_paragraph = false;
    var list_kind: ListKind = .none;
    var heading_index: usize = 0;
    var title_seen = false;
    var skip_this = false;
    var paragraph: std.ArrayListUnmanaged(u8) = .empty;
    var table_rows: std.ArrayListUnmanaged([]const u8) = .empty;

    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (std.mem.startsWith(u8, line, "```")) {
            try flushParagraph(w, ctx, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            if (!in_code) {
                try flushTable(w, ctx, &table_rows);
                in_code = true;
                if (line.len > 3) try w.print("<pre data-lang=\"{s}\">", .{std.mem.trim(u8, line[3..], " ")});
                try w.writeAll("<code>");
            } else {
                in_code = false;
                try w.writeAll("</code></pre>\n");
            }
            continue;
        }
        if (in_code) {
            try escapeHtml(w, line);
            try w.writeAll("\n");
            continue;
        }
        if (line.len == 0) {
            try flushParagraph(w, ctx, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            try flushTable(w, ctx, &table_rows);
            continue;
        }
        if (headingLevel(line)) |level| {
            try flushParagraph(w, ctx, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            try flushTable(w, ctx, &table_rows);
            if (level == 1 and !title_seen) {
                // the page frame already printed this as <h1>
                title_seen = true;
                skip_this = true;
            }
            if (skip_this) {
                skip_this = false;
                continue;
            }
            const title = std.mem.trim(u8, line[level..], " ");
            const lvl = @min(level + 1, 6); // the page already owns h1
            try w.print("<h{d} id=\"", .{lvl});
            try w.writeAll(try anchorFor(arena, heading_index));
            try w.writeAll("\">");
            try renderInline(w, ctx, title);
            try w.print("</h{d}>\n", .{lvl});
            heading_index += 1;
            continue;
        }
        if (isTableRow(line)) {
            try flushParagraph(w, ctx, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            try table_rows.append(arena, line);
            continue;
        }
        if (isBullet(line)) {
            try flushParagraph(w, ctx, &paragraph, &in_paragraph);
            try flushTable(w, ctx, &table_rows);
            if (list_kind != .bullets) {
                try flushList(w, &list_kind);
                try w.writeAll("<ul>");
                list_kind = .bullets;
            }
            try w.writeAll("<li>");
            try renderInline(w, ctx, std.mem.trim(u8, line[2..], " "));
            try w.writeAll("</li>\n");
            continue;
        }
        if (isNumbered(line)) {
            try flushParagraph(w, ctx, &paragraph, &in_paragraph);
            try flushTable(w, ctx, &table_rows);
            if (list_kind != .numbers) {
                try flushList(w, &list_kind);
                try w.writeAll("<ol>");
                list_kind = .numbers;
            }
            const rest = std.mem.trimStart(u8, line, "0123456789. ");
            try w.writeAll("<li>");
            try renderInline(w, ctx, rest);
            try w.writeAll("</li>\n");
            continue;
        }
        try flushList(w, &list_kind);
        try flushTable(w, ctx, &table_rows);
        if (in_paragraph) try paragraph.append(arena, ' ');
        try paragraph.appendSlice(arena, std.mem.trim(u8, line, " "));
        in_paragraph = true;
    }
    try flushParagraph(w, ctx, &paragraph, &in_paragraph);
    try flushList(w, &list_kind);
    try flushTable(w, ctx, &table_rows);
}

const TocEntry = struct { level: usize, title: []const u8, anchor: []const u8 };

fn anchorFor(arena: Allocator, index: usize) ![]const u8 {
    return std.fmt.allocPrint(arena, "sec{d}", .{index});
}

fn headingLevel(line: []const u8) ?usize {
    var n: usize = 0;
    while (n < line.len and line[n] == '#') : (n += 1) {}
    if (n == 0 or n > 4) return null;
    if (n >= line.len or line[n] != ' ') return null;
    return n;
}

fn isBullet(line: []const u8) bool {
    return (std.mem.startsWith(u8, line, "- ") or std.mem.startsWith(u8, line, "* ")) and
        !std.mem.startsWith(u8, line, "***");
}

fn isNumbered(line: []const u8) bool {
    var i: usize = 0;
    while (i < line.len and std.ascii.isDigit(line[i])) : (i += 1) {}
    return i > 0 and i + 1 < line.len and line[i] == '.' and line[i + 1] == ' ';
}

fn isTableRow(line: []const u8) bool {
    return std.mem.startsWith(u8, line, "|") and std.mem.endsWith(u8, line, "|");
}

fn isTableSeparator(line: []const u8) bool {
    for (line) |c| {
        switch (c) {
            '-', '|', ':', ' ' => {},
            else => return false,
        }
    }
    return std.mem.indexOfScalar(u8, line, '-') != null;
}

fn flushParagraph(w: *Io.Writer, ctx: Ctx, paragraph: *std.ArrayListUnmanaged(u8), open: *bool) !void {
    if (!open.*) return;
    try w.writeAll("<p>");
    try renderInline(w, ctx, paragraph.items);
    try w.writeAll("</p>\n");
    paragraph.clearRetainingCapacity();
    open.* = false;
}

const ListKind = enum { none, bullets, numbers };

fn flushList(w: *Io.Writer, kind: *ListKind) !void {
    switch (kind.*) {
        .none => {},
        .bullets => try w.writeAll("</ul>\n"),
        .numbers => try w.writeAll("</ol>\n"),
    }
    kind.* = .none;
}

fn flushTable(w: *Io.Writer, ctx: Ctx, rows: *std.ArrayListUnmanaged([]const u8)) !void {
    if (rows.items.len == 0) return;
    try w.writeAll("<table>\n");
    var first = true;
    for (rows.items) |row| {
        if (isTableSeparator(row)) {
            continue;
        }
        const tag: []const u8 = if (first) "th" else "td";
        try w.writeAll("<tr>");
        var cells = std.mem.splitScalar(u8, std.mem.trim(u8, row, "|"), '|');
        while (cells.next()) |cell| {
            try w.print("<{s}>", .{tag});
            try renderInline(w, ctx, std.mem.trim(u8, cell, " "));
            try w.print("</{s}>", .{tag});
        }
        try w.writeAll("</tr>\n");
        first = false;
    }
    try w.writeAll("</table>\n");
    rows.clearRetainingCapacity();
}

/// Handles `` `code` ``, `**bold**`, `*italic*` and `[text](target)`.
fn renderInline(w: *Io.Writer, ctx: Ctx, text: []const u8) !void {
    var i: usize = 0;
    var plain_start: usize = 0;
    while (i < text.len) {
        if (text[i] == '`') {
            if (std.mem.indexOfScalarPos(u8, text, i + 1, '`')) |end| {
                try escapeHtml(w, text[plain_start..i]);
                try w.writeAll("<code>");
                try escapeHtml(w, text[i + 1 .. end]);
                try w.writeAll("</code>");
                i = end + 1;
                plain_start = i;
                continue;
            }
        }
        if (std.mem.startsWith(u8, text[i..], "**")) {
            if (std.mem.indexOfPos(u8, text, i + 2, "**")) |end| {
                try escapeHtml(w, text[plain_start..i]);
                try w.writeAll("<strong>");
                try renderInline(w, ctx, text[i + 2 .. end]);
                try w.writeAll("</strong>");
                i = end + 2;
                plain_start = i;
                continue;
            }
        }
        if (text[i] == '[') {
            if (std.mem.indexOfScalarPos(u8, text, i, ']')) |mid| {
                if (mid + 1 < text.len and text[mid + 1] == '(') {
                    if (std.mem.indexOfScalarPos(u8, text, mid + 1, ')')) |end| {
                        try escapeHtml(w, text[plain_start..i]);
                        const target = try rewriteTarget(ctx, text[mid + 2 .. end]);
                        try w.print("<a href=\"{s}\">", .{target});
                        try renderInline(w, ctx, text[i + 1 .. mid]);
                        try w.writeAll("</a>");
                        i = end + 1;
                        plain_start = i;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    try escapeHtml(w, text[plain_start..]);
}

const repo_prefix = "https://github.com/umiturtle/ZPU/blob/main/";

/// Rewrites `*.md` links to the generated HTML page and links to source files.
fn rewriteTarget(ctx: Ctx, target: []const u8) ![]const u8 {
    const arena = ctx.arena;
    if (std.mem.startsWith(u8, target, "http://") or std.mem.startsWith(u8, target, "https://") or
        std.mem.startsWith(u8, target, "#"))
    {
        return target;
    }
    if (std.mem.endsWith(u8, target, ".md")) {
        const base = std.fs.path.basename(stripMd(target));
        const here = ctx.spec.lang;
        // A link to a page of the *other* language keeps pointing there, so the
        // two trees can be read side by side.
        const other_lang = std.mem.startsWith(u8, target, "docs/en/");
        if (std.mem.eql(u8, base, "index")) return ctx.prefixed(if (other_lang and here == .ja) "en/index.html" else "index.html");
        if (std.mem.eql(u8, base, "guide")) return ctx.prefixed(if (other_lang and here == .ja) "en/guide.html" else "guide.html");
        if (std.mem.eql(u8, base, "comparison")) return ctx.prefixed(if (other_lang and here == .ja) "en/comparison.html" else "comparison.html");
        if (std.mem.eql(u8, base, "MILESTONES") or std.mem.eql(u8, base, "milestones")) return ctx.prefixed("milestones.html");
        if (std.mem.eql(u8, base, "README") or std.mem.eql(u8, base, "readme")) return ctx.prefixed("readme.html");
        return ctx.prefixed(try std.fmt.allocPrint(arena, "{s}.html", .{base}));
    }
    if (std.mem.startsWith(u8, target, "..")) {
        // an explicit relative hop out of the page's directory
        return ctx.prefixed(target[3..]);
    }
    // a link that already points at a generated page (`03-transforms.html`):
    // it is relative to the file it was written in, so it is left alone
    if (std.mem.endsWith(u8, target, ".html")) return target;
    // a source file or any other repository path: link to the file on GitHub
    return std.fmt.allocPrint(arena, "{s}{s}", .{ repo_prefix, target });
}

fn escapeHtml(w: *Io.Writer, text: []const u8) !void {
    for (text) |c| {
        switch (c) {
            '<' => try w.writeAll("&lt;"),
            '>' => try w.writeAll("&gt;"),
            '&' => try w.writeAll("&amp;"),
            '"' => try w.writeAll("&quot;"),
            else => try w.writeByte(c),
        }
    }
}

// ------------------------------------------------------------------ zig api

/// Turns a Zig source file into an API page: the `//!` module doc, then every
/// `///` block immediately in front of a `pub` declaration.
fn renderZigApi(w: *Io.Writer, ctx: Ctx, text: []const u8) !void {
    const arena = ctx.arena;
    var lines = std.mem.splitScalar(u8, text, '\n');
    var module_doc: std.ArrayListUnmanaged(u8) = .empty;

    // module header
    var rest: std.ArrayListUnmanaged([]const u8) = .empty;
    var in_header = true;
    while (lines.next()) |line| {
        if (in_header and std.mem.startsWith(u8, line, "//!")) {
            try module_doc.appendSlice(arena, std.mem.trimStart(u8, line[3..], " "));
            try module_doc.append(arena, '\n');
            continue;
        }
        if (in_header and line.len == 0 and module_doc.items.len != 0) continue;
        if (in_header and line.len != 0) in_header = false;
        try rest.append(arena, line);
    }
    if (module_doc.items.len != 0) {
        try w.writeAll("<section class=\"module-doc\">");
        try renderMarkdown(w, ctx, module_doc.items, false);
        try w.writeAll("</section>\n");
    }

    try w.print("<h2>{s}</h2>\n<p class=\"hint\">{s}</p>\n", .{ ctx.spec.declarations, ctx.spec.declarations_hint });
    var i: usize = 0;
    var doc: std.ArrayListUnmanaged(u8) = .empty;
    while (i < rest.items.len) : (i += 1) {
        const line = rest.items[i];
        const trimmed = std.mem.trimStart(u8, line, " ");
        if (std.mem.startsWith(u8, trimmed, "///")) {
            const body = std.mem.trim(u8, trimmed[3..], " ");
            if (body.len != 0 and (doc.items.len == 0 or !std.mem.startsWith(u8, body, "-"))) {
                if (doc.items.len != 0) try doc.append(arena, '\n');
                try doc.appendSlice(arena, body);
            }
            continue;
        }
        if (doc.items.len == 0) continue;
        if (!std.mem.startsWith(u8, trimmed, "pub ")) {
            doc.clearRetainingCapacity();
            continue;
        }
        // gather the declaration, which may span several lines
        var sig: std.ArrayListUnmanaged(u8) = .empty;
        var j = i;
        var depth: i32 = 0;
        while (j < rest.items.len and j < i + 12) : (j += 1) {
            const l = std.mem.trim(u8, rest.items[j], " ");
            if (sig.items.len != 0) try sig.append(arena, ' ');
            try sig.appendSlice(arena, l);
            depth += @as(i32, @intCast(std.mem.count(u8, l, "("))) - @as(i32, @intCast(std.mem.count(u8, l, ")")));
            if (std.mem.endsWith(u8, l, "{") or std.mem.endsWith(u8, l, ";") or std.mem.endsWith(u8, l, ",")) {
                if (depth <= 0) break;
            }
        }
        var sig_text: []const u8 = sig.items;
        if (std.mem.endsWith(u8, sig_text, "{")) sig_text = std.mem.trim(u8, sig_text[0 .. sig_text.len - 1], " ");
        if (std.mem.endsWith(u8, sig_text, ";")) sig_text = sig_text[0 .. sig_text.len - 1];
        try w.writeAll("<div class=\"decl\">\n<pre class=\"sig\"><code>");
        try escapeHtml(w, sig_text);
        try w.writeAll("</code></pre>\n<div class=\"doc\">");
        try renderMarkdown(w, ctx, doc.items, false);
        try w.writeAll("</div>\n</div>\n");
        doc.clearRetainingCapacity();
    }
}

// ------------------------------------------------------------------ skeleton

fn writeSkeleton(w: *Io.Writer, arena: Allocator, site: *Site, other: *Site, page: *const Page) !void {
    const up = try depthPrefix(arena, page.out_name);
    try w.writeAll("<!DOCTYPE html>\n");
    try w.print("<html lang=\"{s}\">\n", .{@tagName(site.spec.lang)});
    try w.writeAll(
        \\<head>
        \\<meta charset="utf-8">
        \\<meta name="viewport" content="width=device-width, initial-scale=1">
        \\
    );
    try w.print("<title>{s} — ZPU</title>\n", .{page.title});
    try w.print("<link rel=\"stylesheet\" href=\"{s}style.css\">\n</head>\n<body>\n", .{up});
    try w.print("<header><a class=\"brand\" href=\"{s}index.html\">ZPU</a> <span class=\"sub\">a numerical &amp; graphics library for Zig</span>", .{up});
    try w.writeAll("<span class=\"lang\"><strong>");
    try w.writeAll(site.spec.switcher_here);
    try w.writeAll("</strong>");
    // the same page in the other language (or that language's index)
    const alt = site.counterpart(other, page.out_name) orelse try mk(arena, "{s}index.html", .{other.spec.out_prefix});
    try w.print("<a href=\"{s}{s}\">{s}</a>", .{ up, alt, site.spec.switcher_other });
    try w.writeAll("</span></header>\n<div class=\"layout\">\n<aside>\n");
    var last_group: []const u8 = "\x00";
    var list_open = false;
    for (site.pages.items) |p| {
        if (!std.mem.eql(u8, p.group, last_group)) {
            if (list_open) try w.writeAll("</ul>\n");
            last_group = p.group;
            if (p.group.len != 0) try w.print("<h4>{s}</h4>\n", .{p.group});
            try w.writeAll("<ul>");
            list_open = true;
        }
        try w.writeAll("<li");
        if (std.mem.eql(u8, p.out_name, page.out_name)) try w.writeAll(" class=\"here\"");
        try w.print("><a href=\"{s}{s}\">{s}</a></li>\n", .{ up, p.out_name, p.title });
    }
    if (list_open) try w.writeAll("</ul>\n");
    try w.writeAll("</aside>\n<main>\n");
    if (std.mem.indexOf(u8, page.out_name, "index.html") == null) {
        try w.print("<h1>{s}</h1>\n", .{page.title});
        try w.print("<p class=\"src\">{s}<a href=\"{s}{s}\">{s}</a></p>\n", .{ site.spec.source_line, repo_prefix, page.src_name, page.src_name });
    }
    try w.writeAll(page.body);
    try w.writeAll("</main>\n</div>\n<footer>ZPU — generated by <code>zig build doc</code> (docs/build_docs.zig). No Python involved.</footer>\n</body>\n</html>\n");
}

/// "" for top level pages, "../" for pages in a subdirectory.
fn depthPrefix(arena: Allocator, out_name: []const u8) ![]const u8 {
    var count: usize = 0;
    for (out_name) |c| {
        if (c == '/') count += 1;
    }
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    for (0..count) |_| try buf.appendSlice(arena, "../");
    return buf.items;
}

const style_css =
    \\:root { color-scheme: light dark; --fg: #1b1f23; --bg: #fff; --accent: #0b6bcb; --muted: #667085; --code: #f4f6f8; }
    \\@media (prefers-color-scheme: dark) { :root { --fg: #e6e8eb; --bg: #14171a; --accent: #6db2ff; --muted: #9aa4b2; --code: #1e242a; } }
    \\* { box-sizing: border-box; }
    \\body { margin: 0; font: 16px/1.65 -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif; color: var(--fg); background: var(--bg); }
    \\header { padding: .8rem 1.2rem; border-bottom: 1px solid #8883; display: flex; gap: .6rem; align-items: baseline; }
    \\header .brand { font-weight: 700; font-size: 1.2rem; color: var(--accent); text-decoration: none; }
    \\header .sub { color: var(--muted); font-size: .85rem; }
    \\header .lang { margin-left: auto; font-size: .85rem; }
    \\header .lang a { margin-left: .4rem; }
    \\.layout { display: flex; align-items: flex-start; }
    \\aside { flex: 0 0 15rem; padding: 1rem; border-right: 1px solid #8883; position: sticky; top: 0; max-height: 100vh; overflow: auto; }
    \\aside h4 { margin: 1rem 0 .2rem; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
    \\aside ul { list-style: none; margin: 0; padding: 0; font-size: .92rem; }
    \\aside li { padding: .1rem 0; }
    \\aside a { color: inherit; text-decoration: none; }
    \\aside li.here > a { color: var(--accent); font-weight: 600; }
    \\main { padding: 1.2rem 2rem 4rem; max-width: 56rem; }
    \\h1 { font-size: 1.8rem; } h2 { margin-top: 2rem; border-bottom: 1px solid #8883; padding-bottom: .2rem; }
    \\h3 { margin-top: 1.4rem; }
    \\a { color: var(--accent); }
    \\code { background: var(--code); padding: .1rem .3rem; border-radius: 4px; font-size: .9em; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
    \\pre { background: var(--code); padding: .8rem 1rem; border-radius: 8px; overflow: auto; }
    \\pre code { background: none; padding: 0; }
    \\table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .93rem; }
    \\th, td { border: 1px solid #8884; padding: .35rem .6rem; text-align: left; vertical-align: top; }
    \\th { background: var(--code); }
    \\nav.toc { background: var(--code); padding: .6rem 1rem; border-radius: 8px; font-size: .92rem; }
    \\nav.toc ul { margin: .3rem 0 0; padding-left: 1.2rem; } nav.toc li.l3 { margin-left: 1rem; }
    \\.src { color: var(--muted); font-size: .85rem; }
    \\.hint { color: var(--muted); font-size: .85rem; }
    \\.decl { margin: 1rem 0; border-left: 3px solid var(--accent); padding-left: .8rem; }
    \\.decl pre.sig { margin: 0 0 .3rem; }
    \\.decl .doc > :first-child { margin-top: 0; }
    \\.missing { color: #b42318; }
    \\footer { padding: 1rem 2rem; color: var(--muted); font-size: .85rem; border-top: 1px solid #8883; }
;
