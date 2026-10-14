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
    pages: std.ArrayListUnmanaged(Page) = .empty,

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
    try cwd.createDirPath(io, out_root);
    try cwd.createDirPath(io, out_root ++ "/features");
    try cwd.createDirPath(io, out_root ++ "/api");

    var site = Site{ .io = io, .alloc = arena };

    // ---- hand written pages -------------------------------------------------
    try site.add(.{ .out_name = "index.html", .title = "ZPU", .src_name = "docs/index.md", .group = "" });
    try site.add(.{ .out_name = "guide.html", .title = "ガイド", .src_name = "docs/guide.md", .group = "はじめに" });
    try site.add(.{ .out_name = "comparison.html", .title = "他ライブラリとの違い", .src_name = "docs/comparison.md", .group = "はじめに" });
    try site.add(.{ .out_name = "milestones.html", .title = "マイルストーン", .src_name = "MILESTONES.md", .group = "はじめに" });
    try site.add(.{ .out_name = "readme.html", .title = "README", .src_name = "README.md", .group = "はじめに" });

    // `docs/features/NN-name.md` in name order
    const feature_names = try listMarkdown(io, arena, "docs/features");
    std.mem.sort([]const u8, feature_names, {}, lessThanName);
    for (feature_names) |name| {
        const out_name = try std.fmt.allocPrint(arena, "features/{s}.html", .{stripMd(name)});
        const title = try humanTitle(arena, name);
        const src = try std.fmt.allocPrint(arena, "docs/features/{s}", .{name});
        try site.add(.{ .out_name = out_name, .title = title, .src_name = src, .group = "機能" });
    }

    // ---- API pages from the sources ----------------------------------------
    var source_names: std.ArrayListUnmanaged([]const u8) = .empty;
    try source_names.append(arena, "src/zpu.zig");
    try source_names.append(arena, "src/zgf.zig");
    try source_names.append(arena, "src/backend.zig");
    const mod_names = try listZig(io, arena, "src/zpu");
    std.mem.sort([]const u8, mod_names, {}, lessThanName);
    for (mod_names) |name| {
        try source_names.append(arena, try std.fmt.allocPrint(arena, "src/zpu/{s}", .{name}));
    }
    for (source_names.items) |src| {
        const base = std.fs.path.basename(src);
        const out_name = try std.fmt.allocPrint(arena, "api/{s}.html", .{stripZig(base)});
        const title = try std.fmt.allocPrint(arena, "{s}", .{stripZig(base)});
        try site.add(.{ .out_name = out_name, .title = title, .src_name = src, .group = "API" });
    }

    // ---- render ------------------------------------------------------------
    for (site.pages.items) |*page| {
        const text = readWhole(io, arena, page.src_name) catch |err| {
            std.debug.print("doc: cannot read {s}: {s}\n", .{ page.src_name, @errorName(err) });
            page.body = try std.fmt.allocPrint(arena, "<p class=\"missing\">{s} が読めません ({s})</p>\n", .{ page.src_name, @errorName(err) });
            continue;
        };
        var buf = Io.Writer.Allocating.init(arena);
        if (std.mem.startsWith(u8, page.out_name, "api/")) {
            try renderZigApi(&buf.writer, arena, text);
        } else {
            try renderMarkdown(&buf.writer, arena, text, !std.mem.eql(u8, page.out_name, "index.html"));
        }
        page.body = buf.written();
    }

    for (site.pages.items) |page| {
        const path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ out_root, page.out_name });
        const file = try cwd.createFile(io, path, .{});
        defer file.close(io);
        var html = Io.Writer.Allocating.init(arena);
        try writeSkeleton(&html.writer, arena, &site, &page);
        try file.writeStreamingAll(io, html.written());
        std.debug.print("doc: {s}\n", .{path});
    }

    const css = try std.fmt.allocPrint(arena, "{s}/style.css", .{out_root});
    const css_file = try cwd.createFile(io, css, .{});
    defer css_file.close(io);
    try css_file.writeStreamingAll(io, style_css);
    std.debug.print("doc: {s}\n", .{css});
    std.debug.print("doc: {d} pages\n", .{site.pages.items.len});
}

// ---------------------------------------------------------------- file helpers

fn lessThanName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn stripMd(name: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, name, ".md")) name[0 .. name.len - 3] else name;
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
fn renderMarkdown(w: *Io.Writer, arena: Allocator, text: []const u8, skip_first_heading: bool) !void {
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
        try w.writeAll("<nav class=\"toc\"><strong>目次</strong><ul>");
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
            try flushParagraph(w, arena, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            if (!in_code) {
                try flushTable(w, arena, &table_rows);
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
            try flushParagraph(w, arena, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            try flushTable(w, arena, &table_rows);
            continue;
        }
        if (headingLevel(line)) |level| {
            try flushParagraph(w, arena, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            try flushTable(w, arena, &table_rows);
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
            try renderInline(w, arena, title);
            try w.print("</h{d}>\n", .{lvl});
            heading_index += 1;
            continue;
        }
        if (isTableRow(line)) {
            try flushParagraph(w, arena, &paragraph, &in_paragraph);
            try flushList(w, &list_kind);
            try table_rows.append(arena, line);
            continue;
        }
        if (isBullet(line)) {
            try flushParagraph(w, arena, &paragraph, &in_paragraph);
            try flushTable(w, arena, &table_rows);
            if (list_kind != .bullets) {
                try flushList(w, &list_kind);
                try w.writeAll("<ul>");
                list_kind = .bullets;
            }
            try w.writeAll("<li>");
            try renderInline(w, arena, std.mem.trim(u8, line[2..], " "));
            try w.writeAll("</li>\n");
            continue;
        }
        if (isNumbered(line)) {
            try flushParagraph(w, arena, &paragraph, &in_paragraph);
            try flushTable(w, arena, &table_rows);
            if (list_kind != .numbers) {
                try flushList(w, &list_kind);
                try w.writeAll("<ol>");
                list_kind = .numbers;
            }
            const rest = std.mem.trimStart(u8, line, "0123456789. ");
            try w.writeAll("<li>");
            try renderInline(w, arena, rest);
            try w.writeAll("</li>\n");
            continue;
        }
        try flushList(w, &list_kind);
        try flushTable(w, arena, &table_rows);
        if (in_paragraph) try paragraph.append(arena, ' ');
        try paragraph.appendSlice(arena, std.mem.trim(u8, line, " "));
        in_paragraph = true;
    }
    try flushParagraph(w, arena, &paragraph, &in_paragraph);
    try flushList(w, &list_kind);
    try flushTable(w, arena, &table_rows);
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

fn flushParagraph(w: *Io.Writer, arena: Allocator, paragraph: *std.ArrayListUnmanaged(u8), open: *bool) !void {
    if (!open.*) return;
    try w.writeAll("<p>");
    try renderInline(w, arena, paragraph.items);
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

fn flushTable(w: *Io.Writer, arena: Allocator, rows: *std.ArrayListUnmanaged([]const u8)) !void {
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
            try renderInline(w, arena, std.mem.trim(u8, cell, " "));
            try w.print("</{s}>", .{tag});
        }
        try w.writeAll("</tr>\n");
        first = false;
    }
    try w.writeAll("</table>\n");
    rows.clearRetainingCapacity();
}

/// Handles `` `code` ``, `**bold**`, `*italic*` and `[text](target)`.
fn renderInline(w: *Io.Writer, arena: Allocator, text: []const u8) !void {
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
                try renderInline(w, arena, text[i + 2 .. end]);
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
                        const target = try rewriteTarget(arena, text[mid + 2 .. end]);
                        try w.print("<a href=\"{s}\">", .{target});
                        try renderInline(w, arena, text[i + 1 .. mid]);
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
fn rewriteTarget(arena: Allocator, target: []const u8) ![]const u8 {
    if (std.mem.startsWith(u8, target, "http://") or std.mem.startsWith(u8, target, "https://") or
        std.mem.startsWith(u8, target, "#"))
    {
        return target;
    }
    if (std.mem.endsWith(u8, target, ".md")) {
        const base = std.fs.path.basename(stripMd(target));
        if (std.mem.eql(u8, base, "index")) return "index.html";
        if (std.mem.eql(u8, base, "guide")) return "guide.html";
        if (std.mem.eql(u8, base, "comparison")) return "comparison.html";
        if (std.mem.eql(u8, base, "MILESTONES") or std.mem.eql(u8, base, "milestones")) return "milestones.html";
        if (std.mem.eql(u8, base, "README") or std.mem.eql(u8, base, "readme")) return "readme.html";
        if (std.mem.startsWith(u8, target, "docs/features/")) {
            return std.fmt.allocPrint(arena, "features/{s}.html", .{base});
        }
        return std.fmt.allocPrint(arena, "{s}.html", .{base});
    }
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
fn renderZigApi(w: *Io.Writer, arena: Allocator, text: []const u8) !void {
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
        try renderMarkdown(w, arena, module_doc.items, false);
        try w.writeAll("</section>\n");
    }

    try w.writeAll("<h2>宣言</h2>\n<p class=\"hint\">ソース内の <code>///</code> コメントから生成。</p>\n");
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
        try renderMarkdown(w, arena, doc.items, false);
        try w.writeAll("</div>\n</div>\n");
        doc.clearRetainingCapacity();
    }
}

// ------------------------------------------------------------------ skeleton

fn writeSkeleton(w: *Io.Writer, arena: Allocator, site: *Site, page: *const Page) !void {
    try w.writeAll(
        \\<!DOCTYPE html>
        \\<html lang="ja">
        \\<head>
        \\<meta charset="utf-8">
        \\<meta name="viewport" content="width=device-width, initial-scale=1">
        \\
    );
    try w.print("<title>{s} — ZPU</title>\n", .{page.title});
    try w.writeAll("<link rel=\"stylesheet\" href=\"");
    try w.writeAll(try depthPrefix(arena, page.out_name));
    try w.writeAll("style.css\">\n</head>\n<body>\n<header><a class=\"brand\" href=\"");
    try w.writeAll(try depthPrefix(arena, page.out_name));
    try w.writeAll("index.html\">ZPU</a> <span class=\"sub\">a numerical &amp; graphics library for Zig</span></header>\n<div class=\"layout\">\n<aside>\n");
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
        try w.print("><a href=\"{s}{s}\">{s}</a></li>\n", .{ try depthPrefix(arena, page.out_name), p.out_name, p.title });
    }
    if (list_open) try w.writeAll("</ul>\n");
    try w.writeAll("</aside>\n<main>\n");
    if (!std.mem.eql(u8, page.out_name, "index.html")) {
        try w.print("<h1>{s}</h1>\n", .{page.title});
        try w.print("<p class=\"src\">元ファイル: <a href=\"{s}{s}\">{s}</a></p>\n", .{ repo_prefix, page.src_name, page.src_name });
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
