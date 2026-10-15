//! `zig build run-render3d`
//!
//! A software 3-D renderer: a UV sphere over a checkerboard floor, a perspective
//! camera, a depth buffer, Gouraud-shaded Lambert lighting and a fog term. The
//! vertex transform is one `matmul` on the numeric core — the vertices are a
//! `[N, 4]` tensor and the model-view-projection is a `[4, 4]` tensor, so the
//! projection is the library's own matrix product, not a hand-rolled loop.
//!
//! Triangles that cross the camera plane are cut against the near plane
//! (Sutherland–Hodgman) before the perspective divide, because a vertex with
//! `w` near zero has no screen position at all.
//!
//! The example checks the projection analytically: a sphere of radius `r` at
//! distance `d` covers about `pi * ((r / d) / tan(fov / 2) * height / 2)^2`
//! pixels, and the silhouette has to be that big. On top of that it verifies
//! that the pixel at the projected sphere centre is the near surface at the
//! expected depth, that the floor's far edge ends below the horizon with sky
//! above and floor below it, that fog lightens the floor with distance while
//! the sky darkens towards the horizon, that the highlight faces the light, and
//! that `zpu.matmul` reproduces a hand-written vertex product exactly.
const std = @import("std");
const zpu = @import("zpu");
const zgf = @import("zgf");

const Io = std.Io;
const Image = zgf.Image;
const Color = zgf.Color;
const Tensor = zpu.Tensor;

const width = 192;
const height = 144;
const fov_deg: f64 = 45.0;
const sphere_radius: f64 = 0.5;
const sphere_center = [3]f64{ 0, 0.5, 0 };
const eye = [3]f64{ 0.55, 0.62, 1.65 };
const target = [3]f64{ 0, 0.45, 0 };
/// Points *towards* the light: up, to the right, in front of the camera.
const light_dir = [3]f64{ 0.45, 0.65, 0.62 };

/// A vertex in clip space (the `matmul` result) plus the colour that has to be
/// interpolated along with it.
const ClipVert = struct { c: [4]f64, color: [3]f64 };

/// A triangle corner as the rasteriser wants it: screen position (`z` is the
/// camera-space depth) plus the colour Gouraud interpolation works on.
const TriVert = struct { p: [3]f64, color: [3]f64 };

/// Camera near plane. Everything closer than this is clipped away rather than
/// projected: a vertex with `w` near zero would map to a point at infinity and
/// smear its triangle across the frame.
const near = 0.1;

/// Half the viewport height: the projection scale below is expressed in pixels.
const half_height = @as(f64, height) / 2.0;

const segs = 20;
const rings = 12;
const floor_n = 12;
const max_verts = (rings + 1) * (segs + 1) + (floor_n + 1) * (floor_n + 1);
const max_tris = rings * segs * 2 + floor_n * floor_n * 2;

var checks: usize = 0;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file = Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_file.interface;

    try zpu.init(alloc, .{});
    defer zpu.deinit();

    var scope = try zpu.Scope.init(alloc);
    defer scope.deinit();

    var img = try Image.init(alloc, width, height, 3);
    defer img.deinit();
    var depth: [width * height]f64 = undefined;
    @memset(&depth, std.math.inf(f64));
    // Which surface ended up in front of each pixel. Counting *writes* would
    // count a pixel twice where two triangles overlap (and they do: a sphere's
    // front faces share their edges), so the count comes from here instead.
    var owner: [width * height]u8 = undefined;
    @memset(&owner, 255);
    try background(&img);

    // ---------------------------------------------------------------- geometry
    var world: [max_verts][3]f64 = undefined;
    var normals: [max_verts][3]f64 = undefined;
    var kind: [max_verts]u8 = undefined; // 0 = sphere, 1 = floor
    var tris: [max_tris][3]u32 = undefined;
    var verts: usize = 0;
    var n_tris: usize = 0;

    // UV sphere: rings of latitude, segs of longitude.
    for (0..rings + 1) |i| {
        const theta = std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rings));
        for (0..segs + 1) |j| {
            const phi = std.math.tau * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(segs));
            const n = [3]f64{ @sin(theta) * @cos(phi), @cos(theta), @sin(theta) * @sin(phi) };
            world[verts] = .{
                sphere_center[0] + n[0] * sphere_radius,
                sphere_center[1] + n[1] * sphere_radius,
                sphere_center[2] + n[2] * sphere_radius,
            };
            normals[verts] = n;
            kind[verts] = 0;
            verts += 1;
        }
    }
    const stride = segs + 1;
    for (0..rings) |i| {
        for (0..segs) |j| {
            const a: u32 = @intCast(i * stride + j);
            const b: u32 = @intCast((i + 1) * stride + j);
            tris[n_tris] = .{ a, b, a + 1 };
            n_tris += 1;
            tris[n_tris] = .{ a + 1, b, b + 1 };
            n_tris += 1;
        }
    }

    // Floor: a grid of quads, with a checkerboard shade applied while shading.
    const floor_base: usize = verts;
    // Wide enough that the near plane cuts behind the camera, not inside the
    // visible part of the floor (a clipped triangle is dropped, see below).
    const floor_span: f64 = 4.0;
    for (0..floor_n + 1) |i| {
        for (0..floor_n + 1) |j| {
            const x = -floor_span / 2 + floor_span * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(floor_n));
            const z = -floor_span / 2 + floor_span * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(floor_n));
            world[verts] = .{ x, 0, z };
            normals[verts] = .{ 0, 1, 0 };
            kind[verts] = 1;
            verts += 1;
        }
    }
    const floor_stride = floor_n + 1;
    for (0..floor_n) |i| {
        for (0..floor_n) |j| {
            const a: u32 = @intCast(floor_base + i * floor_stride + j);
            const b: u32 = @intCast(floor_base + (i + 1) * floor_stride + j);
            tris[n_tris] = .{ a, b, a + 1 };
            n_tris += 1;
            tris[n_tris] = .{ a + 1, b, b + 1 };
            n_tris += 1;
        }
    }

    // ------------------------------------------------- the transform (matmul!)
    const mvp = lookAtTimesPerspective();
    const vertex_matrix = try Tensor.zeros(zpu.Shape.initComptime(&.{ max_verts, 4 }), .f64);
    for (0..verts) |i| {
        for (0..3) |c| vertex_matrix.writeF64(i * 4 + c, world[i][c]);
        vertex_matrix.writeF64(i * 4 + 3, 1.0);
    }
    // The vertices are a `[N, 4]` tensor of *row* vectors and `matmul` is NumPy's
    // `a @ b`, so the product computes `v * M` — the transpose of the
    // column-vector `M * v` that `lookAtTimesPerspective` builds. Uploading the
    // transpose makes the two agree; the vertex transform itself is entirely the
    // library's `matmul`, with no loop of ours in it.
    const mvp_tensor = try Tensor.zeros(zpu.Shape.initComptime(&.{ 4, 4 }), .f64);
    for (0..4) |r| for (0..4) |c| mvp_tensor.writeF64(r * 4 + c, mvp[c][r]);

    const clip = try zpu.matmul(vertex_matrix, mvp_tensor);
    const n_verts = try zpu.narrow(clip, 0, 0, verts);

    // The `matmul` gave clip space; the perspective divide and the viewport
    // mapping happen per triangle *after* clipping against the near plane,
    // because a vertex at `w == 0` has no screen position at all.
    const clip_verts = try alloc.alloc([4]f64, verts);
    defer alloc.free(clip_verts);
    var behind: usize = 0;
    for (0..verts) |i| {
        clip_verts[i] = .{
            n_verts.readF64(i * 4 + 0),
            n_verts.readF64(i * 4 + 1),
            n_verts.readF64(i * 4 + 2),
            n_verts.readF64(i * 4 + 3),
        };
        if (clip_verts[i][3] < near) behind += 1;
    }

    // ---------------------------------------------------------------- rasterise
    var shaded: usize = 0;
    var clipped_tris: usize = 0;
    for (0..n_tris) |t| {
        const ia: usize = tris[t][0];
        const ib: usize = tris[t][1];
        const ic: usize = tris[t][2];
        const tri = [3]ClipVert{
            .{ .c = clip_verts[ia], .color = shade(normals[ia], kind[ia], world[ia]) },
            .{ .c = clip_verts[ib], .color = shade(normals[ib], kind[ib], world[ib]) },
            .{ .c = clip_verts[ic], .color = shade(normals[ic], kind[ic], world[ic]) },
        };

        // Sutherland-Hodgman against one plane (`w >= near`), then a triangle
        // fan over whatever is left: 3, 4 or 0 vertices.
        var poly: [4]ClipVert = undefined;
        const n = clipNear(tri, &poly);
        if (n == 0) {
            clipped_tris += 1;
            continue;
        }
        if (n != 3) clipped_tris += 1;
        var fan: usize = 1;
        while (fan + 1 < n) : (fan += 1) {
            const count = rasterize(&img, &depth, &owner, .{
                project(poly[0].c, poly[0].color),
                project(poly[fan].c, poly[fan].color),
                project(poly[fan + 1].c, poly[fan + 1].color),
            }, kind[ia]);
            shaded += count;
        }
    }

    try out.print("zgf — software 3-D renderer\n\n", .{});
    try out.print("{s:<14} {d} vertices, {d} triangles ({d} near-clipped, {d} vertices behind the camera), {d} pixels shaded\n", .{
        "scene", verts, n_tris, clipped_tris, behind, shaded,
    });
    try preview(out, img);

    // ---------------------------------------------------------------- checks
    var sphere_pixels: usize = 0;
    for (0..width * height) |i| {
        if (owner[i] == 0) sphere_pixels += 1;
    }
    const got: f64 = @floatFromInt(sphere_pixels);

    // The analytic check: how many pixels *should* the sphere cover? Seen from
    // distance `d`, a sphere of radius `r` subtends `r / d` radians, which the
    // pinhole projection turns into `(r / d) / tan(fov / 2) * (height / 2)`
    // pixels of radius, hence `pi r_px^2` pixels of silhouette.
    const dist = distance(eye, sphere_center);
    const radius_px = (sphere_radius / dist) / std.math.tan(fov_deg * std.math.pi / 360.0) * half_height;
    const expected = std.math.pi * radius_px * radius_px;

    try out.print("\nsphere covers {d} pixels, the projection predicts {d:.0} (radius {d:.1} px)\n", .{ sphere_pixels, expected, radius_px });
    try gt(out, "the sphere is drawn at all", got, 300);
    try check(out, "the silhouette has the projected size", got / expected, 0.25);

    // The pixel at the *projected* sphere centre is the sphere's nearest
    // surface, so its depth is the eye-to-surface distance.
    const centre = screenOf(mvp, sphere_center);
    const cx: usize = @intFromFloat(centre[0]);
    const cy: usize = @intFromFloat(centre[1]);
    const mid = cy * width + cx;

    try eq(out, "the projected sphere centre belongs to the sphere", owner[mid], 0);
    try check(out, "the centre pixel sits on the near surface", depth[mid] / (dist - sphere_radius), 0.02);

    // Where does the horizon land? Project a point far along the flat forward
    // direction with the same matrix the scene used: the floor converges to that
    // row as the distance goes to infinity.
    const f = normalize(sub(target, eye));
    const flat = normalize([3]f64{ f[0], 0, f[2] });
    const horizon = projectY(mvp, .{ eye[0] + 200.0 * flat[0], eye[1], eye[2] + 200.0 * flat[2] });

    // The floor's far edge: take the vertex on it that lands right of the
    // sphere, so that the sky above that edge is in view.
    var corner: [2]usize = .{ 0, 0 };
    var found = false;
    for (floor_base..verts) |i| {
        if (world[i][2] != -floor_span / 2) continue;
        const s = screenOf(mvp, world[i]);
        const ex: usize = @intFromFloat(s[0]);
        const ey: usize = @intFromFloat(s[1]);
        if (ex > width * 5 / 6 and ex < width - 3 and ey > 2 and ey < height - 3 and (!found or ex > corner[0])) {
            corner = .{ ex, ey };
            found = true;
        }
    }
    try eq(out, "the floor has a far corner in frame, right of the sphere", found, true);
    // A finite distance always projects below the horizon line: the far corner is
    // the last row of real floor, the horizon is where floor would be at an
    // infinite distance. (Both are the same line in the limit — that is why the
    // check is an ordering, not an equality.)
    try gt(out, "the floor's far corner projects below the horizon", @floatFromInt(corner[1]), horizon);
    try eq(out, "the floor fills the pixel below the far corner", owner[(corner[1] + 1) * width + corner[0]], 1);
    try eq(out, "the sky sits above the far corner", owner[(corner[1] - 1) * width + corner[0]], 255);

    // Fog: the floor fades towards the horizon colour as it recedes...
    const col0 = width * 5 / 6 + 2;
    const col1 = width - 3;
    const near_floor = meanRow(&img, height - 6, col0, col1);
    const far_floor = meanRow(&img, corner[1] - 3, col0, col1);
    try gt(out, "the fog lightens the floor with distance", far_floor, near_floor);
    // ... and the sky gradient runs from a dark horizon up to a light zenith.
    const sky_low = @as(usize, @intFromFloat(horizon)) - 4;
    try gt(out, "the sky brightens towards the top of the frame", meanRow(&img, 1, col0, col1), meanRow(&img, sky_low, col0, col1));

    // Lambert: the brightest sphere pixel faces the light, which is up and to
    // the right of the camera.
    var best: f64 = -1;
    var best_x: usize = 0;
    var best_y: usize = 0;
    for (0..height) |y| {
        for (0..width) |x| {
            if (owner[y * width + x] != 0) continue;
            const l = luma(img, x, y);
            if (l > best) {
                best = l;
                best_x = x;
                best_y = y;
            }
        }
    }
    try gt(out, "the highlight is right of the sphere's centre", @floatFromInt(best_x), @floatFromInt(cx));
    try gt(out, "the highlight is above the sphere's centre", @floatFromInt(cy), @floatFromInt(best_y));

    // The vertex transform is the library's `matmul`, so check it against a
    // plain product over the same matrices.
    var worst: f64 = 0;
    for (0..verts) |i| {
        const reference = mulPoint(mvp, world[i]);
        for (0..4) |c| worst = @max(worst, @abs(reference[c] - n_verts.readF64(i * 4 + c)));
    }
    try lt(out, "zpu.matmul matches the hand-written vertex product", worst, 1e-12);

    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};
    const path = "zig-out/render3d.ppm";
    try img.savePpm(io, path);
    try out.print("\nwrote {s} ({d}x{d})\n{d} checks passed\n", .{ path, width, height, checks });
    try out.flush();
}

// ---------------------------------------------------------------------------
// camera and shading
// ---------------------------------------------------------------------------

/// Right-handed look-at, then a perspective projection with the vertical field
/// of view above. Both are ordinary matrices, kept as `[4][4]f64`.
fn lookAtTimesPerspective() [4][4]f64 {
    const f = normalize(sub(target, eye));
    const s = normalize(cross(f, .{ 0, 1, 0 }));
    const u = cross(s, f);
    const view = [4][4]f64{
        .{ s[0], s[1], s[2], -dot(s, eye) },
        .{ u[0], u[1], u[2], -dot(u, eye) },
        .{ -f[0], -f[1], -f[2], dot(f, eye) },
        .{ 0, 0, 0, 1 },
    };
    const fov = fov_deg * std.math.pi / 180.0;
    const proj = [4][4]f64{
        .{ 1.0 / (std.math.tan(fov / 2.0) * aspect()), 0, 0, 0 },
        .{ 0, 1.0 / std.math.tan(fov / 2.0), 0, 0 },
        .{ 0, 0, -(10.0 + 0.1) / (10.0 - 0.1), -2 * 10.0 * 0.1 / (10.0 - 0.1) },
        .{ 0, 0, -1, 0 },
    };
    return mul4(proj, view);
}

fn aspect() f64 {
    return @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
}

/// Lambert, with a soft ambient term and a checkerboard on the floor.
fn shade(n: [3]f64, kind: u8, p: [3]f64) [3]f64 {
    const lambert = @max(0.0, dot(normalize(n), normalize(light_dir)));
    const ambient = 0.22;
    const base: [3]f64 = if (kind == 0)
        .{ 0.95, 0.55, 0.25 }
    else blk: {
        const checker = (@floor(p[0] * 2.0) + @floor(p[2] * 2.0));
        const dark = @mod(checker, 2.0) == 0;
        break :blk if (dark) .{ 0.16, 0.19, 0.26 } else .{ 0.30, 0.34, 0.42 };
    };
    const k = ambient + 0.85 * lambert;
    return .{ base[0] * k, base[1] * k, base[2] * k };
}

/// Clips a triangle against `w >= near`. Returns 3 or 4 vertices when part of it
/// survives, 0 when it is entirely behind the camera, and writes them to `out`.
fn clipNear(tri: [3]ClipVert, out: *[4]ClipVert) usize {
    var count: usize = 0;
    for (0..3) |i| {
        const a = tri[i];
        const b = tri[(i + 1) % 3];
        const a_in = a.c[3] >= near;
        const b_in = b.c[3] >= near;
        if (a_in) {
            out[count] = a;
            count += 1;
        }
        if (a_in != b_in) {
            const t = (near - a.c[3]) / (b.c[3] - a.c[3]);
            out[count] = .{ .c = .{
                a.c[0] + (b.c[0] - a.c[0]) * t,
                a.c[1] + (b.c[1] - a.c[1]) * t,
                a.c[2] + (b.c[2] - a.c[2]) * t,
                near,
            }, .color = .{
                a.color[0] + (b.color[0] - a.color[0]) * t,
                a.color[1] + (b.color[1] - a.color[1]) * t,
                a.color[2] + (b.color[2] - a.color[2]) * t,
            } };
            count += 1;
        }
    }
    return count;
}

/// Clip space to the rasteriser's screen space: divide by `w`, then map NDC to
/// pixels with y pointing down.
fn project(c: [4]f64, color: [3]f64) TriVert {
    const inv = 1.0 / c[3];
    return .{
        .p = .{
            (c[0] * inv * 0.5 + 0.5) * @as(f64, @floatFromInt(width)),
            (0.5 - c[1] * inv * 0.5) * @as(f64, @floatFromInt(height)),
            c[3],
        },
        .color = color,
    };
}

/// One triangle: barycentric test over the bounding box, depth test per pixel,
/// Gouraud-interpolated colour. Returns how many pixels it wrote.
fn rasterize(
    img: *Image,
    depth: *[width * height]f64,
    owner: *[width * height]u8,
    v: [3]TriVert,
    kind: u8,
) usize {
    // No explicit back-face culling: the sphere is convex, so the depth buffer
    // already picks the front surface, and the floor is a single plane.
    const area = edge(v[0].p, v[1].p, v[2].p);
    if (area == 0) return 0;

    const minx: isize = @intFromFloat(@floor(@min(v[0].p[0], @min(v[1].p[0], v[2].p[0]))));
    const maxx: isize = @intFromFloat(@ceil(@max(v[0].p[0], @max(v[1].p[0], v[2].p[0]))));
    const miny: isize = @intFromFloat(@floor(@min(v[0].p[1], @min(v[1].p[1], v[2].p[1]))));
    const maxy: isize = @intFromFloat(@ceil(@max(v[0].p[1], @max(v[1].p[1], v[2].p[1]))));

    var written: usize = 0;
    var y = @max(miny, 0);
    const y_end = @min(maxy, @as(isize, height) - 1);
    while (y <= y_end) : (y += 1) {
        var x = @max(minx, 0);
        const x_end = @min(maxx, @as(isize, width) - 1);
        while (x <= x_end) : (x += 1) {
            const p = [3]f64{ @as(f64, @floatFromInt(x)) + 0.5, @as(f64, @floatFromInt(y)) + 0.5, 0 };
            const w0 = edge(v[1].p, v[2].p, p) / area;
            const w1 = edge(v[2].p, v[0].p, p) / area;
            const w2 = edge(v[0].p, v[1].p, p) / area;
            if (w0 < 0 or w1 < 0 or w2 < 0) continue;
            const z = w0 * v[0].p[2] + w1 * v[1].p[2] + w2 * v[2].p[2];
            const idx: usize = @intCast(y * width + x);
            if (z >= depth[idx]) continue;
            depth[idx] = z;
            owner[idx] = kind;
            const color = Color{
                .r = @floatCast(w0 * v[0].color[0] + w1 * v[1].color[0] + w2 * v[2].color[0]),
                .g = @floatCast(w0 * v[0].color[1] + w1 * v[1].color[1] + w2 * v[2].color[1]),
                .b = @floatCast(w0 * v[0].color[2] + w1 * v[1].color[2] + w2 * v[2].color[2]),
            };
            img.setPixel(x, y, color);
            written += 1;
        }
    }
    return written;
}

fn edge(a: [3]f64, b: [3]f64, p: [3]f64) f64 {
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
}

fn background(img: *Image) !void {
    img.verticalGradient(.{ .r = 0.62, .g = 0.70, .b = 0.85 }, .{ .r = 0.20, .g = 0.24, .b = 0.34 });
}

// ---------------------------------------------------------------------------
// small vector helpers
// ---------------------------------------------------------------------------

fn sub(a: [3]f64, b: [3]f64) [3]f64 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}
fn dot(a: [3]f64, b: [3]f64) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
fn cross(a: [3]f64, b: [3]f64) [3]f64 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}
fn normalize(a: [3]f64) [3]f64 {
    const len = @sqrt(dot(a, a));
    if (len == 0) return .{ 0, 0, 1 };
    return .{ a[0] / len, a[1] / len, a[2] / len };
}
fn distance(a: [3]f64, b: [3]f64) f64 {
    return @sqrt(dot(sub(a, b), sub(a, b)));
}
fn mul4(a: [4][4]f64, b: [4][4]f64) [4][4]f64 {
    var out: [4][4]f64 = undefined;
    for (0..4) |r| {
        for (0..4) |c| {
            var acc: f64 = 0;
            for (0..4) |k| acc += a[r][k] * b[k][c];
            out[r][c] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

fn preview(out: *Io.Writer, img: Image) !void {
    const cols = 64;
    const rows = 20;
    const ramp = " .:-=+*#%@";
    try out.print("\n", .{});
    for (0..rows) |r| {
        try out.print("  ", .{});
        for (0..cols) |c| {
            const x = c * width / cols;
            const y = r * height / rows;
            const p = img.getPixel(x, y);
            const v = 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;
            const idx: usize = @intFromFloat(@round(std.math.clamp(v, 0, 1) * @as(f32, ramp.len - 1)));
            try out.print("{c}", .{ramp[idx]});
        }
        try out.print("\n", .{});
    }
}

/// Mean luminance of one scanline of the image, over the columns `[x0, x1)`.
fn meanRow(img: *const Image, y: usize, x0: usize, x1: usize) f64 {
    var acc: f64 = 0;
    for (x0..x1) |x| acc += luma(img.*, x, y);
    return acc / @as(f64, @floatFromInt(x1 - x0));
}

fn luma(img: Image, x: usize, y: usize) f64 {
    const p = img.getPixel(x, y);
    return 0.299 * @as(f64, p.r) + 0.587 * @as(f64, p.g) + 0.114 * @as(f64, p.b);
}

/// The pixel a world point lands on, through the same matrix the scene used —
/// the perspective divide included, since that is what a projection is.
fn screenOf(m: [4][4]f64, p: [3]f64) [2]f64 {
    const c = mulPoint(m, p);
    return .{
        (c[0] / c[3] * 0.5 + 0.5) * @as(f64, @floatFromInt(width)),
        (0.5 - c[1] / c[3] * 0.5) * @as(f64, @floatFromInt(height)),
    };
}

fn projectY(m: [4][4]f64, p: [3]f64) f64 {
    return screenOf(m, p)[1];
}

fn mulPoint(m: [4][4]f64, v: [3]f64) [4]f64 {
    return .{
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3],
    };
}

fn check(out: *Io.Writer, label: []const u8, ratio: f64, slack: f64) !void {
    if (!(@abs(ratio - 1.0) <= slack)) {
        try out.print("  FAIL {s}: ratio {d} (slack {d})\n", .{ label, ratio, slack });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn gt(out: *Io.Writer, label: []const u8, value: f64, floor: f64) !void {
    if (!(value > floor)) {
        try out.print("  FAIL {s}: {d} is not above {d}\n", .{ label, value, floor });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn lt(out: *Io.Writer, label: []const u8, value: f64, ceiling: f64) !void {
    if (!(value < ceiling)) {
        try out.print("  FAIL {s}: {d} is not below {d}\n", .{ label, value, ceiling });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}

fn eq(out: *Io.Writer, label: []const u8, value: anytype, want: @TypeOf(value)) !void {
    if (value != want) {
        try out.print("  FAIL {s}: {any} != {any}\n", .{ label, value, want });
        try out.flush();
        return error.CheckFailed;
    }
    checks += 1;
}
