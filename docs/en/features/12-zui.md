# 2-D UI library (`zui`)

## Usage

```zig
const zui = @import("zui");

var ui = zui.Ui.init(alloc, .{ .width = 320, .height = 200 });
defer ui.deinit();

// The picture is written as a function of the current state: no tree to build,
// no callbacks to register.
const Body = struct {
    app: *App,

    pub fn run(self: @This(), u: *zui.Ui) zui.Ui.Error!void {
        try u.label("hello, UI", .{});
        if (try u.button("click", .{})) self.app.count += 1;
        _ = try u.checkbox("enabled", &self.app.enabled, .{});
        _ = try u.slider("gain", &self.app.gain, .{ .min = 0, .max = 1 });
        _ = try u.textEdit("name", &self.app.field, .{ .placeholder = "type" });
        try u.progress(self.app.gain, .{ .w = 200 });
    }
};

// One frame = (input, body). Input is a value, so a test and a demo drive the
// same code.
const input = zui.Input{ .pointer = .{ .x = 40, .y = 12 }, .pointer_down = true, .pointer_pressed = true };
const frame = try ui.run(input, Body{ .app = &app });
if (frame.skipped) { /* nothing changed: the display list was reused */ }
```

Layout is nested containers. `body` is any value with a `run(self, ui)` method —
**the value itself is the capture**, which is how Zig spells a closure without
allocating one.

```zig
const Panel = struct {
    pub fn run(_: @This(), u: *zui.Ui) zui.Ui.Error!void {
        _ = try u.button("one", .{ .w = .{ .px = 80 } });
        try u.container("fill", .{ .w = .full, .h = .{ .px = 40 }, .bg = u.theme.surface }, Spacer{});
    }
};

try ui.container("window", .{
    .w = .full, .h = .full,
    .pad = .all(12), .gap = 8, .dir = .y,
    .bg = ui.theme.bg, .radius = 8,
}, Panel{});
```

### Virtual lists and tooltips

For long lists, use `virtualList`: **only the rows you can see are called**, so a
million rows cost one frame of screen, not one frame of data.

```zig
const Rows = struct {
    pub fn run(self: @This(), u: *zui.Ui) zui.Ui.Error!void {
        try u.virtualList("log", 1_000_000, .{
            .style = .{ .w = .full, .h = .{ .px = 200 } },   // it *is* the scroll region
            .row_height = 20,                                 // rows have a fixed height
        }, self, row);
    }
    // A row gets its index and its own box. Nothing measured it, so it paints
    // its own background.
    fn row(_: @This(), u: *zui.Ui, i: usize, box: zui.Rect) zui.Ui.Error!void {
        try u.paintRect(box, if (i % 2 == 0) u.theme.surface else u.theme.bg);
        try u.label("row", .{});
    }
};
```

`virtualList` replaces a `scrollable` (nesting the two would let the outer viewport
eat the wheel event). The fixed row height is a contract, not a compromise: it
turns the visible range into two divisions and removes the measuring pass. A row's
position comes straight from its index (`view.y - scroll + i * (row_height + gap)`),
so a list that skipped rows cannot inherit their space. The tests pin "a million
rows build fewer than six", "rows are exactly 20 px apart", "a 20 px wheel moves
them exactly 20 px" and "a row that scrolled out of view is not built".

`tooltip` is **deferred**: you register it inside the body and `endFrame` paints it
after the body, so it is always on top and cannot be covered by a container or a
later widget — no second pass over the window, which is what a popup would need.
The only condition is "is this id hot", so it adds no state.

```zig
// The host picks the anchor: a widget knows its own box for exactly one frame.
if (textAnchor(u, "Click me")) |at| try u.tooltip(id, Rect.init(at.x - 52, at.y - 9, 104, 18), "sets clicks += 1");
```

The display list turns into a picture, and this is the only part that depends on
`zgf`:

```zig
var img = try zgf.Image.init(alloc, 320, 200, 3);
defer img.deinit();
zui.render.draw(&ui.list, &img, .{ .clear = ui.theme.bg });   // scale = 2 for HiDPI
try img.savePpm(io, "frame.ppm");
```

```
zig build run-ui-demo     # scripted input, nine-panel filmstrip, 31 checks
```

## Implementation intent

### The paradigm: immediate API + retained display list + minimal retained state

`zui` is deliberately a **hybrid**, and the three layers retain different things:

| Layer | Paradigm | Why |
| --- | --- | --- |
| Widget API | **immediate** | `if` and `for` are how Zig wants to be written; a widget tree kept in sync with application state is work that then does not exist |
| Frame (display list) | **retained** | An idle frame runs no body, a GPU can consume it, and a test can inspect the picture |
| Interaction state | **the minimum** | hot/active/focus/scroll/animation/text-edit — exactly what one frame's input cannot recompute |

Three properties of Zig drove the choice:

1. **No closures.** A body is a struct with `run(self, ui) Error!void`. Captures are
   fields, so there is nothing to allocate and no indirect call:
   `container(id, style, body)` monomorphises on the body's type. Without this,
   immediate mode is awkward to write.
2. **No reflection.** You cannot walk a widget tree and diff it. So the retained
   thing is not a tree but a **display list** — an array of commands — and
   rebuilding means calling the body again.
3. **Explicit allocators.** Per-frame scratch (wrapped lines, ids, measurements)
   goes into an arena that is reset each frame; the display list outlives the
   frame, so it lives on a different allocator.

### Rejected alternatives

| Alternative | Why not |
| --- | --- |
| **Fully retained** (widget tree + callbacks) | Zig has no closures, so this means hand-wiring `*anyopaque` + function pointers + a user-side state struct: it hands the user the most error-prone job in UI code, keeping the tree in sync |
| **Fully immediate** (rebuild everything, every frame) | Simple, but it burns CPU and battery on frames where nothing changed, and it makes "compare this frame to that frame" impossible — the thing that makes UI testable |
| **DSL / code generation** | Another language to learn, errors that are unreadable, and a build step outside `zig build` (this repository does not require Python for anything it ships) |
| **Virtual-DOM-style diffing** | No reflection means no general differ, and Zig structs are cheap: rebuilding is faster than diffing what you already have |
| **Third-party dependency** (port an egui-style framework) | Against "zero dependencies"; besides, the existing immediate-mode GUIs are built on C++/Rust closures and templates |

### What is in each layer

* **Display list** (`frame.zig`). Seven commands: `rect`, `rounded`, `line`,
  `triangle`, `text`, `push_clip`, `pop_clip`. `DisplayList` carries the command
  array, the clip depth, a `revision`, and a `reused` flag. Commands are already
  resolved to **absolute coordinates**, so neither the rasteriser nor a GPU needs
  to know anything about layout. For the GPU path they can be expanded to
  `Instance{rect, color, radius, kind, uv}` — one struct for rectangles, rounded
  boxes and glyph quads.

* **Retained state** (`context.zig`). `hot` (under the pointer), `active` (owns
  the press), `focus`, scroll offsets, animation values, and text-edit
  caret/selection. An `Id` is a hash of the widget's name and of the containers it
  sits in, stable across frames, and `forget`/`forgetAll` drops entries on demand
  (a shrinking dynamic list needs that).

* **The idle rule.** If `input.isIdle()` (no press, key, text or wheel), the
  pointer did not move since last frame, nothing is animating and nothing called
  `markDirty()`, then **the body is not run and the display list is reused as it
  stands**. The pointer position needs its own retained copy because a position is
  only "moved" relative to the previous one, which is not a property a single
  `Input` can know. Measured: 60 consecutive idle frames cost ~15 ns each, against
  4–6 µs to rebuild the frame.

### Layout: two passes, parent decides

`container` runs its body **twice**: once measuring (`measuring = true`, draws
nothing, collects what the children want) and once drawing (handing out real
coordinates). `Size.fill` is weighted distribution of what the fixed children left
over; `Size.auto` is content, `Size.px` is fixed, `min_w`/`min_h` are floors.
Placement happens entirely in `occupy`, so a child never needs to know its own
coordinates.

### The rasteriser: one place for shapes, two paths to pixels

Antialiasing is neither multisampling nor a blur pass: it is **analytic coverage
per shape**. And that coverage lives in exactly one function —
`coverage(inst, x, y)` in `src/zui/raster.zig`, which takes a pixel centre and
returns 0..1, the same question a fragment shader is asked.

| Shape (`Instance.Kind`) | How the coverage is computed |
| --- | --- |
| `rect` | the area it overlaps with the pixel square |
| `rounded` | signed distance field of a rounded box → `clamp(0.5 - d, 0, 1)` |
| `glyph` | bilinear sample of the atlas (`uv` in `points[0..4]`) |
| `line` | distance to the segment, `clamp(w/2 + 0.5 - d, 0, 1)`, round caps |
| `triangle` | 4×4 supersampling — small and rare, so correctness over elegance |

**A border is two rounded boxes**: the outer box in the border colour, then the
inner box (shrunk by `border`, radius `radius - border`) in the fill colour.
Two ordinary shapes beat reimplementing an analytic ring (outer SDF − inner SDF)
in every backend, and it needs no border mode in a shader.

The drawing loop is in one place too (`raster.emit`): walk the pixels the
instance's bounding box covers inside its clip, take `coverage`, composite.
`render.zig` (display list straight to pixels) and `pipeline.zig` (display list to
an instance stream, then pixels) both call that same loop.

Three details are what make it fast:

* **No `zgf.Image` element accessors.** An `Image` is a `zpu.Tensor`, and the
  generic `readF64`/`writeF64` take a dtype branch per *component*: measured at
  **~80 ns per write**, which is 15 ms of a 320×200 frame before a single shape is
  drawn. `Tensor.flatSliceMut(f32)` (public API) takes the slice once per frame
  and blends straight into it; a non-`f32` or non-contiguous image falls back to
  the generic path. That change alone took a frame from **19.5 ms to 1.9 ms**.
  The same `Surface` also wraps a raw `f32` buffer (`Surface.fromSlice`), so a
  kernel writes through the identical code.
* **`scale` applies while walking.** Display list and instances are both in
  logical pixels; `Options.scale` is the device ratio. Widgets never learn the DPI.
* **A fully covered opaque pixel is a store**, not a blend: the background is most
  of a frame's pixels.

### The font and the atlas

`font_data.zig` holds 95 ASCII glyphs rasterised at 14 px by `tools/gen_font.py`:
one byte of coverage per pixel, plus the advance and the ink box. The generator is
not part of the build — **its output is committed**, so `zig build` never runs
Python.

A GPU cannot sample 95 textures, and a shader cannot know where a glyph's bytes
start. So `src/zui/atlas.zig` builds the atlas **at comptime**: shelf packing,
128 texels wide, one texel of padding between glyphs, one 8-bit image of coverage.
A glyph becomes the rectangle `uvOf(i)`, so it draws as **one textured quad** whose
`uv` points into the atlas. The padding reads zero, so outside the ink box the
result is transparent — exactly what `font.coverage` reports there, which is one
of the reasons the two paths agree.

The atlas is a `const` array with no runtime init and no allocation, and it uploads
as it stands.

Two invariants connect the generator to the reader, and both are pinned by tests:

1. **The ink box's width is the pixel stride.** A padded bitmap tears every glyph
   diagonally; that bug happened once and a test caught it. The generator re-reads
   the box from the drawn pixels and asserts that all four edges carry ink.
2. **An unknown byte is one replacement glyph**, not one per byte: a multi-byte
   character renders as a single `?`.

### The instance path: the same frame, as data

The display list is the **authoring** format — nested clips, a text run — and a
shader cannot be handed either. `Instance` is the **drawing** format: one shape per
record, no pointers, fixed size, 20 floats (80 bytes, 16-byte aligned, so nothing
has to be packed on the way to a buffer):

```zig
pub const Instance = struct {
    box: [4]f32,     // x, y, w, h - always the true bounding box
    color: [4]f32,
    clip: [4]f32,    // the clip in force for this instance
    param: f32,      // rounded: radius; line: stroke width
    kind: u32,       // 0 rect, 1 rounded, 2 glyph, 3 line, 4 triangle
    points: [6]f32,  // glyph: atlas uv; line: the ends; triangle: the vertices
};
```

* **One walk** (`zui/instances.zig`) is the whole conversion. A border becomes two
  instances; a text run becomes **one instance per glyph** (a space has no ink and
  produces none).
* **Clips are not flattened.** Each instance carries the clip that was in force.
  Reorder the stream, split it into passes or hand it to another thread, and it
  still means the same thing. Nesting and restoration are pinned by tests.
* **Both paths call the same function.** The only difference between `render.zig`
  and `pipeline.zig` is *when* the pixels are written — while walking, or after the
  buffer is built — so "the CPU and the GPU draw the same picture" is a test rather
  than a hope. It runs at 1× and at 2× with an origin offset, and demands
  **zero difference**.

Four things go to a device: `pipeline.pack` (a flat array, 20 floats per instance),
`Params` (a uniform: target size, origin, scale, instance count, atlas size),
`atlasPixels()` (an `r8unorm` texture) and `pipeline.kernel` (the kernel itself).

The kernel is **one invocation per instance**. Compositing is order-dependent — a
later shape covers an earlier one — so it is only correct on a backend that runs
invocations in order. That is why it **deliberately carries no shading-language
source**: `Kernel.canRun(.vulkan)` is false, so `zbackend.dispatch` falls back to
the CPU instead of handing a driver something it would fail on. The demo asserts
that refusal as a check.

On a GPU the same picture is drawn as a **render pass**: the vertex shader expands
one instance into a quad, the fragment shader computes the same `coverage`, and the
hardware's blender puts them in order. That WGSL (`pipeline.render_pass_wgsl`) is
shipped as source but is **not registered as a compute kernel** — registering it
would make `canRun` lie, and a dispatch would try to compile a fragment shader as
compute and fail at the driver.

### The font

`font_data.zig` holds 95 ASCII glyphs rasterised at 14 px by `tools/gen_font.py`:
one byte of coverage per pixel, plus the advance and the ink box. The generator is
not part of the build — **its output is committed**, so `zig build` never runs
Python.

Two invariants connect the generator to the reader, and both are pinned by tests:

1. **The ink box's width is the pixel stride.** A padded bitmap tears every glyph
   diagonally; that bug happened once and a test caught it. The generator re-reads
   the box from the drawn pixels and asserts that all four edges carry ink.
2. **An unknown byte is one replacement glyph**, not one per byte: a multi-byte
   character renders as a single `?`.

### Testing

The widget tests are headless and do nothing but push a **list of inputs**.
Because `Input` is a value, a click and a drag are that list.

```zig
_ = try ui.run(press(20, 13), body);
_ = try ui.run(release(20, 13), body);
try expect(clicks == 1);
```

Assertions read either the display list (rectangle geometry) or the pixels.
"Disabled widgets take no input", "an idle frame does not run the body", "scrolling
clamps at both ends", "the collapsing header remembers", "typing lands at the
caret" are all written that way. `examples/ui_demo.zig` drives the same machinery
with a script and prints a nine-panel filmstrip plus **31 checks** — six of them
about the instance path: the instance count, the device layout round-tripping byte
for byte, the atlas travelling with the frame, the kernel refusing GPU backends,
and **the kernel's pixels being identical to the software renderer's, pixel for
pixel**.

## Performance (`zig build bench -- --filter ui`, this machine)

| Case | Value | Meaning |
| --- | --- | --- |
| Frame rebuild, 320×200 | **~4–6 µs** (170–230k fps) | layout + display list, every widget family |
| Idle frame | **~15 ns** | body skipped, list reused |
| Software raster, 320×200 | **~1.27 ms** (51 Melem/s) | one frame incl. rounded SDFs and text |
| Same at 2× device pixels | about the same (1.21 ms) | the covered area is what costs |
| Display list → instances | **654 ns** | 57 commands → 283 records (text is one per glyph) |
| Packing the instances | **232 ns** | 20 floats × 283 = 22640 B into a flat array |
| Rasterising via instances | **1.51 ms** | walk, scale, then the same emit loop |
| One kernel frame (incl. clear) | **0.98 ms** | dispatched on `zbackend`'s CPU backend |

Rebuild is single-digit microseconds and rasterisation is milliseconds — that
ratio is what the design is built around: **the rasteriser is the part to move to
the GPU**, while layout and the state machine stay on the CPU (milestone ZUI6).

## How it differs from other libraries

* **Dear ImGui / egui (immediate-mode GUIs)**: the writing style is the same, but
  `zui` **retains the frame**. An idle frame does nothing (~15 ns), and the display
  list is a picture, so headless tests are possible. ImGui-style libraries have
  draw lists too, but idling and testability are left to the application.
* **Clay (C immediate-mode layout)**: likewise a single dependency-free library,
  but `zui` carries drawing and interaction state as well, and renders into a
  `zgf` image.
* **Slint / declarative UI**: no separate DSL and no code generation — the UI is
  ordinary Zig, the hierarchy is nested calls, and errors are compiler errors.
* **HTML/CSS**: no selectors and no cascade. The parent decides the children's
  extents; children never know their own coordinates.
* **Its relationship to `zgf`**: `zui` is a *user* of `zgf`. Only
  `zui/render.zig` imports it — layout, widgets and the state machine do not know
  it exists, which is why the same widget code runs against a test (the display
  list), an image (software), or a future GPU (instance buffer).

## Limitations (honestly)

* **The GPU form is not what runs here.** The kernel path (`zbackend`) is correct on
  a backend that preserves invocation order — on this machine, the CPU. The WGSL
  render-pass form is shipped but **cannot be verified in CI** (no GPU in the
  sandbox, so it is not claimed to compile). No mip maps, no texture arrays, no
  variable fonts.
* **One window.** No multi-window, popups or modals — nested containers and clips
  are what is on offer.
* **ASCII text only.** 95 Latin glyphs at 14 px; UTF-8 gets one replacement glyph
  per character (no CJK yet).
* **The host produces the input.** There is no event loop or window handle; you
  hand `run` an `Input`. That is exactly why the tests and the demo run the same
  code.
* **No selection and no IME.** The text field has a caret, insert and delete.
