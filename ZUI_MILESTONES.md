# ZUI milestones — the 2-D UI library

`MILESTONES.md` tracks **zpu** (the numerical core) and **zgf** (the graphics
library). This file tracks **zui**, the 2-D UI library built on top of both, and
it is deliberately a *separate* track: the numerical core does not need to know
that widgets exist, and the UI can move at its own pace.

Same rules as the other track, because they are the project's rules:

* every feature lands with **tests** (headless, no window, no GPU) and, where a
  number is meaningful, a **benchmark**;
* every feature lands with **documentation of usage *and* of implementation
  intent**, in **Japanese and English** (`docs/features/12-zui*.md`,
  `docs/en/features/12-zui*.md`), which `zig build doc` turns into HTML;
* every milestone ends with a **`zig build run-ui-demo`** that is self-checking;
* no Python, no windowing system and no third-party package is required to
  build, test or benchmark any of it.

## The architecture decision (what "方式" this is)

Three layers, each using the paradigm Zig is actually good at. The full
reasoning is in `docs/features/12-zui.md`; the short version:

| Layer | Paradigm | Why |
| --- | --- | --- |
| **Widget API** | *immediate mode* (IMGUI) — you call `ui.button(...)` and it is drawn, every frame | Zig has no closures, no GC and no reflection. A retained widget *tree* would need callbacks, which means hand-passed `*anyopaque` contexts and manual lifetimes everywhere. Immediate mode puts the tree in ordinary `if`/`for` control flow, so it is plain Zig with no hidden machinery. |
| **Frame** | *retained display list* — the immediate calls only append commands | The command list is the one thing worth keeping: it makes "nothing changed, skip the frame" possible, and it is the only boundary a GPU backend can be handed (vertex/index/uniform data, not callbacks). |
| **State** | *retained, but tiny and explicit* — a `Context` of interaction state (hot/active/focus/scroll/animation) plus text-edit state | Some state genuinely cannot be recomputed per frame. Keeping exactly that, keyed by widget id, is the honest minimum: a few hundred bytes, explicitly listed, fully inspectable. |

Everything else follows from Zig's properties rather than from UI fashion:

* **No allocation per frame in steady state** — the frame arena is reset, not
  freed; the persistent maps only grow when a widget is seen for the first time.
* **`comptime` widget capabilities** — a widget says what it needs (focusable,
  want-hover) in its type; the frame code specializes on it instead of
  branching at runtime.
* **Hand-rolled closures** — a "body" is a struct instance with a `run` method.
  Same effect as a closure, zero allocation, visible in the type system.
* **Headless first** — `Input` is a value, output is a display list; the
  software renderer is `zgf`. A window, an IME or a GPU is an *adapter*, never a
  dependency of the core.

## Status legend

`[ ]` not started · `[~]` in progress · `[x]` done and verified.

---

## ZUI0 — the contract before the code

- [x] `ZUI_MILESTONES.md` (this file) is the plan of record for the track.
- [x] `docs/features/12-zui.md` + `docs/en/features/12-zui.md`: the paradigm
      decision above, written out with what was **rejected** and why (retained
      trees + callbacks, a DSL, virtual-DOM diffing, a GPU-first design).
- [x] The layering rule is written down and enforced: `zui` core knows nothing
      about `zgf`; only `zui/render.zig` (the software backend) imports it.

## ZUI1 — core: geometry, input, ids, layout, frame

- [x] `zui/types.zig`: `Vec2`, `Rect` (intersect/contains/merged/inset/place),
      `Edges`, `Color` (own type — no `zgf` in the core), `Size`, `Id`.
- [x] `zui/input.zig`: an `Input` **value** (pointer, buttons, scroll, keys,
      typed text, time, dt) plus edge detection helpers, so a headless test can
      synthesize any interaction.
- [x] `zui/frame.zig`: `DisplayList` of `Command`s (rect, rounded rect, line,
      triangle, text, push/pop clip), arena-owned; `revision`, `reused` and the
      `Instance` form the GPU path will consume. (No `image` command yet: the
      widgets that exist do not need one.)
- [x] The box model lives in `ui.zig` rather than a separate file: `ocupy` is the
      single place that resolves `Size{auto,px,fill}`, padding, gaps, row/column,
      `cross_align`, `justify` and `min_w`/`min_h`; measure-then-place, no solver.
- [x] `zui/context.zig`: the retained minimum — `hot`, `active`, `focus`,
      per-id scroll, per-id animation values, text-edit state — with explicit
      `deinit`.
- [x] Tests: rect math, id scoping (`panel/ok` ≠ `dialog/ok`), layout of
      mixed auto/fill children, clip nesting, skip-frame decision table.

## ZUI2 — painter and the software renderer

- [x] The painter is `Ui.paint*` (`paintRect`, `paintRounded`, `paintLine`,
      `paintTriangle`, `paintText`) — the only way widgets emit, in logical
      pixels, honouring the current clip. A separate `painter.zig` would have
      been a layer with one implementation; it earns its place when a second
      backend arrives, at which point these move.
- [x] `zui/render.zig`: display list → `zgf.Image`. Alpha compositing,
      anti-aliased rounded rects (analytic coverage), 1-px crisp borders, a
      `scale`/`origin` transform for HiDPI, and a fast path that blends straight
      into the image's `f32` slice (the generic element accessors cost ~80 ns per
      component; a frame went 19.5 ms → 1.9 ms).
- [x] **`zgf` update**: `Image.blendPixel` and `Image.blendCoverage` (src-over,
      coverage as a factor separate from the colour's alpha), since the existing
      drawing API was opaque-only. The clip stack lives in the renderer, where
      clipping is a rasterisation concern.
- [x] `tools/gen_font.py` output committed as `src/zui/font_data.zig` (ASCII,
      8-bit coverage); **the build never runs Python**. The generator asserts
      that every glyph box is tight, and two tests in `font.zig` re-assert it
      over the committed table — a padded row used to tear the glyphs.
- [x] Tests: a rectangle lands on exactly its pixels and a half-covered pixel is
      half as bright; a rounded corner is anti-aliased; a border leaves the
      inside alone; clip is respected, including nesting; scaling moves the
      picture by the scale; text puts ink where the metrics say.
- [x] Bench: `ui frame build`, `ui frame idle`, `ui raster 320x200`,
      `ui raster 320x200 @2x` in `bench/bench.zig`. The library modules are
      compiled `ReleaseFast` for the bench (a benchmark of a `Debug` build
      measures the debugger).

## ZUI3 — widgets and the interaction state machine

- [x] `ui.container(id, style, body)` is all three: a panel, a row and a column
      are `.dir` and a `Style` apart. The body is any struct with
      `run(self, ui)` — Zig's closure, with nothing allocated to build one.
- [x] `ui.label`, `ui.button` / `buttonId`, `ui.checkbox`, `ui.radio`,
      `ui.slider`, `ui.progress`, `ui.separator`, `ui.spacer`.
- [x] Hot/active/focus transitions, Tab focus order (against the previous
      frame's ring, so the widget it lands on sees the focus the same frame),
      keyboard activation (`space`/`enter`), hover/press animations.
- [x] `ui.scrollable` + scrollbar, wheel + drag, momentum-free (honest) model.
- [x] `ui.collapsing` (persistent open flag). The tooltip turned out not to need
      the second pass this line predicted: deferring it to `endFrame` puts it on
      top of the frame, which is all a tooltip wants (see ZUI7). Popups and menus
      still do need a pass over the whole window.
- [x] Tests: click lands once (not twice), a press released outside does *not*
      fire, a press that started elsewhere is not adopted, Tab cycles in build
      order and wraps, a disabled widget takes no input and is never hot, a
      slider maps pixels to values monotonically and clamps, scroll clamps at
      both ends.

## ZUI4 — text

- [x] `zui/font.zig`: metrics, `measure`, `wrap` (word wrap + hard breaks),
      glyph lookup, bilinear sampling at any size from the 14 px master, and
      one replacement glyph per multi-byte character.
- [x] `ui.textEdit` (the one genuinely retained widget): caret, insert, delete,
      home/end/left/right, scroll the caret into view. Selection state exists in
      `EditState` but marquee selection and clipboard are **not** implemented —
      they need a host clipboard and are their own milestone.
- [x] Tests: measure against the generated metrics, wrap never exceeds the width
      except for a single over-long word, editing round-trips a string, the caret
      survives frames, typing lands at the caret. `zui/text.zig` covers the
      buffer itself (overflow, caret motion, hit testing).
- [ ] Bench: shaping ns/char, wrap ns/line — **not written**. `ui frame build`
      covers text end to end for now; a per-call number for `measure`/`wrap`
      is the next thing to add here.

## ZUI5 — the hybrid half

- [x] Idle skip: no input, **no pointer movement**, no animation and no
      `markDirty()` ⇒ the body is not re-run and the previous display list is
      reused, flagged through `FrameResult{rebuilt, skipped}` and
      `List.reused`. The pointer position is retained separately, because
      "moved" is a property of two frames and `Input` describes one.
- [x] `ctx.animate(id, target, rate, dt)` (exponential approach with a snap) and
      `Context` retention of values, driven by the input's `dt` so a test can
      advance time by hand.
- [x] Tests: a frame with no input reuses the list and does not move the
      revision, input and `markDirty()` bring it back, an animation keeps frames
      alive until it settles, 60 settled idle frames skip every time, and a
      synthetic session counts skipped frames (the demo reports the ratio).
- [ ] `ui.virtual_list`: **not written**. The scrolling list builds every row and
      clips; a million-row model needs the visible-window API first.
- [ ] Bench: idle frame cost ≈ 0 (done: ~15 ns); a 1M-row list builds a frame in
      O(visible) (not measurable until the API exists).

## ZUI6 — the instance path and the demo

- [x] `zui/frame.zig`: `Instance` — one flat, pointer-free, 20-float (80 B)
      record per shape: `box`, `color`, `clip`, `param`, `kind`, and a six-float
      `points` payload (glyph uv / line ends / triangle vertices). `bounds()`,
      `scaled()` and no-clip default included, so a device buffer needs no
      packing step.
- [x] `zui/instances.zig`: the display list → instance stream conversion, as one
      streaming walk (`each`/`collect`/`count`) with a transform. A border is two
      rounded boxes, a text run is one quad per glyph, a clip is carried *per
      instance* (not flattened), and the whole walk allocates nothing.
- [x] `zui/atlas.zig`: the glyph atlas, built at **comptime** — shelf packing of
      every glyph into one 8-bit image, one texel of padding, `uvOf(i)` and a
      bilinear `sample(u, v)`. No allocation, no init order, uploadable as-is.
- [x] `zui/raster.zig`: the coverage maths in one place — `coverage(inst, x, y)`
      per kind (rect area, rounded SDF, glyph atlas sample, capsule, 4×4
      supersampled triangle) plus `emit`, the pixel loop both render paths run.
      `Surface.fromSlice` lets a kernel write through the identical code.
- [x] `zui/pipeline.zig`: the frame as data — `pack`/`unpack` (20 floats per
      instance), `Params` (the uniform), `atlasPixels()`, and a `zbackend`
      `Kernel` that rasterises the buffer (one invocation per instance). It
      deliberately ships **no** shading-language source, so `canRun` is false on
      every GPU backend and a dispatch falls back to the CPU instead of failing.
- [x] `pipeline.render_pass_wgsl`: the GPU form of the same picture (vertex shader
      expands one quad per instance, fragment shader computes the same coverage,
      hardware blending keeps the order). Shipped as source for a host with a
      `wgpu` device; not registered as a compute kernel, and not claimed to
      compile anywhere CI can check.
- [x] Tests (69 in the `zui` module): the instance stream is a pure function of
      the display list (two walks are byte-identical); the clip stack nests and
      restores; a bordered box is two instances with the inner box shrunk; the
      device layout round-trips field for field; **the instance path draws the
      same picture as the display list path at 1× and at 2× with an origin, with
      zero differing pixels**; and the kernel's output matches the software
      renderer byte for byte.
- [x] Bench: `ui frame -> instances` (654 ns), `ui instance pack` (232 ns),
      `ui raster via instances` (1.51 ms), `ui kernel dispatch` (0.98 ms).
- [x] `examples/ui_demo.zig` (`zig build run-ui-demo`): a small application UI
      driven by *synthetic* input, self-checking (**29 assertions** on geometry,
      pixels, frame accounting and the instance path), writing
      `zig-out/ui-demo.ppm` (a nine-panel filmstrip) and
      `zig-out/ui-demo-kernel.ppm` (the same frame drawn by the kernel).
- [x] Docs: `docs/features/12-zui.md` + `docs/en/features/12-zui.md` describe the
      instance format, the atlas, the parity test and the kernel's honesty rule.

## ZUI7 — the remaining gaps

- [ ] N-frames-in-flight buffer lifetime: a slot ring so a device may hold two
      frames while the CPU keeps building (the instance buffer is the frame, so
      the ring is what makes an asynchronous upload safe).
- [x] `ui.virtualList`: only the visible rows are built (a million-row list paints
      in 798 ns), rows are placed by index with a fixed height, and the scroll
      behaviour is shared with `scrollable` (`scrollRegion`) so the two cannot
      disagree about a wheel event. Tests: fewer than six rows built for a
      million, exactly one pitch apart, exactly 20 px per 20 px of wheel, and a
      row that scrolled out of view is not built at all.
- [x] `ui.tooltip`: deferred — registered in the body, painted by `endFrame`, so
      it is on top of the frame without a second pass over the window.
- [ ] Text selection and the clipboard.
- [ ] Bench: shaping ns/char, wrap ns/line, virtual-list frame cost.
- [ ] A real GPU run of `pipeline.render_pass_wgsl` on a machine with a driver
      (the sandbox has none), plus the Vulkan/WebGPU pipeline objects it needs.
