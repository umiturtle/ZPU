# ZUI/ZAPP Development Roadmap v4 — Zig-Native Architecture

> **📐 アーキテクチャ詳細**: [`docs/zui/ARCHITECTURE.md`](docs/zui/ARCHITECTURE.md)
> **📋 マイルストーン**: [`docs/zui/MILESTONES.md`](docs/zui/MILESTONES.md)
> **🧩 モジュラー構成**: [`docs/zui/MODULAR_ARCHITECTURE.md`](docs/zui/MODULAR_ARCHITECTURE.md)

## Vision: Zigでしか作れない最高のUIライブラリ

**Comptimeでフレームワークのコストをゼロに、明示的パラメータで外部から完全に制御可能に。**

ZUI/ZAPP aims to be the Zig-native equivalent of GPUI (Zed), combining:
- GPUI's GPU scene-graph rendering pipeline
- WebRender's display list + batching architecture
- Vello's compute-shader 2D rendering
- Clay's microsecond arena-based layout
- Slint's compile-time declarative approach (adapted for Zig comptime)

---

## 2026 Landscape: What We Learned

### The State of the Art

| Framework | Language | Architecture | Performance | Key Insight |
|-----------|----------|-------------|-------------|-------------|
| **GPUI/Zed** | Rust | Hybrid immediate+retained, GPU scene graph | 120fps, 2ms input, 0.6s cold start | UI as game engine: scene graph → diff → GPU draw calls |
| **WebRender** | Rust | Display list → batch → GPU rasterize | Firefox-grade, parallel CSS (Stylo) | Separate scene analysis from GPU execution |
| **Vello** | Rust | GPU compute-centric 2D renderer | 177 FPS paris-30k@1600² on M1 Max | Prefix-sum parallelization on GPU, no CPU sorting |
| **Blitz** | Rust | Stylo + Taffy + Vello + wgpu | Browser-grade CSS, native rendering | Reuse Firefox's CSS engine for layout |
| **Xilem** | Rust | Reactive diffing → Masonry widgets → Vello | SwiftUI-like, strongly typed | View tree diff → minimal widget updates |
| **Iced** | Rust | Elm architecture, wgpu/glow | Reactive rendering, 60-80% CPU reduction | Only redraw changed portions |
| **Slint** | Rust/C++ | Declarative DSL → compiled native code | <300KB RAM, embedded-ready | Compile-time optimization of expressions |
| **Clay** | C99 | Arena-based flexbox layout | <100μs for 1000 elements | Zero malloc/free, static arena |
| **DVUI** | Zig | Immediate mode, multi-backend | Variable framerate capable | Only mature native Zig GUI |
| **zigui** | Zig | SwiftUI-like, comptime declarative | SDL_GPU (Metal/Vulkan) | Pure Zig TrueType rasterizer |
| **Gooey** | Zig | GPU-accelerated, Vulkan/Metal/DX12/WebGPU | Hardware-accelerated | Direct GPU layout primitives |

### Key Architectural Patterns (2026 Consensus)

1. **GPU Scene Graph** (GPUI, WebRender): Maintain a scene tree, diff between frames, emit only changed GPU commands
2. **Compute-Shader Rendering** (Vello): Offload sorting, clipping, tile generation to GPU compute shaders
3. **Arena-Based Layout** (Clay): Zero-allocation layout with static arena, microsecond performance
4. **Parallel CSS/Style** (Stylo): Multi-threaded style resolution scales linearly with CPU cores
5. **Incremental/Diff Rendering** (GPUI, Iced): Track dirty regions, only redraw what changed
6. **Compile-Time Optimization** (Slint, Zig comptime): Evaluate layout expressions at compile time
7. **Multi-Pass Layout** (GPUI, SwiftUI): Measure → Resolve → Position → Paint
8. **Text Stack**: Parley (shaping) + Fontique (font resolution) + SDF/MSDF rendering

---

## Our Architecture: Zig-Native GPU Pipeline

### The ZUI Rendering Pipeline (Inspired by GPUI + WebRender)

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION STATE                            │
│   (User's data, owned by the application — like GPUI Views)    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ notify() / dirty tracking
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1: ELEMENT TREE BUILD                        │
│   • Declarative widget calls → element tree                     │
│   • comptime specialization (zero-cost features)                │
│   • Only dirty subtrees rebuilt (GPUI-style incremental)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: LAYOUT (Arena-based, Clay-inspired)       │
│   • Flexbox + Grid + Block algorithms                          │
│   • Static arena allocator (zero malloc/free)                   │
│   • comptime layout for static UIs                              │
│   • Target: <100μs for 1000 elements                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: DISPLAY LIST (WebRender-inspired)         │
│   • Convert laid-out tree → display list                        │
│   • Batch similar operations (minimize GPU state changes)       │
│   • Visibility culling (skip off-screen elements)               │
│   • Sorted render command array                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 4: GPU RENDERING (Vello/GPUI-inspired)       │
│   • Software: ZGF framebuffer (headless/testing)                │
│   • GPU: wgpu (Vulkan/Metal/DX12/WebGPU)                       │
│   • Compute shaders for complex effects                         │
│   • SDF text rendering                                          │
│   • Target: <2ms per frame, 120fps capable                     │
└─────────────────────────────────────────────────────────────────┘
```

### Zig-Specific Advantages We Exploit

#### 1. comptime Scene Graph Specialization
```zig
// Each widget type generates specialized rendering code at compile time
pub fn Widget(comptime Config: WidgetConfig) type {
    return struct {
        pub fn build(self: Self, ctx: *BuildContext) void {
            if (comptime Config.shadows) {
                ctx.emitShadow(self.shadow);
            }
            if (comptime Config.gradients) {
                ctx.emitGradient(self.gradient);
            }
            // Unused features → zero code, zero cost
        }
    };
}
```

#### 2. comptime Layout Caching
```zig
// Static layouts computed entirely at compile time
pub fn staticLayout(comptime spec: LayoutSpec) LayoutResult {
    comptime {
        // Full flexbox computation at compile time
        // Zero runtime cost for static UIs
    }
}
```

#### 3. Arena-First Memory Model
```zig
// Every subsystem uses arena allocation
pub const FrameArena = struct {
    buffer: []u8,
    offset: usize = 0,
    
    pub fn reset(self: *FrameArena) void {
        self.offset = 0; // O(1) frame reset
    }
};
```

#### 4. Error-Union Pipeline
```zig
// Every phase can fail gracefully
pub fn renderFrame(self: *Renderer) !void {
    const tree = try self.buildElementTree();
    const layout = try self.computeLayout(tree);
    const commands = try self.buildDisplayList(layout);
    try self.executeGPU(commands);
}
```

---

## Milestones (Revised for Maximum Aggression)

### 🎯 Milestone 1: GPU Scene Graph Core (P0)
**Inspired by**: GPUI, WebRender

The foundation: scene graph + dirty tracking + display list

#### 1.1 Scene Graph
- [ ] `SceneNode` tree structure (parent/child/sibling)
- [ ] Dirty flag propagation (like GPUI's `notify()`)
- [ ] Incremental tree updates (only rebuild dirty subtrees)
- [ ] Z-ordering and hit testing
- [ ] Clipping regions (nested scissor rects)

#### 1.2 Display List
- [ ] `DisplayCommand` enum (Rect, RoundedRect, Text, Image, Shadow, Gradient, Line)
- [ ] Command batching (group similar operations)
- [ ] Visibility culling (frustum test against viewport)
- [ ] Sort by material/texture (minimize GPU state changes)

#### 1.3 Software Renderer (Testing/Headless)
- [ ] Existing ZGF integration (framebuffer output)
- [ ] Display list → pixel buffer conversion
- [ ] Snapshot testing infrastructure

**Performance Target**: Display list build <500μs for 1000 elements

**Deliverables**:
- `zapp/scene/` — Scene graph, dirty tracking
- `zapp/display_list/` — Command generation, batching
- `zapp/renderer/software/` — Software rasterizer

---

### 🎯 Milestone 2: Arena Layout Engine (P0)
**Inspired by**: Clay, Taffy

Microsecond layout with zero allocations

#### 2.1 Arena Allocator
- [ ] `LayoutArena` — bump allocator for layout data
- [ ] Per-frame reset (O(1), no deallocation)
- [ ] Configurable capacity (4KB-64KB typical)
- [ ] Overflow detection and reporting

#### 2.2 Layout Algorithms
- [ ] Flexbox (row/column, wrap, align, justify, gap)
- [ ] Grid (rows, columns, areas, auto-fill/placement)
- [ ] Block layout (margins, padding, borders)
- [ ] Absolute positioning
- [ ] Intrinsic sizing (min/max content, fit-content)
- [ ] Percentage and fractional units

#### 2.3 Layout Performance
- [ ] comptime layout for static configurations
- [ ] Dirty-region relayout (only recompute changed subtrees)
- [ ] Layout caching (memoize unchanged subtrees)
- [ ] SIMD-optimized measurement passes

**Performance Target**: <100μs for 1000 elements (Clay benchmark)

**Deliverables**:
- `zapp/layout/arena.zig` — Arena allocator
- `zapp/layout/flexbox.zig` — Flexbox engine
- `zapp/layout/grid.zig` — Grid engine
- `zapp/layout/engine.zig` — Multi-pass layout orchestrator

---

### 🎯 Milestone 3: GPU Backend & Command Batch Layer (P0)
**Inspired by**: GPUI, Vello, WebRender

Direct GPU rendering via 4 APIs + intelligent command batching

#### 3.1 GPU Abstraction Layer (自作)
- [ ] `GpuDevice` interface (WebGPU/Vulkan/Metal/DirectX)
- [ ] Platform detection and backend selection
- [ ] Shader compilation pipeline (WGSL → platform-specific)
- [ ] Memory management (buffers, textures, samplers)
- [ ] Synchronization primitives (fences, semaphores)

#### 3.2 Command Batch Layer (comptime制御)
- [ ] `BatchProcessor(Config)` comptime型
- [ ] Command grouping by material/texture
- [ ] Draw call sorting (minimize state changes)
- [ ] Small operation merging (e.g., multiple small rects → single batch)
- [ ] Platform-specific optimization passes
- [ ] **Native**: comptime `.enabled = false` → zero-cost pass-through
- [ ] **Web**: comptime `.enabled = true` → aggressive batching

**Why this layer matters**:
```
Native Desktop:
  1000 draw commands → 1000 API calls → ~1-10ms (acceptable)

Web/WASM:
  1000 draw commands → 1000 JS calls → 10-500ms (critical)
  ↓ batch layer
  1000 draw commands → 50 batched calls → ~1-5ms (acceptable)
```

#### 3.3 Backend Implementations
- [ ] **WebGPU**: Primary for web, fallback for native
- [ ] **Vulkan**: Linux/Windows high-performance
- [ ] **Metal**: macOS/iOS native
- [ ] **DirectX 12**: Windows native
- [ ] Common test suite for all backends

#### 3.4 GPU Rendering Pipeline
- [ ] Vertex/index buffer generation from batched commands
- [ ] Instanced rendering for similar shapes
- [ ] Shader programs:
  - Solid color (rounded rects)
  - Linear/radial gradients
  - Shadow (gaussian blur via compute shader)
  - Text (SDF rendering)
  - Image (textured quads)

**Performance Target**: 
- Native: <2ms per frame for complex UI, 120fps
- Web: <5ms per frame (after batching), 60fps minimum

**Deliverables**:
- `zapp/gpu/device.zig` — GPU abstraction interface
- `zapp/gpu/backends/` — WebGPU, Vulkan, Metal, DirectX implementations
- `zapp/gpu/batch.zig` — Command batch layer (comptime-configurable)
- `zapp/gpu/shaders/` — WGSL shader programs
- `zapp/gpu/pipelines.zig` — Render pipeline management

---

### 🎯 Milestone 4: SDF Text System (P1)
**Inspired by**: Parley, WebRender text

GPU-accelerated text rendering

#### 4.1 Font Pipeline
- [ ] TrueType/OpenType parser (pure Zig)
- [ ] Glyph atlas generation (SDF)
- [ ] Font fallback chain
- [ ] Subpixel positioning
- [ ] Glyph caching (LRU)

#### 4.2 Text Layout
- [ ] Line breaking (Unicode TR#14)
- [ ] Bidirectional text (Unicode TR#9)
- [ ] Kerning and ligatures
- [ ] Text shaping (HarfBuzz integration or pure Zig)
- [ ] Rich text (mixed styles inline)

#### 4.3 Text Editing
- [ ] Multi-line text editor
- [ ] Selection (mouse + keyboard)
- [ ] Cursor navigation
- [ ] Clipboard (copy/paste)
- [ ] Undo/redo (arena-based stack)

**Deliverables**:
- `zapp/text/font.zig` — Font loading, glyph atlas
- `zapp/text/layout.zig` — Text layout engine
- `zapp/text/editor.zig` — Text editing widget
- `zapp/text/sdf.zig` — SDF generation and rendering

---

### 🎯 Milestone 5: Widget Library (P1)
**Inspired by**: GPUI, Material Design

comptime-specialized widgets

#### 5.1 Core Widgets
- [ ] Label (text, rich text)
- [ ] Button (5 variants: filled, elevated, outlined, text, tonal)
- [ ] TextField (single/multi-line, validation)
- [ ] Checkbox / RadioButton / Switch
- [ ] Slider / ProgressBar / Spinner
- [ ] Image (load PNG/WebP/GIF/SVG, display)

#### 5.2 Container Widgets
- [ ] Window (movable, resizable, close/minimize)
- [ ] Panel (docking regions)
- [ ] ScrollView (smooth, momentum)
- [ ] TabView / Accordion
- [ ] SplitView (horizontal/vertical)

#### 5.3 Overlay Widgets
- [ ] Modal dialog
- [ ] Popup menu / Dropdown
- [ ] Context menu (right-click)
- [ ] Tooltip (rich content)
- [ ] Toast notification

#### 5.4 Data Widgets
- [ ] ComboBox / Dropdown
- [ ] ColorPicker (HSV/RGB/HEX)
- [ ] Table (sorting, filtering, virtual scrolling)
- [ ] TreeView (expandable)
- [ ] DraggableValue

**Deliverables**:
- `zapp/widgets/core/` — Basic widgets
- `zapp/widgets/containers/` — Container widgets
- `zapp/widgets/overlays/` — Popup/modal system
- `zapp/widgets/data/` — Data entry/display

---

### 🎯 Milestone 6: Animation & Transition System (P1)
**Inspired by**: Flutter, GPUI

Frame-perfect animations

#### 6.1 Tween Engine
- [ ] Linear interpolation
- [ ] Easing functions (ease-in, ease-out, cubic bezier)
- [ ] Property animation (color, size, position, opacity, rotation)
- [ ] Keyframe sequences
- [ ] Spring physics

#### 6.2 Transition System
- [ ] Page transitions (slide, fade, scale)
- [ ] Widget enter/exit animations
- [ ] Layout animation (smooth reflow)
- [ ] Gesture-driven animation (velocity tracking)

#### 6.3 Animation Controller
- [ ] Play/pause/reverse/repeat
- [ ] Animation groups (parallel, sequence)
- [ ] Interruption handling (smooth transition to new target)
- [ ] vsync-locked timing

**Deliverables**:
- `zapp/animation/tween.zig` — Core tween engine
- `zapp/animation/transitions.zig` — Transition effects
- `zapp/animation/spring.zig` — Physics-based animation

---

### 🎯 Milestone 7: Input & Gesture System (P1)
**Already partially done**

Enhanced input handling

#### 7.1 Event Routing
- [ ] Event capture/bubble phases (DOM-like)
- [ ] Focus management (tab order, programmatic)
- [ ] Keyboard shortcut system (global + local)
- [ ] Input method support (IME)

#### 7.2 Gesture System (Already implemented, enhance)
- [ ] Pinch-to-zoom with focal point
- [ ] Two-finger rotate
- [ ] Edge swipe (back navigation)
- [ ] Long-press with preview
- [ ] Drag & drop (source/target API)

#### 7.3 Accessibility
- [ ] Focus indicators
- [ ] Screen reader integration
- [ ] High contrast mode
- [ ] Keyboard-only navigation (WCAG 2.1 AA)

**Deliverables**:
- `zapp/input/router.zig` — Event routing
- `zapp/input/focus.zig` — Focus management
- `zapp/input/accessibility.zig` — Accessibility layer

---

### 🎯 Milestone 8: Theme & Style System (P2)

comptime theme selection

#### 8.1 Theme Engine
- [ ] Token-based design system (color, spacing, typography)
- [ ] Light/dark theme toggle
- [ ] Custom theme builder
- [ ] comptime theme selection (zero runtime cost)
- [ ] Runtime theme switching

#### 8.2 Style System
- [ ] Style inheritance (cascade)
- [ ] Per-widget style overrides
- [ ] State-based styling (hover, pressed, disabled, focused)
- [ ] Animated style transitions
- [ ] Responsive styles (breakpoints)

**Deliverables**:
- `zapp/theme/tokens.zig` — Design tokens
- `zapp/theme/engine.zig` — Theme engine
- `zapp/theme/style.zig` — Style cascade system

---

### 🎯 Milestone 9: Image Format Completion (P2)

Full format support

#### 9.1 PNG
- [ ] Full decoder (zlib/deflate decompression)
- [ ] Encoder
- [ ] Interlacing
- [ ] APNG (animated)

#### 9.2 WebP
- [ ] Lossy decoder
- [ ] Lossless decoder
- [ ] Animation

#### 9.3 GIF
- [ ] LZW decoder
- [ ] Animation playback
- [ ] Transparency

#### 9.4 SVG
- [ ] SVG parser (XML subset)
- [ ] Shape rendering (rect, circle, path, ellipse, polygon)
- [ ] Path commands (M, L, C, Q, A, Z)
- [ ] Gradients, patterns, transforms
- [ ] Text rendering

**Deliverables**:
- `zapp/image/png.zig` — PNG codec
- `zapp/image/webp.zig` — WebP decoder
- `zapp/image/gif.zig` — GIF codec
- `zapp/image/svg.zig` — SVG renderer

---

### 🎯 Milestone 10: Developer Tools (P2)

World-class debugging experience

#### 10.1 Inspector
- [ ] Hover-to-inspect (highlight widget bounds)
- [ ] Widget tree browser
- [ ] Style viewer (computed styles)
- [ ] Layout debug overlay (margins, padding, constraints)

#### 10.2 Performance
- [ ] Frame time breakdown (layout, paint, GPU)
- [ ] Dirty region visualization
- [ ] Memory usage tracker
- [ ] GPU frame profiler

#### 10.3 Hot Reload
- [ ] Code hot reload (watch files)
- [ ] Style hot reload
- [ ] Layout hot reload
- [ ] State preservation across reloads

**Deliverables**:
- `zapp/devtools/inspector.zig` — Widget inspector
- `zapp/devtools/profiler.zig` — Performance tools
- `zapp/devtools/hot_reload.zig` — Hot reload system

---

## Performance Targets (Aggressive)

| Metric | Target | Inspired By |
|--------|--------|-------------|
| Layout (1000 elements) | <100μs | Clay |
| Frame build (dirty update) | <500μs | GPUI |
| Full frame render | <2ms | GPUI/WebRender |
| Input latency | <2ms (one frame) | GPUI/Zed |
| Cold start | <200ms | Zed (0.6s, but we're smaller) |
| Memory (simple app) | <5MB | Zed (222MB for full editor) |
| FPS (complex UI) | 120fps sustained | GPUI |

---

## Priority Matrix

### P0 (Critical — Architecture Foundation)
1. **GPU Scene Graph Core** — 3 weeks
2. **Arena Layout Engine** — 2 weeks
3. **GPU Backend** — 3 weeks

**Total: 8 weeks** — After this, we have a working GPU-accelerated UI framework

### P1 (High — Usable Framework)
4. **SDF Text System** — 2 weeks
5. **Widget Library** — 2 weeks
6. **Animation System** — 1 week
7. **Input & Gesture** — 1 week

**Total: 6 weeks** — After this, we can build real applications

### P2 (Medium — Polish & Features)
8. **Theme & Style** — 1 week
9. **Image Formats** — 2 weeks
10. **Developer Tools** — 1 week

**Total: 4 weeks** — After this, production-ready

---

## Key Architectural Decisions

### 1. Scene Graph + Display List (Not Pure Immediate Mode)
- **Why**: GPUI proved that hybrid (immediate API, retained scene) gives best of both worlds
- **How**: Widgets called every frame (immediate), but scene graph is retained and diffed
- **Benefit**: Idle frames are nearly free (nothing changed → no work)

### 2. wgpu for GPU (Not Custom Vulkan/Metal)
- **Why**: Cross-platform with single codebase, WebGPU for free
- **How**: wgpu → Vulkan (Linux/Win), Metal (macOS), DX12 (Win), WebGPU (Web)
- **Trade-off**: Slightly more overhead than raw Metal, but massive portability win

### 3. Arena Layout (Not Retained Tree)
- **Why**: Clay proved arena layout is 10-100x faster than tree-based layout
- **How**: Allocate layout data in frame arena, reset every frame
- **Benefit**: Zero allocation overhead, cache-friendly access patterns

### 4. comptime Widget Specialization
- **Why**: Zig's killer feature — features compile away when not used
- **How**: Each widget type parameterized by feature flags, evaluated at comptime
- **Benefit**: Zero-cost abstraction, smallest possible binary

### 5. Incremental Rendering (Not Full Redraw)
- **Why**: GPUI showed dirty tracking is essential for 120fps
- **How**: Track which scene nodes changed, only rebuild those subtrees
- **Benefit**: Complex UIs stay smooth even with many widgets

---

## Comparison: Our Architecture vs. The Best

| Aspect | GPUI/Zed | WebRender | Vello | Clay | **ZUI/ZAPP** |
|--------|----------|-----------|-------|------|--------------|
| Scene graph | ✅ Retained | ✅ Display list | ❌ Immediate | ❌ N/A | ✅ Retained + diff |
| GPU rendering | ✅ Metal/Vulkan/DX | ✅ OpenGL | ✅ Compute shaders | ❌ N/A | ✅ wgpu (all APIs) |
| Layout | Taffy (Rust) | Stylo (Rust) | ❌ N/A | ✅ Flexbox | ✅ comptime + arena |
| Text | CoreText/DWrite/cosmic | wr_glyph_rasterizer | ❌ N/A | External | ✅ SDF + pure Zig |
| Theming | Tailwind-style | ❌ | ❌ | ❌ | ✅ comptime tokens |
| Incremental | ✅ Dirty views | ✅ Dirty rects | ❌ | ❌ | ✅ Dirty scene nodes |
| comptime | ❌ Rust | ❌ | ❌ | ❌ | ✅ Zig comptime! |
| Arena memory | ❌ Heap | ❌ Heap | ❌ Heap | ✅ Arena | ✅ Arena + comptime |

**Our unique advantage**: comptime + arena = compile-time optimization OF compile-time-allocated data. No other framework can do this.

---

## Implementation Order (Critical Path)

```
Phase 1: Scene Graph + Display List + Software Renderer
         (Milestone 1)
         → Can render UI to framebuffer, test headlessly

Phase 2: Arena Layout Engine
         (Milestone 2)
         → Flexbox/Grid layout in <100μs

Phase 3: GPU Backend (wgpu)
         (Milestone 3)
         → Hardware-accelerated rendering

Phase 4: SDF Text System
         (Milestone 4)
         → GPU text rendering

Phase 5: Widget Library
         (Milestone 5)
         → Full widget set

Phase 6: Animation System
         (Milestone 6)
         → Smooth transitions

Phase 7: Input & Gesture Enhancement
         (Milestone 7)
         → Complete interaction system

Phase 8: Theme + Image Formats + Dev Tools
         (Milestones 8-10)
         → Production-ready
```

---

## Current State vs. Ideal Architecture

### ✅ What We Have Now

#### Rendering Layer
- **Software Renderer (ZGF)**: Framebuffer-based pixel manipulation ✅
- **Rich Rendering (rich_render.zig)**: Gradients, shadows, rounded corners, AA ✅
- **Adaptive Quality (adaptive_render.zig)**: Compile-time + runtime quality adjustment ✅
- **Vector Rendering (vector.zig)**: Lines, circles, bezier curves, optimized fill ✅
- **Vector Text (vector_text.zig)**: Simple bitmap font rendering ✅

#### Layout Layer
- **Basic Layout (layout.zig)**: Simple box layout ✅
- **Flexbox (partial)**: Basic flexbox in zui ✅
- **No Arena Allocator**: Uses heap allocation ❌
- **No Grid Layout**: Missing CSS Grid ❌
- **No comptime Layout**: All layout computed at runtime ❌

#### Widget Layer
- **Basic Widgets**: Button, Checkbox, Slider, ProgressBar, TextField ✅
- **Modern UI Components**: Material Design cards, buttons, switches ✅
- **No Container Widgets**: No Window, Panel, ScrollView ❌
- **No Overlay Widgets**: No Modal, Popup, ContextMenu ❌
- **No Data Widgets**: No ComboBox, ColorPicker, Table ❌
- **No comptime Specialization**: Widgets not parameterized by features ❌

#### Text Layer
- **Basic Text Rendering**: Simple bitmap fonts ✅
- **No SDF Rendering**: No signed distance field text ❌
- **No Text Editing**: No multi-line editor, selection, clipboard ❌
- **No Text Shaping**: No kerning, ligatures, bidirectional ❌

#### Input Layer
- **Basic Input (input.zig)**: Keyboard, mouse events ✅
- **Gesture System (gesture.zig)**: Tap, swipe, pinch, pan ✅
- **No Event Routing**: No capture/bubble phases ❌
- **No Focus Management**: No tab order, programmatic focus ❌
- **No Accessibility**: No screen reader, keyboard navigation ❌

#### Animation Layer
- **No Animation System**: No tweens, transitions, springs ❌

#### Theme Layer
- **Material Colors**: Basic color palette ✅
- **No Theme Engine**: No light/dark toggle, no token system ❌
- **No Style Cascade**: No inheritance, no state-based styling ❌

#### Image Layer
- **Image I/O Framework (image_io.zig)**: Format detection, API stubs ✅
- **No PNG Decoder**: Only header parsing, no full decode ❌
- **No WebP/GIF/SVG**: Format stubs only ❌

#### Architecture Layer
- **No Scene Graph**: Immediate rendering, no retained tree ❌
- **No Display List**: Direct rendering, no batching ❌
- **No Dirty Tracking**: Full redraw every frame ❌
- **No GPU Backend**: Software only ❌

#### Developer Tools
- **Snapshot Testing**: Visual regression tests ✅
- **No Inspector**: No widget tree browser ❌
- **No Profiler**: No frame time breakdown ❌
- **No Hot Reload**: No live code/style reload ❌

---

### 🎯 What We Need (Ideal Architecture)

#### Rendering Layer (GPU Scene Graph)
- **Scene Graph**: Retained tree with parent/child relationships
- **Dirty Tracking**: Only rebuild changed subtrees (GPUI-style)
- **Display List**: Batched render commands, visibility culling
- **GPU Backend**: wgpu (Vulkan/Metal/DX12/WebGPU)
- **Software Fallback**: Keep ZGF for testing/headless

**Gap**: ❌ We have software rendering only, no scene graph, no GPU

#### Layout Layer (Arena-Based, Microsecond)
- **Arena Allocator**: Zero-allocation layout, per-frame reset
- **Full Flexbox**: Complete CSS Flexbox spec
- **Grid Layout**: CSS Grid with areas, auto-fill
- **comptime Layout**: Static layouts computed at compile time
- **Dirty Relayout**: Only recompute changed subtrees

**Gap**: ❌ We have basic flexbox, no arena, no grid, no comptime

#### Widget Layer (comptime-Specialized)
- **Container Widgets**: Window, Panel, ScrollView, TabView
- **Overlay Widgets**: Modal, Popup, ContextMenu, Tooltip
- **Data Widgets**: ComboBox, ColorPicker, Table, TreeView
- **comptime Specialization**: Widgets parameterized by features
- **State Management**: Built-in state for complex widgets

**Gap**: ❌ We have basic widgets, no containers/overlays/data widgets

#### Text Layer (SDF, GPU-Accelerated)
- **SDF Rendering**: Signed distance field text on GPU
- **Full Text Editing**: Multi-line, selection, clipboard, undo
- **Text Shaping**: Kerning, ligatures, bidirectional
- **Font Pipeline**: TrueType parser, glyph atlas, fallback chain

**Gap**: ❌ We have basic bitmap text only

#### Input Layer (Event Routing, Accessibility)
- **Event Routing**: Capture/bubble phases (DOM-like)
- **Focus Management**: Tab order, programmatic focus
- **Keyboard Shortcuts**: Global + local shortcut system
- **Accessibility**: Screen reader, keyboard navigation (WCAG 2.1)

**Gap**: ⚠️ We have basic input + gestures, but no routing/accessibility

#### Animation Layer (Tween Engine)
- **Tween Engine**: Interpolation, easing, property animation
- **Transitions**: Page transitions, enter/exit animations
- **Spring Physics**: Physics-based animations
- **Animation Controller**: Play/pause/reverse, groups, interruption

**Gap**: ❌ We have no animation system

#### Theme Layer (Token-Based, comptime)
- **Design Tokens**: Color, spacing, typography tokens
- **Theme Engine**: Light/dark toggle, custom themes
- **Style Cascade**: Inheritance, per-widget overrides
- **State-Based Styling**: Hover, pressed, disabled, focused styles

**Gap**: ⚠️ We have Material colors, but no theme engine/cascade

#### Image Layer (Full Format Support)
- **PNG Decoder**: Full zlib/deflate decompression
- **WebP Decoder**: Lossy + lossless
- **GIF Decoder**: LZW + animation
- **SVG Renderer**: XML parser, shapes, paths, gradients

**Gap**: ⚠️ We have format detection + stubs, no full decoders

#### Architecture Layer (GPU Pipeline)
- **Scene Graph**: Retained tree with dirty tracking
- **Display List**: Batched commands, visibility culling
- **Incremental Rendering**: Only redraw dirty regions
- **Multi-Pass**: Measure → Layout → Paint → Composite

**Gap**: ❌ We have immediate rendering, no scene graph/display list

#### Developer Tools (Inspector, Profiler)
- **Widget Inspector**: Hover-to-inspect, tree browser
- **Layout Debugger**: Show margins, padding, constraints
- **Performance Profiler**: Frame time breakdown, GPU profiler
- **Hot Reload**: Live code/style/layout reload

**Gap**: ⚠️ We have snapshot testing, but no inspector/profiler

---

### 📊 Summary: Current vs. Ideal

| Layer | Current Status | Completion | Priority |
|-------|---------------|------------|----------|
| **Rendering** | Software only, no scene graph | 30% | P0 |
| **Layout** | Basic flexbox, heap allocation | 20% | P0 |
| **Widgets** | Basic widgets, no containers | 40% | P1 |
| **Text** | Bitmap fonts, no editing | 15% | P1 |
| **Input** | Basic + gestures, no routing | 50% | P1 |
| **Animation** | None | 0% | P1 |
| **Theme** | Material colors, no engine | 30% | P2 |
| **Image** | Stubs only | 10% | P2 |
| **Architecture** | Immediate, no GPU | 25% | P0 |
| **Dev Tools** | Snapshot testing | 20% | P2 |

**Overall Completion**: ~25% of ideal architecture

**Biggest Gaps**:
1. ❌ **No GPU rendering** — Software only, massive performance gap
2. ❌ **No scene graph** — Immediate rendering, no dirty tracking
3. ❌ **No arena layout** — Heap allocation, slow layout
4. ❌ **No SDF text** — Bitmap fonts, no text editing
5. ❌ **No animation** — Static UI, no transitions
6. ❌ **No container widgets** — No Window, Panel, ScrollView

**Smallest Gaps**:
1. ⚠️ **Input system** — Have basics + gestures, need routing/accessibility
2. ⚠️ **Widget library** — Have basics, need containers/overlays/data
3. ⚠️ **Theme system** — Have colors, need engine/cascade
4. ⚠️ **Image formats** — Have detection, need full decoders

**Critical Path**:
1. **GPU Scene Graph** (Milestone 1) — Foundation for everything
2. **Arena Layout** (Milestone 2) — Performance foundation
3. **GPU Backend** (Milestone 3) — Hardware acceleration
4. **SDF Text** (Milestone 4) — Text editing requires this
5. **Widget Library** (Milestone 5) — Real applications need this
6. **Animation** (Milestone 6) — Polish and UX

**What's Ready to Use Now**:
- ✅ Software rendering with rich effects (gradients, shadows, AA)
- ✅ Adaptive quality system (compile-time + runtime)
- ✅ Gesture recognition (tap, swipe, pinch, pan)
- ✅ Basic widgets (button, checkbox, slider, etc.)
- ✅ Material Design components (cards, modern buttons)
- ✅ Snapshot testing infrastructure

**What's Blocking Real Applications**:
- ❌ No GPU rendering (too slow for complex UI)
- ❌ No scene graph (no dirty tracking, full redraw)
- ❌ No container widgets (no Window, ScrollView)
- ❌ No text editing (no multi-line editor)
- ❌ No animation (static, lifeless UI)
- ❌ No overlays (no modal dialogs, popups)

---

## Success Criteria

### Demo Applications (Must Build)
1. **Todo App** — Text input, checkboxes, buttons, layout
2. **Settings Panel** — All widget types, theming, scroll
3. **Code Editor** — Text editing, syntax highlighting, SDF text
4. **Dashboard** — Charts, tables, drag & drop
5. **Drawing App** — Canvas, color picker, gestures

### Performance Benchmarks
- [ ] 120fps with 1000 widgets on modern hardware
- [ ] <100μs layout for 1000 elements
- [ ] <2ms full frame time for complex UI
- [ ] <200ms cold start
- [ ] <10MB memory for simple app

### Quality Gates
- [ ] 90%+ test coverage
- [ ] Visual regression tests (snapshot comparison)
- [ ] No memory leaks (arena-based, but verify)
- [ ] Cross-platform CI (Linux, macOS, Windows, Web)
- [ ] Accessibility audit (WCAG 2.1 AA)

---

## Conclusion

**Our bet**: Zig's comptime + arena allocation + GPU rendering = the fastest native UI framework that exists.

**Key insights from 2026 landscape**:
- GPUI proved GPU scene-graph UI works at 120fps
- WebRender proved display list + batching scales to millions of elements
- Vello proved compute-shader 2D rendering is 10x faster than traditional
- Clay proved arena layout achieves microsecond performance
- Slint proved compile-time optimization eliminates runtime overhead
- Zig's comptime is uniquely suited for all of the above

**We don't need to invent anything new** — we need to combine these proven techniques in Zig, exploiting comptime where no other language can.
