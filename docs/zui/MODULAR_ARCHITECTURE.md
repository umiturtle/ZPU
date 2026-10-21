# ZUI Modular Architecture

## Overview

ZUI has been refactored into a collection of loosely-coupled, independently testable modules. This architecture enables:

- **Independent testing**: Each module can be tested in isolation
- **Aggressive optimization**: Changes are localized, reducing risk
- **Clear dependencies**: The dependency graph is explicit and minimal
- **Reusability**: Modules can be used in other projects
- **Compile-time efficiency**: Unused code is eliminated by Zig's comptime

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    High-Level UI (ui.zig)                     │
│  - Widget implementations                                    │
│  - User-facing API                                           │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────┬─────────────┬─────────────┐
             │             │             │             │
        ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
        │ theme   │   │ style   │   │ context │   │  frame  │
        │ (视觉)   │   │ (布局)   │   │ (状态)   │   │ (渲染)   │
        └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │             │
             └─────────────┴──────┬──────┴─────────────┘
                                  │
                          ┌───────▼────────┐
                          │     types      │
                          │ (基础类型)      │
                          └───────┬────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼────┐              ┌─────▼─────┐              ┌────▼────┐
   │  color  │              │    vec2   │              │   id    │
   │ (颜色)   │              │  (向量)    │              │ (标识)  │
   └─────────┘              └─────┬─────┘              └─────────┘
                                  │
                             ┌────▼────┐
                             │  rect   │
                             │ (矩形)  │
                             └─────────┘

独立模块 (无依赖):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  animation  │  │    layout   │  │  event_bus  │
│  (动画)      │  │  (布局算法)  │  │  (事件总线)  │
└─────────────┘  └─────────────┘  └─────────────┘
```

## New Modules

### 1. color.zig (6 tests)
**Purpose**: RGBA color type with hex conversion, interpolation, alpha blending  
**Dependencies**: None  
**Key Features**:
- `Color.hex()` - Create from RGB hex value
- `Color.lerp()` - Linear interpolation
- `Color.withAlpha()` - Adjust transparency
- `Color.approxEql()` - Fuzzy comparison

### 2. vec2.zig (15 tests)
**Purpose**: 2D vector operations  
**Dependencies**: None  
**Key Features**:
- Arithmetic: `add`, `sub`, `scale`
- Geometric: `length`, `normalize`, `dot`
- Comparison: `eql`, `approxEql`
- Utilities: `max`, `min`, `splat`

### 3. rect.zig (14 tests)
**Purpose**: Axis-aligned rectangle operations  
**Dependencies**: vec2.zig  
**Key Features**:
- Construction: `fromPositionSize`, `fromEdges`
- Queries: `contains`, `overlaps`, `intersection`, `union`
- Transformations: `expand`, `shrink`, `translate`
- Properties: `center`, `area`, `size`

### 4. id.zig (7 tests)
**Purpose**: FNV-1a hash-based widget identifiers  
**Dependencies**: None  
**Key Features**:
- `Id.fromString()` - Generate from string
- `Id.combine()` - Hierarchical IDs
- `idFromComptime()` - Compile-time generation
- Zero allocation

### 5. theme.zig (4 tests)
**Purpose**: Visual theme (colors, sizes, spacing)  
**Dependencies**: types.zig  
**Key Features**:
- Built-in themes: `dark`, `light`
- Customization: `with()` method for overrides
- All theme values in one struct
- Comptime-friendly

### 6. style.zig (16 tests)
**Purpose**: Layout and sizing configuration  
**Dependencies**: types.zig  
**Key Features**:
- Size types: `auto`, `fill`, `exact`
- Builder pattern: `row()`, `col()`, `width()`, `padding()`
- Edge definitions: `padX()`, `padY()`, `padding()`
- Alignment: `withAlign()`, `withJustify()`

### 7. layout.zig (14 tests)
**Purpose**: Pure layout calculations  
**Dependencies**: types.zig  
**Key Features**:
- `Constraints` - Available space
- `resolveSize()` - Size resolution
- `alignChild()` - Alignment calculations
- `distributeSpace()` - Space distribution
- `measureText()` - Text measurement (placeholder)

### 8. animation.zig (13 tests)
**Purpose**: Animation math and physics  
**Dependencies**: None  
**Key Features**:
- Easing functions: `linear`, `ease_in`, `ease_out`, `cubic_*`, `elastic`, `bounce`
- `Animation` - Time-based progress
- `Spring` - Physics-based animation
- Utilities: `lerp`, `smoothDamp`, `remap`

### 9. event_bus.zig (6 tests)
**Purpose**: Decoupled event handling  
**Dependencies**: None  
**Key Features**:
- `EventEmitter` - Type-safe event system
- Priority-based handlers
- One-time subscriptions
- `Signal` - Simple callback system

## Existing Modules (Refactored)

### types.zig
**Status**: Still the unified API for basic types  
**Role**: Provides `Vec2`, `Rect`, `Color`, `Id`, etc.  
**Future**: May be refactored to build on top of new standalone modules

### input.zig
**Status**: Unchanged  
**Design**: Already loosely-coupled (input as value, not callback)

## Testing Strategy

Each module follows TDD principles:
1. **Unit tests**: Test individual functions in isolation
2. **No external dependencies**: Tests run without rendering or I/O
3. **Fast execution**: All 95 new tests run in milliseconds
4. **Deterministic**: No randomness or time-dependent behavior

## Usage Example

```zig
const zui = @import("zui");

// Use standalone modules independently
const color = zui.color.Color.hex(0xff8040, 1);
const vec = zui.vec2.Vec2.init(10, 20);
const rect = zui.rect.Rect.fromPositionSize(vec, .{ .x = 100, .y = 50 });

// Use theme system
const theme = zui.theme.Theme.dark.with(.{
    .font_size = 16,
    .accent = zui.color.Color.hex(0x00ff00, 1),
});

// Use animation
var spring = zui.animation.Spring{
    .position = 0,
    .target = 100,
    .stiffness = 200,
    .damping = 20,
};
spring.update(0.016); // 60fps timestep

// Use event bus
var emitter = zui.event_bus.EventEmitter(MyEvent).init(allocator);
defer emitter.deinit();
try emitter.on(myHandler, .normal);
emitter.emit(.{ .value = 42 });
```

## Benefits

### 1. Testability
Each module can be tested without setting up a UI context or rendering pipeline.

### 2. Optimization
Changes to one module don't affect others, enabling aggressive optimization.

### 3. Compile-Time
Zig's comptime eliminates unused modules entirely from the final binary.

### 4. Reusability
Modules like `animation`, `color`, `vec2` can be used in non-UI projects.

### 5. Documentation
Each module is self-contained with clear responsibilities and comprehensive tests.

## Future Work

1. **Extract widgets**: Split `ui.zig` into `widgets/button.zig`, `widgets/label.zig`, etc.
2. **Extract rendering**: Separate display list from widget logic
3. **Parallel testing**: Modules can be tested in parallel
4. **Performance benchmarks**: Benchmark each module independently
5. **Documentation generation**: Auto-generate API docs from module comments

## Migration Guide

The existing API remains unchanged. New modules are additive:

```zig
// Old way (still works)
const color = zui.Color{ .r = 1, .g = 0, .b = 0, .a = 1 };

// New way (recommended for new code)
const color = zui.color.Color.hex(0xff0000, 1);
```

Both approaches are valid. The new modules provide additional functionality and are preferred for new code.
