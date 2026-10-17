# ZUI アーキテクチャ — ZigらしいUIライブラリの設計

## 1. 設計哲学: なぜZigでUIなのか

### Zigだけが持てる武器

| 特徴 | Zig | Rust | C++ | Web (JS/TS) |
|------|-----|------|-----|-------------|
| **comptime** | 言語そのものをコンパイル時に実行 | 型システム + マクロ | テンプレート + constexpr | バンドル時のビルド |
| **隠れた制御フロー** | ゼロ | async/await, Drop | RAII, 例外, 暗黙的変換 | Promise, GC |
| **隠れたアロケーション** | ゼロ（明示的Allocator） | 所有権（隠れたデアロケあり） | RAII（隠れたアロケあり） | GC |
| **エラー伝播** | error union（構文的に強制） | Result + ? | 例外（スタック巻き戻し） | try/catch |
| **I/O分離** | std.Io（明示的注入） | 非同期ランタイム | OS依存 | ブラウザAPI |
| **ポリモーフィズム** | anytype（ゼロコスト） | trait（monomorphization） | virtual/template | dynamic dispatch |

**結論**: Zigのcomptimeは「マクロでもテンプレートでもない、第三の道」。
**普通のZigコード** がコンパイル時に走り、型チェックされ、インライン展開される。
これにより「フレームワークはゼロコスト」が本当に実現できる。

---

## 2. 核心的設計原則

### 原則1: 引数は落ちる、エラーは上がる

```
                    ┌──────────────────┐
                    │  main() / App    │
                    │  allocator, io,  │
                    │  config, theme   │
                    └────────┬─────────┘
                             │ ↓ 引数（Allocator, Io, Config）
                    ┌────────▼─────────┐
                    │     Ui(C)        │ ← comptimeコンテキスト
                    │  init(alloc,io)  │
                    └────────┬─────────┘
                             │ ↓
                    ┌────────▼─────────┐
                    │   body.run()     │ ← ウィジェット呼び出し
                    │  button, label   │
                    └────────┬─────────┘
                             │ ↓
                    ┌────────▼─────────┐
                    │  draw, layout    │ ← 低レベル操作
                    └────────┬─────────┘
                             │ ↓
                    ┌────────▼─────────┐
                    │  pixel, buffer   │ ← 最下層
                    └──────────────────┘

          全関数が error union でエラーを返す
          呼び出し元が「どう扱うか」を決める（唯一の正しい設計）
```

**具体例:**

```zig
// ❌ 悪い例: グローバル状態、エラーを握りつぶす
fn drawButton(label: []const u8) void {
    const color = global_theme.accent;  // グローバル！
    global_image.fillRect(...);          // 副作用が隠れている
}

// ✅ 良い例: すべてが明示的
fn drawButton(
    comptime C: type,           // コンテキスト（comptimeで特殊化）
    ctx: *C,                    // 状態（明示的ポインタ）
    label: []const u8,          // データ
) C.Error!Response {            // エラーは型で宣言
    const theme = ctx.config.theme;
    try ctx.draw.fillRect(...); // エラー伝播
}
```

### 原則2: Comptimeで解決できることはComptimeで解決

```zig
// レンダラーはcomptimeで特殊化される → vtable不要、完全インライン
pub fn Frame(comptime Backend: type) type {
    return struct {
        backend: Backend,
        commands: std.ArrayListUnmanaged(Command),

        pub fn draw(self: *@This(), alloc: Allocator, list: DisplayList) Backend.Error!void {
            for (list.items) |cmd| {
                switch (cmd) {
                    .rect => |r| try self.backend.fillRect(alloc, r),
                    .text => |t| try self.backend.drawText(alloc, t),
                    // comptimeで未知のコマンドはコンパイルエラー → 安全性
                }
            }
        }
    };
}

// ウィジェットセットもcomptimeで登録 → 存在しないウィジェットのコードは生成されない
pub fn Ui(comptime Widgets: type) type {
    return struct {
        // Widgets の各関数が直接インライン展開される
        pub fn button(self: *@This(), label: []const u8, opts: Options) Error!Response {
            return Widgets.button(self, label, opts); // インライン化
        }
    };
}
```

### 原則3: データは一方向、イベントは上方向（UDF）

```
┌─────────────────────────────────────────────────────────┐
│                    アプリケーション状態                    │
│            (ユーザーが所有、ZUIは知らない)                  │
└─────────────────────┬───────────────────────────────────┘
                      │ state (不変)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              body.run(state, input)                      │
│         (純粋関数: state + input → display_list)          │
│                                                          │
│  if (try ui.button("Save", .{}).clicked())               │
│      return .{ .save = {} };  ← イベントを上へ返す        │
└─────────────────────┬───────────────────────────────────┘
                      │ display_list
                      ▼
┌─────────────────────────────────────────────────────────┐
│              layout(display_list, constraints)           │
│              (純粋関数: sizes → positions)                │
└─────────────────────┬───────────────────────────────────┘
                      │ positioned_list
                      ▼
┌─────────────────────────────────────────────────────────┐
│              render(positioned_list, target)             │
│              (ピクセルへの変換)                             │
└─────────────────────────────────────────────────────────┘
```

**重要**: `body` は**決定論的**でなければならない。
同じ state + input → 同じ display_list。
これはZigの `if`, `for`, `switch` がそのまま使える理由。

---

## 3. 2段階構成: Comptime層 vs Runtime層

### Phase 1: Comptime層（コンパイル時に解決）

| 何を | どうやって | メリット |
|------|-----------|---------|
| **ウィジェットセット** | `comptime Widgets: type` パラメータ | 使わないウィジェットのコードがバイナリに入らない |
| **バックエンド** | `comptime Backend: type` パラメータ | vtable不要、完全インライン、分岐ゼロ |
| **品質レベル** | `comptime quality: Quality` | 低品質パスのコードが完全に消える |
| **レイアウトアルゴリズム** | `comptime Algorithm` | flexboxを使わないならflexboxのコードが消える |
| **テーマ定数** | `comptime` theme values | 色やサイズの定数が即値に畳み込まれる |
| **ID生成** | `idFromComptime("name")` | FNV-1aハッシュがバイナリに直接埋め込み |
| **シェイプ生成** | `comptime regularPolygon(6)` | 多角形の頂点配列が定数配列に |
| **エラー型** | `Backend.Error` union | 各バックエンド固有のエラーのみを含む |

**Zigでしか実現できない理由:**
- Rust: trait resolutionはmonomorphizationだが、コンパイル時に「完全に畳み込む」制御が弱い
- C++: テンプレートは強力だが、エラーメッセージとデバッグが地獄
- Web: バンドル時のtree-shakingは実行時のコードまでは消せない
- **Zig: `comptime` は普通のコードをコンパイル時に実行 → 完全制御**

### Phase 2: Runtime層（実行時に計算）

| 何を | どうやって | 注意点 |
|------|-----------|--------|
| **ユーザー状態** | アプリケーションが所有 | ZUIは状態を持たない（ウィジェットレベルの最小状態を除く） |
| **入力処理** | 毎フレーム `Input` 値として受け取る | テストでは任意の入力を注入可能 |
| **body実行** | 毎フレーム `body.run()` を呼ぶ | 決定論的であること |
| **レイアウト計算** | 制約 + サイズ → 位置 | 純粋関数、キャッシュ可能 |
| **アニメーション** | 時間 + パラメータ → 値 | 純粋関数、テスト可能 |
| **レンダリング** | ディスプレイリスト → ピクセル | バックエンド固有 |
| **アロケーション** | すべて `Allocator` で制御 | フレーム毎にarenaリセット |

### 構成図: 何がどこで解決されるか

```
┌──────────────────── COMPTIME（コンパイル時に確定） ────────────────────┐
│                                                                        │
│  const App = Ui(                                                       │
│      SoftwareBackend,      // バックエンド → specialized renderer      │
│      MyWidgets,            // ウィジェット → specialized widgets       │
│      .low,                 // 品質 → specialized draw paths            │
│      .{                    // Config → comptime定数                     │
│          .theme = Theme.dark,                                           │
│          .font_size = 14,                                               │
│      },                                                                │
│  );                                                                    │
│                                                                        │
│  // 結果: 完全に特殊化された型、ゼロコスト抽象                           │
│  // vtable不要、分岐不要、未使用コード不要                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

┌──────────────────── RUNTIME（実行時に計算） ──────────────────────────┐
│                                                                        │
│  pub fn main() !void {                                                 │
│      var gpa = std.heap.GeneralPurposeAllocator(.{}){};                 │
│      defer _ = gpa.deinit();                                           │
│      const alloc = gpa.allocator();                                    │
│                                                                        │
│      var app = try App.init(alloc, .{.width = 800, .height = 600});    │
│      defer app.deinit();                                               │
│                                                                        │
│      while (true) {                                                    │
│          const input = readInput();          // ← 入力（変化する）       │
│          const result = try app.run(input);  // ← body実行（変化する）   │
│          try app.present();                  // ← 表示（変化する）       │
│      }                                                                 │
│  }                                                                     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. データフロー設計

### 4.1 全体のフロー（1フレーム）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        1 FRAME                                     │
│                                                                     │
│  ① INPUT                                                            │
│  ┌───────────┐                                                      │
│  │ Input     │ ← OS/テスト/スクリプトから注入                        │
│  │ (value)   │   (ポインタ位置、キー、スクロール、時刻)              │
│  └─────┬─────┘                                                      │
│        │                                                            │
│  ② BEGIN FRAME                                                      │
│  ┌─────▼─────────────────────────────────────────────┐              │
│  │ Ui.beginFrame(input)                              │              │
│  │   • アリーナリセット（前フレームの文字列を解放）     │              │
│  │   • display_listクリア                             │              │
│  │   • ホット/アクティブ状態更新                       │              │
│  │   • アニメーション更新                              │              │
│  └─────┬─────────────────────────────────────────────┘              │
│        │                                                            │
│  ③ BODY EVALUATION（決定論的）                                       │
│  ┌─────▼─────────────────────────────────────────────┐              │
│  │ body.run(&ui)                                     │              │
│  │   • ウィジェット呼び出し                           │              │
│  │   • 各ウィジェット:                                │              │
│  │     1. ID生成（comptime文字列 + カウンタ）         │              │
│  │     2. 相互作用判定（ホット？アクティブ？）         │              │
│  │     3. ディスプレイリストにコマンド追加             │              │
│  │     4. Response返す                               │              │
│  └─────┬─────────────────────────────────────────────┘              │
│        │                                                            │
│  ④ LAYOUT（純粋関数）                                                │
│  ┌─────▼─────────────────────────────────────────────┐              │
│  │ layout.compute(display_list, constraints)         │              │
│  │   • measure pass: 自然サイズ計算                   │              │
│  │   • arrange pass: 位置確定                         │              │
│  │   • アリーナ使用（フレーム終了で解放）              │              │
│  └─────┬─────────────────────────────────────────────┘              │
│        │                                                            │
│  ⑤ RENDER（特殊化済み）                                              │
│  ┌─────▼─────────────────────────────────────────────┐              │
│  │ backend.render(positioned_list, target)           │              │
│  │   • コマンドをピクセルに変換                       │              │
│  │   • comptime特殊化で分岐ゼロ                      │              │
│  └─────┬─────────────────────────────────────────────┘              │
│        │                                                            │
│  ⑥ RESULT                                                           │
│  ┌─────▼─────┐                                                      │
│  │ Frame     │ ← 呼び出し元に返す                                    │
│  │ Result    │   (イベント、次のフレームが必要か)                     │
│  └───────────┘                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 ウィジェット内部のフロー

```zig
// ウィジェットは「2度呼ばれる」: measure pass と draw pass
// これは container が子のサイズを知らないとレイアウトできないため

fn button(self: *Ui, label: []const u8, opts: Options) Error!Response {
    const id = self.nextId(label);
    const is_hot = self.isHot(id);
    const is_active = self.isActive(id);

    switch (self.pass) {
        .measure => {
            // テキストサイズを計算 → 自然サイズを返す
            // 描画しない、状態を変更しない
            const text_size = self.measureText(label);
            self.reserveSpace(text_size.add(padding));
        },
        .draw => {
            // 実際のサイズがわかっているので描画できる
            const rect = self.allocateSpace();
            const color = if (is_active) self.theme.accent
                         else if (is_hot) self.theme.widget_hover
                         else self.theme.widget;
            try self.draw.fillRoundedRect(rect, color, self.theme.radius);
            try self.draw.text(rect.center(), label, text_color);
        },
    }

    return .{
        .clicked = is_active and self.input.pointer_released,
        .hovered = is_hot,
    };
}
```

### 4.3 コンテキスト型のcomptime特殊化

```zig
// 各モジュールがcomptimeパラメータで特殊化される例

// レンダラー: バックエンドで特殊化
pub fn Renderer(comptime Backend: type) type {
    return struct {
        backend: Backend,

        pub fn fillRect(self: *@This(), alloc: Allocator, rect: Rect, color: Color) Backend.Error!void {
            // Backend が SoftwareRenderer なら → 直接ピクセル操作にインライン展開
            // Backend が GpuRenderer なら → GPU コマンド生成にインライン展開
            return self.backend.fillRect(alloc, rect, color);
        }
    };
}

// UIコンテキスト: ウィジェットセット + バックエンドで特殊化
pub fn Ui(comptime Config: UiConfig) type {
    return struct {
        allocator: Allocator,
        frame: Frame(Config.Backend),
        state: State,
        config: Config,

        // ウィジェット関数は comptime Config.Widgets から生成
        pub fn button(self: *@This(), label: []const u8, opts: Options) Error!Response {
            return Config.Widgets.button(self, label, opts);
        }
    };
}
```

---

## 5. 他フレームワークとの比較

### 5.1 2026年の主要UIフレームワークの設計

| フレームワーク | データフロー | 状態管理 | レイアウト | 特殊機構 |
|---------------|-------------|---------|-----------|---------|
| **GPUI (Zed)** | Entity→View→Element、毎フレーム再構築 | Entity\<T\> (Rc的スマートポインタ) | Taffy (flexbox/grid) | Rust所有権で状態管理 |
| **Iced** | Elm (Model→Update→View) | メッセージベース、不変状態 | Taffy | 型安全メッセージ |
| **egui** | Immediate mode、毎フレーム全UI | IDベースの_retained state_ | カスタム | シンプルさ重視 |
| **Slint** | 宣言的DSL、コンパイル時最適化 | プロパティバインディング | CSS subset | コンパイル時評価 |
| **Compose** | UDF (State↓ Event↑) | StateFlow/ViewModel | ConstraintLayout | recomposition |
| **SwiftUI** | 宣言的struct、@Observable | Property wrappers | Stack/Grid | diff-based update |
| **Flutter** | Widget tree、setState | StatefulWidget | Flex/Stack | Element tree |
| **React (2026)** | Signals/Compiler | useState + compiler | CSS | 自動メモ化 |
| **Fern (Zig)** | Elm (Model→Update→View) | 厳密なDAG依存 | Flexbox | comptime verify |
| **Gooey (Zig)** | Hybrid immediate/retained | Cx-based context | Flexbox | GPU直接 |
| **zigui (Zig)** | comptime declarative | State\<T\> + Binding | Measure/Arrange | SwiftUI風 |

### 5.2 ZUIの立ち位置

```
                  Retained Mode
                      ↑
                      │
        GPUI ─────────┼──────── Iced
                      │
   Hybrid ────────────┼─────────── Elm/MVU
                      │
        ZUI ──────────┼──────── Fern
     (comptime hybrid)│
                      │
   Immediate ─────────┼──────── egui
                      │
                      ↓
                  Immediate Mode

  ZUIは「comptime hybrid」:
  - APIはImmediate Mode（書きやすさ）
  - 内部はRetained（パフォーマンス）
  - Comptimeで「フレームワークのコスト」をゼロに
```

**ZUI独自の価値（Zigでしか作れない理由）:**

1. **Comptime Context Specialization**: GPUIやIcedはruntime dispatchが必要。ZUIはcomptimeで完全特殊化。
2. **Zero-Cost Widget Set**: 使わないウィジェットはバイナリに存在しない（tree-shaking以上の完全排除）
3. **Explicit Everything**: Allocator, Io, Error — すべてが明示的で、テストと推論が容易
4. **Procedural UI**: `if`, `for`, `switch` がそのままUIロジックになる。DSL不要。
5. **Arena Layout**: Clayと同じアプローチ — レイアウトはアリーナで、フレーム毎にリセット

---

## 6. モジュール構成（最終形）

```
┌────────────────────────────────────────────────────────────────────┐
│                      LEVEL 3: COMPOSITION                         │
│                                                                     │
│   ui.zig         Ui(C) を構築、body.run() を実行                   │
│   widgets/       button.zig, label.zig, checkbox.zig, ...          │
│                  各ウィジェット = 純粋な関数 + comptime登録          │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                      LEVEL 2: ORCHESTRATION                        │
│                                                                     │
│   frame.zig      DisplayList、Command生成                         │
│   context.zig    状態管理（hot, active, focus, scroll, animation）  │
│   layout.zig     レイアウト計算（constraints → positions）          │
│   render.zig     Renderer(Backend) で特殊化                        │
│   pipeline.zig   全体のフレームパイプライン                        │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                      LEVEL 1: FOUNDATION                           │
│                                                                     │
│   types.zig      Vec2, Rect, Color, Id, Size, Edges, Alignment    │
│   input.zig      Input (value type, 1フレーム分の入力)             │
│   font.zig       フォントメトリクス                                │
│   text.zig       テキストレイアウト                                │
│   atlas.zig      グリフアトラス                                    │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                      LEVEL 0: PURE (独立、ゼロ依存)                 │
│                                                                     │
│   color.zig      RGBA、hex、lerp、alpha                            │
│   vec2.zig       2Dベクトル演算                                    │
│   rect.zig       矩形操作、contains、overlaps、intersection         │
│   id.zig         FNV-1aハッシュ、comptime生成                      │
│   theme.zig      ビジュアルテーマ（色、サイズ、スペース）           │
│   style.zig      レイアウト設定（Size、Edges、方向）                │
│   animation.zig  イージング、スプリング、lerp、smoothDamp          │
│   geometry.zig   パス、ベジェ、comptime図形生成                    │
│   primitives.zig ピクセル操作、線、矩形、円                        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. エラー伝播設計

```zig
// 各層のエラー型
pub const BackendError = error{
    OutOfMemory,
    InvalidTexture,
    ShaderCompileFailed,
};

pub const LayoutError = error{
    Overflow,        // コンテナが子を収められない
    InvalidConstraint,
};

pub const WidgetError = error{
    IdCollision,     // 同一フレームで同じIDが重複
    MaxDepthExceeded,
};

// Ui(C) のエラー型はこれらのunion
pub fn Ui(comptime C: type) type {
    return struct {
        pub const Error = C.Backend.Error || LayoutError || WidgetError;

        pub fn button(self: *@This(), ...) Error!Response {
            // widget error → caller
            try self.reserveId(id);
            // layout error → caller
            try self.layout.measure(...);
            // backend error → caller
            try self.draw.fillRect(...);
        }
    };
}

// main() で最終的にエラーを処理
pub fn main() !void {
    var app = try App.init(...);
    while (true) {
        const result = app.run(input) catch |err| {
            switch (err) {
                .OutOfMemory => return error.OutOfMemory,
                .IdCollision => @panic("bug: duplicate widget ID"),
                else => log.err("frame error: {}", .{err}),
            }
        };
    }
}
```

---

## 8. テスト戦略

```zig
// LEVEL 0 モジュール: 完全に独立、外部依存ゼロ
test "vec2 - normalize" {
    const v = Vec2.init(3, 4).normalize();
    try expectApproxEqAbs(1.0, v.length(), 0.0001);
}

// レイアウト: 純粋関数テスト
test "layout - row with gap" {
    const result = layoutRow(
        .{ .children = &.{ .{.w = 100}, .{.w = 100} }, .gap = 10 },
        .{ .width = 500 },
    );
    try expectEqual(@as(f32, 0), result[0].x);
    try expectEqual(@as(f32, 110), result[1].x);
}

// ウィジェット: Input注入で対話テスト
test "button - click" {
    var ui = TestUi.init(testing.allocator);
    defer ui.deinit();

    // Frame 1: hover
    const r1 = try ui.run(.{ .pointer = .{.x = 50, .y = 13} });
    try expect(r1.hovered);
    try expect(!r1.clicked);

    // Frame 2: press + release
    const r2 = try ui.run(.{ .pointer = .{.x = 50, .y = 13}, .pointer_pressed = true, .pointer_released = true });
    try expect(r2.clicked);
}

// レンダリング: スナップショットテスト
test "render - produces correct pixels" {
    var buffer = [_]u32{0} ** (100 * 100);
    var backend = SoftwareBackend.init(&buffer, 100, 100);
    var renderer = Renderer(SoftwareBackend){ .backend = backend };

    try renderer.fillRect(.{}, .{ .x = 10, .y = 10, .w = 50, .h = 30 }, .white);
    try expectEqual(@as(u32, 0xFFFFFFFF), buffer[10 * 100 + 10]);
}
```
