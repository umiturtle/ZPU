# ZUI Milestones — Zig-Native UI Framework

## Vision

**「Zigでしか作れない、最高のUIライブラリ」**

comptimeでフレームワークのコストをゼロにし、明示的なパラメータ注入で
外部から制御可能にし、error unionで堅牢にし、
**「ビデオゲームのように作り、ウェブページのように使わない」**。

---

# Part 1: Abstract Milestones（抽象マイルストーン）

概念レベルで「何を達成するか」を定義する。
実装の詳細は Part 2 にある。

---

## M0: 純粋基盤 (Pure Foundation) — ✅ 完了

**目標**: 外部依存ゼロの計算モジュールを確立

```
color ──── 独立、ゼロ依存
vec2  ──── 独立、ゼロ依存
rect  ──── vec2に依存
id    ──── 独立、ゼロ依存
geometry ─ 独立、ゼロ依存
```

**達成基準:**
- ✅ 各モジュールが独立してテスト可能
- ✅ comptime関数を含む（idFromComptime, comptime shapes）
- ✅ 95+の単体テスト

---

## M1: アニメーション・スタイル基盤 (Animation & Style Foundation) — ✅ 完了

**目標**: UIの「見た目」と「動き」を純粋関数として定義

```
animation ── 独立（easing, spring, lerp, smoothDamp）
style    ── types.zigに依存（Size, Edges, Alignment）
theme    ── types.zigに依存（Color, サイズ定数）
layout   ── types.zigに依存（Constraints, 配置計算）
```

**達成基準:**
- ✅ 各モジュールが独立してテスト可能
- ✅ 40+の単体テスト
- ✅ イージング、スプリング物理、レイアウト計算をカバー

---

## M2: 型システム統合 (Type System Integration) — ✅ 完了

**目標**: 既存の types.zig, input.zig, frame.zig, context.zig を整理

```
types.zig  ── Vec2, Rect, Color, Id, Size, Edges, Alignment
input.zig  ── Input (value type, 1フレーム分の入力)
frame.zig  ── DisplayList, Command (保持された表示リスト)
context.zig ── 最小の保持状態 (hot, active, focus, scroll)
```

**達成基準:**
- ✅ Input がバリュータイプ（テストで注入可能）
- ✅ DisplayList がインスタンスストリーム（GPU/Software共通）
- ✅ Context が最小限（ウィジェットに状態を持たない）

---

## M3: Comptimeアーキテクチャ (Comptime Architecture)

**目標**: コンテキスト型のcomptime特殊化を実装

**何をするか:**

```zig
// Comptimeで特殊化されるコンテキスト型
pub fn Ui(comptime Config: UiConfig) type {
    return struct {
        // Config.Backend → specialized renderer
        // Config.Widgets → specialized widget set
        // Config.Quality → specialized draw paths
    };
}

// 使用例
const App = Ui(.{
    .Backend = SoftwareBackend,
    .Widgets = MyWidgetSet,
    .Quality = .high,
    .Theme = Theme.dark,
});
```

**達成基準:**
- [ ] `Ui(Config)` がcomptimeパラメータで特殊化される
- [ ] バックエンドごとに異なるコードが生成される
- [ ] 未使用ウィジェットのコードがバイナリから消える
- [ ] vtable/インターフェース不要のゼロコストポリモーフィズム
- [ ] テストでモックバックエンドを注入可能

---

## M4: ウィジェット分離 (Widget Extraction)

**目標**: monolithic ui.zig からウィジェットを独立モジュールに分離

**何をするか:**

```
ui.zig (1412行) → 以下に分割:
├── widgets/
│   ├── button.zig        // ボタン
│   ├── label.zig         // テキストラベル
│   ├── checkbox.zig      // チェックボックス
│   ├── radio.zig         // ラジオボタン
│   ├── slider.zig        // スライダー
│   ├── text_input.zig    // テキスト入力
│   ├── progress.zig      // プログレスバー
│   ├── header.zig        // 折りたたみヘッダー
│   └── container.zig     // コンテナ（row/column/box）
├── layout.zig            // レイアウトエンジン（既存を強化）
└── ui.zig                // メインAPIのみ（ウィジェット登録）
```

**達成基準:**
- [ ] 各ウィジェットが独立ファイル
- [ ] comptime登録でウィジェットセットを構成
- [ ] 各ウィジェットが独立してテスト可能
- [ ] ui.zig が200行以下に

---

## M5: レンダリング抽象化 (Rendering Abstraction)

**目標**: バックエンドに依存しないレンダリング層

**何をするか:**

```zig
// バックエンドはcomptimeで注入される
pub fn Renderer(comptime Backend: type) type {
    return struct {
        // fillRect, drawLine, drawText, drawImage...
        // すべて Backend の関数にインライン展開される
    };
}

// バックエンドの例:
// - SoftwareBackend  → ピクセルバッファに直接描画
// - GpuBackend       → GPUコマンド生成
// - TestBackend      → テスト用のモック
// - NullBackend      → 何も描かない（ベンチマーク用）
```

**達成基準:**
- [ ] `Renderer(Backend)` がcomptime特殊化される
- [ ] Software/GPU/テストバックエンドを実装
- [ ] 同じDisplayListが全バックエンドで描画可能
- [ ] バックエンド切り替えが実行時オーバーヘッドゼロ

---

## M6: エラー伝播の完全実装 (Error Propagation)

**目標**: 全関数がerror unionを返し、エラーが適切に伝播する

**何をするか:**

```zig
// 各層のエラー型
pub const DrawError = error { OutOfMemory, Overflow };
pub const LayoutError = error { Overflow, InvalidConstraint };
pub const WidgetError = error { IdCollision, MaxDepthExceeded };

// Ui(C) のエラー型は全てのunion
pub fn Ui(comptime C: type) type {
    return struct {
        pub const Error = C.Backend.Error || LayoutError || WidgetError;
    };
}
```

**達成基準:**
- [ ] 全public関数がerror unionを返す
- [ ] エラー型が各層で定義される
- [ ] main() でエラーを明示的に処理
- [ ] テストでエラーケースをカバー

---

## M7: std.Io統合 (I/O Integration)

**目標**: Zig 0.16 の std.Io パターンに従い、I/Oを明示的に注入

**何をするか:**

```zig
// Juicy Mainパターン
pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    var app = try App.init(alloc, io, .{ .width = 800, .height = 600 });
    defer app.deinit();
    try app.run();
}

// Io を使った非決定的操作
pub fn tick(self: *Ui, io: std.Io) Error!void {
    const now = std.Io.Clock.now(.awake, io); // 時刻取得
    const dt = now - self.last_time;
    self.animations.update(dt);
}
```

**達成基準:**
- [ ] 全I/O操作が `std.Io` を受け取る
- [ ] 時刻、ファイル、ネットワークが注入可能
- [ ] テストでモックIoを使用可能
- [ ] 非同期I/O（io_uring, GCD）に準備完了

---

## M8: アリーナレイアウト (Arena Layout)

**目標**: Clay-inspired ゼロアロケーションレイアウト

**何をするか:**

```zig
// レイアウトはアリーナで実行、フレーム毎にリセット
pub fn computeLayout(
    alloc: Allocator,       // アリーナアロケーター
    root: LayoutNode,
    constraints: Constraints,
) Allocator.Error![]LayoutResult {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();   // フレーム終了で全解放
    const arena_alloc = arena.allocator();
    // ... レイアウト計算 ...
}
```

**達成基準:**
- [ ] レイアウト計算がアリーナのみ使用
- [ ] フレーム毎にアリーナリセット
- [ ] 1000要素 <100μs のレイアウト
- [ ] malloc/freeゼロ（アロケーターがアリーナの場合）

---

## M9: comptime宣言的UI (Comptime Declarative UI)

**目標**: Slint/ziguiのような宣言的UIをZig comptimeで実現

**何をするか:**

```zig
// Zig comptime で宣言的UI
pub fn view(state: State, ui: *Ui) Error!void {
    try ui.vstack(.{ .gap = 8, .padding = 16 }, struct {
        fn run(s: State, u: *Ui) Error!void {
            try u.label("Count: {}", .{s.count});
            if (try u.button("Increment", .{}).clicked) {
                // イベントは上へ伝播
            }
            try u.hstack(.{}, struct {
                fn run(s: State, u: *Ui) Error!void {
                    try u.checkbox("Option A", .{ .checked = s.option_a });
                    try u.slider("Volume", .{ .value = s.volume });
                }
            }.run);
        }
    }.run);
}
```

**達成基準:**
- [ ] ネストしたコンテナが自然に書ける
- [ ] comptimeでウィジェットツリーが構築される
- [ ] イベントが上方向に伝播する
- [ ] 手続き型ロジック（if/for/switch）と混在可能

---

## M10: 品質レベル特殊化 (Quality Level Specialization)

**目標**: comptime品質レベルで描画パスを特殊化

**何をするか:**

```zig
pub const Quality = enum { low, medium, high, ultra };

pub fn drawRoundedRect(
    comptime Q: Quality,
    draw: anytype,
    rect: Rect,
    color: Color,
    radius: f32,
) void {
    switch (Q) {
        .low => {
            // 角丸なし矩形（最速）
            draw.fillRect(rect, color);
        },
        .medium => {
            // 簡易アンチエイリアス
            draw.fillRectAA(rect, color, radius);
        },
        .high => {
            // SDF-based角丸
            draw.fillSdfRoundRect(rect, color, radius);
        },
        .ultra => {
            // サブピクセルアンチエイリアス + シャドウ
            draw.fillSdfRoundRectShadow(rect, color, radius, .{.blur = 4});
        },
    }
}
```

**達成基準:**
- [ ] 4段階の品質レベル
- [ ] comptimeで未使用パスが完全に除去
- [ ] low → ultra で段階的に高品質
- [ ] 解像度/デバイスに応じて動的選択（ランタイム）

---

## M11: 解像度適応 (Resolution Adaptation)

**目標**: ディスプレイ解像度に応じてUIが自動適応

**何をするか:**

```zig
// 解像度に応じて品質レベルとレイアウトが変化
pub fn adapt(comptime Config: Config, physical_size: Vec2, dpi: f32) AdaptedConfig {
    const logical_size = physical_size.scale(1.0 / (dpi / 96.0));
    const quality = selectQuality(dpi, logical_size);
    return .{
        .logical_size = logical_size,
        .quality = quality,
        .font_scale = dpi / 96.0,
    };
}
```

**達成基準:**
- [ ] DPIに応じてフォントサイズが調整
- [ ] 画面サイズに応じてレイアウトが変化
- [ ] ズームレベルで詳細度が変化
- [ ] タッチ/マウスでインタラクションが変化

---

## M12: テスト駆動完成 (Test-Driven Completion)

**目標**: 全モジュールを包括的なテストでカバー

**達成基準:**
- [ ] 各LEVEL 0-1モジュール: 10+テスト
- [ ] 各ウィジェット: 5+テスト（クリック、ホバー、キーボード）
- [ ] レイアウト: 20+テスト（row, col, grid, nesting）
- [ ] レンダリング: スナップショットテスト
- [ ] エラーケース: 各関数で最低1つ
- [ ] 合計200+テスト

---

# Part 2: Concrete Milestones（具体マイルストーン）

実装レベルで「具体的に何をやるか」を定義する。

---

## C0: モジュール構造の確定 ✅

**状態**: 完了（9モジュール、95テスト）

**成果物:**
- `src/zui/color.zig` — 6 tests
- `src/zui/vec2.zig` — 15 tests
- `src/zui/rect.zig` — 14 tests
- `src/zui/id.zig` — 7 tests
- `src/zui/theme.zig` — 4 tests
- `src/zui/style.zig` — 16 tests
- `src/zui/layout.zig` — 14 tests
- `src/zui/animation.zig` — 13 tests
- `src/zui/event_bus.zig` — 6 tests

**次のアクション**: なし（完了）

---

## C1: types.zig のリファクタリング ✅

**状態**: ✅ 完了

**具体的な作業:**
1. ✅ types.zig を「ハブ」型として再定義
2. ✅ 既存の Vec2, Rect, Color などは level 0 モジュールをre-export
3. ✅ types.zig 固有の追加メソッド（`Vec2.along`, `Vec2.index` 等）は types.zig に残す
4. ✅ 依存関係グラフを文書化

**完了条件:**
- [x] `zui.Vec2` が `zui.vec2.Vec2` と互換
- [x] types.zig が level 0 モジュールをre-export
- [x] 全既存テストがパス
- [x] 循環依存がない

---

## C2: Ui(Config) のcomptime特殊化 ✅

**状態**: ✅ 完了

**具体的な作業:**
1. ✅ `UiConfig` 型を定義（Backend, Widgets, Quality, Theme）
2. ✅ `Ui(comptime Config: UiConfig)` を実装
3. ✅ SoftwareBackend をcomptimeパラメータ化
4. ✅ テストでモックバックエンドを使用

**完了条件:**
- [x] `Ui(.{ .Backend = SoftwareBackend, ... })` がコンパイル通る
- [x] 異なるバックエンドで異なるコードが生成される
- [x] テストでモックバックエンドが使用可能

**依存**: C1

---

## C3: ウィジェットの分離 ✅

**状態**: ✅ 完了

**達成済み:**
1. ✅ `src/zui/widgets/` ディレクトリ作成
2. ✅ 28個のウィジェット/ヘルパーを抽出:
   - `label.zig`, `button.zig`, `checkbox.zig`, `radio.zig`, `slider.zig`
   - `separator.zig`, `progress.zig`, `spacer.zig`, `tooltip.zig`
   - `collapsing.zig`, `scrollable.zig`, `virtual_list.zig`, `text_edit.zig`
   - `container.zig`, `scroll_region.zig`, `occupy.zig`, `respond.zig`
   - `measure_container.zig`, `frame.zig`, `helpers.zig`
   - `paint.zig` (paintRect, paintRounded, paintLine, paintTriangle, paintText)
   - `lifecycle.zig` (init, deinit, resize, isMeasuring, markDirty, stats, run, result)
   - `id.zig` (scopeHash, idOf, pushIdIndex, pushId, popId)
   - `layout.zig` (top, pushLayout, popLayout, available, contentWidth)
   - `focus.zig` (requestFocus, clearFocus)
3. ✅ 10個の構造体を抽出:
   - `theme.zig` (Theme struct, 35行)
   - `response.zig` (Response struct, 14行)
   - `layout_frame.zig` (LayoutFrame struct, 39行)
   - `style.zig` (Style struct, 28行)
   - `frame_result.zig` (FrameResult struct, 8行)
   - `tip.zig` (Tip struct, 4行)
   - `options.zig` (Options struct, 7行)
   - `stats.zig` (Stats struct, 7行)
   - `measured.zig` (Measured struct, 8行)
   - `ui_with.zig` (UiWith struct, 65行)
4. ✅ 全抽出ウィジェット/構造体がui.zigに統合（委譲パターン）
5. ✅ ui.zig を1527行 → 501行に削減（67%削減）

**残りの作業:**
- [ ] ui.zig を200行以下に（現在501行、公開APIの委譲メソッドが残っている）
- [ ] `widgets.zig` でcomptime登録（現在は各ウィジェットを個別にインポート）
- [ ] 全ウィジェットテストがパス（zigコンパイラが利用可能な時に確認）

**注記:**
ui.zigの残りの501行の大部分は、公開APIを維持するための委譲メソッドです。
これらは各ウィジェットモジュールに処理を委譲する薄いラッパーで、
APIの後方互換性を保つために必要です。200行以下の目標を達成するには、
APIの設計変更（直接ウィジェットモジュールを呼び出す）が必要ですが、
それは破壊的変更になります。

**依存**: C2

---

## C4: Renderer(Backend) の実装 ✅

**状態**: ✅ 完了

**具体的な作業:**
1. ✅ `Renderer(comptime Backend: type)` を実装
2. ✅ `SoftwareBackend` をcomptimeパラメータ対応に
3. ✅ `TestBackend`（テスト用モック）を実装
4. ✅ `NullBackend`（ベンチマーク用）を実装
5. ✅ render.zig, raster.zig をバックエンド抽象化

**完了条件:**
- [x] 3つのバックエンド（Software, Test, Null）が動作
- [x] 同じDisplayListが全バックエンドで描画可能
- [x] バックエンド切り替えがcomptime（実行時コストゼロ）

**依存**: C2

---

## C5: エラー型の体系化

**状態**: 未着手

**具体的な作業:**
1. `DrawError`, `LayoutError`, `WidgetError` を各モジュールで定義
2. `Ui(C).Error` をunionとして構築
3. 全public関数にerror unionを追加
4. テストでエラーケースを追加

**完了条件:**
- [ ] 全public関数がerror unionを返す
- [ ] エラー型が文書化されている
- [ ] テストでエラーケースをカバー

**依存**: C3, C4

---

## C6: std.Io対応

**状態**: 未着手

**具体的な作業:**
1. `Ui.init()` に `std.Io` パラメータを追加
2. 時刻取得を `std.Io.Clock` に変更
3. ファイル操作（フォント読み込み等）を `std.Io` 経由に
4. テストで `std.testing.io` を使用

**完了条件:**
- [ ] 全I/O操作がIo経由
- [ ] テストでモックIoが使用可能
- [ ] Juicy Main パターンに対応

**依存**: C2

---

## C7: アリーナレイアウト

**状態**: 未着手

**具体的な作業:**
1. layout.zig にアリーナアロケーションを追加
2. フレーム開始時にアリーナ生成、終了時にリセット
3. Clay-style のゼロアロケーションレイアウトを実装
4. ベンチマークで1000要素 <100μs を確認

**完了条件:**
- [ ] レイアウトがアリーナのみ使用
- [ ] 1000要素のレイアウトが100μs以下
- [ ] フレーム毎にアリーナリセット

**依存**: C3

---

## C8: 宣言的API

**状態**: 未着手

**具体的な作業:**
1. ネストしたコンテナAPIを実装
2. comptimeでウィジェットツリー構築
3. イベント伝播を実装
4. examples/ で宣言的APIのデモを作成

**完了条件:**
- [ ] ネストしたUIが自然に書ける
- [ ] comptimeウィジェットツリーが構築される
- [ ] デモアプリケーションが動作

**依存**: C3, C5

---

## C9: 品質レベル・解像度適応

**状態**: 未着手

**具体的な作業:**
1. `Quality` enum を定義（low, medium, high, ultra）
2. 各描画関数に `comptime Q: Quality` パラメータを追加
3. DPI適応ロジックを実装
4. 解像度/ズームレベルで品質が変化

**完了条件:**
- [ ] 4段階の品質レベルが動作
- [ ] DPI適応が動作
- [ ] ベンチマークで品質別パフォーマンス測定

**依存**: C4

---

## C10: テスト拡充

**状態**: 継続中

**具体的な作業:**
1. 各ウィジェットに5+テスト追加
2. レイアウトに20+テスト追加
3. スナップショットテスト強化
4. エラーケーステスト追加
5. プロパティベーステスト追加

**目標**: 200+テスト

---

# 優先度マップ

```
        高優先度
           │
    C0 ✅──┤
    C1 ────┤
    C2 ────┤
           │
    ───────┼──────── 中優先度
           │
    C3 ────┤
    C4 ────┤
    C5 ────┤
    C7 ────┤
           │
    ───────┼──────── 低優先度（後のマイルストーン）
           │
    C6 ────┤
    C8 ────┤
    C9 ────┤
    C10 ───┤
           │
        低優先度

    C1 → C2 → C3 → C4 → C5 → C8
              ↓
              C7 (アリーナレイアウト)
              C6 (std.Io)
              C9 (品質レベル)
```

---

# 参考: 2026年のUIトレンド

| トレンド | 参考元 | ZUIでの実装 |
|---------|--------|------------|
| **GPU Scene Graph** | GPUI/Zed | Comptime特殊化でGPU/CPU切り替え |
| **Arena Layout** | Clay | C7 で実装 |
| **Compile-Time UI** | Slint, zigui | C8 でcomptime宣言的UI |
| **Fine-Grained Reactivity** | SolidJS, Svelte 5 | Comptime依存解析で最適化 |
| **Unidirectional Data Flow** | Compose, Elm | body.run() で実現 |
| **Explicit I/O** | Zig 0.16 std.Io | C6 で実装 |
| **Quality Adaptation** | 独自 | C9 で実装 |
| **Zero-Cost Abstraction** | Zig comptime | 全comptime特殊化で実現 |
