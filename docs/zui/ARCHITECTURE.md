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

## 3. 関数の分類: シグネチャが関数の正体を語る

Zigの最大の強みの一つは **「関数を見れば何をする関数かわかる」** こと。
これは偶然ではなく、Zigの設計思想そのもの。
ZUIでもこの特性を最大限に活かす。

### 3.1 4層の関数分類

```
┌─────────────────────────────────────────────────────────────────────┐
│ 分類        │ シグネチャ特徴              │ 例                      │
├─────────────┼─────────────────────────────┼─────────────────────────┤
│ PURE        │ 値 → 値                    │ lerp(), distance()      │
│ (純粋)      │ エラーなし                  │ normalize(), dot()      │
│             │ ポインタなし                │ clamp(), hash()         │
├─────────────┼─────────────────────────────┼─────────────────────────┤
│ COMPUTE     │ 値 + Allocator → 値 or エラー│ layout(), measure()     │
│ (計算)      │ アロケはあるが副作用なし     │ resolveStyle(), sort()  │
│             │                             │                         │
├─────────────┼─────────────────────────────┼─────────────────────────┤
│ MUTATE      │ *T + 値 → void or エラー    │ update(), setState()    │
│ (状態変更)  │ 可変ポインタが副作用の印     │ advanceAnimation()      │
│             │                             │ pushCommand()           │
├─────────────┼─────────────────────────────┼─────────────────────────┤
│ EFFECT      │ *T + Io → エラー            │ readFile(), connect()   │
│ (外部効果)  │ Io = 非決定性の印            │ writePixel(), sleep()   │
│             │                             │ sendNetwork()           │
└─────────────┴─────────────────────────────┴─────────────────────────┘

見分け方:
• 引数にポインタなし → PURE（入力を変えるだけ）
• Allocatorがある → COMPUTE（メモリ使うが状態は変えない）
• *Tがある → MUTATE（状態を変える）
• std.Ioがある → EFFECT（非決定性/外部I/O）
```

### 3.2 具体例: 同じ操作、異なる分類

```zig
// ============================================================
// PURE: 値 → 値、何も変えない、エラーなし
// ============================================================

/// 2点間の距離（純粋な数学）
pub fn distance(a: Vec2, b: Vec2) f32 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return @sqrt(dx * dx + dy * dy);
}

/// 色の線形補間（純粋な数学）
pub fn lerpColor(a: Color, b: Color, t: f32) Color {
    return .{
        .r = a.r + (b.r - a.r) * t,
        .g = a.g + (b.g - a.g) * t,
        .b = a.b + (b.b - a.b) * t,
        .a = a.a + (b.a - a.a) * t,
    };
}

/// IDハッシュ（純粋な計算）
pub fn hashId(str: []const u8) Id {
    // FNV-1a — 入力が同じなら常に同じ結果
    var h: u64 = 0xcbf29ce484222325;
    for (str) |c| { h ^= c; h *%= 0x100000001b3; }
    return .{ .value = h };
}


// ============================================================
// COMPUTE: Allocator使うが、状態は変えない
// ============================================================

/// レイアウト計算（アリーナ使うが、引数のstateは変えない）
pub fn computeLayout(
    alloc: Allocator,           // ← アロケータあり = メモリ使う
    nodes: []const LayoutNode,  // ← const = 読み取り専用
    constraints: Constraints,   // ← 値 = 変更なし
) Allocator.Error![]Rect {     // ← エラー = アロケ失敗の可能性
    var results = try alloc.alloc(Rect, nodes.len);
    // ... 計算 ...
    return results;
}

/// テキスト計測（アリーナ使うが、状態は変えない）
pub fn measureText(
    alloc: Allocator,
    font: *const Font,          // ← *const = 読み取り専用ポインタ
    text: []const u8,
) Allocator.Error!Vec2 {
    // フォントファイルは読むが、UI状態は変えない
}


// ============================================================
// MUTATE: 状態を変える（可変ポインタが印）
// ============================================================

/// アニメーション更新（状態を変える = *Animation）
pub fn tickAnimation(
    self: *Animation,           // ← 可変ポインタ = 状態変更！
    current_time: f64,
) void {
    self.elapsed = current_time - self.start_time;
    self.progress = @min(self.elapsed / self.duration, 1.0);
}

/// ディスプレイリストにコマンド追加（状態を変える = *DisplayList）
pub fn pushCommand(
    self: *DisplayList,         // ← 可変ポインタ = 状態変更！
    cmd: Command,
) Allocator.Error!void {
    try self.commands.append(self.allocator, cmd);
}

/// スクロール位置更新
pub fn updateScroll(
    self: *ScrollState,         // ← 可変ポインタ
    delta: Vec2,
) void {
    self.offset = Vec2.add(self.offset, delta);
}


// ============================================================
// EFFECT: 非決定性/外部I/O（std.Ioが印）
// ============================================================

/// フォントファイル読み込み（ファイルI/O = std.Io）
pub fn loadFont(
    io: std.Io,                 // ← Io = 外部I/O！
    alloc: Allocator,
    path: []const u8,
) FontError!Font {
    const data = try io.readFileAlloc(alloc, path);
    return parseFont(alloc, data);
}

/// ウィンドウに表示（画面I/O = std.Io）
pub fn presentFrame(
    io: std.Io,                 // ← Io = 外部I/O！
    backend: *Backend,
    buffer: []const u32,
) PresentError!void {
    try io.writeDisplay(buffer);
}
```

### 3.3 comptime で関数の性質を検証する

Zigのcomptimeを使えば、「純粋関数にポインタ引数がないこと」などを
**コンパイル時に検証** できる。これもZigだけの特徴。

```zig
/// 関数が PURE（値のみ入出力）であることをcomptimeで検証
fn assertPure(comptime F: type) void {
    const info = @typeInfo(F).@"fn";
    for (info.params) |param| {
        const T = param.type.?;
        // ポインタ型があったらコンパイルエラー
        if (@typeInfo(T) == .pointer) {
            @compileError("Pure function cannot take pointer parameter: " ++ @typeName(T));
        }
    }
}

/// 関数が MUTATE（可変ポインタあり）であることをcomptimeで検証
fn assertMutate(comptime F: type) void {
    const info = @typeInfo(F).@"fn";
    var has_mutable = false;
    for (info.params) |param| {
        const T = param.type.?;
        if (@typeInfo(T) == .pointer) {
            if (!@typeInfo(T).pointer.is_const) {
                has_mutable = true;
            }
        }
    }
    if (!has_mutable) {
        @compileError("Mutate function must have at least one mutable pointer parameter");
    }
}

// 使用例:
comptime {
    assertPure(@TypeOf(distance));         // ✅ 通る
    assertPure(@TypeOf(lerpColor));        // ✅ 通る
    // assertPure(@TypeOf(tickAnimation)); // ❌ コンパイルエラー！
    assertMutate(@TypeOf(tickAnimation));  // ✅ 通る
}
```

### 3.4 DisplayList = 「副作用の記録」

ZUIの重要な設計: **ウィジェットは直接描画しない**。
描画コマンドをDisplayListに「記録」する。
これにより「何が描かれるか」がデータになり、テスト・変換・最適化が可能になる。

```zig
// ウィジェットは「描きたいもの」を返す（直接は描かない）
pub fn button(self: *Ui, label: []const u8) Error!Response {
    // ... 相互作用計算 ...

    // DisplayList にコマンドを「記録」（MUTATE: 状態を変える）
    try self.display_list.push(.{ .fill_rect = .{
        .rect = rect,
        .color = bg_color,
        .radius = self.theme.radius,
    }});
    try self.display_list.push(.{ .text = .{
        .pos = text_pos,
        .content = label,
        .color = text_color,
    }});

    return .{ .clicked = was_clicked };
}

// 後で一度にレンダリング（EFFECT: 外部I/O）
pub fn renderAll(
    io: std.Io,                     // ← Io = ここで初めて外部I/O
    list: *const DisplayList,       // ← *const = 読み取り専用
    target: *RenderBackend,
) RenderError!void {
    for (list.commands.items) |cmd| {
        switch (cmd) {
            .fill_rect => |r| try target.fillRect(r),
            .text => |t| try target.drawText(t),
        }
    }
    try target.present(io);         // ← ここで画面に出力
}
```

**この設計のメリット:**
1. **テスト容易**: DisplayListはデータなので、アサーションで検証可能
2. **変換可能**: 同じDisplayListをSoftware/GPU/テストバックエンドで描画
3. **最適化可能**: コマンドを並び替え、バッチング、カリング
4. **再現可能**: 同じInput → 同じDisplayList → 同じピクセル（決定的）

### 3.5 型で効果を示すパターン集

```zig
// ============================================================
// 値型 (Value Types) — 不変、コピー可能、PURE操作のみ
// ============================================================
const Vec2 = struct { x: f32, y: f32 };          // 値
const Color = struct { r: f32, g: f32, b: f32, a: f32 };  // 値
const Rect = struct { x: f32, y: f32, w: f32, h: f32 };   // 値
const Input = struct { pointer: Vec2, ... };      // 値（毎フレーム生成）

// ============================================================
// 状態型 (State Types) — 可変ポインタで操作、MUTATE
// ============================================================
const DisplayList = struct { commands: ArrayList(Command) };
const Context = struct { hot: ?Id, active: ?Id, ... };
const Animation = struct { progress: f32, ... };

// ============================================================
// 設定型 (Config Types) — comptimeで特殊化、不変
// ============================================================
const UiConfig = struct {
    Backend: type,     // comptime型
    Widgets: type,     // comptime型
    Quality: Quality,  // comptime値
};

// ============================================================
// 効果型 (Effect Types) — Io/Allocatorが必要
// ============================================================
const Font = struct { data: []const u8 };  // loadFont(io, alloc) → Font
const Image = struct { pixels: []u32 };    // loadImage(io, alloc) → Image

// ============================================================
// Union で効果の種類を明示
// ============================================================
const Command = union(enum) {
    fill_rect: struct { rect: Rect, color: Color, radius: f32 },
    stroke_rect: struct { rect: Rect, color: Color, width: f32 },
    fill_circle: struct { center: Vec2, radius: f32, color: Color },
    text: struct { pos: Vec2, content: []const u8, color: Color },
    image: struct { rect: Rect, image_id: u32 },
    clip: Rect,
    restore_clip: void,
};
// → Commandは「描画効果」の列挙。どれか1つだけ。
```

### 3.6 「シグネチャを読む」練習

```zig
// 問題: 以下の関数はそれぞれ何者？

// (a) 引数: 値のみ、戻り値: 値、エラーなし
fn smoothStep(edge0: f32, edge1: f32, x: f32) f32;
// → PURE: 純粋な数学関数。入力が同じなら常に同じ結果。

// (b) 引数: *const（読み取り専用）、Allocator、エラー: アロケ
fn buildDisplayList(alloc: Allocator, widgets: []const Widget) Allocator.Error!DisplayList;
// → COMPUTE: メモリを使うが、入力状態は変えない。新しい値を生成して返す。

// (c) 引数: *T（可変ポインタ）、値、エラーなし
fn advanceFrame(self: *Ui, input: Input) void;
// → MUTATE: Uiの状態を変える。外部I/Oはしない。

// (d) 引数: std.Io、*T（可変）、Allocator、エラー
fn loadTheme(io: std.Io, alloc: Allocator, path: []const u8) ThemeError!Theme;
// → EFFECT: ファイルI/O（非決定性）あり。結果は実行環境で変わる可能性。

// (e) 引数: comptime T、値のみ
fn renderWidget(comptime T: type, widget: T, rect: Rect) []Command;
// → PURE + comptime: コンパイル時に特殊化。実行時は純粋計算。
//    comptimeがあるので、Tごとに別の関数が生成される。
```

### 3.7 まとめ: ZUIの関数分類ルール

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  引数に何があるかで関数の性質が100%わかる:                   │
│                                                             │
│  なし (値のみ)        = PURE     「計算するだけ」           │
│  + Allocator          = COMPUTE  「メモリ使うが状態不变」    │
│  + *T (可変)          = MUTATE   「状態を変える」           │
│  + std.Io             = EFFECT   「外部と通信する」         │
│  + comptime T: type   = SPECIAL  「コンパイル時に特殊化」   │
│                                                             │
│  戻り値で関数の結果がわかる:                                 │
│                                                             │
│  値                   = 計算結果                            │
│  !T (error union)     = 失敗する可能性                      │
│  void                 = 状態を変えた（戻り値なし）           │
│                                                             │
│  このルールを知っていれば、初めて見る関数でも                │
│  シグネチャだけで「何をする関数か」が即座にわかる。          │
│  これがZigの設計思想であり、ZUIもこれに従う。               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.8 Zig固有のイディオムをZUIに活かす

#### `inline for` — コンパイル時ループでコード生成

```zig
/// comptimeでウィジェット関数を直接インライン展開
/// 各ウィジェットが個別の関数呼び出しとして生成される（dispatch不要）
pub fn registerWidgets(comptime widgets: anytype) type {
    return struct {
        pub fn dispatch(ui: anytype, id: WidgetId) anyerror!Response {
            inline for (widgets) |W| {
                if (W.id == id) {
                    return W.draw(ui); // ← 各ウィジェットが直接インライン化
                }
            }
            return error.UnknownWidget;
        }
    };
}

/// comptimeで品質レベルに応じたdraw関数を生成
pub fn specializedDraw(comptime Q: Quality) type {
    return struct {
        pub fn roundedRect(draw: anytype, rect: Rect, color: Color, radius: f32) void {
            if (Q == .low) {
                draw.fillRect(rect, color); // low = 角丸なし、最速
            } else {
                draw.fillRoundedRect(rect, color, radius);
            }
        }
    };
}
```

#### `errdefer` — エラー時のクリーンアップを明示

```zig
/// フォント読み込み: エラーなら自動でアリーナ解放
pub fn loadFont(alloc: Allocator, data: []const u8) FontError!Font {
    var arena = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();  // ← エラーで抜けたら自動解放
    const arena_alloc = arena.allocator();

    const glyphs = try parseGlyphs(arena_alloc, data);
    const atlas = try buildAtlas(arena_alloc, glyphs);

    return .{
        .arena = arena,
        .glyphs = glyphs,
        .atlas = atlas,
    };
    // 成功 → arenaはFontと一緒に生きる
    // エラー → errdeferがarena.deinit()を呼ぶ
}

/// Ui初期化: 途中で失敗したら全部クリーンアップ
pub fn init(comptime C: type, alloc: Allocator) InitError!Ui(C) {
    var self: Ui(C) = .{
        .allocator = alloc,
        .display_list = .empty,
        .state = .{},
    };
    errdefer self.deinit();

    self.font = try font.load(alloc);
    errdefer self.font.deinit(alloc);

    self.atlas = try atlas.init(alloc, self.font);
    // 全部成功 → 返す
    // どこかで失敗 → errdeferが逆順で全部クリーンアップ
    return self;
}
```

#### `unreachable` — 到達不可能を明示

```zig
/// switch で全ケース処理後の「絶対にここに来ない」
pub fn axisIndex(v: Vec2, comptime axis: Axis) f32 {
    return switch (axis) {
        .x => v.x,
        .y => v.y,
        // Axis は enum で x,y しかないので、これ以外ありえない
        // 将来 Axis に .z が増えたらコンパイルエラーで気づく
    };
}

/// デバッグビルドではパニック、リリースでは最適化ヒント
fn findWidget(self: *Context, id: Id) ?*WidgetState {
    for (self.widgets.items) |*w| {
        if (w.id.eql(id)) return w;
    }
    if (comptime std.debug.runtime_safety) {
        unreachable; // デバッグ: バグ検出
    }
    return null; // リリース: 安全フォールバック
}
```

#### Tagged Union + exhaustiveness — 網羅性でバグを防ぐ

```zig
/// レイアウト結果のunion: 全ケースを明示的に処理
pub const LayoutResult = union(enum) {
    leaf: struct { size: Vec2 },
    row: struct { children: []const LayoutResult, gap: f32 },
    col: struct { children: []const LayoutResult, gap: f32 },
    stack: struct { children: []const LayoutResult },

    /// 全ケース処理。新しいケース追加時にコンパイルエラーで気づく
    pub fn totalSize(self: LayoutResult) Vec2 {
        return switch (self) {
            .leaf => |l| l.size,
            .row => |r| blk: {
                var w: f32 = 0;
                var max_h: f32 = 0;
                for (r.children, 0..) |child, i| {
                    const s = child.totalSize();
                    w += s.x;
                    if (i > 0) w += r.gap;
                    max_h = @max(max_h, s.y);
                }
                break :blk .{ .x = w, .y = max_h };
            },
            .col => |c| blk: {
                // ... 同様
            },
            .stack => |s| blk: {
                // ...
            },
        };
    }
};

/// Command union: 新しいコマンド追加時にレンダラー側のswitchがコンパイルエラー
pub fn renderCommand(backend: anytype, cmd: Command) anyerror!void {
    switch (cmd) {
        .fill_rect => |r| try backend.fillRect(r.rect, r.color, r.radius),
        .text => |t| try backend.drawText(t.pos, t.content, t.color),
        // .new_command を追加したら → ここがコンパイルエラー → 忘れない
    }
}
```

#### `anytype` + duck typing — インターフェース宣言ゼロのポリモーフィズム

```zig
/// 「描けるもの」は型宣言不要。fillRectを持っていればそれでいい。
pub fn drawWidget(drawer: anytype, widget: Widget, rect: Rect) @TypeOf(drawer).Error!void {
    // drawer が fillRect, drawText を持っていればコンパイル通る
    // 持っていなければコンパイルエラー（明確なメッセージ）
    try drawer.fillRect(rect, widget.bg_color, widget.radius);
    try drawer.drawText(rect.center(), widget.label, widget.text_color);
}

// 上記は以下すべてで使える（それぞれ別の関数が生成される）:
// - SoftwareRenderer → 直接ピクセル操作にインライン展開
// - GpuRenderer → GPUコマンド生成にインライン展開
// - TestMock → テスト用記録にインライン展開
// - NullRenderer → 完全no-opに最適化（リリースで全コード除去）
//
// Rust: trait + impl が必要
// C++: template concept が必要
// Zig: anytype だけでOK、コンパイル時に型チェック
```

#### `comptime` 型生成 — 設定から型そのものを作る

```zig
/// 品質レベルから専用のRenderer型を生成
pub fn RendererFor(comptime Q: Quality, comptime Backend: type) type {
    return struct {
        backend: Backend,

        pub fn fillRoundedRect(self: *@This(), rect: Rect, color: Color, radius: f32) !void {
            if (Q == .low) {
                // low品質: 角丸を無視、矩形のみ描画
                return self.backend.fillRect(rect, color);
            }
            // medium/high: 実際の角丸描画
            return self.backend.fillRoundedRect(rect, color, radius);
        }

        pub fn fillShadow(self: *@This(), rect: Rect, color: Color, blur: f32) !void {
            if (Q == .low or Q == .medium) {
                return; // shadow描画なし → コードが完全に消える
            }
            return self.backend.fillShadow(rect, color, blur);
        }
    };
}

// 使用時: コンパイル時に型が確定、ランタイムコストゼロ
const MyRenderer = RendererFor(.high, SoftwareBackend);
// → .low/.medium のコードはバイナリに含まれない
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

---

## 9. 全パターン統合: ボタンウィジェットの完全実装

ここでは、セクション3で説明した全てのZigイディオムを使った
ボタンウィジェットの完全な実装を示す。
**シグネチャを読むだけで関数の性質がわかる** ことを確認してほしい。

```zig
// src/zui/widgets/button.zig
const std = @import("std");

// ============================================================
// 型定義: すべて値型、副作用なし
// ============================================================

pub const Options = struct {
    style: ButtonStyle = .primary,
    disabled: bool = false,
};

pub const ButtonStyle = enum { primary, secondary, ghost, danger };

pub const Response = struct {
    clicked: bool,     // このフレームでクリックされた
    hovered: bool,     // 現在ホバー中
    focused: bool,     // 現在フォーカス中
};

// ============================================================
// メイン関数: シグネチャから全てがわかる
// ============================================================

/// ボタンを描画し、相互作用を返す
///
/// シグネチャの読み方:
///   *Ui(C)          → MUTATE: Uiの状態を変える
///   []const u8      → PURE: ラベルは読み取り専用
///   Options          → PURE: オプションは値（コピー）
///   Error!Response   → 失敗する可能性あり、Responseを返す
pub fn button(
    comptime C: type,              // SPECIAL: コンパイル時に特殊化
    self: *Ui(C),                  // MUTATE: 状態を変える
    label: []const u8,             // PURE: 読み取り専用
    opts: Options,                 // PURE: 値コピー
) Ui(C).Error!Response {
    // --- ID生成 (PURE: comptime文字列 + カウンタ) ---
    const id = self.nextId("button");

    // --- 相互作用判定 (PURE: 入力とIDから計算) ---
    const is_hot = self.isHot(id);
    const is_active = self.isActive(id);
    const is_focused = self.isFocused(id);

    // --- クリック判定 (PURE: 入力イベントから計算) ---
    const just_clicked = is_active and self.input.pointer_released and is_hot;

    // --- レイアウト計算 (COMPUTE: アリーナ使用) ---
    const text_size = try measureText(self.allocator, self.font, label, self.theme.font_size);
    const content_size = text_size.add(.{ .x = self.theme.pad * 2, .y = self.theme.pad * 2 });
    const rect = try self.allocateSpace(content_size);

    // --- 色決定 (PURE: 状態から計算) ---
    const bg = colorForStyle(self.theme, opts.style, .{
        .hovered = is_hot,
        .active = is_active,
        .disabled = opts.disabled,
    });

    // --- 描画コマンド生成 (MUTATE: DisplayListに記録) ---
    // 直接描画せず、コマンドを「記録」する
    try self.display_list.push(self.allocator, .{ .fill_rect = .{
        .rect = rect,
        .color = bg,
        .radius = self.theme.radius,
    }});

    // --- テキスト描画コマンド (MUTATE: DisplayListに記録) ---
    const text_pos = rect.center().sub(text_size.scale(0.5));
    try self.display_list.push(self.allocator, .{ .text = .{
        .pos = text_pos,
        .content = label,
        .font_size = self.theme.font_size,
        .color = textColorForStyle(self.theme, opts.style, opts.disabled),
    }});

    // --- レスポンス (値を返す、副作用なし) ---
    return .{
        .clicked = just_clicked and !opts.disabled,
        .hovered = is_hot and !opts.disabled,
        .focused = is_focused,
    };
}

// ============================================================
// 補助関数: すべてPURE（値 → 値）
// ============================================================

/// スタイルと状態から背景色を計算
/// シグネチャ: 値のみ → 値 → エラーなし = PURE
fn colorForStyle(
    theme: Theme,
    style: ButtonStyle,
    state: struct { hovered: bool, active: bool, disabled: bool },
) Color {
    if (state.disabled) return theme.widget;
    if (state.active) return theme.accent;
    if (state.hovered) return theme.widget_hover;
    return switch (style) {
        .primary => theme.accent,
        .secondary => theme.widget,
        .ghost => .{ .a = 0 },
        .danger => Color.hex(0xff4444, 1),
    };
}

/// スタイルからテキスト色を計算
fn textColorForStyle(theme: Theme, style: ButtonStyle, disabled: bool) Color {
    if (disabled) return theme.text_dim;
    return switch (style) {
        .primary => theme.accent_text,
        .danger => .{ .r = 1, .g = 1, .b = 1, .a = 1 },
        else => theme.text,
    };
}

// ============================================================
// テスト: 各関数の性質が明確
// ============================================================

// PURE関数のテスト: 外部依存ゼロ
test "colorForStyle - primary default" {
    const color = colorForStyle(Theme.dark, .primary, .{ .hovered = false, .active = false, .disabled = false });
    try std.testing.expectEqual(Theme.dark.accent.r, color.r);
}

// PURE関数のテスト: 状態変化を網羅
test "colorForStyle - hover changes color" {
    const normal = colorForStyle(Theme.dark, .primary, .{ .hovered = false, .active = false, .disabled = false });
    const hovered = colorForStyle(Theme.dark, .primary, .{ .hovered = true, .active = false, .disabled = false });
    try std.testing.expect(!normal.approxEql(hovered, 0.01));
}

// MUTATE関数のテスト: TestBackendで状態変化を検証
test "button - full click cycle" {
    var ui = TestUi.init(std.testing.allocator);
    defer ui.deinit();

    // Frame 1: ポインタがボタン上に
    const r1 = try ui.run(.{ .pointer = .{ .x = 50, .y = 13 } });
    try std.testing.expect(r1.hovered);
    try std.testing.expect(!r1.clicked);

    // Frame 2: クリック（press + release）
    const r2 = try ui.run(.{
        .pointer = .{ .x = 50, .y = 13 },
        .pointer_pressed = true,
        .pointer_released = true,
    });
    try std.testing.expect(r2.clicked);
}
```

### この実装から読み取れること

```
1. button() のシグネチャ:
   fn button(comptime C, self: *Ui(C), label: []const u8, opts: Options) Error!Response
   → comptime 特殊化 + 状態変更 + 読み取り専用データ + 値 + エラー可能性 + 結果

2. 補助関数のシグネチャ:
   fn colorForStyle(theme: Theme, style: ButtonStyle, state: State) Color
   → 値のみ + 値のみ + 値 → 値 = 完全に純粋、テスト容易

3. 描画は「記録」:
   try self.display_list.push(allocator, .{ .fill_rect = ... })
   → MUTATE（DisplayListを変える）、EFFECTではない（まだ描いていない）
   → 実際の描画は render() で行う（ここで初めて std.Io が必要）

4. テストの分類:
   - colorForStyle テスト → 純粋、Allocator不要、Io不要
   - button テスト → TestBackend を注入、Io不要（DisplayList検証）
   - render テスト → SoftwareBackend + スナップショット比較
```

---

## 10. まとめ: ZUIのZigらしさチェックリスト

| # | 特徴 | ZUIでの実装 | 他の言語で同等が可能か |
|---|------|-----------|---------------------|
| 1 | シグネチャが関数の性質を語る | 値=PURE, *T=MUTATE, Io=EFFECT | 部分的（Rustの&mut程度） |
| 2 | comptime で型を生成 | Ui(Config), Renderer(Backend) | C++ テンプレートが近い |
| 3 | comptime でコードを生成 | inline for, WidgetSet dispatch | C++ constexpr が近い |
| 4 | comptime で検証 | assertPure(), exhaustive switch | 不可（Zig独有） |
| 5 | 隠れたアロケーションゼロ | 全関数にAllocator明示 | なし |
| 6 | 隠れたI/Oゼロ | 全I/Oにstd.Io明示 | なし（Zig 0.16+） |
| 7 | エラー伝播が構文的 | error union + try + catch | Rust Result が近い |
| 8 | errdefer でクリーンアップ | 全初期化関数で errdefer | C++ RAII が近い |
| 9 | 副作用をデータで記録 | DisplayList (Command union) | 独自（UI分野） |
| 10 | anytype でゼロコスト抽象 | drawWidget(anytype) | C++ template が近い |
| 11 | tagged union で網羅性 | Command, LayoutResult switch | Rust enum が近い |
| 12 | unreachable で不可能を宣言 | exhaustive switch の最後 | 独自（最適化ヒント） |
| 13 | 固定長配列 vs スライス | InlineBuffer [64]Command | C/C++ と同じ |
| 14 | 手続き型UI | if/for/switch がそのままUI | 独自（DSL不要） |

**全14項目中、Zig独有が3、他言語に近いがZigが最优が7、同等が4。**
**特に comptime検証（#4）と Io明示（#6）はZig 0.16+でしか実現できない。**
