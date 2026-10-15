# 宣言的UIのAPI設計（フェーズ0.5 - 設計のみ）

## 現行DSLの問題点

### 1. 実行時オーバーヘッド
- `Node` 共用体型によるパターンマッチングが実行時に発生
- `layout()` 関数がツリーを再帰的に走査
- メモリアロケーションが頻繁に発生

### 2. 型安全性の欠如
- `NodeOpts` のフィールドが任意
- ウィジェット固有のプロパティが共用体バリアントに散在
- コンパイル時の検証が不十分

### 3. 開発者体験（DX）
- 冗長な記述（例: `.label("text")` vs `label("text")`）
- スタイルとレイアウトの混在
- イベントハンドラの記述が直感的でない

---

## 参考調査

### React (JSX)
```jsx
function App() {
  return (
    <div className="container">
      <h1>Hello</h1>
      <Button onClick={handleClick}>Click</Button>
    </div>
  );
}
```
- 仮想DOMによる差分更新
- コンポーネントの組み合わせ
- フックによる状態管理

### SwiftUI
```swift
struct ContentView: View {
  @State var count = 0
  
  var body: some View {
    VStack {
      Text("Count: \(count)")
      Button("Increment") { count += 1 }
    }
  }
}
```
- 宣言的構文
- 状態とビューの自動連携
- 型安全なビルダー

### Compose (Kotlin)
```kotlin
@Composable
fun Greeting(name: String) {
  Column {
    Text("Hello $name!")
    Button(onClick = { /* ... */ }) {
      Text("Click me")
    }
  }
}
```
- コンポーザブル関数
- 状態の自動追跡
- リコンポジション

---

## Zig向けAPI設計案

### 設計目標
1. **コンパイル時計算の活用** (`comptime`)
2. **ゼロコスト抽象化**（実行時オーバーヘッド最小）
3. **型安全**（不正なプロパティはコンパイルエラー）
4. **直感的な記述**（冗長性を排除）

### 案1: コマンドパターン（現行DSLの改善）
```zig
const ui = @import("zapp").dsl;

// 現行
var tree = .vstack(.{ .spacing = 8 }, &.{
    .label("Hello"),
    .button("Click"),
});

// 改善案: ビルダーパターン
var layout = ui.layout(.vertical, .{ .spacing = 8 });
layout.child(.label("Hello"));
layout.child(.button("Click"));
```

### 案2: comptimeマクロ
```zig
// comptimeでUIツリーを構築
const ui = comptime ui!{
    .vertical(spacing: 8) {
        .label("Hello"),
        .button("Click"),
    }
};
```

### 案3: 関数型ビルダー
```zig
// 高階関数でUIを構築
const view = ui.vstack(.{ .spacing = 8 }, .{
    ui.label("Hello"),
    ui.button("Click", .{ .onClick = handleClick }),
});
```

### 推奨: 案3の改良版
```zig
// コンポーネント定義
const Button = struct {
    text: []const u8,
    onClick: ?*const fn () void = null,
    style: ButtonStyle = .default,
};

// レイアウト関数
pub fn vstack(comptime options: anytype, children: anytype) Node {
    // comptimeで型チェック
    // 実行時は最小コスト
}

// 使用例
const view = ui.vstack(.{ .spacing = 8 }, .{
    .{ .label = .{ .text = "Hello" } },
    .{ .button = .{ .text = "Click", .onClick = handleClick } },
});
```

---

## コンポーネントカタログ設計

### ウィジェット一覧
| ウィジェット | プロパティ | 状態 |
|-------------|-----------|------|
| `Label` | text, style | - |
| `Button` | text, onClick, style | hover, active, disabled |
| `TextField` | value, placeholder, onChange | focused |
| `Checkbox` | checked, onChange | hover |
| `Slider` | value, min, max, onChange | dragging |
| `ProgressBar` | value, min, max | - |
| `Image` | source, size | - |

### コンテナ
| コンテナ | 特徴 |
|---------|------|
| `VStack` | 垂直配置 |
| `HStack` | 水平配置 |
| `ZStack` | 重ね合わせ |
| `Grid` | グリッドレイアウト |

---

## 実装方針

### 1. 型安全なプロパティ
```zig
pub const LabelProps = struct {
    text: []const u8,
    alignment: Alignment = .start,
    style: TextStyle = .default,
};

pub fn label(props: LabelProps) Node {
    // コンパイル時に型チェック
}
```

### 2. 状態管理
```zig
// シンプルな状態バケツ
pub const State = struct {
    value: f32 = 0,
    focused: bool = false,
    hovered: bool = false,
};

// 状態変更はイベント経由
pub fn update(state: *State, event: Event) void {
    switch (event) {
        .focus => state.focused = true,
        .blur => state.focused = false,
        else => {},
    }
}
```

### 3. イベントシステム
```zig
pub const Event = union(enum) {
    click,
    hover: bool,
    focus: bool,
    key_press: Key,
    value_change: f32,
};

// コールバック型
pub const EventHandler = *const fn (event: Event) void;
```

---

## 次ステップ

1. **プロトタイプ実装**
   - 最小限のウィジェット（Label, Button）でAPIを試す
   - comptime活用方法を検証

2. **パフォーマンス計測**
   - 現行DSLとの比較
   - メモリアロケーション回数
   - レンダリング時間

3. **開発者フィードバック**
   - 実際の使用感を確認
   - APIの改善点を洗い出す

---

## 参考資料

- [SwiftUI Documentation](https://developer.apple.com/xcode/swiftui/)
- [React Documentation](https://react.dev/)
- [Jetpack Compose](https://developer.android.com/jetpack/compose)
- [Zig Language Reference](https://ziglang.org/documentation/master/)
