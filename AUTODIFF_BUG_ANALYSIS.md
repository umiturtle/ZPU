# 自動微分のバグ分析とテストケース追加

## 発見された潜在的な問題

### 1. メモリリーク問題

**場所**: `src/zpu/autodiff.zig` の `contribution` 関数

**問題**: 一時的なテンソルの解放漏れ

多くのVJPルールで、`ops.*`関数を呼び出して一時的なテンソルを作成しているが、それらが `release` または `defer release` で解放されていない。

**例**:
```zig
.sqrt => blk: {
    const two = try scalarLike(a, 2);
    const d = try ops.mul(y, two);  // `two` と `d` が解放されていない
    break :blk try ops.div(g, d);
},
```

**修正が必要**: 以下のように `defer release` を追加すべき：
```zig
.sqrt => blk: {
    const two = try scalarLike(a, 2);
    defer release(two);
    const d = try ops.mul(y, two);
    defer release(d);
    break :blk try ops.div(g, d);
},
```

### 2. ブロードキャストの勾配処理

**場所**: `reduceToShape` と `unbroadcast` 関数

**問題**: 複雑なブロードキャストパターンで勾配が正しく伝播しない可能性

**テストケース**:
```zig
// 複雑なブロードキャスト: [3, 1, 5] + [1, 4, 5]
const a = try Tensor.full(try Shape.init(&.{3, 1, 5}), .f32, 2.0);
const b = try Tensor.full(try Shape.init(&.{1, 4, 5}), .f32, 3.0);
const c = try zpu.add(a, b);  // 結果は [3, 4, 5]

// 勾配は正しく [3, 1, 5] と [1, 4, 5] に縮約されるべき
```

### 3. 高階微分の問題

**場所**: `jvp` と `hvp` の実装

**問題**: "reverse over reverse" 実装で、メモリ管理と参照カウントに問題がある可能性

**症状**: Issue #8 で報告されているヘシアンのメモリリーク

### 4. 離散操作の勾配

**場所**: `contribution` 関数の `.argmax`, `.argmin` などのケース

**問題**: 離散操作は勾配が0または未定義だが、テストが不十分

## 追加すべきテストケース

### 基本操作の勾配テスト

```zig
// 1. 二項演算の勾配
test "gradient: binary operations" {
    // add, sub, mul, div の両方の入力に対する勾配
}

// 2. 縮約操作の勾配
test "gradient: reductions" {
    // sum, mean, max, min の axis 指定と AXIS_NONE
}

// 3. 線形代数の勾配
test "gradient: linear algebra" {
    // matmul, solve, inv, det の勾配
}

// 4. ブロードキャストの勾配
test "gradient: broadcasting patterns" {
    // 様々なブロードキャストパターン
}
```

### 高階微分のテスト

```zig
test "second derivative: basic" {
    // f(x) = x^3 の2階微分
    // f'(x) = 3x^2, f''(x) = 6x
}

test "hessian: quadratic form" {
    // f(x) = x^T A x のヘシアンは 2A
}

test "jacobian: linear function" {
    // f(x) = Ax のヤコビアンは A
}
```

### エッジケースのテスト

```zig
test "gradient: empty tensors" {
    // 空テンソルの勾配
}

test "gradient: scalar tensors" {
    // スカラーテンソルの勾配
}

test "gradient: NaN and Inf handling" {
    // 特殊値の扱い
}
```

## 推奨される修正

### 1. メモリ管理の改善

全ての `contribution` ルールで、一時的なテンソルを `defer release` で解放する：

```zig
pub fn contribution(node: *const Node, i: usize, g: Tensor) Error!?Tensor {
    // ...
    const raw: ?Tensor = switch (node.prim) {
        .square => blk: {
            const two = try scalarLike(a, 2);
            defer release(two);  // 追加
            const two_a = try ops.mul(two, a);
            defer release(two_a);  // 追加
            break :blk try ops.mul(g, two_a);
        },
        // ... 全てのケースに同様の修正を適用
    };
}
```

### 2. テストの追加

`tests_grad.zig` に以下のテストを追加：

```zig
test "gradient: comprehensive binary ops" {
    // 両方の入力に対する勾配をテスト
    inline for (.{f32, f64}) |dt| {
        // add, sub, mul, div の勾配チェック
    }
}

test "gradient: broadcasting stress test" {
    // 複雑なブロードキャストパターン
}

test "gradient: high-order derivatives" {
    // 2階、3階微分のテスト
}
```

### 3. デバッグツールの追加

勾配チェックを容易にするユーティリティ：

```zig
pub fn numericalGradient(
    f: *const fn (Tensor) anyerror!Tensor,
    x: Tensor,
    eps: f64,
) !Tensor {
    // 数値微分による勾配計算
}

pub fn compareGradients(
    auto_grad: Tensor,
    num_grad: Tensor,
    tolerance: f64,
) !void {
    // 自動微分と数値微分の比較
}
```

## 次のステップ

1. ✅ 包括的な勾配テストスイートを作成
2. ⏳ 自動微分のメモリリークを修正
3. ⏳ ブロードキャストの勾配テストを追加
4. ⏳ 高階微分のテストを実装
5. ⏳ エッジケースのテストカバレッジを向上

## 実装優先度

### 高優先度
- メモリリークの修正（全VJPルール）
- 基本的な勾配テストの追加

### 中優先度
- ブロードキャストのテスト
- 線形代数の勾配テスト

### 低優先度
- 高階微分のテスト
- エッジケースのカバレッジ向上
