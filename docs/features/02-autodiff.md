# 自動微分 — `prims` / `tape` / `autodiff` / `transforms`

ZPU の自動微分は JAX の設計から大きく影響を受けている:
**関数変換**として `grad` / `jvp` / `vjp` / `jacobian` / `hessian` / `hvp` / `vmap` / `jit`
を提供し、それらが**合成可能**であること（`jit(grad(f))`、`jacobian(grad(f))` = `hessian(f)`）
が一番の売り。このページは「使える」だけなら要らない — `transforms` の API だけ見ればよい。
ここには「規則をどう足すか」「高階微分がなぜ動くか」「JAX と何が違うか」を書く。

## 変換の一覧

| 変換 | 型 | 説明 |
|------|-----|------|
| `grad(f, args, wrt)` | `(R^n → R) → R^n` | スカラー関数の勾配。逆方向モード。 |
| `valueAndGrad(f, args, wrt)` | `→ (R, R^n)` | 関数値と勾配を同時に返す。 |
| `vjp(f, args, wrt, cotangent)` | `→ (R, R^n)` | 明示的な余接ベクトルでの VJP。 |
| `jvp(f, args, tangents)` | `→ (R^m, R^m)` | 接ベクトル積。順方向モード。 |
| `jacobian(f, args, wrt)` | `→ R^{m×n}` | ヤコビアン（各出力の勾配を並べた行列）。 |
| `hessian(f, args)` | `→ R^{n×n}` | ヘシアン（`jacobian(grad(f))` として実装）。 |
| `hvp(f, args, wrt, v)` | `→ R^n` | ヘシアン-ベクトル積。逆方向の逆方向。 |
| `vmap(f, args)` | `→ R^{k×...}` | 先頭軸のベクトル化マップ。 |
| `jit(f)` | `→ Plan` | トレース → 実行計画。定数畳み込み付き。 |

JAX と違うのは **`jacobian` と `hessian` がトップレベルにある**こと。JAX では
`jax.jacobian` / `jax.hessian` は `vjp` / `jvp` の合成として定義されるが、
ZPU でも内部実装は同じ（`hessian = jacobian ∘ grad`）でも、よく使うものは
直接呼べるようにしてある。

## 使い方（規則を足す側）

1. `prims.Prim` に op を足す。
2. `ops` にカーネルを書く（最後に `ops.finish(prim, inputs, out, attrs)` を呼ぶ）。
3. `autodiff.contribution(node, k, g)` に **VJP 規則**（`k` 番目の入力に関する
   コタンジェント）を書く。前向きモードが要るなら `jvp` 用の規則も足す。

```zig
// VJP の例: y = exp(x) なら dx = g * exp(x)
.exp => blk: {
    const x = node.inputs[0];
    const e = try ops.exp(x);            // 前向きの値を再計算する（保存しない）
    defer ops.releaseTemp(e);
    break :blk try ops.mul(g, e);
},
```

### 規則の分類（comptime で検査）

すべての `Prim` は 3 つのいずれかに分類される:

* **`.rule`** — `contribution` に case がある。`exp`, `sin`, `matmul`, ...
* **`.zero`** — 勾配は常に 0。`floor`, `ceil`, `eq`, `lt`, `argmax`, ...
* **`.unimplemented`** — 実行時 `Error.NotSupported`。`cholesky`, `conv`, ...

この分類は `classify(prim)` 関数で comptime に検証され、未分類の
primitive を足すとビルドが落ちる。`.reverse`（= `flip`）と `.copy` の
規則漏れはこの仕組みで見つけた。

## 実装意図

### 保存しないで再計算する（checkpointing の代替）

テープは値を持たず、primitive と属性と入力の id だけを記録する（`tape.Node`）。
VJP は必要な中間値をその場で作り直す。メモリが読みやすく、`jit` の定数
畳み込みとも相性が良い。その代わり計算量は増えるので、`softmax` のように
「安定な形で再計算できる」演算はその形で書く。

JAX の `jax.checkpoint` / `remat` に相当するものはまだ無い（#8 で検討中）。
原理的には tape の node に `checkpoint` フラグを立てて、VJP のときに
`f` を再実行して中間値を作れば実装できる。

### 勾配は `Engine` が所有する

`autodiff.Engine` は `id -> Tensor` のマップを持ち、`seed(t, g)` で
加算合成する。既存の勾配があれば `add` で足して古い参照を解放、
無ければそのまま格納する。マップの値は `retain()` されており、
`Engine.deinit` が直接解放する（`releaseTemp` ではないのは、`grad` が
スコープを開いているときに「スコープのものではない参照」を確実に
落とすため）。

### 関連性の判定

`run` は最初に `wrt` から逆向きに辿って「勾配が要るノード」を印付ける
（`needed` マップ）。以降の掃引はそのマークが付いたノードだけを見る。
`wrt` に関係しない枝は計算しない。ループの中では毎回
`tape.nodes.items[idx]` を読み直す — コタンジェントの計算自体が op を
呼び、その op が同じテープにノードを追加して領域を再確保することが
あるため。

### tape はスタック

`MAX_TAPE_DEPTH = 8` の固定長スタックで、`pushTape` / `popTape` / `setTape`
と、丸ごと退避する `save` / `restore` がある。

`grad` は「前向き計算を内側のテープに記録し、掃引はそのテープを外して行う」。
外すのは、掃引自身の op がこれから読むテープに追記されないようにするため。

`hvp` は逆に、掃引の op を**外側のテープに載せたい**ので
`saved = context.save()` の後に `pushTape(&outer)` してから
`valueAndGrad` を呼ぶ。これが高階微分の仕組み: 外側のテープに
「勾配計算そのもの」を記録し、さらにそれを微分する。

### JVP（前向きモード）

`jvp` は JAX と同じ「`J · v` を返す」変換だが、内部実装は
**逆方向を m 回**（m は出力の要素数）呼んで合成する:

```
for j in 0..m:
    cotangent_j = one_hot(j)
    grad_j = vjp(f, args, cotangent_j)
    out_tangent[j] = sum(grad_j * tangent for each arg)
```

出力が小さい（スカラーなど）ときは逆方向の方が速く、出力が大きい
ときは順方向（本当の JVP）の方が速い。JAX は両方持っているが、
ZPU はまだ逆方向の実装のみ。

### `jacobian` と `hessian`

`jacobian(f, args, wrt)` は各出力要素について VJP を 1 回ずつ呼んで
ヤコビアンの行を構成する。`O(m)` 回の VJP 評価。

`hessian(f, args)` は `jacobian(grad(f))` として実装される:
まず `grad(f)` を計算する関数を作り、そのヤコビアンを取る。
内部では `valueAndGrad` を m 回呼ぶので `O(n²)`。
ヘシアンが必要な場面（Newton 法、サドル点判定、不確実性推定）は
パラメータ数が小さいことが多いので、実用的。

## テストの見方

`src/zpu/tests.zig` の勾配テストは 3 種類の根拠で書かれている。

* 解析解（`d/dx Σ exp(x²) = 2x·exp(x²)` など）
* 有限差分との比較（`gradcheck`: 多種類の op を通した関数）
* 恒等式（`grad grad sum(x*x) == 2`、`grad grad quad == (3, 4)` など）

さらに `autodiff.zig` の中に「VJP 規則が全部コンパイルできる」テストがあり、
網羅漏れ（`switch` の網羅性）がその場でビルドエラーになる。

`src/zpu/tests_grad.zig` は dtype × op の勾配テーブル:
33 個の elementwise unary、12 個の binary、12 個の reduction、
11 個の shape/indexing、6 個の linalg、12 個の nn レイヤー・損失を
f64/f32 の両方で有限差分と比較する。

## JAX との対応表

| JAX | ZPU | 備考 |
|-----|-----|------|
| `jax.grad` | `zpu.grad` | 同じ |
| `jax.value_and_grad` | `zpu.valueAndGrad` | 同じ |
| `jax.jacobian` | `zpu.jacobian` | ZPU ではトップレベル |
| `jax.hessian` | `zpu.hessian` | `jacobian(grad(f))` で実装 |
| `jax.jvp` | `zpu.jvp` | 内部は VJP の m 回呼び出し |
| `jax.vjp` | `zpu.vjp` | 明示的な余接ベクトル |
| `jax.vmap` | `zpu.vmap` | `batch_depth` で実装 |
| `jax.jit` | `zpu.jit` | トレース + 定数畳み込み |
| `jax.custom_jvp` | **未実装** | #6 |
| `jax.custom_vjp` | **未実装** | #6 |
| `jax.checkpoint` | **未実装** | #8 |

## 既知の制限と課題

* **`conv2d` の VJP** — `NotSupported`。微分可能な形で書き直すか、
  `conv_transpose` を使った VJP の実装が必要（#5）。
* **`scatter_add` の VJP** — 入力 `x` の勾配が `g` になっていたが、
  正しくはゼロ（`x` の値は出力に使われない）（#3、修正済）。
* **`prod(axis, keepdims=false)` の VJP** — `expandReduced` の形状扱いに
  問題があり `ShapeMismatch`（#4）。
* **`hessian` のメモリリーク** — `GradWrapper` が返すテンソルの所有権が
  スコープとミスマッチし、6 アロケーションがリークする。機能的には正しいが、
  所有権モデルの改善が必要（#8）。
