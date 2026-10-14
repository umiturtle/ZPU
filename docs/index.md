# ZPU ドキュメント

**Zig だけで書かれた数値計算ライブラリ (`zpu`) と汎用グラフィックライブラリ (`zgf`)。**
外部依存ゼロ、ビルドは `zig build` ひとつ、ドキュメント生成もテストもベンチも Zig で書かれている。

```sh
zig build test     # ユニット + 数値テスト（リーク検査込み）
zig build check    # zig fmt --check + テスト
zig build bench    # 計測（Zig のみ）
zig build doc      # この HTML を生成
```

## どこから読むか

| 目的 | ページ |
| --- | --- |
| まず動かす | [ガイド](guide.html) |
| 設計の意図と他ライブラリとの違い | [他ライブラリとの違い](comparison.html) |
| これから作るもの | [マイルストーン](milestones.html) |
| 機能ごとの使い方と実装意図 | 左の「機能」 |
| ソースから生成した API 一覧 | 左の「API」 |

## 30 秒でわかる全体像

* **`zpu`** — N 次元配列（`Tensor`）と eager な演算（`ops`）、自動微分（`tape` + `autodiff`）、
  関数変換（`transforms`: `grad` / `jvp` / `vmap` / `jit`）、実行計画（`graph` + `jit`）、
  ニューラルネット層（`nn`）、最適化（`optim`）、乱数（`rng`）、線形代数（`linalg`）。
* **`zgf`** — 画像と 2D/3D 描画。`zpu` のテンソルをそのまま入力に取る（フィルタ・ぼかし・
  微分可能レンダリング）。GPU 実行は `zbackend` 経由。
* **`zbackend`** — WebGPU / Metal / Vulkan / DirectX 12 を `std.DynLib` で検出し、
  同じ `Kernel` / `Buffer` / `dispatch` インターフェースで扱う。無ければ CPU に落ちる。

```zig
const zpu = @import("zpu");

try zpu.init(alloc, .{});
defer zpu.deinit();

const x = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, zpu.Shape.initComptime(&.{ 2, 2 }));

// eager な演算は生成した場所（Scope）が所有する
var scope = try zpu.Scope.init(alloc);
defer scope.deinit();
const y = try zpu.tanh(try zpu.matmul(x, x));

// 変換の戻り値は呼び出し側が所有する
const g = try zpu.grad(alloc, loss, null, &.{x}, &.{0});
defer alloc.free(g);
```

## このドキュメント自体について

* 原本はリポジトリ内のファイル（このページは `docs/index.md`）。HTML は `zig build doc` が
  `docs/build_docs.zig` で生成する。Python も静的サイトジェネレータも使っていない。
* 各 API ページはソースの `//!` と `///` コメントから生成される。つまり
  **doc コメントが唯一の情報源**で、HTML とソースが食い違うことはない。
