# ZPU マイルストーン

ZPU は 2 つのライブラリを 1 つのリポジトリで育てる。

| ライブラリ | 立ち位置 | 現状 |
| --- | --- | --- |
| `zpu` | JAX の立ち位置 — N 次元配列・自動微分・関数変換・実行計画 | 中核は動作。24/24 テスト green、リーク無し |
| `zgf` | 汎用グラフィックライブラリ — 将来の GPU バックエンドを共有 | プレースホルダ (`src/zgf.zig`) |

運用ルール:

* **テスト駆動** — すべての機能は `src/zpu/tests.zig`（および各モジュール内の `test`）に
  期待値つきで先に落とす。`std.testing.allocator` の下で走るのでテストは同時に
  リーク検査でもある。
* **ベンチマーク駆動** — 性能に触る変更は `bench/` の数値で前後比較する。
  ベンチも Zig のみ（Python 依存なし）。
* **ドキュメント** — 機能ごとに「使い方」と「実装意図」を書く。原本はコミットされた
  ファイル（`docs/*.md` とソース内の `///`）で、HTML は `zig build doc` が生成する。
* **履歴** — jujutsu (jj) で管理。意味のある単位でコミットし、検証できた時点で push。

---

## M0 — 基盤 (完了)

* [x] Zig 0.17 のビルドスクリプト（モジュールグラフ `zbackend` → `zpu` → `zgf`）
* [x] 固定ランク `Shape`（`MAX_RANK` 次元、コンパイル時 shape 構築）
* [x] 64 バイト整列の参照カウント付き `Buffer`
* [x] ストライド付き N 次元 `Tensor`、view / コピー / 読み書き
* [x] `Scope` / `Tracker` による生存管理（`releaseTemp`、`untrack`、`keepGrad`）
* [x] `zig build test` / `bench` / `doc` / `check`（`zig fmt` + テスト）

受け入れ条件: `zig build test` が green、`zig fmt --check` が clean。

## M1 — eager な数値カーネル (完了)

* [x] 要素演算（単項 43 種、二項 25 種）とブロードキャスト
* [x] リダクション（sum/mean/max/min/var/std、軸指定、keepdims）
* [x] 形状操作（reshape / transpose / broadcastTo / squeeze / expandDims / concat / stack）
* [x] 添字操作（slice / gather / scatter / pad / take / where）
* [x] `matmul`（バッチ対応）、`linalg.Singular` を含む線形代数
* [x] `dtype` の一貫性（`scalarLike` は入力 dtype を保つ。`.bool` のみ f32 へ）

受け入れ条件: NumPy で計算した期待値と 1e-12 で一致（`tests.zig`）。

## M2 — 自動微分 (完了)

* [x] `tape`（primitive ごとの VJP 規則、テープ スタック `MAX_TAPE_DEPTH = 8`）
* [x] `autodiff.Engine` による逆方向スイープ（勾配は engine 所有、`get` は retain して返す）
* [x] 前方向モード（JVP）と `jvp` 変換
* [x] 高階微分（`grad(grad(f))`）、`hvp`（reverse-over-reverse）

受け入れ条件: 有限差分との一致、`d/dx Σexp(x²)` の解析値、
`grad grad sum(x*x) = 2`、`grad grad quad = (3, 4)`。

## M3 — 関数変換 (完了)

* [x] `grad` / `valueAndGrad` / `keepGrad`
* [x] `vmap`（先頭軸のベクトル化、vmap 内の `matmul` は深さを保持）
* [x] `jit`（トレース → 定数畳み込みつき実行計画、`numSteps` / `traceCount`）
* [x] 変換の入れ子（`jit(grad(f))`、`vmap(grad(f))`）と参照カウントの確立

受け入れ条件: 24/24 テスト green、リーク 0（2026-10 時点で達成）。

## M4 — ニューラルネット層と学習 (完了)

* [x] `nn`: layerNorm / dropout / maxPool2d / avgPool2d / oneHot / conv2d / 初期化子
* [x] `optim`: SGD（momentum）/ Adam / RMSProp（状態は `untrack` でスコープ外へ）
* [x] `rng`: uniform / normal（再現可能なシード）
* [x] 学習ループが実際に損失を下げるテスト

## M5 — ベンチマーク基盤 (進行中)

* [x] `bench/` に Zig のみのハーネス（`zig build bench`）
* [x] 各カーネルのスループット（GFLOP/s、GB/s）と理論ピーク比
* [ ] NumPy / PyTorch との比較は **CI で生成した数値表を取り込む**形にする
      （リポジトリのツールは Zig のみ。比較は「外部で測った数値」として文書化）
* [ ] 回帰検出: `bench/baseline.txt` と比較して閾値以上に落ちたら失敗させる

## M6 — ドキュメント生成 (進行中)

* [x] 原本: `docs/*.md`（ガイド・実装意図・他ライブラリとの違い）+ ソース内 `///`
* [x] `tools/docgen.zig` — Python を使わない Markdown サブセット → HTML 生成器
* [x] 機能ごとの「使い方 + 実装意図」ページ（`docs/features/*.md`）
* [x] `zig build doc` で `docs/out/` に静的 HTML（索引・目次・API 一覧）
* [ ] 生成物の視覚確認と、ソース doc コメントの抜け拾い

## M7 — バックエンド層 (進行中)

* [x] `src/backend.zig`: `Backend` enum（cpu / webgpu / metal / vulkan / d3d12）、
      `Capabilities`、`Runtime`、`probe()` / `open()`（`std.DynLib` による動的ロード）
* [x] `Kernel` / `Source` / `Buffer` / `dispatch` の共通インターフェース
* [ ] 実機バックエンドの最小 dispatch 実装
  * [ ] Vulkan（`libvulkan.so.1` を dlopen、compute pipeline、SSBO）
  * [ ] Direct3D 12（`d3d12.dll`、compute root signature、UAV）
  * [ ] Metal（`libMetal.dylib`、`MTLComputePipelineState`）
  * [ ] WebGPU（`wgpu-native` / Dawn の C API）
* [ ] CPU バックエンドを同じインターフェースで包み、`zgf` から使えるようにする
* [ ] バックエンド選択のフォールバック規則（無い環境では cpu へ）
* [ ] 各バックエンドの probe をテストする（ライブラリが無くても SKIP で green）

## M8 — `zgf` (未着手)

* [ ] 画像バッファ（RGBA8 / f32）と色変換、リサイズ、合成
* [ ] 2D 図形のラスタライズ（線・三角形・円）とブレンド
* [ ] テキスト描画（ビットマップフォント）
* [ ] `zpu` のテンソルを入力に取るフィルタ（ぼかし、Sobel、トーンカーブ）
* [ ] `zbackend` 経由の GPU 実行（同じカーネルを CPU/GPU 両方で）
* [ ] PNG のエンコード / デコード（zlib を Zig で、依存なし）

## M9 — 仕上げ (未着手)

* [ ] Zig 0.17 のイディオム見直し（`std.Io`、エラーセット、`@splat`、`inline for`）
* [ ] 公開 API の doc コメント網羅（`zig build doc` の警告ゼロ）
* [ ] README に「他ライブラリとの違い」と実測ベンチを載せる
* [ ] `examples/` の充実（チュートリアルから写経できる形）
* [ ] タグ付けとリリースノート

---

## 進め方 (ワークフロー)

```sh
zig build test            # 24/24 + リーク検査
zig build check           # zig fmt --check + テスト
zig build bench           # 計測 (Zig のみ)
zig build doc             # docs/out/ に HTML
zig build run-<example>   # examples/<name>.zig
```

コミットは jj:

```sh
jj status
jj commit -m "…"
jj bookmark set arena/fe3b1e92-zpu -r @- && jj git push -b arena/fe3b1e92-zpu
```
