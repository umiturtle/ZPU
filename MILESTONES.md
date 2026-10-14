# ZPU / ZGF マイルストーン

ZPU は 2 つのライブラリを 1 つのリポジトリで育てる。

| ライブラリ | 立ち位置 | 現状 |
| --- | --- | --- |
| `zpu` | JAX の立ち位置 — N 次元配列・自動微分・関数変換・実行計画 | 中核は動作。72 テスト green、リーク無し |
| `zgf` | 汎用グラフィックライブラリ — GPU バックエンドを `zbackend` 経由で共有 | 画像・描画・フィルタまで実装 |
| `zbackend` | 実行バックエンド層 — CPU 実行 + WebGPU / Metal / Vulkan / DirectX 12 の検出と起動 | CPU / Vulkan / WebGPU が実行可能（GPU 実機での確認待ち）。Metal / D3D12 は検出のみ |

運用ルール:

* **テスト駆動** — すべての機能は先に期待値つきでテストを書き、`std.testing.allocator`
  の下で走らせる。テストがそのままリーク検査になる。
* **ベンチマーク駆動** — 性能に触る変更は `bench/` の数値で前後比較する。ベンチも
  Zig のみ（Python 依存なし）。
* **comptime 優先** — 一貫性の検査は実行時ではなくコンパイル時に持っていく。
  例: すべての `Prim` が微分規則を持つかどうかは `comptime` ブロックで検証され、
  規則の無い primitive を足すとビルドが落ちる（`src/zpu/autodiff.zig`）。
  調査用の実行時計測コードはライブラリに残さない。
* **ドキュメント** — 機能ごとに「使い方」と「実装意図」を書く。日本語版と英語版の
  両方をコミットされた原本として持ち、HTML は `zig build doc` が生成する。
* **履歴** — jujutsu (jj) で管理。意味のある単位でコミットし、検証できたら即 push。

---

## M0 — 基盤 (完了)

* [x] Zig 0.17 のビルドスクリプト（モジュールグラフ `zbackend` → `zpu` → `zgf`）
* [x] 固定ランク `Shape`（`MAX_RANK` 次元、コンパイル時 shape 構築）
* [x] 64 バイト整列の参照カウント付き `Buffer`
* [x] ストライド付き N 次元 `Tensor`、view / コピー / 読み書き
* [x] `Scope` / `Tracker` による生存管理（`releaseTemp`、`untrack`、`keepGrad`）
* [x] `zig build test` / `bench` / `doc` / `check`（`zig fmt` + テスト）
* [x] `build.zig.zon` は依存ゼロ（`.dependencies` を持たない）、MIT ライセンス

受け入れ条件: `zig build check` が無音で成功し、テストがリークを報告しないこと。

## M1 — eager な数値カーネル (完了)

* [x] 要素演算（単項 43 種、二項 25 種）とブロードキャスト
* [x] リダクション（sum/mean/max/min/var/std、軸指定、keepdims）
* [x] 形状操作（reshape / transpose / broadcastTo / squeeze / expandDims / concat / stack）
* [x] 添字操作（slice / gather / scatter / pad / take / where）
* [x] `matmul`（バッチ対応）、`linalg.Singular` を含む線形代数
* [x] `dtype` の一貫性（`scalarLike` は入力 dtype を保つ。`.bool` のみ f32 へ）

受け入れ条件: 参照値と 1e-12 で一致（`src/zpu/tests.zig`）。

## M2 — 自動微分 (完了)

* [x] `tape`（primitive ごとの VJP 規則、テープ スタック `MAX_TAPE_DEPTH = 8`）
* [x] `autodiff.Engine` による逆方向スイープ（勾配は engine 所有、`get` は retain して返す）
* [x] 前方向モード（JVP）と `jvp` 変換
* [x] 高階微分（`grad(grad(f))`）、`hvp`（reverse-over-reverse）
* [x] **微分規則の網羅を comptime で保証**: `Prim` を「規則あり / 勾配ゼロ /
      未実装」に分類し、未分類の primitive があるとコンパイルエラー。
      これにより `.reverse`（= `flip`）と `.copy` の規則漏れを発見して修正し、
      `.cholesky` / `.triangular_solve` は「未実装」として実行時エラーにした
      （黙ってゼロ勾配を返さない）。

受け入れ条件: 有限差分との一致、`d/dx Σexp(x²)` の解析値、
`grad grad sum(x*x) = 2`、`grad grad quad = (3, 4)`。

## M3 — 関数変換 (完了)

* [x] `grad` / `valueAndGrad` / `keepGrad`
* [x] `vmap`（先頭軸のベクトル化、vmap 内の `matmul` は深さを保持）
* [x] `jit`（トレース → 定数畳み込みつき実行計画、`numSteps` / `traceCount`）
* [x] 変換の入れ子（`jit(grad(f))`、`vmap(grad(f))`）と参照カウントの確立

受け入れ条件: 24/24 テスト green、リーク 0。

## M4 — ニューラルネット層と学習 (完了)

* [x] `nn`: layerNorm / dropout / maxPool2d / avgPool2d / oneHot / conv2d / 初期化子
* [x] `optim`: SGD（momentum）/ Adam / RMSProp（状態は `untrack` でスコープ外へ）
* [x] `rng`: uniform / normal（再現可能なシード）
* [x] 学習ループが実際に損失を下げるテスト

## M5 — ベンチマーク基盤 (完了)

* [x] `bench/` に Zig のみのハーネス（`zig build bench`）
* [x] 各カーネルのスループット（GFLOP/s、GB/s）と理論ピーク比
* [x] `bench/baseline.txt` による回帰検出（閾値以上に落ちたら失敗）
* [x] NumPy / PyTorch との比較は **外部で測った数値表を取り込む**形にする
      （リポジトリのツールは Zig のみ。比較は「外で測った数値」として文書化）

## M6 — ドキュメント生成 (完了)

* [x] 原本: `docs/*.md`（ガイド・実装意図・他ライブラリとの違い）+ ソース内 `///`
* [x] `docs/build_docs.zig` — Python を使わない Markdown サブセット → HTML 生成器
* [x] 機能ごとの「使い方 + 実装意図」ページ（`docs/features/*.md`）
* [x] `zig build doc` で `zig-out/doc/` に静的 HTML（索引・目次・API 一覧）
* [x] **英語版と日本語版の両方**を生成する（言語切り替えつきの HTML、
      `zig build doc` が 36 + 34 ページを出す）
* [ ] 生成物の視覚確認と、ソース doc コメントの抜け拾い
* [ ] `jit` / `graph` の実装意図ページ（`docs/features/` の残り）

## M7 — バックエンド層 (進行中)

* [x] `src/backend.zig`: `Backend` enum（cpu / webgpu / metal / vulkan / d3d12）、
      `Capabilities`、`Runtime`、`probe()` / `open()`（`std.DynLib` による動的ロード）
* [x] `Kernel` / `Source` / `Buffer` / `dispatch` の共通インターフェース
* [x] CPU バックエンド（`dispatch` と `dispatchThreaded`、カーネル内エラーの伝播）
* [x] Vulkan の検出（ローダ → インスタンス → 物理デバイス列挙 → 名前と API 版）
* [x] デバイス層: `Device`（バックエンドごとのタグ付き共用体）、`openDevice`、
      `dispatchOn`、ロックフリーなプロセス単位キャッシュ（`device` / `shutdownDevices`）
* [x] **Vulkan**: ローダのみを相手に、論理デバイス + 計算キュー + コマンドプール、
      ホスト可視メモリ、SPIR-V からの compute pipeline、SSBO、push constant、
      フェンス待ち、読み戻し（`src/backend/vulkan.zig`）
* [x] **WebGPU**: Dawn / wgpu-native を `wgpuGetProcAddress` か直接シンボルで解決、
      アダプタ / デバイス要求、WGSL モジュール、ストレージバッファ、bind group、
      compute pass、`map_async` 読み戻し（`src/backend/webgpu.zig`）
* [x] GPU が無い環境では `openDevice` が `null` を返し、`dispatch` は CPU に落ちる
      （テストはそのまま green）
* [x] **実デバイスでの検証**: `zig build run-gpu-probe` が各デバイスを開いて
      1024 要素のカーネルを実行し数値を検査する。この開発環境には GPU が無いので
      ソフトウェア Vulkan（SwiftShader）を入れ、`ZPU_VULKAN_LIB=` で
      ドライバを指定して実際にディスパッチを通した（実測値は docs/features/09
      に記載）。GPU のあるマシンではローダと ICD が既にあるため
      `zig build run-gpu-probe` だけで同じ検査が走る
* [x] Vulkan の段階解決（global → instance → device、`vkGetDeviceProcAddr`）と
      ICD 直接ロード、Linux の multiarch ディレクトリ探索、ドライバの明示
      （`setDriverLibrary` / `ZPU_VULKAN_LIB` / `ZPU_WEBGPU_LIB` /
      `VK_ICD_FILENAMES` / `VK_DRIVER_FILES`）
* [x] パイプラインのキャッシュ（SPIR-V のハッシュ + バッファ数で 4 スロット、
      記述子セット・プール・コマンドバッファ・フェンスも使い回す。実測で
      1975–2160 µs → 60–85 µs/回）
* [x] デバイス常駐バッファ（`createDeviceBuffer` / `writeDeviceBuffer` /
      `dispatchDeviceBuffers` / `readDeviceBuffer`、デバイスごとのバッファプール。
      実測で往復 52–55 µs ↔ 常駐 37–43 µs/回）
* [ ] 数値コアのテンソルをデバイス常駐させる（今はカーネルを直接呼ぶ経路のみ）
* [ ] Direct3D 12 / Metal の実行（上記 2 つが固まってから）
* [ ] `zgf` から `zbackend` 経由で GPU 実行する経路

## M8 — `zgf` (進行中)

* [x] 画像 (`Image`) — `[H, W, C]` f32、RGB / RGBA / mono、8 bit sRGB、PPM/PGM 入出力
* [x] 2D 図形のラスタライズ（点・矩形・円・線・グラデーション）
* [x] `zpu` のテンソルを入力に取るフィルタ（ぼかし、Sobel、シャープ、明度、コントラスト）
* [x] 微分可能なフィルタ（`grad` でフィルタのパラメータを微分できる）
* [ ] テキスト描画（ビットマップフォント）
* [ ] PNG のエンコード / デコード（zlib を Zig で、依存なし）
* [ ] `zbackend` 経由の GPU 実行（同じカーネルを CPU/GPU 両方で）

## M9 — CI/CD (進行中)

* [x] GitHub Actions: Linux / macOS / Windows で `zig build check` / `test` / `doc` /
      `examples` / `bench -- --quick`（`.github/workflows/ci.yml`）、
      `zig-out/doc` を成果物として保存
* [x] GitHub Pages への配信（`.github/workflows/pages.yml`、`main` への push で公開）
* [x] **CI が 3 OS すべて green**（ubuntu / macOS / Windows + docs）。
      Windows は `std.DynLib` が無い（0.17 では `@compileError`）ので、動的
      ロードを `src/backend/lib.zig`（`kernel32` の `LoadLibraryA`）に寄せて解決
* [x] **GPU の実行経路を CI で検査**（`gpu` ジョブ）: lavapipe を入れて
      `ZPU_REQUIRE_GPU=vulkan zig build run-gpu-probe`。GPU の無いランナーでも
      SPIR-V を実行するデバイスが無いと落ちるので、CPU フォールバックが回帰を
      隠せない
* [x] Zig のバージョンは 0.17.0 に固定（CI が ziglang.org から取得）
* [x] `build.zig.zon` の `minimum_zig_version` を 0.17.0 に合わせる（済）
* [ ] ベンチ結果を CI のログに残す（回帰検出は `--check` で失敗させる）。
      ベンチ自体は CI で走っており、そこが見つけた不具合もある
      （`reshape` が非連続入力を実体化したとき返すビューが参照を持たず、
      `conv2d` が解放済みメモリを読んでいた → `37d4757b`）
* [ ] Pages の有効化（リポジトリ設定）と、最初の green な run の確認

## M10 — 仕上げ (進行中)

* [x] README に「他ライブラリとの違い」と実測ベンチを載せる
* [x] 調査用の実行時計測コードを残さない（一貫性の検査は comptime へ。
      `build.zig` に残っていた未使用の調査オプションも削除した）
* [ ] Zig 0.17 のイディオム見直し（`std.Io`、エラーセット、`@splat`、`inline for`）
* [ ] 公開 API の doc コメント網羅（`zig build doc` の警告ゼロ）
* [ ] `examples/` の充実 — `build.zig` が名前を挙げている 8 本のうち、
      書けたのは `gpu-probe` の 1 本。残り 7 本（`tensors` / `autodiff` /
      `train-mlp` / `images` / `vector-art` / `render3d` / `diff-render`）を
      書く（`serve_demo` は別枠で既にある）
* [ ] タグ付けとリリースノート

---

## 進め方 (ワークフロー)

```sh
zig build test            # ユニット + 数値 + リーク検査
zig build check           # zig fmt --check + テスト
zig build bench           # 計測 (Zig のみ)
zig build doc             # zig-out/doc/ に HTML
zig build run-<example>   # examples/<name>.zig
```

コミットは jj:

```sh
jj status
jj commit -m "…"
jj bookmark set arena/fe3b1e92-zpu -r @-
jj git push -b arena/fe3b1e92-zpu
```
