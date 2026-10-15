# ZPU

**Zig で書いた数値計算ライブラリ (`zpu`)・汎用グラフィックライブラリ (`zgf`)・2D UI ライブラリ (`zui`)。**
Zig 0.17 のみ。外部依存もコード生成もビルド時の Python も無い。すべて `zig build` で完結する。

```sh
zig build test     # 205/205 green。std.testing.allocator の下で走るのでリーク検査も兼ねる
zig build check    # zig fmt --check + テスト
zig build bench    # カーネル別スループット (Zig のみ)
zig build doc      # docs/out/ に HTML を生成
```

## ライブラリ構成

依存関係は一方通行で、下層は上層を知らない:

```
zpu  ←  zgf  ←  zui
 ↑                 ↑
 └── zbackend ─────┘
```

| | 中身 | 状態 |
| --- | --- | --- |
| `zpu` | JAX の立ち位置: N 次元配列・自動微分・関数変換・実行計画。**純粋な数値計算ライブラリ。GPU バックエンドに依存しない** | 中核完成 (M0–M4) |
| `zbackend` | WebGPU / Metal / Vulkan / DirectX 12 を dlopen で扱うバックエンド層。`zpu` とは独立 | インターフェース完成 (M7) |
| `zgf` | 汎用グラフィックライブラリ。`zpu` のテンソルを使い、必要に応じて `zbackend` で GPU 加速 | 立ち上げ中 (M8) |
| `zui` | 2D UI ライブラリ。即時モード API + 保持される表示リスト + 最小の保持状態。`zpu`・`zgf`・`zbackend` を使う | ソフトウェア経路が動作 ([`ZUI_MILESTONES.md`](ZUI_MILESTONES.md)) |

計画と進捗は [`MILESTONES.md`](MILESTONES.md)（`zui` は別トラックの
[`ZUI_MILESTONES.md`](ZUI_MILESTONES.md)）、機能ごとの説明は [`docs/`](docs/) にある。

## 30 秒でわかる `zpu`

```zig
const std = @import("std");
const zpu = @import("zpu");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try zpu.init(gpa.allocator(), .{});
    defer zpu.deinit();

    const S = zpu.Shape;
    const x = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, S.initComptime(&.{ 2, 2 }));

    // 逆方向モードの自動微分
    const Loss = struct {
        fn f(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
            _ = ud;
            const sq = try zpu.square(args[0]);
            defer zpu.releaseTemp(sq);
            return zpu.sumAll(zpu.tanh(sq));
        }
    };
    const grads = try zpu.grad(gpa.allocator(), Loss.f, null, &.{x}, &.{0});
    defer gpa.allocator().free(grads);
    defer zpu.keepGrad(gpa.allocator(), grads, 0).deinit(); // 使う分だけ取り出す

    std.debug.print("{d}\n", .{grads[0].readF64(0)});
}
```

### モジュール

| モジュール | 役割 |
| --- | --- |
| `dtype` / `shape` / `buffer` | 固定ランクのメタデータ、64 B 整列の参照カウント付きストレージ |
| `tensor` | ストライド付き N 次元配列。`Scope` / `Tracker` / `untrack` / `releaseTemp` |
| `ops` | eager な演算（要素演算・リダクション・形状・添字・`matmul`） |
| `linalg` | 分解・逆行列・`solve`・`det`（`Error.Singular` を返す） |
| `prims` / `tape` / `autodiff` | primitive と VJP/JVP 規則、逆方向スイープのエンジン |
| `graph` / `jit` | 記録した DAG と、定数畳み込みつき実行計画 |
| `transforms` | `grad` / `valueAndGrad` / `vjp` / `jvp` / `jacobian` / `hessian` / `hvp` / `vmap` / `jit` |
| `nn` / `optim` / `rng` | layerNorm・dropout・pooling・conv2d、SGD/Adam/RMSProp、乱数 |

### 変換

```zig
const g   = try zpu.grad(alloc, f, null, args, &.{0});            // []Tensor（呼び出し側が所有）
const vg  = try zpu.valueAndGrad(alloc, f, null, args, &.{0});    // .value と .grads
const vjp = try zpu.vjp(alloc, f, null, args, &.{0}, cotangent);  // 明示的な余接ベクトル
const jvp = try zpu.jvp(alloc, f, args, &.{v});                   // .value と .tangent
const jac = try zpu.jacobian(alloc, f, null, args, 0);            // R^{m×n} のヤコビアン
const hes = try zpu.hessian(alloc, f, null, args);                // R^{n×n} のヘシアン
const hv  = try zpu.hvp(alloc, f, null, args, 0, v);              // H f(x) · v
const vm  = try zpu.vmap(alloc, f, null, args);                   // 先頭軸をベクトル化

var j = zpu.jit(alloc, f, null);                                  // 1 回トレースして再生
defer j.deinit();
const y = try j.call(&.{x});
```

`jit(grad(f))` や `vmap(grad(f))` のような入れ子も動く。テープはスタックになっており
(`MAX_TAPE_DEPTH = 8`)、外側のテープに内側の前向き計算と後ろ向きスイープの両方が記録される
— `hvp` はこの仕組みだけで実装されている（トリック無し）。

## メモリの持ち方

Zig には GC が無いので、テンソルの生存は明示的に決める。ZPU の規則は 3 つだけ。

1. **生成した場所が所有する。** すべての生成 (`Tensor.empty` など) は所有者が 1 つだけ持つ
   参照カウントを返す。
2. **`Scope` はそのスコープ内で生まれた参照をまとめて解放する。** 変換 (`grad` など) は
   内部でスコープを開くので、途中のテンソルは呼び出し側で解放しなくてよい。
   スコープを持ち出したい値は `zpu.untrack(t)` で切り離す（オプティマイザの状態がこれを使う）。
3. **変換の戻り値は呼び出し側が所有する。** `grad` の `[]Tensor`、`jit.call` の結果は
   呼び出し側の責任。`zpu.keepGrad` は 1 要素だけ持ち出して残りを解放する。

`std.testing.allocator` の下でテストを走らせているので、この規約が破れると即座に失敗する。
現在のテストはリーク 0 で通っている。

## 他のライブラリと何が違うか

| | JAX / PyTorch / TF | NumPy | Zig エコシステムの数値ライブラリ | **ZPU** |
| --- | --- | --- | --- | --- |
| 言語・依存 | Python + C++/XLA | Python + BLAS | Zig (多くは C の BLAS に依存) | **Zig のみ。依存ゼロ**（標準ライブラリだけ） |
| 形状 | 動的、実行時ポリモーフィズム | 動的 | 動的 or comptime | **`MAX_RANK` 次の固定ランク `Shape`**。添字ミスは早く落ちる |
| メモリ | GC / 参照カウント + deleter | GC | 手動 | **スコープ所有 + 明示的な参照カウント。GC 無し、リーク検査つき** |
| 自動微分 | コンパイラ側の変換 (XLA / autograd) | 無し | 一部に手書き勾配 | **ライブラリ内で完結する変換**。`grad`・`jvp`・`vmap`・`jit` がすべて Zig の関数合成 |
| 入れ子の変換 | 対応 (トレーサ) | — | — | **テープのスタック + グラフの差し替え**で対応。`jit(grad(f))` も `hvp` も同じ仕組み |
| バックエンド | XLA / CUDA / Metal など | BLAS | 個別 | **`zbackend`: WebGPU / Metal / Vulkan / D3D12 を dlopen で検出**。CPU も同じ `Kernel` インターフェース |
| ビルド | pip / conda、コード生成 | pip | プロジェクトごと | **`zig build` だけ。Python も生成コードも不要** |

具体的に意識している違い:

* **動的形状を持たない。** JAX は `jax.jit` が shape の変更で再トレースするのに対し、ZPU は
  そもそもランクを `MAX_RANK` の範囲で固定として扱い、`Shape.initComptime` で comptime に
  決まる形状はそのままコンパイル時に検査する。実行時形状は `Shape.init` で作る。
* **所有権が API の一部。** 変換の戻り値・`Scope`・`untrack` がドキュメント化された契約で、
  テストがそれを検証する。解放漏れも二重解放もテストで落ちる。
* **バックエンドは後付けではなく同じ層。** `zgf` と `zpu` が同じ `zbackend`
  （`Backend` / `Capabilities` / `Kernel` / `dispatch`）を使う。GPU が無ければ CPU に落ちる。
* **「測ってから速くする」。** `bench/` に Zig のみの計測ハーネスを持ち、変更前後の数値を
  リポジトリ内で比較する。

## テストとベンチ

* `src/zpu/tests.zig` — 期待値は NumPy、解析解、有限差分から取っている。
  テストは機能ごとに分かれており、1 テスト 1 機能。
* 各モジュールの `test` — そのモジュールだけで閉じる不変条件（ブロードキャスト結果、
  VJP の形、`Engine` の所有権など）。
* `bench/bench.zig` — カーネル別のスループットを表示し、`bench/baseline.txt` と比較して
  回帰を検出する（`zig build bench -- --check`）。

## ライセンス

MIT。
