# ガイド

`zpu` を実際に使うときの手順と、はまりやすい点をまとめる。

## 1. ビルドと実行

Zig 0.17 だけが必要。追加のパッケージも生成コードも無い。

```sh
zig build test                  # テスト（リーク検査つき）
zig build test -Dtest-filter=matmul   # 一部だけ
zig build check                 # zig fmt --check + テスト
zig build bench                 # ベンチ
zig build bench -- --filter matmul --check
zig build doc                   # zig-out/doc/ に HTML
zig build examples              # examples/ を全部ビルド
zig build run-tensors           # 1 つ動かす
```

`zpu` を自分のプロジェクトから使う場合:

```zig
// build.zig.zon
.dependencies = .{ .zpu = .{ .path = "../ZPU" } },
```

```zig
// build.zig
const zpu = b.dependency("zpu", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("zpu", zpu.module("zpu"));
```

グラフィックも使うなら `zgf` を、バックエンドを直接触るなら `zbackend` を追加する。

## 2. 初期化

```zig
try zpu.init(gpa.allocator(), .{});      // アロケータとスレッド数を設定
defer zpu.deinit();
```

* `zpu.init` は**アロケータを登録するだけ**。テンソルの生存は `Scope` が決める。
* スレッドを使う要素演算は `zpu.initThreaded(alloc, n)` / `.threads = n`。
  既定は 1（インライン）で、ベンチの数値はこの設定で測っている。
* `std.testing.allocator` を渡してテストすると、解放漏れがそのままテスト失敗になる。
  だから `zpu` のテストは「数値が合っている」と「メモリが漏れていない」を同時に検査する。

## 3. テンソルの作り方

```zig
const S = zpu.Shape;

const a = try zpu.Tensor.zeros(S.initComptime(&.{ 2, 3 }), .f32);     // comptime 形状
const dims = [_]usize{ 2, 3 };
const b = try zpu.Tensor.ones(try S.init(&dims), .f64);               // 実行時形状
const c = try zpu.Tensor.fromArray(f64, &.{ 1, 2, 3, 4 }, S.initComptime(&.{ 2, 2 }));
const d = try zpu.Tensor.full(.{}, .f64, 3.0);                        // スカラー
const e = try zpu.arange(10, .f64);
```

`Shape` は `MAX_RANK` 次元までの固定ランク。ブロードキャストは NumPy と同じ規則で、
rank の違うテンソル同士の `add` などは自動で揃う。ただし `add` は**同じ rank を要求**し、
rank が違う場合は `broadcastPair` で揃えてから渡す（内部的には変換がそうしている）。

## 4. 生存管理（いちばん大事なところ）

`zpu` は参照カウントを明示的に扱う。覚える規則は 3 つだけ。

1. **作った場所が所有する。** すべての生成は 1 つの所有参照を返す。
2. **`Scope` はそのスコープ内で生まれた参照をまとめて解放する。**

```zig
var scope = try zpu.Scope.init(alloc);
defer scope.deinit();                 // ここまでに作ったテンソルを全部解放
const y = try zpu.tanh(x);            // 解放しなくてよい
```

3. **変換の戻り値は呼び出し側が所有する。**

```zig
const grads = try zpu.grad(alloc, f, null, args, &.{0});
defer alloc.free(grads);              // スライス
defer zpu.keepGrad(alloc, grads, 0).deinit();   // 使う分だけ取り出して解放
```

スコープを抜けても残したいテンソル（オプティマイザの状態、モデルのパラメータ）は
`zpu.untrack(t)` で切り離す。逆に、スコープの外で作ったテンソルをスコープに
持ち込むことはない（スコープは自分が生んだ参照しか解放しない）。

`zpu.releaseTemp(t)` は「スコープが無ければ解放し、あれば何もしない」ヘルパで、
演算の途中結果を書くときに使う。モジュール内部でよく出てくる。

## 5. 演算の呼び出し

```zig
const s = try zpu.add(a, b);
const m = try zpu.matmul(a, b);
const r = try zpu.sum(a, -1, false);      // 軸 -1、keepdims なし
const v = try zpu.variance(x, 0, true);
const w = try zpu.where(mask, p, q);
const t = try zpu.transpose(x, &.{ 1, 0 });
const sl = try zpu.slice(x, 0, 1, 3, .{ .step = 2 });
```

* 軸は `i32`（負は末尾から）。`zpu.AXIS_LAST` は `-1`、`zpu.AXIS_NONE` はリダクション無し。
* dtype は入力から決まる。`scalarLike(t, v)` は `t` と同じ dtype のスカラーを作る
  （`bool` だけは `f32` に落ちる）。混ざった dtype の二項演算は上位 dtype に昇格する。

## 6. 微分する

```zig
fn loss(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
    _ = ud;
    const h = try zpu.matmul(args[0], args[1]);
    defer zpu.releaseTemp(h);
    return zpu.sumAll(zpu.square(h));
}

const grads = try zpu.grad(alloc, loss, null, &.{ x, w }, &.{ 0, 1 });
```

* `grad` は**逆方向モード**。`wrt` は「勾配が欲しい引数の添字」。
* `valueAndGrad` は値と勾配を同時に返す（前向き計算を 1 回で済ませたいとき）。
* `jvp` は前向きモード（`f(x + εv) - f(x)` を 1 回の掃引で）。出力が少数のときに強い。
* `hvp` は `H f(x) · v`。`grad(grad(f))` と同じ仕組みで、テープのスタックで実現している。
* 微分できない op（`conv2d` の VJP など）は実行時に `Error.NotSupported` を返す。

## 7. 速くする: `vmap` と `jit`

```zig
var j = zpu.jit(alloc, loss, null);
defer j.deinit();
const y = try j.call(&.{x, w});      // 初回だけトレース、以降は再生
std.debug.print("{d} steps\n", .{j.numSteps()});
```

* `jit` は初回呼び出しで関数を**トレース**し、定数を畳み込み、依存の無い計算を消す。
  以降は `args` の id が同じなら計画を再生する（形状が変われば再トレース）。
* 再生中の一時テンソルは計画ではなく呼び出し側のスコープが所有する。
  返り値だけは呼び出し側が `deinit` する。
* `vmap` は先頭軸をベクトル化する。`vmap` の中の `matmul` は
  「vmap の深さ（`batch_depth`）」を保って計算するので、`(B, M, K) @ (B, K, N)` の
  つもりが `(M, K) @ (K, N)` になる、といった事故は起きない。

## 8. 学習ループ

```zig
var opt = zpu.optim.Adam.init(alloc, .{ .lr = 1e-2 });
defer opt.deinit();

var params = [_]zpu.Tensor{ w1, b1, w2, b2 };
for (0..steps) |_| {
    const g = try zpu.grad(alloc, loss, null, &params, &.{ 0, 1, 2, 3 });
    defer {
        for (g) |*t| { var tt = t.*; tt.deinit(); }
        alloc.free(g);
    }
    try opt.step(&params, g);
}
```

`optim` の状態（Adam の 1 次・2 次モーメント）は `untrack` でスコープの外に置かれるので、
`Scope` の中から呼んでも消えない。

## 9. はまりどころ

* **スコープが所有しているテンソルを自分で `deinit` しない。** 二重解放になり、
  `SafeAllocator` ならその場で落ちる。`deinit` してよいのは変換の戻り値だけ。
* `Tensor.item()` はスカラーでないと panic する。読むだけなら `readF64(i)`、
  スカラーは `readF64(0)`。
* 整数テンソルは `readF64` で読めるが、微分の対象にはできない（`requires_grad` が立たない）。
* `Scope` は入れ子にできるが、**内側のスコープは外側が生きている間は何も所有しない**
  （記録は外側に集約される）。長生きさせたい値は `untrack` で明示的に外へ出す。
* テストで `defer` を使ってスコープ内のテンソルを解放すると、スコープの解放と二重になる。
  テストでは「解放しない」が正解。
