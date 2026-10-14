# 関数変換（`transforms`）

`grad` / `valueAndGrad` / `jvp` / `hvp` / `vmap` / `jit`。すべて `zpu.zig` から
そのまま使える。

## 使い方

```zig
fn f(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
    _ = ud;
    const h = try zpu.matmul(args[0], args[1]);
    defer zpu.releaseTemp(h);
    return zpu.sumAll(zpu.tanh(h));
}

// 逆方向モード: 戻り値は []Tensor（呼び出し側が所有）
const grads = try zpu.grad(alloc, f, null, &.{ x, w }, &.{ 0, 1 });
defer alloc.free(grads);
defer zpu.keepGrad(alloc, grads, 0).deinit();   // 1 要素だけ持ち出して残りを解放

// 値と勾配を同時に
var vg = try zpu.valueAndGrad(alloc, f, null, &.{x}, &.{0});
defer vg.deinit();
// vg.value, vg.grads[0]

// 前向きモード: 接ベクトル v を渡す
var jr = try zpu.jvp(alloc, f, null, &.{x}, &.{v});
defer jr.deinit();
// jr.value, jr.tangent

// H f(x) · v（逆方向の逆方向）
const hv = try zpu.hvp(alloc, f, null, &.{x}, 0, v);
defer { var h = hv; h.deinit(); }

// 先頭軸をベクトル化
const vm = try zpu.vmap(alloc, f, null, &.{ batch });
defer { var m = vm; m.deinit(); }

// トレースして再生
var j = zpu.jit(alloc, f, null);
defer j.deinit();
const y = try j.call(&.{ x, w });     // 初回だけトレース
_ = j.numSteps();                     // 実行計画のステップ数
_ = j.traceCount;                     // 何回トレースしたか
```

## 実装意図

すべての変換は同じ手順を踏む（`transforms.zig` の冒頭にも書いてある）。

1. 変換が作るテンソルを所有する `Scope` を開く。
2. 必要な記録装置（テープ / グラフ）を差し込む。
3. 関数を呼ぶ。
4. **外側の記録状態に戻す** — これが `grad(grad(f))` や `jit(grad(f))` を可能にする。
   変換自身の計算（掃引など）が、外側のグラフやテープに記録されるべきものだから。
5. 外に出す値だけ `retain` する。

* **`grad` は引数をコピーしてから使う。** `copyArgs` が `retainScoped()` で
  各引数の参照をスコープに預けてから `wrt` を選ぶ。
  呼び出し側のテンソルが変換中に解放されないための保険で、
  スコープが終わると同時に解放される。

* **`vmap` は `batch_depth` を進める。** 引数の先頭軸をベクトル軸として、
  その深さをテンソルに記録してから関数を呼ぶ。
  演算側（特に `matmul`）はこの深さを見て「どの軸が行列で、どれがバッチか」を決める。
  出力から深さを戻すときに `withBatchDepth` を使う。

* **`vmap` の中のリダクション。** `keepdims` の有無で形が変わる演算は、
  深さを保ったまま「1 段内側の軸」を畳む必要がある。
  そのため `vmap` は `wrt` の軸を深さぶんずらして解釈する。

* **`jit` はトレースと再生を分ける。**
  `Jit.trace` が記録した `Graph` を `jit.build` に渡すと、`Plan`（`Step` と `Blob` の列）
  ができる。`Step` は primitive と入出力の id、`Blob` は「どの引数にも依存しない、
  畳み込まれた定数」を持つ。
  再生時は `Step` を順に実行するだけで、グラフは見ない。

* **のれん分け（保有参照の受け渡し）。** トレース中の一時テンソルは
  変換のスコープが所有し、`Plan` は自分が必要とする値だけを `retain` する。
  トレース中に関数が *変換* の戻り値（= 呼び出し側が所有する参照）を返した場合は、
  その 1 つの余分な参照を `Jit.trace` が `releaseExtraRef` で落とす。
  「スコープが持っている参照」と「記録ノードが持っている参照」を数えて、
  それでも多い 1 つを返す、という規則で判定している。

* **`hvp` は特別扱いしない。** 外側テープに `∇f(x) · v` を記録し、
  そのテープをもう一度掃引するだけ。専用カーネルは存在しない。

## 現在わかっている制限

* `jit` は引数の **id** が変わると再トレースする。同じ id で形状だけ変えると
  計画が古いままになるので、形状を変えるときは新しいテンソルを作る。
* 変換の戻り値（`grad` の `[]Tensor`、`jit.call` の結果）は
  **呼び出し側が `deinit` する**。スコープ内で作ったテンソルは `deinit` しない。
  この非対称性がこのライブラリで唯一覚えるべき規則で、テストは
  `std.testing.allocator` の下で走るので破ると即座に落ちる。
* `vmap` の `wrt` は先頭軸のみ。入れ子の `vmap` は 2 段まで意図的に動かしている。
