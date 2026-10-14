# 最適化（`optim`）

## 使い方

```zig
var opt = zpu.optim.Adam.init(alloc, .{ .lr = 1e-2 });
defer opt.deinit();

var params = [_]zpu.Tensor{ w1, b1, w2, b2 };
for (0..steps) |_| {
    const g = try zpu.grad(alloc, loss, null, &params, &.{ 0, 1, 2, 3 });
    defer {
        for (g) |*t| {
            var tt = t.*;
            tt.deinit();
        }
        alloc.free(g);
    }
    try opt.step(&params, g);
}
```

`Sgd`（`momentum` つき）も `RmsProp` も同じ形。

```zig
var sgd = zpu.optim.Sgd.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
var rms = zpu.optim.RmsProp.init(alloc, .{ .lr = 1e-3 });
```

## 実装意図

* **状態はパラメータの id で引く。** Adam の 1 次・2 次モーメント、
  momentum、RMS の二乗平均は、いずれも `id -> Tensor` のマップに入る。
  パラメータを増やしても `step` のコードは同じ。

* **状態は `untrack` でスコープの外に出す。** 状態テンソルは
  `Tensor.zeros(...)` で作られるが、これが学習ループの `Scope` の中で起きると
  スコープの終了と同時に解放されてしまう。
  `optim` は生成直後に `zpu.untrack(state)` を呼び、所有権を最適化器自身に移す。
  解放は `opt.deinit()` の責任。

* **更新はその場で書く。** `subInPlace` は連続なパラメータテンソルに対して
  `p[i] -= delta[i]` を直接実行する（`writeF64` / `readF64` 経由）。
  新しいテンソルを作らないので、長いループでも割り当てが増えない。
  dtype が違う勾配が来た場合は `cast` してから引く。

* **勾配の解放は呼び出し側。** `step` は `grads` を借りるだけで所有しない。
  だから上の例のように `defer` で解放する。所有権を曖昧にしないのが
  このライブラリの一貫した方針で、テストは
  `std.testing.allocator` の下で「漏れ」と「二重解放」の両方を検出する。

## テスト

`optimisers minimise a quadratic` が実際に最適化を回して収束を確認する。
`(w-3)^2 + (b+1)^2` を Adam（400 ステップ）で `w=3, b=-1` に、
`(w-3)^2` をモーメンタム付き SGD（200 ステップ）で `w=3` に持っていく。
学習率とステップ数はテストの中に固定してあるので、収束しなくなったら即座に落ちる。

## これから

* 学習率スケジューラ（`cosine`、`one_cycle`）
* 重み減衰（`weight_decay`）と勾配クリッピング
* パラメータのグループ分け（層ごとに学習率を変える）
