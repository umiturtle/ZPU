# 乱数（`rng`）

## 使い方

```zig
var gen = zpu.rng.Rng.init(42);                     // シードは明示的に
const u = try gen.uniform(zpu.Shape.initComptime(&.{ 2, 3 }), .f32);
const n = try gen.normal(zpu.Shape.initComptime(&.{ 4 }), .f64);
_ = gen.nextU64();

// グローバルなヘルパ（暗黙のシードを使う）
zpu.rng.seed(7);
const a = try zpu.rng.uniform(shape, .f64);
const b = try zpu.rng.normal(shape, .f32);
const c = try zpu.rng.randint(shape, 0, 10);
const p = try zpu.rng.permutation(10);              // 0..n-1 のシャッフル
```

## 実装意図

* **状態を持ち回る。** `Rng` は値型で、コピーすると独立したストリームになる。
  グローバルな状態に依存しないので、テストは決定的に書けるし、
  スレッドを増やしても同じ列が出る。

* **一様分布は `f64` の仮数に合わせて作る。** `nextU64` から
  上位ビットを使って `[0, 1)` に写す。下位ビットは線形合同法系の
  周期性に弱いため使わない。

* **正規分布は Box-Muller。** 2 つずつ生成し、奇数個必要なら 1 つ余分に作って捨てる。
  `cos` を使う代わりに `sqrt(-2 ln u)` の形にして、
  `sin` / `cos` の呼び出しを 1 回減らしている。

* **生成は `Tensor` を直接埋める。** 中間配列を作らないので、
  大きなテンソルでも割り当てが 1 回で済む。dtype は
  `f32` / `f64` の両方に対応し、整数が必要なときは `randint` を使う。

* **`dropout` との関係。** `nn.dropout` は `gen.uniform` で
  マスク用の一様乱数を作り、`1/(1-p)` でスケールする。
  乱数生成器を層の引数として渡すので、
  「学習の途中でシードを固定して再現する」がそのまま書ける。

## テスト

`rng: determinism and distribution` が

* 同じシードの 2 つの生成器が同じテンソルを作ること（決定性）
* 20000 個の正規乱数の平均が 0、分散が 1 に近いこと（分布）

を確認する。許容誤差は 0.03 / 0.05 で、これは 20000 サンプルの
標準誤差（平均で約 0.007、分散で約 0.01）より十分大きい。
