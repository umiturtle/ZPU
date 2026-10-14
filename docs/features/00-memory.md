# メモリと形状（`dtype` / `shape` / `buffer` / `tensor`）

## 使い方

```zig
const S = zpu.Shape;

// comptime で決まる形状: 添字はコンパイラが検査する
const x = try zpu.Tensor.zeros(S.initComptime(&.{ 2, 3 }), .f32);

// 実行時に決まる形状
const dims = [_]usize{ rows, cols };
const y = try zpu.Tensor.ones(try S.init(&dims), .f64);

x.shape.dims[0];        // 2
x.rank();               // 2
x.numel();              // 6
x.dtype();              // .f32
x.isContiguous();
x.strides[0];           // 行ストライド

// 読み書きは要素添字で（view でも論理的な順序で扱える）
const tidx = x.offset + 1;
_ = tidx;
try zpu.ops.copyInto(x, y);      // view から view へのコピー
x.readF64(0);
x.writeF64(0, 1.5);
```

スコープによる生存管理:

```zig
var scope = try zpu.Scope.init(alloc);
defer scope.deinit();                 // ここまでに生まれた参照をまとめて解放

const a = try zpu.Tensor.ones(S.initComptime(&.{4}), .f64);   // スコープが所有
const b = try zpu.exp(a);                                     // 同上

// スコープの外に持ち出したい値（モデルのパラメータ、最適化器の状態）
const state = try zpu.Tensor.zeros(S.initComptime(&.{4}), .f64);
zpu.untrack(state);                   // 以後 state は呼び出し側が所有
```

## 実装意図

* **固定ランクの値型 `Shape`。** `MAX_RANK`（既定 8）次元までの `dims` を持ち、
  `initComptime` は comptime 既知の次元をそのまま使い、`init` は実行時の値を詰める。
  どちらも同じ `Shape` になり、ランクは `rank` フィールドで持つ。
  動的な軸数を持つライブラリにありがちな「ランクごとの分岐爆発」を避けるため、
  ブロードキャストなどの計算は `MAX_RANK` の固定長配列で行い、
  `payloadRank()`（vmap の深さを除いたランク）で意味を合わせている。

* **64 バイト整列の `Buffer`。** 参照カウントは `std.atomic.Value(u32)`。
  `createWith` は `alloc.alignedAlloc(u8, .fromByteUnits(ALIGN), n)` でデータを取り、
  `Buffer` 構造体自体も別に確保する。整列を 64 にするのは、
  ベクトル化したカーネルと GPU バッファの両方で都合が良いため。

* **view は別の参照。** `viewOf` は `(shape, strides, offset)` だけを変えた
  `Tensor` を作り、**バッファの参照を 1 つ増やす**。その参照は生成時に
  `track` でスコープに預ける（スコープがあれば）。スコープが無ければ
  呼び出し側がその参照を持つ。`reshape`・`transpose`・`squeeze`・
  `broadcastTo`・`sliceRanges` は view を返せる限り view を返し、
  コピーが必要なときだけ `toContiguous` する。

* **`Scope` と `Tracker`。** `Scope.init` は `Tracker` を作り、
  既に別のトラッカーが生きていれば**何も所有しない**（`Tracker.begin` が false を返す）。
  変換（`grad` など）が内部でスコープを開いても、その中の一時テンソルは
  外側のスコープに集約される、という規則になっている。
  `untrack` は「この参照はスコープのものではない」と宣言し、
  リストから外して呼び出し側に所有権を渡す。`keepGrad` のように
  戻り値だけは外に持ち出す、という使い方をされる。

* **`releaseTemp`。** 「スコープがあれば何もしない／無ければ解放する」。
  モジュール内部で「途中結果を捨てる」意図を書くための補助関数で、
  呼び出し側のスコープの有無でタダの解放が二重にならないようにしてある。

## 注意

* スコープが所有しているテンソルを自分で `deinit` してはいけない（二重解放）。
  解放してよいのは「変換の戻り値」と「`untrack` した値」だけ。
* `Shape.initComptime` に実行時の値を渡すとコンパイラが落ちる。
  実行時形状は `Shape.init`。
* `Tensor.item()` は `numel() == 1` でないと panic する。
  単に読むなら `readF64(0)`。
