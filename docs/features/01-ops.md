# eager な演算（`ops`）

## 使い方

```zig
// 要素演算（単項）
const e = try zpu.exp(x);          // exp / expm1 / log / log1p / sqrt / rsqrt / square /
const r = try zpu.relu(x);         // recip / cbrt / sin / cos / tan / asin / acos / atan /
const t = try zpu.tanh(x);         // sinh / cosh / tanh / asinh / acosh / atanh / erf /
                                   // sigmoid / softplus / relu / floor / ceil / round / sign /
                                   // isnan / isinf / isfinite / logicalNot

// 要素演算（二項、ブロードキャストつき）
const s = try zpu.add(a, b);       // add / sub / mul / div / pow / maximum / minimum /
const p = try zpu.pow(a, b);       // atan2 / hypot / mod / eq / ne / lt / le / gt / ge /
                                   // logicalAnd / logicalOr / logicalXor
const k = try zpu.addScalar(x, 1); // スカラー版: addScalar / subScalar / mulScalar / divScalar
const z = try zpu.scalarLike(x, 2); // x と同じ dtype・形状 {1} の定数

// リダクション（axis は i32、負は末尾から。keepdims で形を保つ）
const total = try zpu.sumAll(x);
const col = try zpu.sum(x, 0, true);
const avg = try zpu.mean(x, -1, false);
const v = try zpu.variance(x, 0, false);
const sd = try zpu.stddev(x, 0, false);
const mx = try zpu.max(x, 1, false);      // max / min / prod / all / any
const am = try zpu.argmax(x, -1);         // argmax / argmin は i64
const sm = try zpu.softmax(x, -1);        // softmax / logSoftmax（数値安定版）

// 形状操作（できる限り view を返す）
const rs = try zpu.reshape(x, S.initComptime(&.{ 6 }));
const tr = try zpu.transpose(x, &.{ 1, 0 });
const bt = try zpu.broadcastTo(x, S.initComptime(&.{ 4, 2, 3 }));
const ex = try zpu.expandDims(x, 0);
const sq = try zpu.squeeze(x, 0);
const fl = try zpu.flatten(x);

// 添字・結合
const sl = try zpu.sliceRanges(x, &.{ .{ .start = 1, .end = 3 }, zpu.Range.all() });
const tk = try zpu.take(x, ids, 0);
const sc = try zpu.scatterAdd(x, ids, updates, 0);
const pd = try zpu.pad(x, &.{ .{ 1, 1 }, .{ 0, 0 } }, 0.0);
const cc = try zpu.concat(a, b, 0);
const st = try zpu.stack(a, b, 1);
const fl2 = try zpu.flip(x, 1);
const wh = try zpu.where(mask, p, q);
const cl = try zpu.clamp(x, -1, 1);
const lp = try zpu.lerp(a, b, 0.25);
const mm = try zpu.matmul(a, b);          // バッチ対応: 最後の 2 次元が行列
```

## 実装意図

* **すべての演算が 1 つの経路を通る。** 各 op は最後に `ops.finish(prim, inputs, out, attrs)` を
  呼び、そこが「テープに記録」「グラフに記録」「スコープへの登録」を一括で行う。
  だから VJP 規則を書けば `grad` が、グラフに乗れば `jit` が自動で使える。
  op を追加するときにやることは「カーネルを書く」「`prims.Prim` を足す」
  「VJP 規則を足す」の 3 つだけで、変換側には手を入れない。

* **ブロードキャストは 1 か所で解決する。** `broadcastPair(a, b)` が結果形状を計算し、
  `mapBinary` が「出力の添字から入力の添字を逆算する」形でループする。
  各演算にブロードキャストのコードを書かないので、規則のずれが起きない。
  なお `add` などの公開 API は**同じ rank** を要求し、rank が違う場合は
  `broadcastPair` を通す（内部の変換や `nn` はそのように呼ぶ）。

* **dtype の昇格。** `binaryOutDType` が 2 つの入力から出力 dtype を決める
  （`f64` > `f32` > 整数 > `bool`）。`scalarLike` は「相手と同じ dtype」のスカラーを作る。
  これが `f32` に固定だった時期があり、`f64` のテストで精度が落ちたので直した
  （`bool` のときだけ `f32` に落とす）。

* **view 優先。** `reshape` は「要求された形状が連続ストライドで実現できるか」を
  `reshapeTarget` で判定し、可能なら view を返す。`transpose`・`squeeze`・
  `broadcastTo`・`sliceRanges` も同様。view は計算量ゼロだが、
  **後段のカーネルがストライドを扱える必要がある**ので、
  読み出しは `readLogical`、書き込みや `matmul` は `toContiguous` を通す。

* **`matmul` は vmap の深さを保つ。** 各テンソルが `batch_depth` を持っており、
  `matmul` は「末尾 2 次元が行列、その手前がバッチ」として、
  vmap のスライス次元（深さぶんの先頭軸）を壊さずに `gemmTyped` /
  `gemmBatched` に渡すストライドを組み立てる。
  1 次元の入力は NumPy と同じく昇格・降格する（`squeeze(-1)` で戻す）。

* **数値の安定性は演算側の責任。** `softmax` は最大値を引いてから `exp` する、
  `logSoftmax` は `x - logsumexp` を使う、`variance` は 2 パス（平均を引いてから二乗平均）。
  これらは VJP を書くときに「前向きの値を再利用する」ためでもある。

## 注意

* `add` などは**同じ rank** を要求する。rank を跨ぐときは `broadcastPair` を使うか、
  片方を `reshape` して揃える。
* 軸は `i32`。`zpu.AXIS_LAST` は `-1`、`zpu.AXIS_NONE` は「リダクションしない」。
* リダクションの結果は `keepdims=false` で次元が落ちる。vmap の中で使うときは
  `keepdims=true` の方が事故が少ない。
