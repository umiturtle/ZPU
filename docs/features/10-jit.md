# 実行計画（`jit`）

`zpu.jit` は「一度記録して、何度も再生する」変換。Zig の関数を一度トレースして
`Plan`（`zpu.plan`）に直し、以降の呼び出しは計画だけを回す。`grad` と入れ子に
できるので、`jit(grad(f))` は backward を含む計画になる。

## 使い方

```zig
fn f(ud: ?*anyopaque, args: []const zpu.Tensor) anyerror!zpu.Tensor {
    _ = ud;
    const h = try zpu.matmul(args[0], args[1]);
    defer zpu.releaseTemp(h);
    return zpu.sumAll(zpu.tanh(h));
}

var j = zpu.jit(alloc, f, null); // まだトレースしていない
defer j.deinit();

const y = try j.call(&.{ x, w }); // 1 回目: 記録してから実行
defer y.deinit();

const y2 = try j.call(&.{ x2, w2 }); // 2 回目以降: 計画の再生だけ
defer y2.deinit();

_ = j.trace_count; // 1（トレースは 1 回だけ）
_ = j.numSteps(); // 畳み込みと除去の後に残ったステップ数
if (j.plan) |p| try p.describe(writer); // 計画の中身を人が読める形で
```

* 引数は**位置**で束ねられる。同じ形の別テンソルを渡すかぎりトレースし直さない。
* **形や dtype を変えるときは `j.trace(&args)` を明示的に呼ぶ。** 畳み込んだ定数は
  そのときの形で確定しているので、黙って別の形を流すと結果が合わない。
* `Plan` も直接使える（`zpu.plan.build(alloc, &graph, arg_ids, out_ids)`）。
  `Graph` を自分で立てて記録したものを計画にしたいときの入口。

## 何をするか

1. **記録** — `Graph` を立てて `f` を一度実行する。`ops` はモードに関係なく
   グラフが立っていれば記録するので、`grad` の中身（backward）も残る。
2. **計画** — `zpu.plan.build` がグラフを 4 段で畳む:
   * *不要コードの除去*: 要求した出力から逆向きにたどり、必要なノードだけ残す。
   * *引数の確定*: 呼び出し側が渡す id（`arg_ids`）は計算しない。
   * *定数畳み込み*: 引数に依存しないノードは**ここで一度だけ**評価し、
     結果を `Blob` として持つ（重み、リテラル、その上のスライスや転置も）。
   * *取り込み*: 残ったステップが読む「引数でも生成物でもない」テンソル
     （関数の中だけで作った定数）は `retain()` して計画が持ち続ける。
3. **再生** — `Plan.run` / `runAll` が引数を位置で束ね、`Step` の列を
   `ops.applyPrim` で実行する。戻り値は `retain()` 済みで、**呼び出し側が所有**
   （他の変換と同じ規約）。

## 実装意図

* **なぜ記録方式か。** Zig には Python のような AST フックが無く、comptime でも
  「任意の関数の中身」は覗けない。そこで *実行しながら記録する*（JAX と同じ
  発想）ことにした。op を 1 回書けば eager でも `grad` でも `jit` でも同じ実装が
  使える、という一点に効く。
* **記録はコンテキストに置く。** グラフはスレッドローカルの `context` に刺さる
  （`context.graph`）。op は「立っていれば書く」だけなので、`jit` の中の
  `grad`、`grad` の中の `jit` でも記録が漏れない。
* **ノードは id を運ぶ。** ポインタをキーにすると、解放されたテンソルの
  アドレスが再利用されたときに取り違える。`Tensor.id` は単調増加のカウンタで、
  解放後も再利用されない。
* **グラフ自身が参照を 1 本持つ。** Zig に GC は無い。トレースを生んだ `Scope`
  が消えても計画が生き残るよう、`record` が入出力を `retain()` し、`Graph.deinit`
  が 1 本ずつ返す。
* **畳み込みは記録を切って行う。** `evalFolded` はテープもグラフも外し、
  `.compiled` モードで評価する。だから畳み込みが自分自身を記録して増え続ける
  ことはない。
* **再生時の確保は 2 つだけ。** テンポラリ用の `Scope` と `id → Tensor` の
  ハッシュマップ。ステップの配列はビルド時に確定しているので、再生で伸びない。
* **計画は形を固定する。** 定数を焼き込む以上、形が変われば作り直しが要る。
  「勝手に再トレースする」より「合わない形は明示的に `trace` し直す」方が、
  どこで高くつくかが読みやすいと考えた。

## テストの見方

* `src/zpu/jit.zig` の `jit folds constants and eliminates dead code` が、
  折り畳み・除去・再生の一致（`trace_count` が増えないこと）を検査する。
* `src/zpu/tests.zig` に `jit` を含む変換のケースがある。
* `zig build bench` の `jit(grad MLP) f32` が、grad 全体を計画にしたときの
  スループットを測る（`--filter jit` で 1 本だけ回せる）。
