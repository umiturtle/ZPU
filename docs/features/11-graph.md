# 記録（`graph`）

`zpu.graph` は `jit` の前半だけを取り出したもの。実行した op を平らな列として
記録し、`zpu.plan` がそれを計画に変換する。利用者が直接触ることは少ないが、
「何がどう記録されているか」を知りたいとき、あるいは自前の実行器を書きたい
ときの入口になる。

## 使い方

```zig
var g = zpu.graph.Graph.init(alloc);
defer g.deinit();

_ = zpu.graph.setActiveGraph(&g); // ここから記録が始まる
defer _ = zpu.graph.setActiveGraph(null); // 使い終わったら外す

const y = try zpu.tanh(x); // ops はグラフが立っていれば必ず記録する
_ = y;

try std.debug.print("{d} nodes\n", .{g.len()});
for (g.nodes.items) |n| {
    try std.debug.print("{s} ({d} inputs)\n", .{ @tagName(n.prim), n.n_inputs });
}

// 消費されているが生成されていない id = 関数の引数（と外から来た定数）
const externals = try g.externalIds(alloc);
defer alloc.free(externals);
```

* `Node` は `prim` と最大 3 本の `inputs`、`out`、`attrs` を持つ。
* `producedBy(id)` は「その id を生成したノードがあるか」を答える。
* 入れ子で使うときは `setActiveGraph` が返す**以前のポインタ**を戻す
  （`jit` の中の `grad` のように、記録の層が重なることがある）。

## 実装意図

* **平らな列にした。** 木ではなく `ArrayListUnmanaged(Node)` の並び。計画を作る側
  （`zpu.plan.build`）は出力から逆向きにたどるだけでよく、再帰も深さ制限も要らな
  い。Zig は再帰でスタックを食うので、平らな方が安全で読みやすい。
* **モードと独立に記録する。** `context.mode` が `.eager` でも、グラフが立って
  いれば op は記録する。これが `jit(grad(f))` を成り立たせている — backward の
  op も同じ経路を通るので、特別扱いが要らない。
* **id で参照する。** ポインタをキーにすると、解放後に同じアドレスへ来た別の
  テンソルと取り違える。`Tensor.id` は単調増加なので、解放後も再利用されない。
* **グラフが参照を 1 本持つ。** `record` が入出力を `retain()` し、
  `Graph.deinit` が 1 本ずつ返す。だからトレースを生んだ `Scope` が消えても、
  記録はそのまま残る（計画を後から組める）。
* **記録は best-effort。** ノードの追加が失敗しても握りつぶす（`catch {}`）。
  記録が無くても実行結果は変わらないので、メモリが苦しいときに計算まで落とす
  必要はない、という判断。
* **`MAX_INPUTS = 3`。** 現在の `prims` はすべて 1〜2 入力なので、3 本で足りる。
  4 本目以降が要る op を足すときはここを上げ、`plan.build` の写し方も合わせる。

## テストの見方

* `src/zpu/graph.zig` の `graph records and releases` が、記録・解放・
  リーク検査（`std.testing.allocator`）を 1 つで見ている。
* 計画への変換は `src/zpu/jit.zig` のテストが、`jit` 全体としては
  `src/zpu/tests.zig` と `zig build bench` の `jit(...)` ケースが見ている。
