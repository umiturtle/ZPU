# 自動微分の中身（`prims` / `tape` / `autodiff`）

`transforms` を使うだけならこのページは読まなくてよい。ここには
「なぜ `grad` が速くも遅くも書けるのか」「新しい op に勾配をどう足すか」を書く。

## 使い方（規則を足す側）

1. `prims.Prim` に op を足す。
2. `ops` にカーネルを書く（最後に `ops.finish(prim, inputs, out, attrs)` を呼ぶ）。
3. `autodiff.contribution(node, k, g)` に **VJP 規則**（`k` 番目の入力に関する
   コタンジェント）を書く。前向きモードが要るなら `jvp` 用の規則も足す。

```zig
// VJP の例: y = exp(x) なら dx = g * exp(x)
.exp => blk: {
    const x = node.inputs[0];
    const e = try ops.exp(x);            // 前向きの値を再計算する（保存しない）
    defer ops.releaseTemp(e);
    break :blk try ops.mul(g, e);
},
```

## 実装意図

* **保存しないで再計算する。** テープは値を持たず、primitive と属性と入力の id だけを
  記録する（`tape.Node`）。VJP は必要な中間値をその場で作り直す。
  メモリが読みやすく、`jit` の定数畳み込みとも相性が良い。
  その代わり計算量は増えるので、`softmax` のように「安定な形で再計算できる」演算は
  その形で書く。

* **勾配は `Engine` が所有する。** `autodiff.Engine` は
  `id -> Tensor` のマップを持ち、`seed(t, g)` で加算合成する。
  既存の勾配があれば `add` で足して古い参照を解放、無ければそのまま格納する。
  マップの値は `retain()` されており、`Engine.deinit` が直接解放する
  （`releaseTemp` ではないのは、`grad` がスコープを開いているときに
  「スコープのものではない参照」を確実に落とすため）。

* **関連性の判定。** `run` は最初に `wrt` から逆向きに辿って
  「勾配が要るノード」を印付ける（`needed` マップ）。以降の掃引は
  そのマークが付いたノードだけを見る。`wrt` に関係しない枝は計算しない。
  ループの中では毎回 `tape.nodes.items[idx]` を読み直す —
  コタンジェントの計算自体が op を呼び、その op が同じテープにノードを
  追加して領域を再確保することがあるため。

* **tape はスタック。** `MAX_TAPE_DEPTH = 8` の固定長スタックで、
  `pushTape` / `popTape` / `setTape` と、丸ごと退避する `save` / `restore` がある。
  `grad` は「前向き計算を内側のテープに記録し、掃引はそのテープを外して行う」。
  外すのは、掃引自身の op がこれから読むテープに追記されないようにするため。
  `hvp` は逆に、掃引の op を**外側のテープに載せたい**ので
  `saved = context.save()` の後に `pushTape(&outer)` してから
  `valueAndGrad` を呼ぶ。

* **JVP（前向きモード）。** 接ベクトルを伝播させる規則は
  `prims` の属性から機械的に決まる部分が多く、
  `jvp` 変換は「入力の接ベクトルを `seed` して前向きに 1 回走査する」だけで済む。
  出力が少ないとき（スカラー出力など）は逆方向より速い。

* **対応していない演算ははっきり失敗させる。** 例えば `conv2d` の VJP は
  `Error.NotSupported` を返す。黙って間違った勾配を返すより、
  微分可能な形（`pad` + スライス + `matmul` の合成）で書き直すよう促す。

## テストの見方

`src/zpu/tests.zig` の勾配テストは 3 種類の根拠で書かれている。

* 解析解（`d/dx Σ exp(x²) = 2x·exp(x²)` など）
* 有限差分との比較（多種類の op を通した関数）
* 恒等式（`grad grad sum(x*x) == 2`、`grad grad quad == (3, 4)` など）

さらに `autodiff.zig` の中に「VJP 規則が全部コンパイルできる」テストがあり、
網羅漏れ（`switch` の網羅性）がその場でビルドエラーになる。
