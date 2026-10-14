# 線形代数（`linalg`）

## 使い方

```zig
// 行列の形からレイアウトを決める（行優先 / 列優先、転置の要否）
const layout = try zpu.linalg.matLayout(a, b);

const x = try zpu.linalg.solve(a, b);        // A x = b
const ai = try zpu.linalg.inv(a);            // A^-1
const d = try zpu.linalg.det(a);             // 行列式（スカラー）
const c = try zpu.linalg.cholesky(a, false); // 下三角
const y = try zpu.linalg.triangularSolve(l, b, false);

var lu = try zpu.linalg.luFactor(a);         // 分解を使い回す
defer lu.deinit();
const s = try zpu.linalg.luSolve(&lu, b);

const dp = try zpu.linalg.dot(u, v);
const nm = try zpu.linalg.norm(x, -1, false);
const tr = try zpu.linalg.trace(m);
const dg = try zpu.linalg.diag(v);
const ot = try zpu.linalg.outer(u, v);
const ct = try zpu.linalg.contract(a, b);    // 最後の軸を縮約
```

## 実装意図

* **エラーは値で返す。** 特異行列は `Error.Singular`。
  判定は `SINGULAR_TOL = 1e-12` のピボット比較で、
  “ゼロに近い” の閾値が 1 か所にまとまっている。
  NaN を返して後段で静かに壊れるより、呼び出し側に判断させる。

* **`det` は分解を使う。** 余因子展開ではなく、LU 分解の対角成分の積。
  途中でピボットが閾値を下回れば 0 を返す（特異）。

* **`solve` は勾配を持つ。** `x = A⁻¹b` の VJP は
  `-A⁻ᵀ (ḡ) A⁻ᵀ ...` ではなく、`solve(Aᵀ, g)` の形で書ける。
  行列の逆行列を明示的に作らないので数値的に安定で、`inv` を使うより速い。

* **転置とレイアウトの吸収。** `matLayout` が「a と b のどちらを
  転置して読むか」を決め、`matmul` / `solve` 系はそれに従って
  カーネルの走査順を変える。`toContiguous` のコピーを避けるための仕掛け。

* **`Lu` は再利用可能。** 同じ行列で右辺だけ変える使い方
  （反復解法、複数の目的関数）では `luFactor` を 1 回だけ呼んで
  `luSolve` を繰り返す方が速い。`Lu` は自分が確保した作業領域を
  `deinit` で解放する。

## テスト

`src/zpu/tests.zig` の `linalg with gradients` が
`solve`・`det`・`inv` の値を解析解と比較し、
さらに `solve` と `det` を微分して有限差分と突き合わせる。
`det([[3,1],[1,2]]) = 5`、`solve([[3,1],[1,2]], [9,8]) = [2,3]` は
手計算で確かめられる値としてテストに書いてある。
