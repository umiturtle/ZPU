# 画像とフィルタ（`zgf`）

## 使い方

```zig
const zgf = @import("zgf");

var img = try zgf.Image.init(alloc, 256, 256, 3);   // 幅, 高さ, チャネル数
defer img.deinit();

img.fill(.{ .r = 0.05, .g = 0.08, .b = 0.12 });
img.verticalGradient(.{ .r = 1, .g = 1, .b = 1 }, .{ .r = 0, .g = 0, .b = 0.4 });
img.drawRect(16, 16, 64, 64, .{ .r = 1, .g = 0.8, .b = 0 });
img.drawCircle(180, 128, 48, .{ .r = 0.9, .g = 0.2, .b = 0.3 });
img.drawLine(0, 0, 255, 255, .{ .r = 1, .g = 1, .b = 1 });
img.drawLineThick(0, 255, 255, 0, 4, .{ .r = 0.2, .g = 1, .b = 0.6 });

const p = img.getPixel(10, 20);

// フィルタはテンソルを返すので、そのまま zpu の世界とつながる
const soft = try img.blur(3);            // 分離型ガウス（σ = radius/2）
const edges = try img.sobel();
const crisp = try img.sharpen(1, 1.5);
const bright = try img.brightness(1.2);
const punchy = try img.contrast(1.4);

// 8 bit sRGB と、バイナリ PPM/PGM の読み書き
const bytes = try img.toPixels(alloc);
defer alloc.free(bytes);
try img.savePpm(init.io, "poster.ppm");
var loaded = try zgf.loadPpm(alloc, init.io, "poster.ppm");
defer loaded.deinit();
```

## 実装意図

* **画像はテンソルそのもの。** `Image` は `[H, W, C]`（0..1 の f32）の
  `zpu.Tensor` を包むだけ。だからフィルタは普通のテンソル式で書けて、
  ピクセル値の関数として `grad` で微分できるし、`vmap` も `jit` も効く。
  `zgf` がバイト列ではなく `zpu` の上から始まっている理由がこれ。

* **フィルタは合成で書く。** `blur` は 1 次元ガウス核（`gaussianKernel`、
  σ = radius/2）を作り、y 方向 → x 方向の 2 回に分けて適用する。
  `pad` で端を埋め、`sliceRanges` で窓を切り出し、核の係数を掛けて足す —
  分離型畳み込みを op の式として書いている。`sobel` も 3×3 の 2 つの演算子で
  畳み込んで勾配の大きさを返すだけ。

* **画像はスコープより長生きする。** `Image.init` は自分のテンソルに
  `zpu.untrack` を呼ぶので、どのスコープで作られても所有者は `img.deinit()` の
  1 人だけ。フィルタが作る中間テンソルは `zpu.releaseTemp` に渡す
  （スコープの中では何もしない、外では解放する）。

* **色は型で持つ。** `Color` は f32 の RGBA。モノクロ画像では輝度を書き込み、
  RGB ではアルファを捨てる。`toPixels` / `fromPixels` が 8 bit sRGB との
  変換で、`toPpm` / `savePpm` / `loadPpm` は依存ゼロで済む最小の形式（P5/P6）を
  話す。

* **標準ライブラリ以外を使わない。** 書き出しは
  `std.Io.Dir.cwd().createFile(io, path, .{})` と `std.fmt` のヘッダ、
  読み込みは `readPositionalAll` 1 回とヘッダのトークン分割だけ。
  zlib が要る PNG はマイルストーン上の課題で、まだツリーには無い。

## 注意

* 描画はアンチエイリアス無し（ピクセルを立てるか立てないか）。
  `drawLine` は太さ 1 ピクセル、`drawLineThick` は同じ Bresenham 経路に
  正方形ブラシを押し当てる。
* 範囲外の座標はエラーにせず無視する。画面の外にはみ出す描画が安全に書ける。
* フィルタの戻り値は「テンソル」であって「画像」ではない。続けて描画したいなら
  画像側に戻す必要がある。

## テスト

`src/zgf.zig` の 6 つのテストが、生成と画素アクセス、8 bit の往復、
小さな P6 ファイルのバイト列、フィルタの振る舞い（一定色はぼかしても一定、
輪郭は Sobel に現れる、明度とコントラストのクリップ）、描画プリミティブ、
ガウス核の正規化と対称性を確認する。すべて `std.testing.allocator` の下で走る。
