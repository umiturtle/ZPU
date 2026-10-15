# ZAPP クロスプラットフォーム一貫性テストガイド

## 概要

このドキュメントでは、ZAPPのクロスプラットフォーム一貫性をテストする方法について説明します。Linux (X11) と Web (Wasm) の両環境で同じUIが正しくレンダリングされることを検証します。

## テストの目的

1. **環境間一貫性**: Linux X11 と Web (Wasm) で同じレンダリング結果を得られるか
2. **バージョン間一貫性**: 異なるバージョン間でレンダリング結果が安定しているか
3. **回帰検出**: 変更によって意図しないレンダリング変化が発生していないか

## 前提条件

### Linux (X11) 環境
```bash
# 依存パッケージ
sudo apt-get install libx11-dev imagemagick

# Zig 0.13.0
wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz
tar -xf zig-linux-x86_64-0.13.0.tar.xz
export PATH="$(pwd)/zig-linux-x86_64-0.13.0:$PATH"
```

### Web (Wasm) 環境
```bash
# Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Node.js（Webテスト実行用）
sudo apt-get install nodejs

# HTTP サーバー（テスト用）
sudo apt-get install python3
```

## テストの実行

### 1. リファレンス画像の生成

まず、既知の正しいレンダリング結果を生成します：

```bash
cd /home/user/ZPU
zig build run-reference-generator
```

これにより、`snapshots/reference/0.2.0/` にリファレンス画像が生成されます。

### 2. 自動テストの実行

クロスプラットフォームテストを自動的に実行します：

```bash
chmod +x test_cross_platform.sh
./test_cross_platform.sh
```

このスクリプトは以下を行います：
- Linux X11 で各シナリオをレンダリング
- Web (Wasm) で各シナリオをレンダリング
- 結果を比較して差分を検出

### 3. 手動テスト

#### Linux X11 でのテスト

```bash
# スナップショットランナーをビルド
zig build run-snapshot-runner -Dplatform=linux_x11

# 実行
./zig-out/bin/snapshot-runner
```

#### Web (Wasm) でのテスト

```bash
# WebAssembly にビルド
zig build-exe test/snapshot_runner.zig \
  --dep zapp zgf zpu \
  -Mroot=test/snapshot_runner.zig \
  -Mzapp=src/zapp.zig \
  -Mzgf=src/zgf.zig \
  -Mzpu=src/zpu.zig \
  -target wasm32-freestanding \
  -O ReleaseSmall \
  -femit-bin=snapshots/web_runner.wasm

# HTTP サーバーを起動
python3 -m http.server 8000

# ブラウザで http://localhost:8000/snapshots/web_test.html を開く
```

## テストシナリオ

### 1. basic_ui

基本的なUI要素のレンダリングをテストします：
- ラベル
- ボタン
- スライダー
- チェックボックス

**期待される結果**: すべての要素が正しい位置と色でレンダリングされる

### 2. interactive

インタラクティブ要素のテスト：
- テキスト入力フィールド
- ドロップダウンメニュー

**期待される結果**: 要素が正しく表示され、ボーダーと背景が適切にレンダリングされる

### 3. animation

アニメーション要素のテスト：
- プログレスバー

**期待される結果**: 指定された進捗率が正しく表示される

## スナップショット比較

### 比較アルゴリズム

各ピクセルを比較し、以下の指標を計算します：

1. **差分ピクセル数**: 許容誤差（1%）を超える差分があるピクセル数
2. **最大ピクセル差分**: すべてのピクセルチャンネル（RGBA）の最大差分
3. **平均ピクセル差分**: すべてのピクセルの平均差分
4. **構造的類似度 (SSIM)**: 画像の構造的類似度（0-1）

### 合格基準

- 構造的類似度 > 0.95（95%以上のピクセルが一致）
- 平均ピクセル差分 < 0.01（1%未満の平均差分）

### 差分の可視化

差分が見つかった場合、`snapshots/diffs/` に差分画像が生成されます：

```bash
# ImageMagick を使用して差分を可視化
compare -metric RMSE \
  snapshots/linux_x11/0.2.0/basic_ui.ppm \
  snapshots/web/0.2.0/basic_ui.ppm \
  snapshots/diffs/basic_ui_diff.png
```

## ディレクトリ構造

```
snapshots/
├── reference/          # リファレンス画像（既知の正しい結果）
│   └── 0.2.0/
│       ├── basic_ui.ppm
│       ├── interactive.ppm
│       └── animation.ppm
├── linux_x11/          # Linux X11 環境でのテスト結果
│   └── 0.2.0/
│       ├── basic_ui.ppm
│       ├── interactive.ppm
│       └── animation.ppm
├── web/                # Web (Wasm) 環境でのテスト結果
│   └── 0.2.0/
│       ├── basic_ui.png
│       ├── interactive.png
│       └── animation.png
└── diffs/              # 差分画像
    ├── basic_ui_diff.png
    ├── interactive_diff.png
    └── animation_diff.png
```

## トラブルシューティング

### Linux X11 でテストが失敗する

**症状**: "No X11 display" エラー

**解決策**:
```bash
# DISPLAY環境変数を確認
echo $DISPLAY

# Xサーバーが起動しているか確認
ps aux | grep Xorg

# 仮想フレームバッファを使用
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### Web テストで画像が生成されない

**症状**: Canvas に何も表示されない

**解決策**:
1. ブラウザのコンソールでエラーを確認
2. WebAssembly モジュールが正しく読み込まれているか確認
3. CORS ポリシーの問題を確認（ローカルサーバーを使用）

### 差分が許容範囲を超える

**症状**: テストは失敗するが、見た目は同じに見える

**原因**: 
- アンチエイリアスの違い
- 浮動小数点演算の精度差
- レンダリング順序の違い

**解決策**:
1. 許容誤差を調整（`test/snapshot_runner.zig` の `tolerance` 変数）
2. リファレンス画像を再生成
3. レンダリングコードを確認

## 継続的統合（CI）

GitHub Actions で自動的にテストを実行する例：

```yaml
name: Cross-Platform Consistency Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Zig
        uses: goto-bus-stop/setup-zig@v2
        with:
          version: 0.13.0
      
      - name: Install X11 dependencies
        run: sudo apt-get install libx11-dev imagemagick
      
      - name: Generate reference images
        run: zig build run-reference-generator
      
      - name: Run cross-platform tests
        run: ./test_cross_platform.sh
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: test-results
          path: snapshots/
```

## ベストプラクティス

### 1. リファレンス画像の管理

- リファレンス画像はGitにコミットする
- バージョンごとにディレクトリを分ける
- 大きな変更後はリファレンス画像を再生成する

### 2. テストの頻度

- 各コミットで自動テストを実行
- リリース前に完全なクロスプラットフォームテストを実行
- 定期的にリファレンス画像を更新

### 3. 許容誤差の設定

- プラットフォーム間の差異を考慮して許容誤差を設定
- 一般的な差異：
  - アンチエイリアス: 1-2%
  - 色空間変換: 0.5-1%
  - 浮動小数点精度: 0.01-0.1%

### 4. 差分の分析

- 差分が見つかった場合は、目視で確認する
- 意図的な変更の場合はリファレンス画像を更新
- 意図しない変更の場合はコードを修正

## 付録

### 画像フォーマット

- **PPM**: Linux X11 環境で使用（シンプルで依存関係なし）
- **PNG**: Web 環境で使用（ブラウザネイティブサポート）

### 変換コマンド

```bash
# PPM → PNG
convert input.ppm output.png

# PNG → PPM
convert input.png output.ppm

# 画像サイズ確認
identify image.ppm
```

### 参考リンク

- [Zig ドキュメント](https://ziglang.org/documentation/0.13.0/)
- [Emscripten ドキュメント](https://emscripten.org/docs/)
- [ImageMagick 使用方法](https://imagemagick.org/script/command-line-tools.php)

## 結論

このテストスイートにより、ZAPPのクロスプラットフォーム一貫性を自動的に検証できます。定期的にテストを実行し、リファレンス画像を更新することで、レンダリングの品質と一貫性を維持できます。
