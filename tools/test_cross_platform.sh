#!/bin/bash
# ZAPP クロスプラットフォーム一貫性テスト
# Linux (X11) と Web (Wasm) の両方でテストを実行し、結果を比較する

set -e

echo "=== ZAPP クロスプラットフォーム一貫性テスト ==="
echo ""

# Zigのパスを確認
if ! command -v zig &> /dev/null; then
    echo "エラー: Zigが見つかりません"
    echo "Zig 0.13.0をインストールしてください:"
    echo "  wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz"
    echo "  tar -xf zig-linux-x86_64-0.13.0.tar.xz"
    echo "  export PATH=\"\$(pwd)/zig-linux-x86_64-0.13.0:\$PATH\""
    exit 1
fi

echo "Zig バージョン: $(zig version)"
echo ""

# スナップショットディレクトリを作成
mkdir -p snapshots/linux_x11/0.2.0
mkdir -p snapshots/web/0.2.0
mkdir -p snapshots/reference/0.2.0
mkdir -p snapshots/diffs

# X11の依存関係を確認
if [ "$(uname)" = "Linux" ]; then
    echo "Linux環境を検出しました"
    if ! dpkg -l | grep -q libx11-dev; then
        echo "警告: libx11-devがインストールされていません"
        echo "インストール: sudo apt-get install libx11-dev"
    fi
fi

# テスト対象のシナリオ
SCENARIOS=("basic_ui" "interactive" "animation")

echo ""
echo "=== Linux (X11) テスト ==="
echo ""

for scenario in "${SCENARIOS[@]}"; do
    echo "シナリオ: $scenario"
    
    # Linux (X11) でテスト実行
    echo "  Linux X11 でテスト実行中..."
    zig build run-snapshot-runner \
        -Dplatform=linux_x11 \
        -Dscenario=$scenario \
        -Doutput=snapshots/linux_x11/0.2.0/${scenario}.ppm \
        2>&1 | grep -E "(✓|✗|エラー)" || true
    
    if [ -f "snapshots/linux_x11/0.2.0/${scenario}.ppm" ]; then
        echo "  ✓ Linux X11 スナップショット保存完了"
    else
        echo "  ✗ Linux X11 スナップショット生成失敗"
    fi
    echo ""
done

echo ""
echo "=== Web (Wasm) テスト ==="
echo ""

# WebAssembly用にビルド
echo "WebAssembly バイナリをビルド中..."
zig build-exe test/snapshot_runner.zig \
    --dep zapp zgf zpu \
    -Mroot=test/snapshot_runner.zig \
    -Mzapp=src/zapp.zig \
    -Mzgf=src/zgf.zig \
    -Mzpu=src/zpu.zig \
    -target wasm32-freestanding \
    -O ReleaseSmall \
    -femit-bin=snapshots/web_runner.wasm \
    2>&1 | grep -E "(✓|✗|エラー)" || true

if [ -f "snapshots/web_runner.wasm" ]; then
    echo "✓ WebAssembly バイナリ生成完了"
else
    echo "✗ WebAssembly バイナリ生成失敗"
fi

# Web実行（Node.jsまたはブラウザ）
if command -v node &> /dev/null; then
    echo ""
    echo "Node.js でWebテスト実行中..."
    
    for scenario in "${SCENARIOS[@]}"; do
        echo "  シナリオ: $scenario"
        
        # Web用のHTMLラッパーを作成
        cat > snapshots/web_test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>ZAPP Web Test</title>
    <style>
        body { margin: 0; padding: 20px; font-family: sans-serif; }
        canvas { border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>ZAPP Web Test - <span id="scenario"></span></h1>
    <canvas id="canvas" width="800" height="600"></canvas>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // JavaScript interop functions
        const jsInterop = {
            canvas_init: (width, height) => {
                canvas.width = width;
                canvas.height = height;
            },
            canvas_put_image: (dataPtr, len, x, y, width, height) => {
                const data = new Uint8Array(memory.buffer, dataPtr, len);
                const imageData = ctx.createImageData(width, height);
                for (let i = 0; i < len; i += 4) {
                    imageData.data[i] = data[i];     // R
                    imageData.data[i+1] = data[i+1]; // G
                    imageData.data[i+2] = data[i+2]; // B
                    imageData.data[i+3] = data[i+3]; // A
                }
                ctx.putImageData(imageData, x, y);
            },
            canvas_clear: (r, g, b, a) => {
                ctx.fillStyle = `rgba(${r*255}, ${g*255}, ${b*255}, ${a})`;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            },
            window_should_close: () => 0,
            get_timestamp: () => Date.now(),
        };
        
        let memory;
        
        // WebAssembly モジュールを読み込み
        WebAssembly.instantiateStreaming(fetch('web_runner.wasm'), {
            env: jsInterop
        }).then(result => {
            memory = result.instance.exports.memory;
            document.getElementById('scenario').textContent = new URLSearchParams(window.location.search).get('scenario') || 'unknown';
            
            // メイン関数を実行
            if (result.instance.exports.main) {
                result.instance.exports.main();
            }
            
            // スクリーンショットを保存
            setTimeout(() => {
                canvas.toBlob(blob => {
                    const scenario = new URLSearchParams(window.location.search).get('scenario');
                    const link = document.createElement('a');
                    link.download = `snapshots/web/0.2.0/${scenario}.png`;
                    link.href = URL.createObjectURL(blob);
                    link.click();
                });
            }, 1000);
        });
    </script>
</body>
</html>
EOF
        
        # シナリオごとにテスト実行
        # 注意: 実際のWeb実行にはブラウザ環境が必要
        echo "  Webテスト実行（ブラウザ環境が必要）"
        echo "  手動で実行: python3 -m http.server 8000"
        echo "  ブラウザで: http://localhost:8000/snapshots/web_test.html?scenario=$scenario"
    done
else
    echo "警告: Node.jsが見つかりません。Webテストをスキップします。"
fi

echo ""
echo "=== 一貫性比較 ==="
echo ""

for scenario in "${SCENARIOS[@]}"; do
    echo "シナリオ: $scenario"
    
    linux_file="snapshots/linux_x11/0.2.0/${scenario}.ppm"
    web_file="snapshots/web/0.2.0/${scenario}.png"
    
    if [ -f "$linux_file" ] && [ -f "$web_file" ]; then
        echo "  Linux X11 と Web の結果を比較中..."
        
        # 画像比較ツールで比較
        if command -v compare &> /dev/null; then
            compare -metric RMSE "$linux_file" "$web_file" "snapshots/diffs/${scenario}_diff.png" 2>&1 | \
                awk '{print "  差分 (RMSE): " $1}'
        else
            echo "  警告: ImageMagickのcompareコマンドが見つかりません"
            echo "  インストール: sudo apt-get install imagemagick"
        fi
    else
        echo "  比較可能なファイルがありません"
        [ ! -f "$linux_file" ] && echo "    - Linux X11 スナップショットなし"
        [ ! -f "$web_file" ] && echo "    - Web スナップショットなし"
    fi
    echo ""
done

echo ""
echo "=== テスト完了 ==="
echo ""
echo "結果:"
echo "  - Linux X11 スナップショット: snapshots/linux_x11/0.2.0/"
echo "  - Web スナップショット: snapshots/web/0.2.0/"
echo "  - 参照スナップショット: snapshots/reference/0.2.0/"
echo "  - 差分画像: snapshots/diffs/"
echo ""
echo "詳細な比較結果は snapshots/ ディレクトリを確認してください。"
