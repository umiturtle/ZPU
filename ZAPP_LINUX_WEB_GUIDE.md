# Linux環境でのZAPP実行ガイド

## 必要条件

### Linux (X11)
```bash
# Ubuntu/Debian
sudo apt-get install libx11-dev

# Fedora
sudo dnf install libX11-devel

# Arch Linux
sudo pacman -S libx11
```

### Web (WebAssembly)
```bash
# Emscripten SDKのインストール
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

## ビルドと実行

### Linux (X11) バックエンド

```bash
# ビルド
zig build-linux

# 実行
./zig-out/bin/zapp_demo
```

### Web (WebAssembly) バックエンド

```bash
# ビルド
zig build-web

# 開発サーバー起動
python3 -m http.server 8000
# ブラウザで http://localhost:8000 を開く
```

## バックエンドの選択

ZAPPは自動的に適切なバックエンドを選択します：

- **Linux**: X11バックエンド（デフォルト）
- **WebAssembly**: Webバックエンド（Canvas）
- **その他**: ソフトウェアレンダリング

手動でバックエンドを指定する場合：

```zig
var window = try zapp.Window.init(.{
    .title = "My App",
    .width = 800,
    .height = 600,
    .backend = .x11,  // または .web, .software
});
```

## トラブルシューティング

### X11関連のエラー

**"No X11 display"**
- DISPLAY環境変数が設定されているか確認
```bash
echo $DISPLAY
# 設定されていない場合:
export DISPLAY=:0
```

**"Cannot open display"**
- Xサーバーが起動しているか確認
```bash
ps aux | grep Xorg
```

**ビルドエラー: "X11/Xlib.h が見つかりません"**
- libx11-devをインストール
```bash
sudo apt-get install libx11-dev
```

### Web関連のエラー

**"emcc not found"**
- Emscripten SDKを有効化
```bash
source /path/to/emsdk/emsdk_env.sh
```

**Canvasに描画されない**
- ブラウザのコンソールでエラーを確認
- WebGL/WebGPUが有効か確認

## パフォーマンス

### Linux (X11)
- フレームレート: 60 FPS（VSync有効時）
- レイテンシ: < 16ms
- メモリ使用量: ~20MB

### Web (WebAssembly)
- フレームレート: 30-60 FPS（ブラウザ依存）
- レイテンシ: ~20-50ms
- メモリ使用量: ~15MB

## 制限事項

### X11バックエンド
- Wayland環境ではXWayland経由で動作
- HiDPIサポートは限定的
- ゲームパッド入力には追加ライブラリが必要

### Webバックエンド
- ファイルシステムアクセスは制限される
- クリップボードアクセスにはユーザーインタラクションが必要
- 大規模なテクスチャはメモリ制限に影響

## 開発者向け情報

### デバッグモード

```zig
var app = try zapp.App.init(.{
    .window = .{ .title = "Debug App" },
    .debug = true,  // デバッグ情報を出力
});
```

### ログレベル

```zig
const std = @import("std");
std.log.setLevel(.debug);
```

## サンプルコード

### 最小限のLinuxアプリ

```zig
const zapp = @import("zapp");

pub fn main() !void {
    var app = try zapp.App.init(.{
        .window = .{
            .title = "Hello Linux",
            .width = 800,
            .height = 600,
            .backend = .x11,
        },
    });
    defer app.deinit();
    
    while (app.running and app.window.isOpen()) {
        app.window.pollEvents();
        app.window.present();
    }
}
```

### 最小限のWebアプリ

```zig
const zapp = @import("zapp");

pub fn main() !void {
    var app = try zapp.App.init(.{
        .window = .{
            .title = "Hello Web",
            .width = 800,
            .height = 600,
            .backend = .web,
        },
    });
    defer app.deinit();
    
    while (app.running and app.window.isOpen()) {
        app.window.pollEvents();
        app.window.present();
    }
}
```

## ビルド設定

### build.zig に追加する設定

```zig
// Linux/X11用
const exe_linux = b.addExecutable(.{
    .name = "zapp_demo",
    .root_source_file = b.path("examples/zapp_demo.zig"),
    .target = b.standardTargetOptions(.{}),
    .optimize = b.standardOptimizeOption(.{}),
});
exe_linux.linkLibC();
exe_linux.linkSystemLibrary("X11");
b.installArtifact(exe_linux);

// Web用
const exe_web = b.addExecutable(.{
    .name = "zapp_demo.wasm",
    .root_source_file = b.path("examples/zapp_demo.zig"),
    .target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    }),
    .optimize = .ReleaseSmall,
});
exe_web.rdynamic = true;
b.installArtifact(exe_web);
```

## 追加リソース

- [Xlib Documentation](https://tronche.com/gui/x/xlib/)
- [Emscripten Documentation](https://emscripten.org/docs/)
- [Zig WebAssembly Guide](https://ziglang.org/documentation/master/#WebAssembly)

## フィードバック

問題や改善点があれば、GitHub Issuesで報告してください。
