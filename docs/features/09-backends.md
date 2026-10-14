# 実行バックエンド（`zbackend`）

## 使い方

```zig
const zpu = @import("zpu");
const zb = zpu.zbackend;

// 1. このマシンに何があるかを聞く（探索結果はキャッシュされる）
const report = zpu.backends();
std.debug.print("best compute backend: {s}\n", .{report.best().name()});
var buf: [16]u8 = undefined;
for (zb.ALL) |b| {
    const info = report.detail(b);
    std.debug.print("{s: <7} available={} devices={d} api={s}\n", .{
        b.name(), info.available, info.device_count, info.versionString(&buf),
    });
    for (0..info.device_count) |i| std.debug.print("  - {s}\n", .{info.deviceName(i)});
}

// 2. カーネルは 1 回だけ書く。Zig の本体と、バックエンドごとのシェーダ
const Scale = struct {
    fn body(ctx: *zb.KernelCtx) void {
        const data = ctx.buffers[0].as(f32);
        data[ctx.id] = data[ctx.id] * 2.0 + 1.0;
    }
};
const kernel = zb.Kernel{
    .name = "scale_offset",
    .body = Scale.body,
    .source = .{
        .wgsl =
        \\@group(0) @binding(0) var<storage, read_write> data: array<f32>;
        \\@compute @workgroup_size({threads})
        \\fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {
        \\    let i = gid.x;
        \\    data[i] = data[i] * 2.0 + 1.0;
        \\}
        ,
        .msl = ..., .glsl = ..., .hlsl = ...,
    },
    .spirv = null, // Vulkan は SPIR-V が要る（下の「注意」を参照）
};

// 3. 実行する。GPU が無ければ CPU の Zig 本体が走る
var buffers = [_]zb.Buffer{.{ .ptr = @ptrCast(data.ptr), .len = data.len * @sizeOf(f32) }};
var ctx = zb.KernelCtx{ .buffers = &buffers, .count = data.len, .threads = 64, .alloc = alloc };
try zb.dispatch(.cpu, kernel, &ctx);              // 逐次
try zb.dispatchThreaded(.cpu, kernel, &ctx, 8);   // 8 スレッドに分割

// 4. デバイスを自分で持つこともできる（毎回開かないので速い）
if (try zb.openDevice(.vulkan)) |dev| {
    var device = dev;
    defer device.deinit();
    try zb.dispatchOn(&device, kernel, &ctx);
}
```

`zig build run-gpu-probe` は上の表を出力し、同じカーネルを 4 つのシェーディング
言語に落とし、CPU のスレッド実行をシリアル実行と突き合わせて検証する。さらに
各バックエンドの**デバイスを実際に開いて**、そのカーネルのコードを持っている
バックエンドでは実行して数値を検査する（GPU の無いマシンでは「not available」と
表示され、CPU だけが実行する — これは失敗ではなく、そのマシンの正しい答え）。

## 実装意図

* **検出は本物、しかし検出だけ。** `probe()` は `std.DynLib` でローダを開き、
  何があるかをランタイムに聞く。Vulkan なら `vkEnumerateInstanceVersion` の後、
  実際に `vkCreateInstance` して `vkEnumeratePhysicalDevices` でデバイスを列挙し、
  `VkPhysicalDeviceProperties` からデバイス名と API バージョンを読み、
  インスタンスを破棄する。WebGPU / Metal / D3D12 は必要なエントリポイント
  （`wgpuCreateInstance` / `MTLCreateSystemDefaultDevice` / `D3D12CreateDevice`）を
  1 つ解決して、そのライブラリのパスを報告する。
  SDK もヘッダもリンク時依存も不要で、ABI 構造体は自前で宣言し、
  エントリポイントは名前で読み込む。

* **無いことは正常な答え。** その場合は `available=false`、`device_count=0`、
  `open()` は `null` を返し、`best()` は CPU に落ちる。
  GPU の無い CI は、GPU のあるノート PC と同じコードの別の枝を通る。

* **1 つのカーネル、複数の言語。** `Kernel` はシェーディング言語ごとのソースと
  任意の SPIR-V バイト列を持つ。`lower()` が置換するのは `{name}` と `{threads}`
  だけで、それ以外の中括弧はそのまま通す（実際のシェーダは中括弧だらけなので）。
  `canRun(backend)` が「このカーネルを実行できるバックエンド」を答える
  （CPU は常に可、GLSL 経路は SPIR-V が必要）。

* **CPU バックエンドが参照実装。** `dispatch` は呼び出し空間を素直に回し、
  `dispatchThreaded` は同じ空間をストライドで分割する — GPU の dispatch と
  同じ分割の仕方で、ワーカーごとにコンテキストをコピーし、失敗は
  `Error.KernelFailed` に集約する。スレッド実行とシリアル実行の結果が違えば、
  そのカーネルは並列に書けていない。例題がその同値を検査している。

* **聞かれる情報は固定長バッファに載せる。** `Info` はローダのパス、API 版、
  最大 8 個のデバイス名を固定配列で持ち、「このマシンには何があるのか」に
  割り当てなしで答えられる。ドキュメント・例題・ベンチがこの問いを立てる。

* **デバイスは 3 段階で扱う。** `probe()`（何があるか）→ `open()`（ドライバの
  記述）→ `openDevice()`（生きた接続）。`Device` はバックエンドごとのタグ付き
  共用体で、`.cpu` / `.vulkan` / `.webgpu` は実行でき、`.metal` / `.d3d12` は
  まだ席だけ。`dispatchOn(&dev, kernel, ctx)` がそのデバイスで実行し、
  `dispatch(backend, kernel, ctx)` は「開けるならデバイス、開けなければ CPU」を
  自分で決める。`device(backend)` はプロセス単位で 1 回だけ開いて使い回す
  （インスタンスとデバイスの作成はミリ秒単位なので、ディスパッチごとに開くと
  計算そのものが誤差になる）。

* **キャッシュはロックフリー。** スロットは
  `unopened → opening → opened | unavailable` の 4 状態で、`cmpxchg` で
  1 スレッドだけが開く。負けたスレッドは待たずにその回のディスパッチを CPU で
  実行する — 数ミリ秒待つより、正しい答えを今返すほうが良い。

* **Vulkan はローダだけを相手にする。** ヘッダも SDK もリンク時依存も無し。
  `VkInstanceCreateInfo` などの ABI 構造体をこのリポジトリ内で宣言し、
  `vkGetInstanceProcAddr` から全エントリポイントを名前で解決する。
  デバイスを作ったら、計算用のキュー族を探し、ホスト可視でコヒーレントな
  メモリ種を選び、バッファは**生成時にマップしたまま**持つ（アップロードも
  読み戻しも `memcpy`）。ディスパッチごとにシェーダモジュール・記述子
  セットレイアウト・パイプライン・記述子セット・コマンドバッファ・フェンスを
  作っては壊す。`count` と `threads` は push constant（16 バイト、
  `{ u32 count; u32 threads; u32 pad[2]; }`）で渡す。

* **WebGPU は `webgpu.h` の流儀に合わせる。** Dawn の `wgpuGetProcAddress`
  でも、wgpu-native のようにシンボルを直接公開する実装でも動くように、
  `Loader` のフィールド名と `wgpu*` のシンボル名を 1 つの表にまとめ、
  **足りないエントリがあれば comptime でビルドが落ちる**
  （実行時の文字列処理も、ましてや計測用のコードも要らない）。
  文字列は NUL 終端でなく `WGPUStringView`（長さ付き）で渡し、
  コールバックは `AllowProcessEvents` モードにして `wgpuInstanceProcessEvents`
  で回す。バッファは `MappedAtCreation` で作り、書いてから unmap、
  読み戻しは `wgpuBufferMapAsync` + ポーリング。

* **カーネルの ABI は 1 つだけ。** `@group(0) @binding(i)` が
  `KernelCtx.buffers[i]` で、**どのバッファも storage buffer** として束ねる。
  エントリポイントは `main` 固定（`Kernel.lower` が `{threads}` を埋め込むので、
  `@workgroup_size` はシェーダ側に持つ）。Vulkan の push constant も含め、
  シェーダ側から見える契約はこの 2 つだけ。

## 現状

| バックエンド | 検出 | 実行 |
| --- | --- | --- |
| CPU | 常に | `dispatch` / `dispatchThreaded` / `dispatchOn` |
| Vulkan | ローダ + インスタンス + 物理デバイス列挙 | **実装済み**: デバイス・計算キュー・コマンドプール、ホスト可視メモリ、SPIR-V パイプライン、SSBO、push constant、フェンス待ち |
| WebGPU | 実装のロード（Dawn / wgpu-native） | **実装済み**: アダプタ要求、デバイス要求、WGSL モジュール、ストレージバッファ、bind group、compute pass、読み戻し |
| Metal | `MTLCreateSystemDefaultDevice` | 予定 |
| Direct3D 12 | `D3D12CreateDevice` | 予定 |

Vulkan と WebGPU を先に完成させる。残り 2 つは検出コードと `Backend.ALL` の席を
保ったまま待つ。

**どこで何を検証しているか。** この開発環境には GPU が無い（`libvulkan` も
Dawn も無く、`/dev/dri` も無い）。だから GPU 経路そのものは実機で
`zig build run-gpu-probe` を走らせて確かめる — このコマンドは各バックエンドの
デバイスを開き、カーネルのコードを持っているバックエンドで実行して数値を検査する。
GPU が無い環境で検証できるのは、(1) ローダ探索とインスタンス作成の成否、
(2) ABI 構造体のレイアウト（comptime のテストでサイズとオフセットを固定）、
(3) エントリポイント表の網羅性（comptime）、(4) 実行できないときに
`null` を返し CPU に落ちること、(5) CPU 実行の同値性、の 5 つ。

## 注意

* `Backend.ALL` は 5 つを固定順で並べるので、UI やログは例外分岐なしで回せる。
* 探索結果はスレッドごとのコンテキストにキャッシュされ、`zpu.deinit()` で捨てる。
  だから `zpu.backends()` はループの中で呼んでも安い。
* カーネルのソースはただの文字列。WGSL を知らないホストでも、
  知っているバックエンドに渡せる。
* **Vulkan は SPIR-V が必須**（`.spirv`）。GLSL ソースは記録として持てるが、
  SPIR-V への変換は Zig の外（`glslangValidator` など）で行い、その成果物を
  貼る — ビルド時に外部ツールを要求しないため。`.spirv` が無いカーネルは
  Vulkan では実行可能と判定されず、CPU に落ちる（`canRun` が正直に答える）。
* **WebGPU のエントリポイントは `main` 固定**で、`@workgroup_size` は
  `{threads}` としてソースに埋め込む。`ctx.threads` は計算領域の分割にだけ効く。
* **uniform は使わない。** 小さな定数も storage buffer として束ねる
  （`var<storage, read> scale: array<f32>;` のように）。バックエンドが全
  バッファを storage として記述子に書くためで、混ぜるとレイアウト不一致になる。
* **GPU 経路はバッファを毎回往復させる。** ディスパッチ 1 回ごとにデバイス
  メモリを確保・転送・解放するので、ホットループでは `dispatchOn` と常駐
  バッファを自分で持つこと（デバイス常駐テンソルは次の段階）。
* デバイスを閉じるのは `Device.deinit`（`openDevice` が返した値）か
  `zpu.shutdownDevices()`（プロセス内キャッシュ）。ディスパッチ中に閉じない。

* `Backend.ALL` は 5 つを固定順で並べるので、UI やログは例外分岐なしで回せる。
* 探索結果はスレッドごとのコンテキストにキャッシュされ、`zpu.deinit()` で捨てる。
  だから `zpu.backends()` はループの中で呼んでも安い。
* カーネルのソースはただの文字列。WGSL を知らないホストでも、
  知っているバックエンドに渡せる。
