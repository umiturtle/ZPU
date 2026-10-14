# 実行バックエンド（`zbackend`）

## 使い方

```zig
const zpu = @import("zpu");

const report = zpu.backends();                 // 一度だけ探索して以降はキャッシュ
std.debug.print("compute backend: {s}\n", .{@tagName(report.best())});

for (zbackend.Backend.ALL) |b| {
    const info = report.detail(b);
    std.debug.print("{s: <7} available={} devices={d} api={s}\n", .{
        b.name(), info.available, info.device_count, info.versionString(&buf),
    });
    for (0..info.device_count) |i| std.debug.print("  - {s}\n", .{info.deviceName(i)});
}

// カーネルは 1 回だけ書く
const scale_offset = zbackend.Kernel{
    .name = "scale_offset",
    .workgroup = .{ .x = 64 },
    .wgsl =
    \\@group(0) @binding(0) var<storage, read_write> data: array<f32>;
    \\@compute @workgroup_size({threads})
    \\fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\    let i = gid.x;
    \\    data[i] = data[i] * 2.0 + 1.0;
    \\}
    ,
    .msl = ...,
    .glsl = ...,
    .hlsl = ...,
};

var ctx = zbackend.KernelCtx{
    .id = 0,
    .buffers = &.{ .{ .handle = data.buf, .role = .storage, .bytes = data.numel() * 4 } },
};
try zpu.dispatchKernel(.cpu, scale_offset, &ctx);            // シングルスレッド
try zpu.dispatchKernelThreaded(.cpu, scale_offset, &ctx, 8); // 8 スレッドで分割
```

`zig build run-gpu-probe` は上の表を出力し、同じカーネルを 4 つのシェーディング
言語に落とし、CPU のスレッド実行をシリアル実行と突き合わせて検証する。

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

## 現状

| バックエンド | 検出 | 実行 |
| --- | --- | --- |
| CPU | 常に | `dispatch` / `dispatchThreaded` |
| Vulkan | ローダ + インスタンス + 物理デバイス列挙 | 実装中（デバイス・キュー・パイプライン・SSBO） |
| WebGPU | `wgpuCreateInstance` | 実装中（バッファと compute pass） |
| Metal | `MTLCreateSystemDefaultDevice` | 予定 |
| Direct3D 12 | `D3D12CreateDevice` | 予定 |

Vulkan と WebGPU を先に完成させる。残り 2 つは検出コードと
`Backend.ALL` の席を保ったまま待つ。

## 注意

* `Backend.ALL` は 5 つを固定順で並べるので、UI やログは例外分岐なしで回せる。
* 探索結果はスレッドごとのコンテキストにキャッシュされ、`zpu.deinit()` で捨てる。
  だから `zpu.backends()` はループの中で呼んでも安い。
* カーネルのソースはただの文字列。WGSL を知らないホストでも、
  知っているバックエンドに渡せる。
