# ZUI/ZAPP 開発メモ

> **このファイルは、プロジェクトの進捗、実装意図、次のステップを記録するものです。**
> **AIエージェントの引き継ぎや、人間の開発者の参考のために使用します。**
> **毎コミットで更新してください。**

---

## 最終更新日

2026-10-17

---

## プロジェクト概要

ZUI/ZAPPは、Zig言語で構築する**次世代UIライブラリ**です。

**ビジョン**: 「Zigでしか作れない、最高のUIライブラリ」
- Comptimeでフレームワークのコストをゼロに
- 明示的なパラメータ注入で外部から完全に制御可能
- エラー伝播で堅牢
- GPU高速レンダリング

**参考フレームワーク**: GPUI (Zed), WebRender, Vello, Clay, Slint

---

## アーキテクチャ設計

### 核心原則

1. **引数は落ちる、エラーは上がる**
   - Allocator, Io, Config はパラメータで注入
   - エラーは error union で伝播

2. **Comptimeで解決できることはComptimeで**
   - バックエンド、ウィジェット、品質レベルをcomptimeパラメータに
   - vtable不要、分岐ゼロ、未使用コード完全除去

3. **データは一方向（UDF）**
   - State↓ Event↑
   - bodyは純粋関数

### 関数の4層分類

| シグネチャ | 分類 | 例 |
|-----------|------|-----|
| 値 → 値 | **PURE** | `lerp()`, `distance()` |
| + Allocator | **COMPUTE** | `layout()`, `measure()` |
| + *T（可変） | **MUTATE** | `update()`, `push()` |
| + std.Io | **EFFECT** | `loadFile()`, `present()` |

### 2段階構成

- **Comptime層**: バックエンド特殊化、ウィジェット登録、品質パス、ID生成
- **Runtime層**: ユーザー状態、入力、レイアウト計算、レンダリング

---

## 現在の進捗

### 完了済み ✅

#### Phase 0: 基盤構築
- [x] 9つの疎結合モジュール（color, vec2, rect, id, theme, style, layout, animation, event_bus）
- [x] 95個の単体テスト
- [x] アーキテクチャ設計文書（ARCHITECTURE.md）
- [x] 詳細なロードマップ（ROADMAP.md）
- [x] マイルストーン定義（MILESTONES.md）
- [x] 開発ガイドライン（AGENTS.md）
- [x] CI/CD設定（GitHub Pages自動デプロイ）

#### 実装済みモジュール
- `src/zui/color.zig` — RGBA色、hex変換、補間
- `src/zui/vec2.zig` — 2Dベクトル演算
- `src/zui/rect.zig` — 矩形操作
- `src/zui/id.zig` — FNV-1aハッシュID
- `src/zui/theme.zig` — ビジュアルテーマ
- `src/zui/style.zig` — レイアウト設定
- `src/zui/layout.zig` — 基本的なレイアウト計算
- `src/zui/animation.zig` — イージング、スプリング
- `src/zui/event_bus.zig` — イベント処理

#### 既存機能（ZUI v1）
- 基本ウィジェット（button, label, checkbox, slider, etc.）
- ソフトウェアレンダリング
- 即座モードAPI
- ディスプレイリスト
- 基本的なレイアウト（row/col/box）

### 未実装 ❌

#### Phase 1: シーングラフ基盤（最優先）
- [ ] Arena Allocator — フレーム毎リセットの超高速アロケーター
- [ ] DisplayCommand — 描画コマンド型
- [ ] DisplayList — コマンドの集合体、バッチ処理
- [ ] Command Batcher — 類似コマンドの統合
- [ ] SceneNode — UIツリーのノード
- [ ] Dirty Tracking — 差分検出
- [ ] Software Renderer — DisplayList → ピクセル変換

#### Phase 2: アリーナレイアウトエンジン
- [ ] LayoutNode — レイアウト計算用ノード
- [ ] Flexbox Layout — CSS Flexbox実装
- [ ] Grid Layout — CSS Grid実装
- [ ] Layout Engine — 複数アルゴリズムの統合

#### Phase 3: GPUバックエンド
- [ ] GpuDevice — 4つのAPIを統一的に扱う抽象層
- [ ] WebGPU Backend
- [ ] Vulkan Backend
- [ ] Metal Backend
- [ ] DirectX 12 Backend
- [ ] Command Batch Layer（comptime制御）

#### Phase 4-10: 追加機能
- [ ] SDF Text System
- [ ] Widget Library（コンテナ、オーバーレイ、データ）
- [ ] Animation System
- [ ] Input & Gesture Enhancement
- [ ] Theme & Style System
- [ ] Image Formats
- [ ] Developer Tools

---

## 実装意図と背景

### なぜZigなのか？

1. **Comptime** — Rustのtrait、C++のtemplateを超えるメタプログラミング
2. **明示的パラメータ** — Allocator, Io, Configを明示的に渡す設計
3. **エラー伝播** — error unionで堅牢なエラーハンドリング
4. **ゼロコスト抽象** — vtable不要、完全インライン展開
5. **手続き型UI** — if/for/switchがそのままUIロジック

### なぜこのアーキテクチャなのか？

**2026年のUIフレームワーク調査結果**:
- GPUI/Zed: GPU Scene Graphで120fps達成
- WebRender: Display List + Batchingで大規模UIに対応
- Vello: Compute Shaderで高速2Dレンダリング
- Clay: Arena Allocatorで<100μsのレイアウト
- Slint: コンパイル時最適化でランタイムコストゼロ

**Zigの独自性**:
- Comptimeで全ての最適化をコンパイル時に実行
- 他の言語では実現できない「フレームワークコストゼロ」を実現

### GPUコマンドバッチング層が必要な理由

**問題**:
```
Native Desktop: 1000 draw commands → 1-10ms（許容）
Web/WASM:       1000 draw commands → 10-500ms（致命的）
```

**解決策**:
```zig
const WebConfig = .{ .enabled = true };   // 積極的バッチング
const NativeConfig = .{ .enabled = false }; // ゼロオーバーヘッド

// comptimeで制御 → 不要なら完全に消滅
```

**結果**:
- Web: 1000 commands → 50 batched calls → 1-5ms ✅
- Native: 1000 commands → 1000 direct calls → 1-10ms ✅

---

## 次のステップ

### 最優先タスク

**Task 1.1: Arena Allocator**（Phase 1の最初）

**ゴール**: フレーム毎にリセットできる超高速アロケーター

**実装方針**:
1. TDDで進める（テストファースト）
2. 以下の機能をテスト駆動で実装:
   - `Arena.init(capacity)` — 指定バイト数確保
   - `arena.alloc(size)` — メモリ割り当て
   - `arena.reset()` — 使用領域を0に戻す
   - オーバーフロー処理
3. パフォーマンス目標: 1000回alloc/reset が 1ms 以下

**ファイル**: `src/zui/arena.zig`

**依存**: なし（他のモジュールに依存しない）

---

## 開発ルール

### 必須

1. **AGENTS.mdに従うこと**
2. **実装前にARCHITECTURE.mdを読むこと**
3. **TDD or ベンチマーク駆動で進めること**
4. **テストはこれでもかというほど追加すること**
5. **ドキュメントも同時に更新すること**
6. **memory.mdも更新すること**

### コミット

- 1コミット1変更
- 明確なコミットメッセージ
- ドキュメント変更も含める

### テスト

- 各関数に5-10個のテストケース
- 正常系・異常系・エッジケースをカバー
- プロパティベーステスト
- パフォーマンステスト

---

## 技術的なメモ

### Zig 0.17の新機能

- `std.Io` — 明示的なI/O注入
- `std.process.Init` — Juicy Main パターン
- comptimeの改善
- 型解決の再設計

### 既存のコードベース

**ZUI v1（既存）**:
- `src/zui/ui.zig` — メインAPI（1412行）
- `src/zui/types.zig` — 基本型（Vec2, Rect, Color等）
- `src/zui/frame.zig` — DisplayList
- `src/zui/context.zig` — 状態管理
- `src/zui/render.zig` — ソフトウェアレンダラー

**ZUI v2（新規、疎結合モジュール）**:
- `src/zui/color.zig`, `vec2.zig`, `rect.zig`, etc.
- これらは既存の`types.zig`とは独立
- 将来的にv1と統合する予定

### ビルドシステム

- `zig build test` — テスト実行
- `zig build doc` — ドキュメント生成（Markdown → HTML）
- `zig build bench` — ベンチマーク
- `zig build run-<name>` — サンプル実行

### CI/CD

- GitHub Actions で自動テスト
- mainブランチにプッシュするとGitHub Pagesに自動デプロイ
- Zig 0.17.0 を使用

---

## 問題と解決策

### 発生した問題

1. **Zigがインストールされていない**
   - 解決策: `tools/install_zig.sh` を使用（作成済み）
   - または手動でダウンロード

2. **既存のコードと新しいモジュールの競合**
   - 現状: `types.zig`（v1）と`color.zig`など（v2）が共存
   - 解決策: 将来的にv1をv2ベースに書き換え

3. **テスト環境の制約**
   - 現状: Arena.ai環境ではZigがプリインストールされていない
   - 解決策: インストールスクリプトを用意

### 今後の課題

1. **GPUバックエンドの実装**
   - 4つのAPI（WebGPU/Vulkan/Metal/DX12）の抽象化
   - プラットフォーム固有の最適化

2. **パフォーマンス最適化**
   - Arena Allocatorの高速化
   - レイアウトエンジンの最適化
   - GPUバッチングの最適化

3. **テストカバレッジの向上**
   - 既存モジュールのテスト追加
   - 統合テストの充実
   - スナップショットテストの拡充

---

## リソース

### ドキュメント

- `AGENTS.md` — 開発ガイドライン
- `ROADMAP.md` — 詳細なロードマップ
- `docs/zui/ARCHITECTURE.md` — アーキテクチャ設計
- `docs/zui/MILESTONES.md` — マイルストーン定義
- `docs/zui/MODULAR_ARCHITECTURE.md` — モジュール構成

### 参考リンク

- [Zig 0.17 Release Notes](https://ziglang.org/download/0.17.0/release-notes.html)
- [GPUI (Zed)](https://github.com/zed-industries/zed)
- [WebRender](https://github.com/servo/webrender)
- [Vello](https://github.com/linebender/vello)
- [Clay](https://github.com/nicbarker/clay)

---

## 変更履歴

### 2026-10-17
- AGENTS.md作成
- architecture.md更新（関数分類、Zigイディオム追加）
- roadmap.md更新（GPUバッチング層追加）
- CI設定更新（Zig 0.17.0）
- memory.md作成

---

**このファイルは常に最新の状態に保ってください。**
**次の実装者は、このファイルを読んで現状を把握してから作業を開始してください。**
