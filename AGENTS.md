# AGENTS.md — ZUI/ZAPP 開発ガイドライン

> **このドキュメントは、AIエージェントがZUI/ZAPPプロジェクトの実装を行う際のガイドラインです。**
> **人間の開発者も参照してください。**

---

## 1. 実装フロー

### 1.1 実装の進め方

実装を行う際は、必ず以下のフローに従うこと：

```
1. 📋 ROADMAP.md を確認
   └─ どのマイルストーンを実装するか選択

2. 📖 docs/zui/ARCHITECTURE.md を読む
   └─ アーキテクチャ設計を理解する
   └─ 実装する機能がどのレイヤーに属するか確認

3. 🔍 ROADMAP.md の詳細版を確認
   └─ 選択したマイルストーンの詳細な要件を確認
   └─ 実装すべき機能を明確にする

4. 📝 実装計画を立てる
   └─ 詳細版をもっと砕いて、小さなタスクに分解
   └─ 各タスクの完了基準を明確にする
   └─ 依存関係を考慮して順序を立てる

5. 🔧 実装
   └─ テスト駆動またはベンチマーク駆動で進める
   └─ 1つのタスクごとにコミット

6. 📚 ドキュメント更新
   └─ コード変更に合わせてドキュメントも更新
   └─ ARCHITECTURE.md, ROADMAP.md, その他関連ドキュメント
```

### 1.2 アーキテクチャドキュメントの参照

**実装を開始する前に、必ず `docs/zui/ARCHITECTURE.md` を読むこと。**

このドキュメントには以下の重要情報が含まれている：
- 設計原則（引数は落ちる、エラーは上がる）
- 関数の分類（PURE / COMPUTE / MUTATE / EFFECT）
- comptimeパターンの使用法
- データフロー設計
- モジュール構成

**新しい機能を実装する際は、既存の設計原則に従っていることを確認すること。**

---

## 2. 開発手法

### 2.1 テスト駆動開発 (TDD) と ベンチマーク駆動開発

**目標に応じて、適切な開発手法を選択すること：**

#### テスト駆動開発 (TDD) を選ぶべき場合：
- 正しい性が重要な機能（レイアウトアルゴリズム、データ構造）
- 入出力が明確な機能（パーサー、バリデーター）
- 複雑なロジックを含む機能

**TDDサイクル：**
```
1. 🔴 RED:    失敗するテストを書く
2. 🟢 GREEN:  テストを通す最小限の実装
3. 🔵 REFACTOR: コードを改善
4. 📝 COMMIT: コミット
```

#### ベンチマーク駆動開発を選ぶべき場合：
- パフォーマンスが重要な機能（レンダラー、レイアウトエンジン）
- 最適化が必要な機能（メモリ使用量、速度）
- 既存の実装を改善する場合

**ベンチマーク駆動サイクル：**
```
1. 📊 BASELINE: 現在のパフォーマンスを測定
2. 🎯 TARGET: 目標パフォーマンスを設定
3. 🔧 OPTIMIZE: 最適化を実装
4. 📊 MEASURE: 再度測定して改善を確認
5. 📝 COMMIT: コミット
```

### 2.2 テストは「これでもか」というほど増やす

**テストの数量和種類を最大限にすること：**

#### 必要なテストの種類：

1. **単体テスト (Unit Tests)**
   - 各関数に対して複数のテストケース
   - 正常系だけでなく、異常系もカバー
   - エッジケースを網羅
   
2. **統合テスト (Integration Tests)**
   - 複数モジュールの連携
   - 実際の使用シナリオ
   
3. **プロパティベーステスト (Property-based Tests)**
   - 任意の入力に対して成り立つ不変条件
   - ランダム生成による網羅的テスト
   
4. **スナップショットテスト (Snapshot Tests)**
   - 視覚的な回帰検出
   - レンダリング結果の比較
   
5. **パフォーマンステスト (Performance Tests)**
   - ベンチマーク
   - メモリ使用量の測定
   
6. **ストレステスト (Stress Tests)**
   - 大量のデータでの動作確認
   - 極端な条件下的でのテスト

**目標：各関数に対して最低5-10個のテストケース**

#### テストの書き方：

```zig
// ✅ 良い例：網羅的なテスト
test "arena - basic allocation" { ... }
test "arena - reset and reuse" { ... }
test "arena - overflow handling" { ... }
test "arena - alignment" { ... }
test "arena - typed allocation" { ... }
test "arena - create single object" { ... }
test "arena - remaining capacity" { ... }
test "arena - multiple allocations" { ... }
test "arena - reset preserves capacity" { ... }
test "arena - zero-size allocation" { ... }

// ❌ 悪い例：テストが少なすぎる
test "arena - works" { ... }  // これだけでは不十分
```

---

## 3. ドキュメント管理

### 3.1 ドキュメントの更新

**コードを変更したら、必ず関連するドキュメントも更新すること。**

更新が必要なドキュメント：
- `docs/zui/ARCHITECTURE.md` — アーキテクチャ変更時
- `ROADMAP.md` — マイルストーンの進捗更新
- `docs/zui/MILESTONES.md` — 具体的な実装状況
- `docs/zui/MODULAR_ARCHITECTURE.md` — モジュール構成変更時
- 各モジュールのドキュメントコメント

### 3.2 ドキュメントとコードの整合性

**ドキュメントとコードの不一致を機械的に検出する仕組みを導入する。**

#### 実装済み：

1. **ドキュメント自動生成**
   - `zig build doc` でMarkdownからHTMLを自動生成
   - 実装: `docs/build_docs.zig`
   - 出力先: `zig-out/doc/`

2. **整合性チェック**
   - `zig build check` でコードフォーマット＋テスト＋ドキュメント整合性をチェック
   - 手動チェック: `python3 tools/check_docs.py`

#### ドキュメント更新のルール：

**コードを変更したら、必ず関連するドキュメントも更新すること。**

チェック項目：
- [ ] コード内の関数シグネチャとドキュメントの説明が一致しているか
- [ ] モジュール構成とドキュメントの記述が一致しているか
- [ ] APIの変更がドキュメントに反映されているか
- [ ] 新しい機能のドキュメントが追加されているか

**整合性チェックでエラーが出たら、修正してからコミットすること。**

---

## 4. 開発環境

### 4.1 Zigのインストール

**Zig 0.17（最新版）をインストールすること。**

この開発環境（Arena.ai）では、pip経由でZigをインストールする必要があります：

```bash
# Pip経由でZig 0.17.0をインストール
pip install --break-system-packages ziglang

# 確認
python3 -m ziglang version  # 0.17.0 と表示されればOK

# または、直接バイナリを実行
/usr/local/lib/python3.11/dist-packages/ziglang/zig version
```

**ビルドコマンド**:
```bash
# テスト実行
python3 -m ziglang build test

# ドキュメント生成
python3 -m ziglang build doc

# ベンチマーク
python3 -m ziglang build bench
```

**重要**: CIはZig 0.17.0を使用しています。ローカルでも同じバージョンを使用してください。

**テストを実行する前に、必ずZigをインストールすること。**

---

## 5. 品質保証

### 5.1 CI/CD と自動デプロイ

**mainブランチにプッシュすると、自動的にWebページがアップデートされる。**

GitHub Actions (`.github/workflows/ci.yml`) により：

1. **テスト実行** — `zig build test`
2. **サンプルビルド** — 各デモアプリのビルド
3. **WebAssemblyビルド** — Web版のビルド
4. **スナップショット比較** — 視覚的リグレッションテスト
5. **GitHub Pagesデプロイ** — 以下を自動公開：
   - `zig build doc` で生成したドキュメント（HTML）
   - WebAssembly版デモアプリ
   - スナップショット画像

**URL**: `https://<username>.github.io/ZPU/`

**開発フロー**:
```
コード変更 → プッシュ → CI自動実行 → Webページ自動更新
```

**注意**: ドキュメントを変更したら、ローカルで `zig build doc` を実行して確認してからプッシュすること。

### 5.2 最高品質を目指す

**時間をかけてゆっくり考えながら、最高品質のコードを書くこと。**

#### 品質チェックリスト：

- [ ] **設計原則に従っているか**
  - 関数の分類（PURE / COMPUTE / MUTATE / EFFECT）が適切か
  - エラー伝播が正しいか
  - comptimeを活用しているか

- [ ] **テストは十分か**
  - 正常系・異常系・エッジケースをカバーしているか
  - プロパティベーステストはあるか
  - パフォーマンステストはあるか

- [ ] **ドキュメントは更新されたか**
  - コード変更に合わせてドキュメントも更新したか
  - 整合性チェックはパスしたか

- [ ] **パフォーマンスは目標を達成しているか**
  - ベンチマーク目標を達成しているか
  - メモリ使用量は適切か

### 5.2 実装方針への異議

**実装方針に異議があれば、ちゃんと伝えること。**

- 設計原則に矛盾する実装を見つけたら指摘する
- テストが不十分な場合は追加を提案する
- パフォーマンス目標が達成できていない場合は報告する
- ドキュメントとコードの不一致を見つけたら修正する

**「言わなきゃいけないことを言う」ことが最高品質への道。**

---

## 6. コミットガイドライン

### 6.1 コミットの粒度

**1つのコミットで1つの意味のある変更を行うこと。**

良い例：
- `Add Arena allocator with comprehensive tests`
- `Implement DisplayCommand union type`
- `Optimize layout engine for 1000 elements`

悪い例：
- `Update code` — 何を変更したかわからない
- `Fix bugs` — どのバグを修正したかわからない

### 6.2 コミットメッセージ

```
<type>: <subject>

<body>

<footer>
```

**Type:**
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント変更
- `test`: テスト追加・修正
- `perf`: パフォーマンス改善
- `refactor`: リファクタリング

**例：**
```
feat(arena): Add Arena allocator with frame-based reset

- Implement bump allocator with alignment support
- Add typed allocation methods
- Include comprehensive tests (10 test cases)
- Performance: <1ms for 1000 alloc/reset cycles

Closes #123
```

---

## 7. 現状と今後の予定

### 実装済み ✅

- `AGENTS.md` — このガイドライン
- `docs/build_docs.zig` — Markdown → HTML 自動生成
- `tools/check_docs.py` — ドキュメント整合性チェック（手動実行用）
- `ROADMAP.md` — 詳細なロードマップ
- `docs/zui/ARCHITECTURE.md` — アーキテクチャ設計

### 今後の予定 📋

1. **整合性チェックのZig化**
   - 現状: `tools/check_docs.py`（Python、手動実行）
   - 目標: `zig build check-docs`（Zig、自動実行）
   - `docs/build_docs.zig` を拡張して実装予定

2. **CI/CD統合**
   - GitHub Actions で自動チェック
   - プルリクエスト時に整合性検証

---

## 8. メモと引き継ぎ

### memory.md

**`memory.md` はプロジェクトの進捗、実装意図、次のステップを記録するファイルです。**

**目的**:
- AIエージェントの引き継ぎ
- 人間の開発者の参考
- 実装の背景と意図の記録

**更新ルール**:
- 毎コミットで更新
- 実装意図を明記
- 次のステップを明確に
- 問題と解決策を記録

**内容**:
- プロジェクト概要
- アーキテクチャ設計
- 現在の進捗（完了済み/未実装）
- 実装意図と背景
- 次のステップ
- 技術的なメモ
- 問題と解決策

**実装を開始する前に、必ず `memory.md` を読むこと。**

---

## 9. 次のステップ

実装を開始する際は、このドキュメントに従って：

1. `memory.md` を読んで現状を把握
2. `ROADMAP.md` を読んで実装するマイルストーンを選ぶ
3. `docs/zui/ARCHITECTURE.md` を読んで設計原則を理解する
4. マイルストーンを小さなタスクに分解する
5. テスト駆動 or ベンチマーク駆動で実装する
6. テストはこれでもかというほど追加する
7. ドキュメントも同時に更新する
8. `memory.md` も更新する
9. 品質チェックリストを確認する

**最高のUIライブラリを作るために、最高の品質を目指そう。**
