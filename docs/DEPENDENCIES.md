# 依存関係の整理と整合性保証フレームワーク

## 変更概要

### 1. 依存関係の分離

**Before:**
```
zpu → zbackend
zgf → zpu, zbackend
zui → zpu, zgf, zbackend
```

**After:**
```
zpu (独立)
zbackend (独立)
zgf → zpu
zui → zpu, zgf, zbackend
```

**理由:**
- `zpu` は純粋な数値計算ライブラリであり、GPU バックエンドに依存すべきではない
- `zbackend` はオプションの機能拡張であり、`zpu` のコア機能とは独立
- `zgf` は `zbackend` を使っていない（不要な import を削除）
- `zui` のみ GPU 描画パイプラインのために `zbackend` を必要とする

### 2. 変更内容

#### コード変更
- `src/zpu/context.zig`: `zbackend` の import と `backends()`, `bestBackend()` 関数を削除
- `src/zpu.zig`: `zbackend` の再エクスポート（`DeviceBuffer`, `createDeviceBuffer` 等）を削除
- `build.zig`: `zpu` と `zgf` から `zbackend` の import を削除
- `examples/gpu_probe.zig`: `zpu.zbackend` ではなく `@import("zbackend")` を使用
- `src/zpu/tests.zig`: バックエンド関連のテストを削除（`tests/snap/backend.zig` に移行済み）

#### ドキュメント変更
- `README.md`: 依存関係図を追加、テスト数を 205 に更新
- `MILESTONES.md`: 依存関係のカラムを追加、テスト数を更新、M0 の説明を更新

### 3. 整合性保証フレームワーク

#### `tools/validate_consistency.sh`

以下のチェックを自動化:

1. **zpu の独立性**: `src/zpu/context.zig` と `src/zpu.zig` が `zbackend` を参照していない
2. **build.zig の依存関係**: 
   - `zpu` が `zbackend` を import していない
   - `zgf` が `zbackend` を import していない
   - `zui` が `zbackend` を import している（必要）
3. **テスト数の整合性**: `MILESTONES.md` と `README.md` のテスト数が一致

#### CI 統合

`.github/workflows/ci.yml` に整合性チェックを追加:
```yaml
- name: Consistency validation
  shell: bash
  run: ./tools/validate_consistency.sh 2>&1 | tee -a ci.log
```

これにより、依存関係やドキュメントの drift を自動的に検出。

## 使い方

### ローカルでの整合性チェック

```bash
./tools/validate_consistency.sh
```

### テスト数の確認

```bash
zig build test --summary all
```

出力の `Build Summary` 行から実際のテスト数を確認し、`MILESTONES.md` と `README.md` を更新。

### 依存関係の確認

```bash
# build.zig の依存関係を確認
grep -A 5 "addImport" build.zig | grep -E "zpu|zgf|zui|zbackend"
```

## 今後の運用

1. **新機能追加時**: 
   - `zpu` に機能を追加する場合、`zbackend` に依存しないこと
   - `zbackend` の機能を使う場合は `zui` または別のレイヤーで実装

2. **テスト数変更時**:
   - `zig build test --summary all` で実際の数を確認
   - `MILESTONES.md` と `README.md` の両方を更新
   - `./tools/validate_consistency.sh` で整合性を確認

3. **依存関係変更時**:
   - `build.zig` の `addImport` を変更
   - `README.md` の依存関係図を更新
   - `MILESTONES.md` の依存関係カラムを更新
   - `./tools/validate_consistency.sh` で整合性を確認

## 参照

- PR: https://github.com/umiturtle/ZPU/pull/2
- Commit: `5283c6e`
- Issue: 依存関係の整理と整合性保証フレームワークの整備
