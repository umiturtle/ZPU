# ZAPP デザイン哲学

## 核となる考え

**「過去に囚われず、新鮮さと使いやすさを両立し、最終的に完全なカスタマイズ性を実現する」**

ZAPPのデザインは、既存のデザインシステム（マテリアル、フラット、ニュートラルなど）の模倣ではありません。Zigという言語が持つ哲学を、UIという形で表現することを目指します。

同時に、**「新鮮さ」**（最新トレンド、視覚的な魅力）と**「使いやすさ」**（直感的、学習成本低）の両立を追求します。最終的には、ユーザーが自由にUIをカスタマイズできることを目標とします。

---

## Zigの哲学とUIへの反映

### 1. 明示性（Explicit is better than implicit）
- **UIでの表現**: 状態が一目で分かる
- 例: ボタンの状態（normal/hover/active/disabled）が明確に区別される
- 例: フォーカス位置が視覚的に明瞭

### 2. 効率性（No hidden control flow, no hidden allocations）
- **UIでの表現**: 動作が予測可能、応答が高速
- 例: ユーザー操作へのフィードバックが即座に返る
- 例: アニメーションが軽量で卡顿しない

### 3. 安全性（Correctness first）
- **UIでの表現**: ユーザーを誤った操作に導かない
- 例: 破壊的操作には確認ダイアログ
- 例: 無効な状態は視覚的に無効化

### 4. シンプルさ（One way to do it）
- **UIでの表現**: 過剰な装飾を避ける
- 例: 必要な情報だけを適切な階層で提示
- 例: コンポーネントのバリエーションを絞る

---

## 現代的なUIのトレンド調査

### 調査対象（2024-2026）

#### プロダクトUI
| プロダクト | 特徴 | 参考点 |
|-----------|------|--------|
| Linear | ダーク基調、キーボードショートカット重視、ミニマル | DX重視の設計 |
| Vercel | グラデーション、ガラスモーフィズム、大胆なタイポグラフィ | モダンな視覚表現 |
| Raycast | コマンドパレット中心、高速な操作感 | キーボードファースト |
| Arc Browser | サイドバー、タブ管理、空間的UI | 情報階層の可視化 |
| Notion | ブロックエディタ、柔軟なレイアウト | コンテンツ中心 |
| Figma | ツールバー、プロパティパネル、キャンバス | 高機能ツールのUI |

#### デザイントレンド
| トレンド | 説明 | 採用検討 |
|---------|------|---------|
| ベントーUI（Bento UI） | グリッド状のカードレイアウト、Apple風 | ○ 情報整理に有用 |
| グラスモーフィズム | 半透明、ぼかし、層状表現 | △ パフォーマンス考慮 |
| ネオモーフィズム | 押し出し/凹み表現 | ✗ 視認性に課題 |
| ブルータリズム | 大胆なタイポグラフィ、_raw_な表現 | △ Zigの精神に合う可能性 |
| マイクロインタラクション | 小さな動きで状態を伝達 | ○ 必須 |

---

## ZAPPのデザイン方向性

### コンセプト: 「Code-Native UI」

Zigコードを書く体験を、UIにも反映させます。

#### 特徴
1. **タイポグラフィファースト**
   - コードのように読みやすい
   - モノスペースフォントの活用（数値、コード表示）
   - 明確な階層構造

2. **高密度情報表示**
   - IDEのように多くの情報を効率的に表示
   - 余白を最小化、でも窮屈にならない
   - コンパクトなコンポーネント

3. **ダークテーマ標準**
   - 開発者の目に優しい
   - コードエディタとの統一感
   - ライトテーマはオプション

4. **キーボードファースト**
   - ショートカットが豊富
   - タブ移動、コマンドパレット
   - マウス不要で操作可能

5. **サブtleな動き**
   - 派手なアニメーションは避ける
   - 状態変化を小さく、確実に伝える
   - パフォーマンスを損なわない動き

---

## カラーパレット案

### ダークテーマ（デフォルト）

```
Background:
  - Base:       #0D0D0F  (ほぼ黒、わずかに青み)
  - Surface:    #16161A  (カード、パネル)
  - Elevated:   #1F1F23  (ドロップダウン、モーダル)
  - Overlay:    rgba(0,0,0,0.7)

Text:
  - Primary:    #E4E4E7  (ほぼ白)
  - Secondary:  #A1A1AA  (説明テキスト)
  - Tertiary:   #71717A  (無効テキスト)
  - Inverse:    #09090B  (明るい背景上)

Accent:
  - Primary:    #A78BFA  (紫系 - Zigのロゴカラーに近い)
  - Success:    #34D399  (緑)
  - Warning:    #FBBF24  (黄)
  - Error:      #F87171  (赤)
  - Info:       #60A5FA  (青)

Border:
  - Default:    #27272A
  - Active:     #A78BFA
  - Focus:      #A78BFA (ring)
```

### ライトテーマ（オプション）

```
Background:
  - Base:       #FAFAFA
  - Surface:    #FFFFFF
  - Elevated:   #F4F4F5

Text:
  - Primary:    #18181B
  - Secondary:  #52525B
  - Tertiary:   #A1A1AA

Accent:
  - Primary:    #7C3AED  (濃い紫)
```

---

## タイポグラフィ

### フォントファミリー

```
Default (UI):
  - macOS:    -apple-system, BlinkMacSystemFont, "SF Pro Text"
  - Windows:  "Segoe UI", system-ui
  - Linux:    "Cantarell", "DejaVu Sans", system-ui
  - Fallback: sans-serif

Monospace (コード、数値):
  - Preferred: "JetBrains Mono", "Fira Code", "SF Mono"
  - Fallback:  "Menlo", "Consolas", monospace

Japanese (日本語):
  - Preferred: "Noto Sans JP", "Hiragino Kaku Gothic Pro"
  - Fallback:  "Meiryo", "Yu Gothic", sans-serif
```

### サイズスケール

```
xs:   11px  (キャプション、バッジ)
sm:   13px  (ボタン、入力フィールド)
base: 14px  (本文)
lg:   16px  (サブタイトル)
xl:   18px  (タイトル)
2xl:  22px  (見出し)
3xl:  28px  (大きな見出し)
```

### ウェイト

```
Regular: 400  (本文)
Medium:  500  (強調、ボタン)
SemiBold: 600 (見出し)
Bold:    700  (大きな見出し)
```

---

## スペーシング

4pxベースのグリッドシステム:

```
0:   0px
1:   4px
2:   8px
3:   12px
4:   16px
5:   20px
6:   24px
8:   32px
10:  40px
12:  48px
16:  64px
20:  80px
24:  96px
```

---

## 形状

### 角丸

```
none:    0px    (シャープ、コードライク)
sm:      4px    (ボタン、入力フィールド)
md:      8px    (カード、パネル)
lg:      12px   (モーダル、ドロップダウン)
xl:      16px   (大きなカード)
full:    9999px (アバター、バッジ)
```

### ボーダー

```
thin:    1px    (デフォルト)
medium:  2px    (強調)
thick:   3px    (フォーカスリング)
```

---

## 動き

### トランジション時間

```
instant:  0ms    (即座)
fast:     100ms  (ホバー、小さな変化)
normal:   200ms  (標準)
slow:     300ms  (モーダル表示)
slower:   500ms  (ページ遷移)
```

### イージング関数

```
ease-in:    cubic-bezier(0.4, 0, 1, 1)
ease-out:   cubic-bezier(0, 0, 0.2, 1)
ease-in-out: cubic-bezier(0.4, 0, 0.2, 1)
spring:     (独自実装、物理ベース)
```

---

## コンポーネントデザイン

### ボタン

```
Primary:
  - Background: accent-primary
  - Text: white
  - Border: none
  - Hover: brightness(1.1)
  - Active: brightness(0.9)
  - Disabled: opacity(0.5)

Secondary:
  - Background: surface
  - Text: text-primary
  - Border: 1px border-default
  - Hover: background(elevated)

Ghost:
  - Background: transparent
  - Text: text-primary
  - Border: none
  - Hover: background(surface)
```

### 入力フィールド

```
Default:
  - Background: surface
  - Border: 1px border-default
  - Focus: border(accent) + ring(2px, accent, 0.2)
  - Error: border(error)
  - Placeholder: text-tertiary
```

### カード

```
Default:
  - Background: surface
  - Border: 1px border-default
  - Shadow: 0 1px 3px rgba(0,0,0,0.1)
  - Hover: shadow(0 4px 12px rgba(0,0,0,0.15))
```

---

## 新鮮さと使いやすさの両立

### 方針

ZAPPは「新鮮なUI」と「使いやすいUI」の両立を目指します。これらは相反するように見えますが、適切な設計で両立可能です。

### 新鮮さの要素
- **視覚的な魅力**: グラデーション、微妙な影、モダンなタイポグラフィ
- **最新トレンド**: ベントーUI、ガラスモーフィズム（適切に使用）
- **独自性**: Zigらしさを表現する独自の視覚言語

### 使いやすさの要素
- **直感的な操作**: 一般的なUIパターンに従う
- **低い学習コスト**: 初めてでも使える
- **予測可能性**: 動作が一貫している

### 両立方法
1. **層状アプローチ**
   - 基本構造は慣れ親しんだパターン（使いやすさ）
   - 視覚表現をモダンに（新鮮さ）
   
2. **プログレッシブエンハンスメント**
   - 基本機能はシンプルに
   - 詳細なカスタマイズはオプションに

3. **コンテキスト依存**
   - 開発者向けツール: 高密度、コードライク
   - 一般向けアプリ: シンプル、直感的

### カスタマイズ性による解決

最終的には、ユーザーが「新鮮さ」と「使いやすさ」のバランスを自分で調整できるようにします。

- **テーマシステム**: 視覚スタイルを完全にカスタマイズ
- **コンポーネント差し替え**: 動作を変更可能
- **レイアウト自由化**: ユーザーが配置を制御

---

## 次のステップ

1. **モックアップ作成**
   - 基本コンポーネントのデザイン
   - ダークテーマ、ライトテーマ両方
   - 「新鮮さ」と「使いやすさ」の両立を検証
   - Figmaまたはコードでのプロトタイプ

2. **コミュニティレビュー**
   - Zigコミュニティからのフィードバック
   - 複数の方向性を提示して選択

3. **デザインシステムの実装**
   - トークンをコードに定義
   - コンポーネントライブラリの構築

4. **カスタマイズシステムの設計**
   - テーマシステムのAPI設計
   - ユーザー定義コンポーネントの仕組み

---

## 参考リンク

- [Zig Programming Language](https://ziglang.org/)
- [Linear App](https://linear.app/)
- [Vercel Design System](https://vercel.com/design)
- [Raycast](https://www.raycast.com/)
- [Arc Browser](https://arc.net/)
