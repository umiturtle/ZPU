//! ZAPP レスポンシブレイアウトシステム
//!
//! 画面サイズに基づいてレイアウトを動的に調整するシステム。
//! CSS Flexbox/Gridにインスパイアされた設計。
//!
//! ## 機能
//! - ブレークポイントベースのレスポンシブデザイン
//! - Flexbox風のレイアウト
//! - アスペクト比の維持
//! - 最小/最大サイズ制約

const std = @import("std");

// ============================================================================
// ブレークポイント
// ============================================================================

/// 画面サイズのブレークポイント
pub const Breakpoint = enum {
    xs,     // ~576px (手机)
    sm,     // 576px~768px
    md,     // 768px~992px
    lg,     // 992px~1200px
    xl,     // 1200px~
    
    pub fn fromWidth(width: u32) Breakpoint {
        return if (width < 576) .xs
            else if (width < 768) .sm
            else if (width < 992) .md
            else if (width < 1200) .lg
            else .xl;
    }
    
    pub fn minWidth(self: Breakpoint) u32 {
        return switch (self) {
            .xs => 0,
            .sm => 576,
            .md => 768,
            .lg => 992,
            .xl => 1200,
        };
    }
};

/// レスポンシブ値 (ブレークポイントごとに異なる値)
pub fn ResponsiveValue(comptime T: type) type {
    return struct {
        xs: ?T = null,
        sm: ?T = null,
        md: ?T = null,
        lg: ?T = null,
        xl: ?T = null,
        default: T,
        
        pub fn get(self: @This(), breakpoint: Breakpoint) T {
            return switch (breakpoint) {
                .xs => self.xs orelse self.default,
                .sm => self.sm orelse self.xs orelse self.default,
                .md => self.md orelse self.sm orelse self.xs orelse self.default,
                .lg => self.lg orelse self.md orelse self.sm orelse self.xs orelse self.default,
                .xl => self.xl orelse self.lg orelse self.md orelse self.sm orelse self.xs orelse self.default,
            };
        }
        
        pub fn init(default_val: T) @This() {
            return .{ .default = default_val };
        }
    };
}

// ============================================================================
// Flexboxレイアウト
// ============================================================================

/// Flex方向
pub const FlexDirection = enum {
    row,
    row_reverse,
    column,
    column_reverse,
};

/// Flex折り返し
pub const FlexWrap = enum {
    nowrap,
    wrap,
    wrap_reverse,
};

/// Flex配置
pub const JustifyContent = enum {
    flex_start,
    flex_end,
    center,
    space_between,
    space_around,
    space_evenly,
};

/// Flex整列
pub const AlignItems = enum {
    flex_start,
    flex_end,
    center,
    stretch,
    baseline,
};

/// Flexコンテナスタイル
pub const FlexContainer = struct {
    direction: FlexDirection = .row,
    wrap: FlexWrap = .nowrap,
    justify: JustifyContent = .flex_start,
    align_items: AlignItems = .stretch,
    gap: f32 = 0,
};

/// Flexアイテムスタイル
pub const FlexItem = struct {
    flex_grow: f32 = 0,
    flex_shrink: f32 = 1,
    flex_basis: ?f32 = null,
    align_self: ?AlignItems = null,
    order: i32 = 0,
};

// ============================================================================
// グリッドレイアウト
// ============================================================================

/// グリッドトラックサイズ
pub const GridTrackSize = union(enum) {
    fixed: f32,
    fraction: f32,    // fr単位
    auto: void,
    min_content: void,
    max_content: void,
    minmax: struct { min: f32, max: f32 },
};

/// グリッドコンテナ
pub const GridContainer = struct {
    columns: []const GridTrackSize,
    rows: []const GridTrackSize,
    gap_x: f32 = 0,
    gap_y: f32 = 0,
    justify_items: JustifyContent = .flex_start,
    align_items: AlignItems = .stretch,
};

/// グリッドアイテム
pub const GridItem = struct {
    column_start: ?u32 = null,
    column_end: ?u32 = null,
    row_start: ?u32 = null,
    row_end: ?u32 = null,
    column_span: u32 = 1,
    row_span: u32 = 1,
};

// ============================================================================
// レイアウトエンジン
// ============================================================================

/// レイアウト制約
pub const LayoutConstraints = struct {
    min_width: f32 = 0,
    max_width: f32 = std.math.inf(f32),
    min_height: f32 = 0,
    max_height: f32 = std.math.inf(f32),
    preferred_width: ?f32 = null,
    preferred_height: ?f32 = null,
};

/// レイアウト結果
pub const LayoutResult = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    
    pub fn contains(self: LayoutResult, px: f32, py: f32) bool {
        return px >= self.x and px <= self.x + self.width and
               py >= self.y and py <= self.y + self.height;
    }
};

/// レスポンシブレイアウト計算
pub const ResponsiveLayout = struct {
    container_width: u32,
    container_height: u32,
    breakpoint: Breakpoint,
    
    pub fn init(width: u32, height: u32) ResponsiveLayout {
        return .{
            .container_width = width,
            .container_height = height,
            .breakpoint = Breakpoint.fromWidth(width),
        };
    }
    
    /// Flexboxレイアウトを計算
    pub fn calculateFlexLayout(
        self: ResponsiveLayout,
        container: FlexContainer,
        items: []const LayoutConstraints,
    ) []LayoutResult {
        _ = self;
        _ = container;
        _ = items;
        // TODO: Implement flex layout calculation
        return &[_]LayoutResult{};
    }
    
    /// グリッドレイアウトを計算
    pub fn calculateGridLayout(
        self: ResponsiveLayout,
        grid: GridContainer,
        items: []const LayoutConstraints,
    ) []LayoutResult {
        _ = self;
        _ = grid;
        _ = items;
        // TODO: Implement grid layout calculation
        return &[_]LayoutResult{};
    }
};

// ============================================================================
// アスペクト比
// ============================================================================

/// アスペクト比の維持
pub const AspectRatio = struct {
    width: u32,
    height: u32,
    
    pub fn init(w: u32, h: u32) AspectRatio {
        const gcd_val = gcd(w, h);
        return .{ .width = w / gcd_val, .height = h / gcd_val };
    }
    
    pub fn ratio(self: AspectRatio) f32 {
        return @as(f32, @floatFromInt(self.width)) / @as(f32, @floatFromInt(self.height));
    }
    
    /// 指定された幅から高さを計算
    pub fn calculateHeight(self: AspectRatio, width: f32) f32 {
        return width / self.ratio();
    }
    
    /// 指定された高さから幅を計算
    pub fn calculateWidth(self: AspectRatio, height: f32) f32 {
        return height * self.ratio();
    }
    
    /// 指定された領域に収まる最大サイズを計算
    pub fn fitIn(self: AspectRatio, max_width: f32, max_height: f32) struct { width: f32, height: f32 } {
        const width_from_height = self.calculateWidth(max_height);
        if (width_from_height <= max_width) {
            return .{ .width = width_from_height, .height = max_height };
        } else {
            return .{ .width = max_width, .height = self.calculateHeight(max_width) };
        }
    }
};

fn gcd(a: u32, b: u32) u32 {
    if (b == 0) return a;
    return gcd(b, a % b);
}

// ============================================================================
// テスト
// ============================================================================

test "layout - Breakpoint detection" {
    try std.testing.expectEqual(Breakpoint.xs, Breakpoint.fromWidth(400));
    try std.testing.expectEqual(Breakpoint.sm, Breakpoint.fromWidth(600));
    try std.testing.expectEqual(Breakpoint.md, Breakpoint.fromWidth(800));
    try std.testing.expectEqual(Breakpoint.lg, Breakpoint.fromWidth(1000));
    try std.testing.expectEqual(Breakpoint.xl, Breakpoint.fromWidth(1400));
}

test "layout - ResponsiveValue" {
    const rv = ResponsiveValue(f32){
        .default = 10,
        .sm = 20,
        .xl = 50,
    };
    
    try std.testing.expectEqual(@as(f32, 10), rv.get(.xs));
    try std.testing.expectEqual(@as(f32, 20), rv.get(.sm));
    try std.testing.expectEqual(@as(f32, 20), rv.get(.md));
    try std.testing.expectEqual(@as(f32, 20), rv.get(.lg));
    try std.testing.expectEqual(@as(f32, 50), rv.get(.xl));
}

test "layout - AspectRatio" {
    const ar = AspectRatio.init(16, 9);
    try std.testing.expectEqual(@as(u32, 16), ar.width);
    try std.testing.expectEqual(@as(u32, 9), ar.height);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0 / 9.0), ar.ratio(), 1e-6);
}

test "layout - AspectRatio fit" {
    const ar = AspectRatio.init(16, 9);
    const fitted = ar.fitIn(800, 600);
    try std.testing.expectApproxEqAbs(@as(f32, 800), fitted.width, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 450), fitted.height, 1e-6);
}

test "layout - LayoutResult contains" {
    const rect = LayoutResult{ .x = 10, .y = 20, .width = 100, .height = 50 };
    try std.testing.expect(rect.contains(50, 40));
    try std.testing.expect(!rect.contains(5, 40));
}
