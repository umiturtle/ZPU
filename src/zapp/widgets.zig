//! ZAPP Widgets Module
//!
//! 個別のウィジェットモジュールを統合してエクスポートする。
//!
//! 各ウィジェットは独立したモジュールとして実装されており、
//! 必要に応じて個別にインポートすることも可能。

pub const drawing = @import("widgets/drawing.zig");
pub const font = @import("widgets/font.zig");
pub const label = @import("widgets/label.zig");
pub const button = @import("widgets/button.zig");
pub const slider = @import("widgets/slider.zig");
pub const checkbox = @import("widgets/checkbox.zig");
pub const progress_bar = @import("widgets/progress_bar.zig");
pub const text_input = @import("widgets/text_input.zig");
pub const separator = @import("widgets/separator.zig");
pub const spacer = @import("widgets/spacer.zig");

// 便利なエイリアス
pub const FontConfig = font.FontConfig;
pub const FontSize = font.FontSize;
pub const LabelConfig = label.LabelConfig;
pub const ButtonConfig = button.ButtonConfig;
pub const ButtonState = button.ButtonState;
pub const SliderConfig = slider.SliderConfig;
pub const CheckboxConfig = checkbox.CheckboxConfig;
pub const ProgressBarConfig = progress_bar.ProgressBarConfig;
pub const TextInputConfig = text_input.TextInputConfig;
pub const TextInputState = text_input.TextInputState;
pub const SeparatorConfig = separator.SeparatorConfig;
pub const SeparatorOrientation = separator.SeparatorOrientation;
pub const SpacerConfig = spacer.SpacerConfig;
