//! Widgets module - comptime-registered widget set
//!
//! This module aggregates all widgets and provides comptime registration.
//! Widgets are extracted from ui.zig for better modularity.

const std = @import("std");

// Widget modules
pub const label = @import("widgets/label.zig");
pub const button = @import("widgets/button.zig");
pub const separator = @import("widgets/separator.zig");
pub const checkbox = @import("widgets/checkbox.zig");
pub const radio = @import("widgets/radio.zig");
pub const slider = @import("widgets/slider.zig");

// Re-export widget types and functions
pub const LabelOptions = label.LabelOptions;
pub const labelWidget = label.label;

pub const ButtonOptions = button.ButtonOptions;
pub const buttonWidget = button.button;
pub const buttonIdWidget = button.buttonId;

pub const ToggleOptions = checkbox.ToggleOptions;
pub const checkboxWidget = checkbox.checkbox;
pub const radioWidget = radio.radio;

pub const SliderOptions = slider.SliderOptions;
pub const sliderWidget = slider.slider;

pub const separatorWidget = separator.separator;

// Widget registry for comptime introspection
pub const WidgetRegistry = struct {
    pub const widgets = .{
        .label,
        .button,
        .separator,
        .checkbox,
        .radio,
        .slider,
    };

    pub fn count() usize {
        return widgets.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "widget registry" {
    try std.testing.expectEqual(@as(usize, 6), WidgetRegistry.count());
}
