//! Layout occupation helper - calculates widget rect based on style and content size.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Rect = types.Rect;
const Style = ui_mod.Style;
const Vec2 = types.Vec2;

/// Calculates the widget's rectangle based on style and content size.
/// Updates the layout frame state and returns the positioned rect.
pub fn occupy(self: *Ui, style: Style, content_size: Vec2) Rect {
    const lf = self.top();
    const main = lf.dir;
    const cross = main.other();

    const min_main = style.minAlong(main);
    const min_cross = style.minAlong(cross);
    const wanted_main = @max(min_main, content_size.index(main) + style.pad.sum(main));
    const wanted_cross = @max(min_cross, content_size.index(cross) + style.pad.sum(cross));
    const main_req = style.sizeAlong(main);
    const cross_req = style.sizeAlong(cross);

    if (lf.mode == .measure) {
        switch (main_req) {
            .auto => lf.fixed_main += wanted_main,
            .px => |v| lf.fixed_main += @max(v, min_main),
            .fill => |w| {
                lf.fill_content += wanted_main;
                lf.weight += w;
            },
        }
        if (lf.children > 0) {
            lf.fixed_main += lf.gap;
            lf.fill_content += lf.gap;
            lf.gaps += lf.gap;
        }
        const want_cross: f32 = switch (cross_req) {
            .auto, .fill => wanted_cross,
            .px => |v| @max(v, min_cross),
        };
        lf.max_cross = @max(lf.max_cross, want_cross);
        lf.children += 1;
        return Rect.fromSize(0, 0);
    }

    const main_extent: f32 = @max(min_main, switch (main_req) {
        .auto => wanted_main,
        .px => |v| v,
        .fill => |w| if (lf.weight > 0) lf.fill_extent * (w / lf.weight) else lf.remaining(),
    });
    const cross_extent: f32 = @max(min_cross, switch (cross_req) {
        .px => |v| v,
        .fill => lf.cross,
        .auto => if (lf.cross_align == .stretch) lf.cross else wanted_cross,
    });

    const start = if (lf.children == 0) lf.cursor else lf.cursor + lf.gap;
    lf.cursor = start + main_extent;
    lf.children += 1;

    const slack = @max(0, lf.cross - cross_extent);
    const cross_off: f32 = switch (cross_req) {
        .fill => 0,
        .px, .auto => switch (lf.cross_align) {
            .start, .stretch => 0,
            .center => slack * 0.5,
            .end => slack,
        },
    };

    return if (main == .x)
        Rect.init(start, lf.content.y + cross_off, main_extent, cross_extent)
    else
        Rect.init(lf.content.x + cross_off, start, cross_extent, main_extent);
}

// ============================================================================
// Tests
// ============================================================================

test "occupy exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(occupy);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
