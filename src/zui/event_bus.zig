//! Event bus for ZUI
//!
//! Decoupled event handling: publishers don't know about subscribers.
//! Fully testable and independent of rendering.

const std = @import("std");

/// Event priority for ordering
pub const Priority = enum(u8) {
    highest = 0,
    high = 64,
    normal = 128,
    low = 192,
    lowest = 255,
};

/// Event handler result
pub const HandlerResult = enum {
    /// Continue processing
    continue_processing,
    /// Stop propagation
    stop_propagation,
    /// Event was handled, prevent default
    handled,
};

/// Generic event handler function type
pub const Handler = *const fn (event: anytype) HandlerResult;

/// Event subscription
pub const Subscription = struct {
    id: u64,
    priority: Priority = .normal,
    once: bool = false,
};

/// Simple event emitter (type-safe, comptime-based)
pub fn EventEmitter(comptime EventType: type) type {
    return struct {
        const Self = @This();

        pub const HandlerFn = *const fn (event: EventType) HandlerResult;

        handlers: std.ArrayListUnmanaged(HandlerEntry) = .empty,
        allocator: std.mem.Allocator,

        const HandlerEntry = struct {
            handler: HandlerFn,
            priority: Priority,
            once: bool,
        };

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.handlers.deinit(self.allocator);
        }

        pub fn on(self: *Self, handler: HandlerFn, priority: Priority) !void {
            try self.handlers.append(self.allocator, .{
                .handler = handler,
                .priority = priority,
                .once = false,
            });
            // Sort by priority (lower value = higher priority)
            std.mem.sort(HandlerEntry, self.handlers.items, {}, comparePriority);
        }

        pub fn once(self: *Self, handler: HandlerFn, priority: Priority) !void {
            try self.handlers.append(self.allocator, .{
                .handler = handler,
                .priority = priority,
                .once = true,
            });
            std.mem.sort(HandlerEntry, self.handlers.items, {}, comparePriority);
        }

        pub fn emit(self: *Self, event: EventType) void {
            var to_remove = std.ArrayListUnmanaged(usize).empty;
            defer to_remove.deinit(self.allocator);

            for (self.handlers.items, 0..) |entry, i| {
                _ = entry.handler(event);
                if (entry.once) {
                    to_remove.append(self.allocator, i) catch {};
                }
            }

            // Remove one-time handlers (in reverse to maintain indices)
            var i = to_remove.items.len;
            while (i > 0) {
                i -= 1;
                _ = self.handlers.orderedRemove(to_remove.items[i]);
            }
        }

        pub fn clear(self: *Self) void {
            self.handlers.clearRetainingCapacity();
        }

        fn comparePriority(_: void, a: HandlerEntry, b: HandlerEntry) bool {
            return @intFromEnum(a.priority) < @intFromEnum(b.priority);
        }
    };
}

/// Signal: simpler event with no data
pub const Signal = struct {
    count: u32 = 0,
    handlers: std.ArrayListUnmanaged(*const fn () void) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Signal {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Signal) void {
        self.handlers.deinit(self.allocator);
    }

    pub fn connect(self: *Signal, handler: *const fn () void) !void {
        try self.handlers.append(self.allocator, handler);
    }

    pub fn emit(self: *Signal) void {
        self.count += 1;
        for (self.handlers.items) |handler| {
            handler();
        }
    }

    pub fn clear(self: *Signal) void {
        self.handlers.clearRetainingCapacity();
    }
};

// ============================================================================
// Tests
// ============================================================================

const TestEvent = struct {
    value: i32,
};

test "event emitter - basic emit" {
    var emitter = EventEmitter(TestEvent).init(std.testing.allocator);
    defer emitter.deinit();

    // Simple handler registration test
    const handler = struct {
        fn handle(_: TestEvent) HandlerResult {
            return .continue_processing;
        }
    }.handle;

    try emitter.on(handler, .normal);
    
    // emit that does nothing but verify it doesn't crash
    emitter.emit(.{ .value = 42 });
    
    // Verify handler was registered
    try std.testing.expectEqual(@as(usize, 1), emitter.handlers.items.len);
}

test "event emitter - priority ordering" {
    var emitter = EventEmitter(TestEvent).init(std.testing.allocator);
    defer emitter.deinit();

    const handler_low = struct {
        fn handle(_: TestEvent) HandlerResult {
            return .continue_processing;
        }
    }.handle;

    const handler_high = struct {
        fn handle(_: TestEvent) HandlerResult {
            return .continue_processing;
        }
    }.handle;

    try emitter.on(handler_low, .low);
    try emitter.on(handler_high, .high);

    // High priority should be first
    try std.testing.expect(emitter.handlers.items.len == 2);
    try std.testing.expect(@intFromEnum(emitter.handlers.items[0].priority) < @intFromEnum(emitter.handlers.items[1].priority));
}

test "event emitter - once" {
    var emitter = EventEmitter(TestEvent).init(std.testing.allocator);
    defer emitter.deinit();

    const handler = struct {
        fn handle(_: TestEvent) HandlerResult {
            return .continue_processing;
        }
    }.handle;

    try emitter.once(handler, .normal);
    emitter.emit(.{ .value = 1 });
    emitter.emit(.{ .value = 2 });

    // Handler should only be called once, so it should be removed after first emit
    try std.testing.expect(emitter.handlers.items.len == 0);
}

test "event emitter - clear" {
    var emitter = EventEmitter(TestEvent).init(std.testing.allocator);
    defer emitter.deinit();

    const handler = struct {
        fn handle(_: TestEvent) HandlerResult {
            return .continue_processing;
        }
    }.handle;

    try emitter.on(handler, .normal);
    try std.testing.expect(emitter.handlers.items.len == 1);

    emitter.clear();
    try std.testing.expect(emitter.handlers.items.len == 0);
}

test "signal - basic" {
    var signal = Signal.init(std.testing.allocator);
    defer signal.deinit();

    try std.testing.expectEqual(@as(u32, 0), signal.count);

    signal.emit();
    try std.testing.expectEqual(@as(u32, 1), signal.count);

    signal.emit();
    try std.testing.expectEqual(@as(u32, 2), signal.count);
}

test "signal - clear" {
    var signal = Signal.init(std.testing.allocator);
    defer signal.deinit();

    const handler = struct {
        fn handle() void {}
    }.handle;

    try signal.connect(handler);
    try std.testing.expect(signal.handlers.items.len == 1);

    signal.clear();
    try std.testing.expect(signal.handlers.items.len == 0);
}
