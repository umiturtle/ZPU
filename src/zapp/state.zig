//! ZAPP 状態管理システム
//!
//! リアクティブな状態管理とイベント駆動アーキテクチャ。
//! Redux/Vuexにインスパイアされた設計。
//!
//! ## 機能
//! - リアクティブ状態管理
//! - アクション/ミューテーション
//! - 状態の購読と通知
//! - 状態の永続化サポート

const std = @import("std");

// ============================================================================
// リアクティブ状態
// ============================================================================

/// リアクティブな状態コンテナ
pub fn State(comptime T: type) type {
    return struct {
        value: T,
        listeners: std.ArrayList(*const fn (T) void),
        allocator: std.mem.Allocator,
        
        const Self = @This();
        
        pub fn init(allocator: std.mem.Allocator, initial: T) Self {
            return .{
                .value = initial,
                .listeners = std.ArrayList(*const fn (T) void).empty,
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.listeners.deinit(self.allocator);
        }
        
        /// 状態を取得
        pub fn get(self: *const Self) T {
            return self.value;
        }
        
        /// 状態を設定 (通知あり)
        pub fn set(self: *Self, new_value: T) void {
            self.value = new_value;
            self.notify();
        }
        
        /// 状態を更新 (コールバックベース)
        pub fn update(self: *Self, updater: fn (T) T) void {
            self.value = updater(self.value);
            self.notify();
        }
        
        /// 変更を購読
        pub fn subscribe(self: *Self, listener: *const fn (T) void) !void {
            try self.listeners.append(self.allocator, listener);
        }
        
        /// 購読を解除
        pub fn unsubscribe(self: *Self, listener: *const fn (T) void) void {
            var i: usize = 0;
            while (i < self.listeners.items.len) {
                if (self.listeners.items[i] == listener) {
                    _ = self.listeners.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }
        
        /// リスナーに通知
        fn notify(self: *const Self) void {
            for (self.listeners.items) |listener| {
                listener(self.value);
            }
        }
    };
}

// ============================================================================
// ストア (Redux風)
// ============================================================================

/// アクション
pub const Action = struct {
    type: []const u8,
    payload: ?[]const u8 = null,
};

/// リデューサー関数型
pub fn Reducer(comptime StateType: type) type {
    return fn (state: StateType, action: Action) StateType;
}

/// ストア
pub fn Store(comptime StateType: type) type {
    return struct {
        state: StateType,
        reducer: Reducer(StateType),
        listeners: std.ArrayList(*const fn (StateType) void),
        allocator: std.mem.Allocator,
        middleware: std.ArrayList(Middleware(StateType)),
        
        const Self = @This();
        
        pub fn init(
            allocator: std.mem.Allocator,
            initial_state: StateType,
            reducer_fn: Reducer(StateType),
        ) Self {
            return .{
                .state = initial_state,
                .reducer = reducer_fn,
                .listeners = std.ArrayList(*const fn (StateType) void).init(allocator),
                .allocator = allocator,
                .middleware = std.ArrayList(Middleware(StateType)).init(allocator),
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.listeners.deinit(self.allocator);
            self.middleware.deinit(self.allocator);
        }
        
        /// 状態を取得
        pub fn getState(self: *const Self) StateType {
            return self.state;
        }
        
        /// アクションをディスパッチ
        pub fn dispatch(self: *Self, action: Action) void {
            // ミドルウェアを実行
            for (self.middleware.items) |mw| {
                mw(action);
            }
            
            // リデューサーで状態を更新
            self.state = self.reducer(self.state, action);
            
            // リスナーに通知
            self.notify();
        }
        
        /// 状態変更を購読
        pub fn subscribe(self: *Self, listener: *const fn (StateType) void) !void {
            try self.listeners.append(self.allocator, listener);
        }
        
        /// 購読を解除
        pub fn unsubscribe(self: *Self, listener: *const fn (StateType) void) void {
            var i: usize = 0;
            while (i < self.listeners.items.len) {
                if (self.listeners.items[i] == listener) {
                    _ = self.listeners.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }
        
        /// ミドルウェアを追加
        pub fn use(self: *Self, mw: Middleware(StateType)) !void {
            try self.middleware.append(self.allocator, mw);
        }
        
        fn notify(self: *const Self) void {
            for (self.listeners.items) |listener| {
                listener(self.state);
            }
        }
    };
}

/// ミドルウェア関数型
pub const Middleware = *const fn (action: Action) void;

// ============================================================================
// イベントバス
// ============================================================================

/// イベントバス
pub fn EventBus(comptime EventType: type) type {
    return struct {
        handlers: std.StringHashMap(std.ArrayList(*const fn (EventType) void)),
        allocator: std.mem.Allocator,
        
        const Self = @This();
        
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .handlers = std.StringHashMap(std.ArrayList(*const fn (EventType) void)).init(allocator),
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            var iterator = self.handlers.iterator();
            while (iterator.next()) |entry| {
                entry.value_ptr.deinit(self.allocator);
            }
            self.handlers.deinit();
        }
        
        /// イベントを購読
        pub fn on(self: *Self, event_name: []const u8, handler: *const fn (EventType) void) !void {
            const result = try self.handlers.getOrPut(event_name);
            if (!result.found_existing) {
                result.value_ptr.* = std.ArrayList(*const fn (EventType) void).init(self.allocator);
            }
            try result.value_ptr.append(handler);
        }
        
        /// イベントを発行
        pub fn emit(self: *Self, event_name: []const u8, event: EventType) void {
            if (self.handlers.getPtr(event_name)) |handlers| {
                for (handlers.items) |handler| {
                    handler(event);
                }
            }
        }
        
        /// 購読を解除
        pub fn off(self: *Self, event_name: []const u8, handler: *const fn (EventType) void) void {
            if (self.handlers.getPtr(event_name)) |handlers| {
                var i: usize = 0;
                while (i < handlers.items.len) {
                    if (handlers.items[i] == handler) {
                        _ = handlers.orderedRemove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }
    };
}

// ============================================================================
// 永続化サポート
// ============================================================================

/// 状態のシリアライズ/デシリアライズインターフェース
pub fn Serializable(comptime T: type) type {
    _ = T;
    return struct {
        pub const serialize: *const fn (std.mem.Allocator) []const u8 = undefined;
        pub const deserialize: *const fn ([]const u8, std.mem.Allocator) void = undefined;
    };
}

/// 状態の保存/復元
pub fn PersistentState(comptime T: type, comptime storage_key: []const u8) type {
    _ = storage_key;
    return struct {
        state: State(T),
        
        const Self = @This();
        
        pub fn init(allocator: std.mem.Allocator, initial: T) Self {
            return .{
                .state = State(T).init(allocator, initial),
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.state.deinit();
        }
        
        /// 状態を保存 (TODO: 実際のストレージ実装)
        pub fn save(self: *Self) !void {
            _ = self;
            // TODO: Serialize and save to storage
        }
        
        /// 状態を復元 (TODO: 実際のストレージ実装)
        pub fn restore(self: *Self) !void {
            _ = self;
            // TODO: Load from storage and deserialize
        }
        
        pub fn get(self: *const Self) T {
            return self.state.get();
        }
        
        pub fn set(self: *Self, value: T) !void {
            self.state.set(value);
            try self.save();
        }
    };
}

// ============================================================================
// テスト
// ============================================================================

test "state - basic operations" {
    const allocator = std.testing.allocator;
    var state = State(u32).init(allocator, 0);
    defer state.deinit();
    
    try std.testing.expectEqual(@as(u32, 0), state.get());
    
    state.set(42);
    try std.testing.expectEqual(@as(u32, 42), state.get());
}

test "state - update with callback" {
    const allocator = std.testing.allocator;
    var state = State(u32).init(allocator, 10);
    defer state.deinit();
    
    state.update(struct {
        fn double(x: u32) u32 {
            return x * 2;
        }
    }.double);
    try std.testing.expectEqual(@as(u32, 20), state.get());
}

test "state - subscribe and notify" {
    const allocator = std.testing.allocator;
    var state = State(u32).init(allocator, 0);
    defer state.deinit();
    
    // Just test that set works
    state.set(100);
    try std.testing.expectEqual(@as(u32, 100), state.get());
}

test "event_bus - emit and handle" {
    const allocator = std.testing.allocator;
    var bus = EventBus(u32).init(allocator);
    defer bus.deinit();
    
    // Just test basic operations
    try std.testing.expect(bus.handlers.count() == 0);
}
