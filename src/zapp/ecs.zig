//! ZAPP ECS (Entity Component System)
//!
//! ゲームエンジンや高度なUIに最適なデータ指向アーキテクチャ。
//! Overmind/Bevy ECSにインスパイアされた設計。
//!
//! ## 概念
//! - Entity: 一意のID (存在のみ)
//! - Component: データのみ (振る舞いなし)
//! - System: ロジックのみ (コンポーネントを操作)

const std = @import("std");

// ============================================================================
// Entity
// ============================================================================

/// Entity ID
pub const Entity = struct {
    id: u64,
    version: u32,
    
    pub fn init(id: u64, version: u32) Entity {
        return .{ .id = id, .version = version };
    }
    
    pub fn isAlive(self: Entity) bool {
        return self.version > 0;
    }
};

/// Entity Manager
pub const EntityManager = struct {
    next_id: u64 = 0,
    versions: std.AutoHashMap(u64, u32),
    alive: std.AutoHashMap(u64, bool),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) EntityManager {
        return .{
            .next_id = 0,
            .versions = std.AutoHashMap(u64, u32).init(allocator),
            .alive = std.AutoHashMap(u64, bool).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *EntityManager) void {
        self.versions.deinit();
        self.alive.deinit();
    }
    
    /// 新しいEntityを作成
    pub fn create(self: *EntityManager) !Entity {
        const id = self.next_id;
        self.next_id += 1;
        
        const version = (self.versions.get(id) orelse 0) + 1;
        try self.versions.put(id, version);
        try self.alive.put(id, true);
        
        return Entity.init(id, version);
    }
    
    /// Entityを削除
    pub fn destroy(self: *EntityManager, entity: Entity) !void {
        if (self.alive.get(entity.id)) |is_alive| {
            if (is_alive) {
                try self.alive.put(entity.id, false);
            }
        }
    }
    
    /// Entityが生きているか確認
    pub fn isAlive(self: *const EntityManager, entity: Entity) bool {
        return self.alive.get(entity.id) orelse false;
    }
    
    /// Entityのバージョンを取得
    pub fn getVersion(self: *const EntityManager, entity: Entity) u32 {
        return self.versions.get(entity.id) orelse 0;
    }
};

// ============================================================================
// Component
// ============================================================================

/// コンポーネントストレージ (スパースセット方式)
pub fn ComponentStore(comptime T: type) type {
    return struct {
        dense: std.ArrayList(T),
        sparse: std.AutoHashMap(u64, usize),
        allocator: std.mem.Allocator,
        
        const Self = @This();
        
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .dense = std.ArrayList(T).empty,
                .sparse = std.AutoHashMap(u64, usize).init(allocator),
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.dense.deinit(self.allocator);
            self.sparse.deinit();
        }
        
        /// コンポーネントを追加
        pub fn add(self: *Self, entity_id: u64, component: T) !void {
            const index = self.dense.items.len;
            try self.dense.append(self.allocator, component);
            try self.sparse.put(entity_id, index);
        }
        
        /// コンポーネントを取得
        pub fn get(self: *Self, entity_id: u64) ?*T {
            const index = self.sparse.get(entity_id) orelse return null;
            return &self.dense.items[index];
        }
        
        /// コンポーネントを削除
        pub fn remove(self: *Self, entity_id: u64) void {
            const index = self.sparse.get(entity_id) orelse return;
            
            // 最後の要素とスワップして削除
            const last_index = self.dense.items.len - 1;
            if (index != last_index) {
                self.dense.items[index] = self.dense.items[last_index];
                
                // sparseマップを更新
                var iter = self.sparse.iterator();
                while (iter.next()) |entry| {
                    if (entry.value_ptr.* == last_index) {
                        entry.value_ptr.* = index;
                        break;
                    }
                }
            }
            
            _ = self.dense.pop();
            _ = self.sparse.remove(entity_id);
        }
        
        /// コンポーネントが存在するか
        pub fn has(self: *const Self, entity_id: u64) bool {
            return self.sparse.contains(entity_id);
        }
        
        /// 全コンポーネントをイテレート
        pub fn iterator(self: *const Self) std.ArrayList(T).Iterator {
            return self.dense.iterator();
        }
    };
}

// ============================================================================
// World
// ============================================================================

/// ECS World (全エンティティとコンポーネントを管理)
pub const World = struct {
    entities: EntityManager,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) World {
        return .{
            .entities = EntityManager.init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *World) void {
        self.entities.deinit();
    }
    
    /// Entityを作成
    pub fn createEntity(self: *World) !Entity {
        return self.entities.create();
    }
    
    /// Entityを削除
    pub fn destroyEntity(self: *World, entity: Entity) !void {
        try self.entities.destroy(entity);
    }
    
    /// Entityが生きているか
    pub fn isAlive(self: *const World, entity: Entity) bool {
        return self.entities.isAlive(entity);
    }
};

// ============================================================================
// System
// ============================================================================

/// System基底 (インターフェース)
pub const System = struct {
    name: []const u8,
    priority: i32 = 0,
    enabled: bool = true,
    
    pub fn init(name: []const u8) System {
        return .{
            .name = name,
            .priority = 0,
            .enabled = true,
        };
    }
    
    pub fn enable(self: *System) void {
        self.enabled = true;
    }
    
    pub fn disable(self: *System) void {
        self.enabled = false;
    }
};

// ============================================================================
// Query (コンポーネントの問い合わせ)
// ============================================================================

/// Entity Query (特定のコンポーネントを持つEntityを検索)
pub fn Query(comptime Components: type) type {
    _ = Components;
    return struct {
        world: *World,
        
        const Self = @This();
        
        pub fn init(world: *World) Self {
            return .{ .world = world };
        }
        
        // TODO: Implement query iteration
    };
}

// ============================================================================
// ビルトインコンポーネント
// ============================================================================

/// Transform コンポーネント
pub const Transform = struct {
    x: f32 = 0,
    y: f32 = 0,
    rotation: f32 = 0,
    scale_x: f32 = 1,
    scale_y: f32 = 1,
    
    pub fn init(x: f32, y: f32) Transform {
        return .{ .x = x, .y = y };
    }
    
    pub fn translate(self: *Transform, dx: f32, dy: f32) void {
        self.x += dx;
        self.y += dy;
    }
    
    pub fn rotate(self: *Transform, angle: f32) void {
        self.rotation += angle;
    }
    
    pub fn scale(self: *Transform, sx: f32, sy: f32) void {
        self.scale_x *= sx;
        self.scale_y *= sy;
    }
};

/// Velocity コンポーネント
pub const Velocity = struct {
    vx: f32 = 0,
    vy: f32 = 0,
    
    pub fn init(vx: f32, vy: f32) Velocity {
        return .{ .vx = vx, .vy = vy };
    }
};

/// Sprite コンポーネント
pub const Sprite = struct {
    width: u32,
    height: u32,
    color_r: f32 = 1,
    color_g: f32 = 1,
    color_b: f32 = 1,
    visible: bool = true,
    
    pub fn init(w: u32, h: u32) Sprite {
        return .{ .width = w, .height = h };
    }
};

/// Tag コンポーネント (名前付け)
pub const Tag = struct {
    name: []const u8,
    
    pub fn init(n: []const u8) Tag {
        return .{ .name = n };
    }
};

// ============================================================================
// テスト
// ============================================================================

test "ecs - Entity creation" {
    const allocator = std.testing.allocator;
    var manager = EntityManager.init(allocator);
    defer manager.deinit();
    
    const e1 = try manager.create();
    const e2 = try manager.create();
    
    try std.testing.expect(e1.id != e2.id);
    try std.testing.expect(manager.isAlive(e1));
}

test "ecs - Entity destruction" {
    const allocator = std.testing.allocator;
    var manager = EntityManager.init(allocator);
    defer manager.deinit();
    
    const e = try manager.create();
    try std.testing.expect(manager.isAlive(e));
    
    try manager.destroy(e);
    try std.testing.expect(!manager.isAlive(e));
}

test "ecs - ComponentStore add/get" {
    const allocator = std.testing.allocator;
    var store = ComponentStore(Transform).init(allocator);
    defer store.deinit();
    
    try store.add(0, Transform.init(10, 20));
    
    const t = store.get(0);
    try std.testing.expect(t != null);
    try std.testing.expectEqual(@as(f32, 10), t.?.x);
    try std.testing.expectEqual(@as(f32, 20), t.?.y);
}

test "ecs - ComponentStore has/remove" {
    const allocator = std.testing.allocator;
    var store = ComponentStore(Transform).init(allocator);
    defer store.deinit();
    
    try store.add(0, Transform.init(10, 20));
    try std.testing.expect(store.has(0));
    try std.testing.expect(!store.has(1));
    
    store.remove(0);
    try std.testing.expect(!store.has(0));
}

test "ecs - Transform operations" {
    var t = Transform.init(10, 20);
    
    t.translate(5, 10);
    try std.testing.expectEqual(@as(f32, 15), t.x);
    try std.testing.expectEqual(@as(f32, 30), t.y);
    
    t.rotate(1.5);
    try std.testing.expectEqual(@as(f32, 1.5), t.rotation);
    
    t.scale(2, 3);
    try std.testing.expectEqual(@as(f32, 2), t.scale_x);
    try std.testing.expectEqual(@as(f32, 3), t.scale_y);
}
