// ホットリロードサポートモジュール
// ファイル変更監視と自動リロード機能を提供

const std = @import("std");

/// ファイル変更イベント
pub const FileEvent = struct {
    path: []const u8,
    event_type: EventType,
    timestamp: i64,
    
    pub const EventType = enum {
        created,
        modified,
        deleted,
    };
};

/// ファイルウォッチャー
/// ファイルシステムの変更を監視し、コールバックを呼び出す
pub const FileWatcher = struct {
    allocator: std.mem.Allocator,
    watched_files: std.StringHashMap(FileInfo),
    on_change: ?*const fn (FileEvent) void,
    
    const FileInfo = struct {
        last_modified: i64,
        size: u64,
    };
    
    pub fn init(allocator: std.mem.Allocator) FileWatcher {
        return .{
            .allocator = allocator,
            .watched_files = std.StringHashMap(FileInfo).init(allocator),
            .on_change = null,
        };
    }
    
    pub fn deinit(self: *FileWatcher) void {
        self.watched_files.deinit();
    }
    
    /// ファイルを監視リストに追加
    pub fn watch(self: *FileWatcher, path: []const u8) !void {
        const file = std.fs.cwd().openFile(path, .{}) catch return;
        defer file.close();
        
        const stat = try file.stat();
        const info = FileInfo{
            .last_modified = @intCast(stat.mtime),
            .size = stat.size,
        };
        
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);
        
        try self.watched_files.put(owned_path, info);
    }
    
    /// ファイルを監視リストから削除
    pub fn unwatch(self: *FileWatcher, path: []const u8) void {
        if (self.watched_files.fetchRemove(path)) |kv| {
            self.allocator.free(kv.key);
        }
    }
    
    /// 変更をチェック
    /// 注意: この実装はポーリングベース。本当のファイルシステム通知ではない
    pub fn checkForChanges(self: *FileWatcher) !void {
        var iterator = self.watched_files.iterator();
        while (iterator.next()) |entry| {
            const path = entry.key_ptr.*;
            const old_info = entry.value_ptr.*;
            
            const file = std.fs.cwd().openFile(path, .{}) catch {
                // ファイルが削除された
                if (self.on_change) |callback| {
                    callback(.{
                        .path = path,
                        .event_type = .deleted,
                        .timestamp = std.time.milliTimestamp(),
                    });
                }
                continue;
            };
            defer file.close();
            
            const stat = try file.stat();
            const new_mtime: i64 = @intCast(stat.mtime);
            
            if (new_mtime != old_info.last_modified or stat.size != old_info.size) {
                // ファイルが変更された
                entry.value_ptr.* = .{
                    .last_modified = new_mtime,
                    .size = stat.size,
                };
                
                if (self.on_change) |callback| {
                    callback(.{
                        .path = path,
                        .event_type = .modified,
                        .timestamp = std.time.milliTimestamp(),
                    });
                }
            }
        }
    }
    
    /// 変更コールバックを設定
    pub fn setOnChange(self: *FileWatcher, callback: *const fn (FileEvent) void) void {
        self.on_change = callback;
    }
};

/// リロード可能なリソースマネージャー
/// 設定ファイル、シェーダー、アセットなどをホットリロード可能にする
pub const ReloadableResourceManager = struct {
    allocator: std.mem.Allocator,
    resources: std.StringHashMap(ResourceEntry),
    watcher: FileWatcher,
    
    const ResourceEntry = struct {
        data: []const u8,
        version: u32,
        path: []const u8,
    };
    
    pub fn init(allocator: std.mem.Allocator) ReloadableResourceManager {
        return .{
            .allocator = allocator,
            .resources = std.StringHashMap(ResourceEntry).init(allocator),
            .watcher = FileWatcher.init(allocator),
        };
    }
    
    pub fn deinit(self: *ReloadableResourceManager) void {
        var iterator = self.resources.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
            self.allocator.free(entry.value_ptr.path);
            self.allocator.free(entry.key_ptr.*);
        }
        self.resources.deinit();
        self.watcher.deinit();
    }
    
    /// リソースを読み込む
    pub fn loadResource(self: *ReloadableResourceManager, name: []const u8, path: []const u8) ![]const u8 {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const data = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        errdefer self.allocator.free(data);
        
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);
        
        const entry = ResourceEntry{
            .data = data,
            .version = 1,
            .path = owned_path,
        };
        
        try self.resources.put(owned_name, entry);
        try self.watcher.watch(path);
        
        return data;
    }
    
    /// リソースを取得
    pub fn getResource(self: *ReloadableResourceManager, name: []const u8) ?[]const u8 {
        const entry = self.resources.getPtr(name) orelse return null;
        return entry.data;
    }
    
    /// 変更をチェックしてリロード
    pub fn checkAndReload(self: *ReloadableResourceManager) !void {
        _ = self;
        // TODO: Implement actual reload logic
    }
};

/// 開発モードフラグ
/// comptimeで判定し、本番ビルドでは最適化される
pub const dev_mode = blk: {
    const build_mode = @import("builtin").mode;
    const mode_int: u32 = @intFromEnum(build_mode);
    break :blk mode_int == 0 or mode_int == 1; // Debug or ReleaseSafe
};

/// ホットリロードが有効かどうか
pub fn isHotReloadEnabled() bool {
    return dev_mode;
}

// ============================================================================
// テスト
// ============================================================================

test "hot_reload - FileWatcher init and deinit" {
    const allocator = std.testing.allocator;
    var watcher = FileWatcher.init(allocator);
    defer watcher.deinit();
    
    try std.testing.expect(watcher.on_change == null);
}

test "hot_reload - dev mode detection" {
    // Debugビルドではdev_modeがtrue
    try std.testing.expect(dev_mode);
    try std.testing.expect(isHotReloadEnabled());
}

test "hot_reload - ReloadableResourceManager init" {
    const allocator = std.testing.allocator;
    var manager = ReloadableResourceManager.init(allocator);
    defer manager.deinit();
    
    try std.testing.expect(manager.resources.count() == 0);
}
