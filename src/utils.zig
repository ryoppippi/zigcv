const std = @import("std");
const c = @import("c_api.zig");

pub fn fromCStructsToArrayList(from_array: anytype, from_array_length: i32, comptime ToType: type, allocator: std.mem.Allocator) !std.ArrayList(ToType) {
    const len = @as(usize, @intCast(from_array_length));
    var arr = try std.ArrayList(ToType).initCapacity(allocator, len);
    {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const elem = blk: {
                const elem = ToType.initFromC(from_array[i]);
                break :blk switch (comptime @typeInfo(@TypeOf(elem))) {
                    .ErrorUnion => try elem,
                    else => elem,
                };
            };
            try arr.append(elem);
        }
    }
    return arr;
}

//  note: cannot implicitly cast double pointer '[*c]?*anyopaque' to anyopaque pointer '?*anyopaque'
pub fn ensurePtrNotNull(ptr: anytype) !@TypeOf(ptr) {
    if (ptr == null) return error.AllocationError;
    return ptr.?;
}

pub fn ensureFileExists(path: []const u8, allow_zero_byte: bool) !void {
    const stat = std.fs.cwd().statFile(path) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("File not found: {s}\n", .{path});
            return error.FileNotFound;
        },
        else => return err,
    };
    if (stat.size == 0 and !allow_zero_byte) {
        std.debug.print("File is empty: {s}\n", .{path});
        return error.FileEmpty;
    }
}

pub fn ensureModelFile(filename: []const u8, allocator: std.mem.Allocator) !void {
    const cache_dir = "zig-cache/tmp";
    const dest_rel = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ cache_dir, filename });
    defer allocator.free(dest_rel);

    try std.fs.cwd().makePath(cache_dir);

    if (std.fs.cwd().statFile(dest_rel)) |_| {
        return;
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    const model_dir = std.process.getEnvVarOwned(allocator, "ZIGCV_MODEL_DIR") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return error.FileNotFound,
        else => return err,
    };
    defer allocator.free(model_dir);

    var src_dir = try std.fs.cwd().openDir(model_dir, .{});
    defer src_dir.close();
    var dest_dir = try std.fs.cwd().openDir(cache_dir, .{});
    defer dest_dir.close();

    var src_file = try src_dir.openFile(filename, .{});
    defer src_file.close();

    var dest_file = try dest_dir.createFile(filename, .{ .truncate = true });
    defer dest_file.close();

    var buffer: [16 * 1024]u8 = undefined;
    while (true) {
        const read_bytes = try src_file.read(&buffer);
        if (read_bytes == 0) break;
        try dest_file.writeAll(buffer[0..read_bytes]);
    }
}

pub fn downloadFile(url: []const u8, dir: []const u8, allocator: std.mem.Allocator) !void {
    if (dir[dir.len - 1] != '/') unreachable;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var split_url = std.mem.split(u8, url, "/");
    var filename: []const u8 = undefined;
    while (split_url.next()) |s| {
        filename = s;
    }
    const dir_filename = try std.fmt.allocPrint(
        arena_allocator,
        "{s}{s}",
        .{ dir, filename },
    );
    var child = std.ChildProcess.init(
        &.{
            "curl",
            url,
            "-Lo",
            dir_filename,
            "--create-dirs",
        },
        arena_allocator,
    );
    child.stderr = std.io.getStdErr();
    child.stdout = std.io.getStdOut();

    ensureFileExists(dir_filename, false) catch |err| switch (err) {
        error.FileNotFound, error.FileEmpty => {
            _ = try child.spawnAndWait();
            return;
        },
        else => return err,
    };
}

test "ensureNotNull" {
    var ptr: ?*u8 = null;
    try std.testing.expectError(error.AllocationError, ensurePtrNotNull(ptr));
}
