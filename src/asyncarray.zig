const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const Mat = core.Mat;

pub const AsyncArray = struct {
    ptr: c.AsyncArray,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.AsyncArray_New();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.AsyncArray) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.AsyncArray_Close(self.ptr);
        self.ptr = null;
    }

    pub fn get(self: Self, mat: *Mat) !void {
        const result = c.AsyncArray_GetAsync(self.ptr, mat.*.ptr);
        if (result[0] != 0) {
            return error.RuntimeError;
        }
    }
};

test "AsyncArray" {
    var aa = try AsyncArray.init();
    defer aa.deinit();
    try std.testing.expect(aa.ptr != null);
}

//*    implementation done
//*    pub const AsyncArray = ?*anyopaque;
//*    pub extern fn AsyncArray_New(...) AsyncArray;
//*    pub extern fn AsyncArray_GetAsync(async_out: AsyncArray, out: Mat) [*c]const u8;
//*    pub extern fn AsyncArray_Close(a: AsyncArray) void;
//*    pub extern fn Net_forwardAsync(net: Net, outputName: [*c]const u8) AsyncArray;
