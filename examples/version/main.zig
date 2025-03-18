const std = @import("std");
const cv = @import("zigcv");

pub fn main() !void {
    std.debug.print("version via zig binding:\t{s}\n", .{cv.openCVVersion()});
    std.debug.print("version via c api directly:\t{s}\n", .{cv.c.openCVVersion()});
}
