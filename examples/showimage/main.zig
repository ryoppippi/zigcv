const std = @import("std");
const cv = @import("zigcv");
const ma = @import("zigcv").ma;
const cv_c_api = cv.c_api;

pub fn main() anyerror!void {
    ma.init(.{ .allocator = std.heap.page_allocator });
    defer ma.deinit();

    var args = try std.process.argsWithAllocator(std.heap.page_allocator);
    defer args.deinit();
    const prog = args.next();
    const img_PATH = args.next() orelse {
        std.log.err("usage: {s} [image_PATH]", .{prog.?});
        std.os.exit(1);
    };

    // open display window
    const window_name = "Show Image";
    var window = try cv.Window.init(window_name);
    defer window.deinit();

    var img = try cv.imRead(img_PATH, .unchanged);
    defer img.deinit();
    while (true) {
        window.imShow(img);
        if (window.waitKey(1) >= 0) {
            break;
        }
    }
}
