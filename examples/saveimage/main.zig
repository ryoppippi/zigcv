const std = @import("std");
const cv = @import("zigcv");

pub fn main() anyerror!void {
    const allocator = std.heap.page_allocator;
    var args = try std.process.argsWithAllocator(allocator);
    const prog = args.next();
    const device_id_char = args.next() orelse {
        std.log.err("usage: {s} [cameraID]", .{prog.?});
        std.process.exit(1);
    };
    args.deinit();

    const device_id = try std.fmt.parseUnsigned(i32, device_id_char, 10);

    // open webcam
    var webcam = try cv.VideoCapture.init();
    try webcam.openDevice(device_id);
    defer webcam.deinit();

    var img = try cv.Mat.init();
    defer img.deinit();

    try webcam.read(&img);

    if (img.isEmpty()) return error.NoImage;

    try cv.imWrite("saveimg.png", img);
}
