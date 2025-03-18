const std = @import("std");
const cv = @import("zigcv");

pub fn main() anyerror!void {
    // open webcam
    var webcam = try cv.VideoCapture.init();
    try webcam.openDevice(0);
    defer webcam.deinit();

    // open display window
    const window_name = "Hello";
    var window = try cv.Window.init(window_name);
    defer window.deinit();

    var img = try cv.Mat.init();
    defer img.deinit();

    var img2 = try cv.Mat.init();
    defer img2.deinit();

    while (true) {
        webcam.read(&img) catch {
            std.debug.print("capture failed", .{});
            std.process.exit(1);
        };
        cv.detailEnhance(img, &img2, 100, 0.5);

        window.imShow(img2);
        if (window.waitKey(1) >= 0) {
            break;
        }
    }
}
