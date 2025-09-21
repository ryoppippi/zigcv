// This example uses a deep neural network to perform object detection.
// It can be used with either the Caffe face tracking or Tensorflow object detection models that are
// included with OpenCV 3.4
//
// To perform face tracking with the Caffe model, make sure the pinned
// assets are synced locally (e.g. `devenv shell -- download-models`).
//
// To perform object tracking with the Tensorflow model:
//
// Download and extract the model file named "frozen_inference_graph.pb" from:
// http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
//
// You will also need the pbtxt config file:
// https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt
//

const std = @import("std");
const cv = @import("zigcv");
// const downloadFile = cv.utils.downloadFile;
const Mat = cv.Mat;
const Size = cv.Size;
const c_api = cv.c_api;

const cache_dir = "./zig-cache/tmp/";
const model_path = cache_dir ++ "res10_300x300_ssd_iter_140000.caffemodel";
const config_path = cache_dir ++ "deploy.prototxt";

pub fn main() anyerror!void {
    var allocator = std.heap.page_allocator;
    var args = try std.process.argsWithAllocator(allocator);
    const prog = args.next();
    const device_id_char = args.next() orelse {
        std.log.err("usage: {s} [cameraID]", .{prog.?});
        std.os.exit(1);
    };
    args.deinit();

    const device_id = try std.fmt.parseUnsigned(i32, device_id_char, 10);

    // open webcam
    var webcam = try cv.VideoCapture.init();
    try webcam.openDevice(device_id);
    defer webcam.deinit();

    // open display window
    const window_name = "DNN Detection";
    var window = try cv.Window.init(window_name);
    defer window.deinit();

    // prepare image matrix
    var img = try cv.Mat.init();
    defer img.deinit();

    // open DNN object tracking model
    try cv.utils.ensureModelFile("res10_300x300_ssd_iter_140000.caffemodel", allocator);
    try cv.utils.ensureModelFile("deploy.prototxt", allocator);
    var net = cv.Net.readNet(model_path, config_path) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        std.os.exit(1);
    };
    defer net.deinit();

    if (net.isEmpty()) {
        std.debug.print("Error: could not load model\n", .{});
        std.os.exit(1);
    }

    net.setPreferableBackend(.default);
    net.setPreferableTarget(.cpu);

    const ratio: f64 = 1;
    const mean = cv.Scalar.init(104, 177, 123, 0);
    const swap_rgb = false;

    while (true) {
        webcam.read(&img) catch {
            std.debug.print("capture failed", .{});
            std.os.exit(1);
        };
        if (img.isEmpty()) {
            continue;
        }

        var blob = try cv.Blob.initFromImage(
            img,
            ratio,
            Size.init(300, 300),
            mean,
            swap_rgb,
            false,
        );
        defer blob.deinit();

        net.setInput(blob, "");

        var prob = try net.forward("");
        defer prob.deinit();

        var prob_flattened = try prob.reshape(1, 1);
        defer prob_flattened.deinit();
        performDetection(&img, prob_flattened);

        window.imShow(img);
        if (window.waitKey(1) >= 0) {
            break;
        }
    }
}

// performDetection analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of detections, and each detection
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
fn performDetection(frame: *Mat, results: Mat) void {
    const green = cv.Color{ .g = 255 };
    var i: usize = 0;
    while (i < results.total()) {
        var confidence = results.get(f32, 0, i + 2);
        const cols: f32 = @floatFromInt(frame.cols());
        const rows: f32 = @floatFromInt(frame.rows());
        if (confidence > 0.5) {
            var left: i32 = @intFromFloat(results.get(f32, 0, i + 3) * cols);
            var top: i32 = @intFromFloat(results.get(f32, 0, i + 4) * rows);
            var right: i32 = @intFromFloat(results.get(f32, 0, i + 5) * cols);
            var bottom: i32 = @intFromFloat(results.get(f32, 0, i + 6) * rows);
            cv.rectangle(frame, cv.Rect{ .x = left, .y = top, .width = right, .height = bottom }, green, 2);
        }
        i += 7;
    }
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
