const std = @import("std");
const testing = std.testing;
const test_allocator = std.testing.allocator;
const core = @import("../core.zig");
const dnn = @import("../dnn.zig");
const imgcodecs = @import("../imgcodecs.zig");
const utils = @import("../utils.zig");
const Mat = core.Mat;
const Size = core.Size;
const Scalar = core.Scalar;
const Rect = core.Rect;
const Net = dnn.Net;
const Blob = dnn.Blob;

const img_dir = "./libs/gocv/images/";
const cache_dir = "./zig-cache/tmp/";

const caffe_model_url = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel";
const caffe_model_file = cache_dir ++ "bvlc_googlenet.caffemodel";
const caffe_prototext_url = "https://raw.githubusercontent.com/opencv/opencv_extra/20d18acad1bcb312045ea64a239ebe68c8728b88/testdata/dnn/bvlc_googlenet.prototxt";
const caffe_prototext_file = cache_dir ++ "bvlc_googlenet.prototxt";
const tensorflow_model_zip_url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
const tensorflow_model_zip_file = cache_dir ++ "inception5h.zip";
const tensorflow_model_filename = "tensorflow_inception_graph.pb";
const tensorflow_model_file = cache_dir ++ tensorflow_model_filename;
const onnx_model_url = "https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx";
const onnx_model_file = cache_dir ++ "googlenet-9.onnx";

// pub fn downloadModel(url: []const u8, allocator_: std.mem.Allocator) !void {
//     try utils.downloadFile(url, cache_dir, allocator_);
// }

var fixtures_ready = false;

fn ensureDnnFixtures() !void {
    if (fixtures_ready) return;

    try utils.ensureModelFile("bvlc_googlenet.caffemodel", test_allocator);
    try utils.ensureModelFile("bvlc_googlenet.prototxt", test_allocator);
    try utils.ensureModelFile(tensorflow_model_filename, test_allocator);
    try utils.ensureModelFile("imagenet_comp_graph_label_strings.txt", test_allocator);
    try utils.ensureModelFile("googlenet-9.onnx", test_allocator);
    try utils.ensureModelFile("res10_300x300_ssd_iter_140000.caffemodel", test_allocator);
    try utils.ensureModelFile("deploy.prototxt", test_allocator);

    fixtures_ready = true;
}

fn checkNet(net: *Net, allocator: std.mem.Allocator) !void {
    net.setPreferableBackend(.default);
    net.setPreferableTarget(.cpu);

    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var blob = try Blob.initFromImage(
        img,
        1.0,
        Size.init(224, 224),
        Scalar.init(0, 0, 0, 0),
        false,
        false,
    );
    defer blob.deinit();
    try testing.expectEqual(false, blob.mat.isEmpty());

    net.setInput(blob, "data");

    var layer = try net.getLayer(0);
    defer layer.deinit();

    try testing.expectEqual(@as(i32, -1), layer.inputNameToIndex("notthere"));
    try testing.expectEqual(@as(i32, -1), layer.outputNameToIndex("notthere"));
    try testing.expectEqualStrings("_input", layer.getName());
    try testing.expectEqualStrings("", layer.getType());

    var ids = try net.getUnconnectedOutLayers(allocator);
    defer ids.deinit();
    try testing.expectEqual(@as(usize, 1), ids.items.len);
    try testing.expectEqual(@as(i32, 142), ids.items[0]);

    var lnames = try net.getLayerNames(allocator);
    defer lnames.deinit();

    try testing.expectEqual(@as(usize, 142), lnames.items.len);

    var err_happend = false;
    try testing.expectEqualStrings("conv1/relu_7x7", lnames.items[1]);
    var cs = [_][]const u8{"prob"};
    var prob = try net.forwardLayers(&cs, allocator);
    defer prob.deinit();
    try testing.expect(prob.list.items.len > 0);
    try testing.expectEqual(false, prob.list.items[0].isEmpty());

    var prob_mat = try prob.list.items[0].reshape(1, 1);
    defer prob_mat.deinit();

    const minmax = prob_mat.minMaxLoc();

    try testing.expectApproxEqRel(@as(f64, 0.9998), minmax.max_val, 0.00005);
    try testing.expectEqual(@as(i32, 955), minmax.min_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.min_loc.y);
    try testing.expectEqual(@as(i32, 812), minmax.max_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.max_loc.y);

    const perf = net.getPerfProfile();
    try testing.expect(@as(usize, 0) != perf);

    if (err_happend) return error.SkipZigTest;
}

test "dnn read net from disk" {
    try ensureDnnFixtures();
    // try downloadModel(caffe_model_url, test_allocator);
    // try downloadModel(caffe_prototext_url, test_allocator);
    var net = try Net.readNet(
        caffe_model_file,
        caffe_prototext_file,
    );
    defer net.deinit();
    try testing.expectEqual(false, net.isEmpty());

    try checkNet(&net, test_allocator);
}

test "dnn read net from memory" {
    try ensureDnnFixtures();
    // try downloadModel(caffe_model_url, test_allocator);
    // try downloadModel(caffe_prototext_url, test_allocator);

    var model_file = try std.fs.cwd().openFile(caffe_model_file, .{});
    const m_stat = try std.fs.cwd().statFile(caffe_model_file);
    defer model_file.close();
    var model = try model_file.reader().readAllAlloc(
        test_allocator,
        m_stat.size,
    );
    defer test_allocator.free(model);

    var config_file = try std.fs.cwd().openFile(caffe_prototext_file, .{});
    const c_stat = try std.fs.cwd().statFile(caffe_prototext_file);
    defer config_file.close();
    var config = try config_file.reader().readAllAlloc(
        test_allocator,
        c_stat.size,
    );
    defer test_allocator.free(config);

    var net = try Net.readNetFromBytes("caffe", model, config);
    defer net.deinit();

    try checkNet(&net, test_allocator);
}

fn checkCaffeNet(net: *Net) !void {
    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var blob = try Blob.initFromImage(
        img,
        1.0,
        Size.init(224, 224),
        Scalar.init(0, 0, 0, 0),
        false,
        false,
    );
    defer blob.deinit();
    try testing.expectEqual(false, blob.mat.isEmpty());

    net.setInput(blob, "data");

    var prob = try net.forward("prob");
    defer prob.deinit();
    try testing.expectEqual(false, prob.isEmpty());
    var prob_mat = try prob.reshape(1, 1);
    defer prob_mat.deinit();

    const minmax = prob_mat.minMaxLoc();

    try testing.expectApproxEqRel(@as(f64, 0.9998), minmax.max_val, 0.00005);
    try testing.expectEqual(@as(i32, 955), minmax.min_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.min_loc.y);
    try testing.expectEqual(@as(i32, 812), minmax.max_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.max_loc.y);
}

test "dnn read caffe disk" {
    try ensureDnnFixtures();
    // try downloadModel(caffe_model_url, test_allocator);
    // try downloadModel(caffe_prototext_url, test_allocator);
    var net = try Net.readNetFromCaffe(
        caffe_prototext_file,
        caffe_model_file,
    );
    defer net.deinit();

    try checkCaffeNet(&net);
}

test "dnn read caffe memory" {
    try ensureDnnFixtures();
    // try downloadModel(caffe_model_url, test_allocator);
    // try downloadModel(caffe_prototext_url, test_allocator);

    var model_file = try std.fs.cwd().openFile(caffe_model_file, .{});
    const m_stat = try std.fs.cwd().statFile(caffe_model_file);
    defer model_file.close();
    var model = try model_file.reader().readAllAlloc(
        test_allocator,
        m_stat.size,
    );
    defer test_allocator.free(model);

    var config_file = try std.fs.cwd().openFile(caffe_prototext_file, .{});
    const c_stat = try std.fs.cwd().statFile(caffe_prototext_file);
    defer config_file.close();
    var config = try config_file.reader().readAllAlloc(
        test_allocator,
        c_stat.size,
    );
    defer test_allocator.free(config);

    var net = try Net.readNetFromCaffeBytes(config, model);
    defer net.deinit();

    try checkCaffeNet(&net);
}

fn checkTensorflow(net: *Net) !void {
    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .color);
    defer img.deinit();

    var blob = try Blob.initFromImage(
        img,
        1.0,
        Size.init(224, 224),
        Scalar.init(0, 0, 0, 0),
        true,
        false,
    );
    defer blob.deinit();

    net.setInput(blob, "input");
    var prob = try net.forward("softmax2");
    defer prob.deinit();
    try testing.expectEqual(false, prob.isEmpty());

    var prob_mat = try prob.reshape(1, 1);
    defer prob_mat.deinit();

    const minmax = prob_mat.minMaxLoc();

    try testing.expectApproxEqRel(@as(f64, 1.0), minmax.max_val, 0.00005);
    try testing.expectEqual(@as(i32, 481), minmax.min_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.min_loc.y);
    try testing.expectEqual(@as(i32, 234), minmax.max_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.max_loc.y);
}

fn downloadTFModel() !void {
    try ensureDnnFixtures();
}

test "dnn read tensorflow disk" {
    try downloadTFModel();
    var net = try Net.readNetFromTensorflow(tensorflow_model_file);
    defer net.deinit();

    try checkTensorflow(&net);
}

test "dnn read tensorflow memory" {
    try downloadTFModel();
    var model_file = try std.fs.cwd().openFile(tensorflow_model_file, .{});
    const m_stat = try std.fs.cwd().statFile(tensorflow_model_file);
    defer model_file.close();
    var model = try model_file.reader().readAllAlloc(
        test_allocator,
        m_stat.size,
    );
    defer test_allocator.free(model);

    var net = try Net.readNetFromTensorflowBytes(model);
    defer net.deinit();

    try checkTensorflow(&net);
}

fn checkONNX(net: *Net) !void {
    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var blob = try Blob.initFromImage(
        img,
        1.0,
        Size.init(224, 224),
        Scalar.init(0, 0, 0, 0),
        true,
        false,
    );
    defer blob.deinit();
    try testing.expectEqual(false, blob.mat.isEmpty());

    net.setInput(blob, "data_0");
    var prob = try net.forward("prob_1");
    defer prob.deinit();
    try testing.expectEqual(false, prob.isEmpty());

    var prob_mat = try prob.reshape(1, 1);
    defer prob_mat.deinit();

    const minmax = prob_mat.minMaxLoc();

    try testing.expectApproxEqRel(@as(f64, 0.9965), minmax.max_val, 0.0005);
    try testing.expectEqual(@as(i32, 955), minmax.min_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.min_loc.y);
    try testing.expectEqual(@as(i32, 812), minmax.max_loc.x);
    try testing.expectEqual(@as(i32, 0), minmax.max_loc.y);
}

test "dnn read onnx disk" {
    try ensureDnnFixtures();
    // try downloadModel(onnx_model_url, test_allocator);
    var net = try Net.readNetFromONNX(onnx_model_file);
    defer net.deinit();

    try checkONNX(&net);
}

test "dnn read onnx memory" {
    try ensureDnnFixtures();
    // try downloadModel(onnx_model_url, test_allocator);
    var model_file = try std.fs.cwd().openFile(onnx_model_file, .{});
    const m_stat = try std.fs.cwd().statFile(onnx_model_file);
    defer model_file.close();
    var model = try model_file.reader().readAllAlloc(
        test_allocator,
        m_stat.size,
    );
    defer test_allocator.free(model);

    var net = try Net.readNetFromONNXBytes(model);
    defer net.deinit();

    try checkONNX(&net);
}

test "dnn blob initFromImage Grayscale" {
    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var blob = try dnn.Blob.initFromImage(
        img,
        1.0,
        Size.init(100, 100),
        Scalar.init(0, 0, 0, 0),
        false,
        false,
    );
    defer blob.deinit();
    try testing.expectEqual(false, blob.mat.isEmpty());
}

test "dnn blob initFromImage Channel" {
    var img = try Mat.initSize(100, 100, .cv32fc3);
    defer img.deinit();

    var blob = try Blob.initFromImage(
        img,
        1.0,
        Size.init(100, 100),
        Scalar.init(0, 0, 0, 0),
        true,
        false,
    );
    defer blob.deinit();

    var ch2 = try blob.getChannel(0, 1);
    defer ch2.deinit();

    try testing.expectEqual(false, ch2.isEmpty());
    try testing.expectEqual(@as(i32, 100), ch2.rows());
    try testing.expectEqual(@as(i32, 100), ch2.cols());
}

test "dnn blob initFromImage Size" {
    var img = try Mat.initSize(100, 100, .cv32fc3);
    defer img.deinit();

    var blob = try Blob.initFromImage(
        img,
        1.0,
        Size.init(100, 100),
        Scalar.init(0, 0, 0, 0),
        true,
        false,
    );
    defer blob.deinit();

    var sz = (try blob.getSize()).toArray();

    try testing.expectEqual(@as(f64, 1), sz[0]);
    try testing.expectEqual(@as(f64, 3), sz[1]);
    try testing.expectEqual(@as(f64, 100), sz[2]);
    try testing.expectEqual(@as(f64, 100), sz[3]);
}

test "dnn blob initFromImages" {
    var imgs = std.ArrayList(Mat).init(test_allocator);
    defer imgs.deinit();

    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    try imgs.append(img);
    try imgs.append(img);

    var blob = try Blob.initFromImages(
        imgs.items,
        1.0,
        Size.init(25, 25),
        Scalar.init(0, 0, 0, 0),
        false,
        false,
        .cv32fc1,
    );
    defer blob.deinit();

    var sz = (try blob.getSize()).toArray();
    try testing.expectEqual(@as(f64, 2), sz[0]);
    try testing.expectEqual(@as(f64, 3), sz[1]);
    try testing.expectEqual(@as(f64, 25), sz[2]);
    try testing.expectEqual(@as(f64, 25), sz[3]);
}

test "dnn blob getImages" {
    var img = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var imgs = [_]Mat{ img, img, img };

    var blob = try Blob.initFromImages(
        &imgs,
        1.0,
        Size.init(25, 25),
        Scalar.init(0, 0, 0, 0),
        false,
        false,
        .cv32fc1,
    );
    defer blob.deinit();

    var imgs_from_blob = try blob.getImages(test_allocator);
    defer imgs_from_blob.deinit();

    {
        var i: usize = 0;
        while (i < imgs_from_blob.list.items.len) : (i += 1) {
            var imgi = imgs_from_blob.list.items[i];
            var img_from_blob = try Mat.init();
            defer img_from_blob.deinit();
            imgi.convertTo(&img_from_blob, imgi.getType());
            try testing.expectEqual(false, img_from_blob.isEmpty());
            var diff = try Mat.init();
            defer diff.deinit();
            Mat.compare(imgi, img_from_blob, &diff, .ne);
            const nz = Mat.countNonZero(diff);
            try testing.expectEqual(@as(i32, 0), nz);
        }
    }
}

test "dnn nmsboxes" {
    var img = try imgcodecs.imRead(img_dir ++ "face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    img.convertTo(&img, .cv32fc1);

    comptime var bboxes = [_]Rect{
        Rect.init(53, 47, 589 - 53, 451 - 47),
        Rect.init(118, 54, 618 - 118, 450 - 54),
        Rect.init(53, 66, 605 - 53, 480 - 66),
        Rect.init(111, 65, 630 - 111, 480 - 65),
        Rect.init(156, 51, 640 - 156, 480 - 51),
    };
    comptime var scores = [_]f32{ 0.82094115, 0.7998236, 0.9809663, 0.99717456, 0.89628726 };
    const score_threshold: f32 = 0.5;
    const nms_threshold: f32 = 0.4;
    const max_index: usize = 1;

    var indices = try dnn.nmsBoxes(
        bboxes[0..],
        scores[0..],
        score_threshold,
        nms_threshold,
        max_index,
        test_allocator,
    );
    defer indices.deinit();

    try testing.expectEqual(@as(i32, 3), indices.items[0]);
}

test "dnn nmsboxesWithParams" {
    var img = try imgcodecs.imRead(img_dir ++ "face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    img.convertTo(&img, .cv32fc1);

    comptime var bboxes = [_]Rect{
        Rect.init(53, 47, 589 - 53, 451 - 47),
        Rect.init(118, 54, 618 - 118, 450 - 54),
        Rect.init(53, 66, 605 - 53, 480 - 66),
        Rect.init(111, 65, 630 - 111, 480 - 65),
        Rect.init(156, 51, 640 - 156, 480 - 51),
    };
    comptime var scores = [_]f32{ 0.82094115, 0.7998236, 0.9809663, 0.99717456, 0.89628726 };
    const score_threshold: f32 = 0.5;
    const nms_threshold: f32 = 0.4;
    const max_index: usize = 1;
    const eta: f32 = 1.0;
    const top_k: i32 = 0;

    var indices = try dnn.nmsBoxesWithParams(
        bboxes[0..],
        scores[0..],
        score_threshold,
        nms_threshold,
        eta,
        top_k,
        max_index,
        test_allocator,
    );
    defer indices.deinit();

    try testing.expectEqual(@as(i32, 3), indices.items[0]);
}
