const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const ensureFileExists = utils.ensureFileExists;
const Mat = core.Mat;
const Mats = core.Mats;
const Size = core.Size;
const Scalar = core.Scalar;
const Rect = core.Rect;
const AsyncArray = @import("asyncarray.zig").AsyncArray;

pub const Net = struct {
    ptr: c.Net,

    const Self = @This();

    pub const BackendType = enum(i32) {
        /// NetBackendDefault is the default backend.
        default = 0,

        /// NetBackendHalide is the Halide backend.
        halide = 1,

        /// NetBackendOpenVINO is the OpenVINO backend.
        openvino = 2,

        /// NetBackendOpenCV is the OpenCV backend.
        opencv = 3,

        /// NetBackendVKCOM is the Vulkan backend.
        vkcom = 4,

        /// NetBackendCUDA is the Cuda backend.
        cuda = 5,

        /// ParseNetBackend returns a valid NetBackendType given a string. Valid values are:
        /// - halide
        /// - openvino
        /// - opencv
        /// - vulkan
        /// - cuda
        /// - default
        pub fn toEnum(backend: []const u8) BackendType {
            return std.meta.stringToEnum(BackendType, backend) orelse .default;
        }
    };

    pub const TargetType = enum(i32) {
        /// NetTargetCPU is the default CPU device target.
        cpu = 0,

        /// NetTargetFP32 is the 32-bit OpenCL target.
        fp32 = 1,

        /// NetTargetFP16 is the 16-bit OpenCL target.
        fp16 = 2,

        /// NetTargetVPU is the Movidius VPU target.
        vpu = 3,

        /// NetTargetVulkan is the NVIDIA Vulkan target.
        vulkan = 4,

        /// NetTargetFPGA is the FPGA target.
        fpga = 5,

        /// NetTargetCUDA is the CUDA target.
        cuda = 6,

        /// NetTargetCUDAFP16 is the CUDA target.
        cuda_fp16 = 7,

        pub fn toEnum(target: []const u8) TargetType {
            return std.meta.stringToEnum(TargetType, target) orelse .cpu;
        }
    };

    fn initFromC(ptr: c.Net) !Self {
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.Net_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn readNet(model: []const u8, config: []const u8) !Self {
        _ = try ensureFileExists(model, false);
        _ = try ensureFileExists(config, false);
        const nn_ptr = c.Net_ReadNet(@as([*]const u8, @ptrCast(model)), @as([*]const u8, @ptrCast(config)));
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromBytes(framework: []const u8, model: []u8, config: []u8) !Self {
        if (framework.len == 0) return error.InvalidFramework;
        if (model.len == 0) return error.ModelIsNotProvided;
        if (config.len == 0) return error.ConfigIsNotProvided;
        const c_model = core.toByteArray(model);
        const c_config = core.toByteArray(config);
        const c_framework = @as([*]const u8, @ptrCast(framework));
        const nn_ptr = c.Net_ReadNetBytes(c_framework, c_model, c_config);
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromCaffe(prototxt: []const u8, caffe_model: []const u8) !Self {
        _ = try ensureFileExists(prototxt, false);
        _ = try ensureFileExists(caffe_model, false);
        const nn_ptr = c.Net_ReadNetFromCaffe(@as([*]const u8, @ptrCast(prototxt)), @as([*]const u8, @ptrCast(caffe_model)));
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromCaffeBytes(prototxt: []u8, caffe_model: []u8) !Self {
        const c_prototxt = core.toByteArray(prototxt);
        const c_caffe_model = core.toByteArray(caffe_model);
        const nn_ptr = c.Net_ReadNetFromCaffeBytes(c_prototxt, c_caffe_model);
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromTensorflow(model: []const u8) !Self {
        _ = try ensureFileExists(model, false);
        const nn_ptr = c.Net_ReadNetFromTensorflow(@as([*]const u8, @ptrCast(model)));
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromTensorflowBytes(model: []u8) !Self {
        const c_model = core.toByteArray(model);
        const nn_ptr = c.Net_ReadNetFromTensorflowBytes(c_model);
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromTorch(model: []const u8) !Self {
        _ = try ensureFileExists(model, false);
        const nn_ptr = c.Net_ReadNetFromTorch(utils.castZigU8ToC(model));
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromONNX(model: []const u8) !Self {
        _ = try ensureFileExists(model, false);
        const nn_ptr = c.Net_ReadNetFromONNX(@as([*]const u8, @ptrCast(model)));
        return try initFromC(nn_ptr);
    }

    pub fn readNetFromONNXBytes(model: []u8) !Self {
        const c_model = core.toByteArray(model);
        const nn_ptr = c.Net_ReadNetFromONNXBytes(c_model);
        return try initFromC(nn_ptr);
    }

    pub fn isEmpty(self: Self) bool {
        if (self.ptr == null) return false;
        return c.Net_Empty(self.ptr);
    }

    pub fn setInput(self: *Self, blob: Blob, name: []const u8) void {
        _ = c.Net_SetInput(self.ptr, blob.mat.toC(), @as([*]const u8, @ptrCast(name)));
    }

    pub fn forward(self: *Self, output_name: []const u8) !Mat {
        const mat_ptr = c.Net_Forward(self.ptr, @as([*]const u8, @ptrCast(output_name)));
        return try Mat.initFromC(mat_ptr);
    }

    pub fn forwardLayers(self: Self, output_blob_names: [][]const u8, allocator: std.mem.Allocator) !Mats {
        var c_mats: c.Mats = undefined;
        var c_string_array = try allocator.alloc([*]const u8, output_blob_names.len);
        defer allocator.free(c_string_array);
        for (output_blob_names, 0..) |name, i| c_string_array[i] = @as([*]const u8, @ptrCast(name));

        const c_strings = c.CStrings{
            .strs = @as([*c][*c]const u8, @ptrCast(c_string_array.ptr)),
            .length = @as(i32, @intCast(output_blob_names.len)),
        };

        c.Net_ForwardLayers(self.ptr, &c_mats, c_strings);
        return try Mat.toArrayList(c_mats, allocator);
    }

    pub fn forwardAsync(self: *Self, output_name: []const u8) !AsyncArray {
        const aa_ptr = c.Net_ForwardAsync(self.*.ptr, utils.castZigU8ToC(output_name));
        return try AsyncArray.initFromC(aa_ptr);
    }

    // SetPreferableBackend ask network to use specific computation backend.
    //
    // For further details, please see:
    // https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a7f767df11386d39374db49cd8df8f59e
    //
    pub fn setPreferableBackend(self: *Self, backend: BackendType) void {
        _ = c.Net_SetPreferableBackend(self.ptr, @intFromEnum(backend));
    }

    // SetPreferableTarget ask network to make computations on specific target device.
    //
    // For further details, please see:
    // https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a9dddbefbc7f3defbe3eeb5dc3d3483f4
    pub fn setPreferableTarget(self: *Self, target: TargetType) void {
        _ = c.Net_SetPreferableTarget(self.ptr, @intFromEnum(target));
    }

    // GetPerfProfile returns overall time for inference and timings (in ticks) for layers
    //
    // For further details, please see:
    // https://docs.opencv.org/master/db/d30/classcv_1_1dnn_1_1Net.html#a06ce946f675f75d1c020c5ddbc78aedc
    //
    pub fn getPerfProfile(self: Self) f64 {
        return @as(f64, @floatFromInt(c.Net_GetPerfProfile(self.ptr)));
    }

    pub fn getUnconnectedOutLayers(self: Self, allocator: std.mem.Allocator) !std.ArrayList(i32) {
        var res: c.IntVector = undefined;
        defer c.IntVector_Close(res);
        _ = c.Net_GetUnconnectedOutLayers(self.ptr, &res);
        var return_res = std.ArrayList(i32).init(allocator);
        {
            var i: usize = 0;
            while (i < res.length) : (i += 1) {
                try return_res.append(res.val[i]);
            }
        }
        return return_res;
    }

    pub fn getLayerNames(self: *Self, allocator: std.mem.Allocator) !struct {
        items: []const []const u8,
        arena: std.heap.ArenaAllocator,

        pub fn deinit(self_: *@This()) void {
            self_.arena.deinit();
        }
    } {
        var arena = std.heap.ArenaAllocator.init(allocator);
        var arena_allocator = arena.allocator();

        var c_strs: c.CStrings = undefined;
        defer c.CStrings_Close(c_strs);
        c.Net_GetLayerNames(self.ptr, &c_strs);
        const len = @as(usize, @intCast(c_strs.length));
        const return_array = try arena_allocator.alloc([]const u8, len);
        for (return_array, 0..) |*item, i| {
            item.* = try arena_allocator.dupe(u8, std.mem.span(c_strs.strs[i]));
        }
        return .{
            .items = return_array,
            .arena = arena,
        };
    }

    pub fn getLayer(self: Self, layerid: i32) !Layer {
        return try Layer.initFromC(c.Net_GetLayer(self.ptr, layerid));
    }
};

pub const Blob = struct {
    mat: Mat,

    const Self = @This();

    // BlobFromImage creates 4-dimensional blob from image. Optionally resizes and crops
    // image from center, subtract mean values, scales values by scalefactor,
    // swap Blue and Red channels.
    //
    // For further details, please see:
    // https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga152367f253c81b53fe6862b299f5c5cd
    //
    pub fn initFromImage(
        image: Mat,
        scalefactor: f64,
        size: Size,
        mean: Scalar,
        swap_rb: bool,
        crop: bool,
    ) !Self {
        const new_c_blob = c.Net_BlobFromImage(
            image.toC(),
            scalefactor,
            size.toC(),
            mean.toC(),
            swap_rb,
            crop,
        );
        const new_blob_mat = try Mat.initFromC(new_c_blob);
        return try initFromMat(new_blob_mat);
    }

    // BlobFromImages Creates 4-dimensional blob from series of images.
    // Optionally resizes and crops images from center, subtract mean values,
    // scales values by scalefactor, swap Blue and Red channels.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga2b89ed84432e4395f5a1412c2926293c
    //
    pub fn initFromImages(
        images: []const Mat,
        scalefactor: f64,
        size: Size,
        mean: Scalar,
        swap_r_b: bool,
        crop: bool,
        ddepth: Mat.MatType,
    ) !Self {
        var new_blob_mat = try Mat.init();
        const c_mats = try Mat.toCStructs(images);
        c.Net_BlobFromImages(
            c_mats,
            new_blob_mat.toC(),
            scalefactor,
            size.toC(),
            mean.toC(),
            swap_r_b,
            crop,
            @intFromEnum(ddepth),
        );
        return try initFromMat(new_blob_mat);
    }

    pub fn initFromMat(mat: Mat) !Self {
        _ = try epnn(mat.ptr);
        return Self{ .mat = mat };
    }

    pub fn deinit(self: *Self) void {
        self.mat.deinit();
    }

    pub fn getSize(self: Self) !Scalar {
        return Scalar.initFromC(c.Net_GetBlobSize(self.mat.toC()));
    }

    pub fn getChannel(self: Self, imgidx: i32, chnidx: i32) !Mat {
        return try Mat.initFromC(c.Net_GetBlobChannel(self.mat.toC(), imgidx, chnidx));
    }

    /// ImagesFromBlob Parse a 4D blob and output the images it contains as
    /// 2D arrays through a simpler data structure (std::vector<cv::Mat>).
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga4051b5fa2ed5f54b76c059a8625df9f5
    ///
    pub fn getImages(self: Self, allocator: std.mem.Allocator) !Mats {
        var c_mats: c.Mats = undefined;
        _ = c.Net_ImagesFromBlob(self.mat.toC(), &c_mats);
        return try Mat.toArrayList(c_mats, allocator);
    }
};

pub const Layer = struct {
    ptr: c.Layer,

    const Self = @This();

    pub fn initFromC(ptr: c.Layer) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.Layer_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn toC(self: Self) c.Layer {
        return self.ptr;
    }

    pub fn inputNameToIndex(self: *Self, name: []const u8) i32 {
        return c.Layer_InputNameToIndex(self.ptr, @as([*]const u8, @ptrCast(name)));
    }

    pub fn outputNameToIndex(self: *Self, name: []const u8) i32 {
        return c.Layer_OutputNameToIndex(self.ptr, @as([*]const u8, @ptrCast(name)));
    }

    pub fn getName(self: Self) []const u8 {
        return std.mem.span(c.Layer_GetName(self.ptr));
    }

    pub fn getType(self: Self) []const u8 {
        return std.mem.span(c.Layer_GetType(self.ptr));
    }
};

/// NMSBoxes performs non maximum suppression given boxes and corresponding scores.
///
/// For futher details, please see:
/// https://docs.opencv.org/4.4.0/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee
pub fn nmsBoxes(
    bboxes: []Rect,
    scores: []f32,
    score_threshold: f32,
    nms_threshold: f32,
    max_index: usize,
    allocator: std.mem.Allocator,
) !std.ArrayList(i32) {
    var c_bboxes_array = try allocator.alloc(c.Rect, bboxes.len);
    defer allocator.free(c_bboxes_array);
    for (bboxes, 0..) |bbox, i| c_bboxes_array[i] = bbox.toC();
    const c_bboxes_struct = c.Rects{
        .rects = @as([*]c.Rect, @ptrCast(c_bboxes_array.ptr)),
        .length = @as(i32, @intCast(bboxes.len)),
    };

    const c_scores_struct = c.FloatVector{
        .val = @as([*]f32, @ptrCast(scores.ptr)),
        .length = @as(i32, @intCast(scores.len)),
    };

    var indices_vector: c.IntVector = undefined;
    indices_vector.length = @as(i32, @intCast(max_index));

    c.NMSBoxes(
        c_bboxes_struct,
        c_scores_struct,
        score_threshold,
        nms_threshold,
        &indices_vector,
    );
    defer c.IntVector_Close(indices_vector);

    const len = @as(usize, @intCast(indices_vector.length));
    var indices = try std.ArrayList(i32).initCapacity(allocator, len);
    {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            try indices.append(indices_vector.val[i]);
        }
    }
    return indices;
}

/// NMSBoxesWithParams performs non maximum suppression given boxes and corresponding scores.
///
/// For futher details, please see:
/// https://docs.opencv.org/4.4.0/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee
pub fn nmsBoxesWithParams(
    bboxes: []Rect,
    scores: []f32,
    score_threshold: f32,
    nms_threshold: f32,
    eta: f32,
    top_k: i32,
    max_index: usize,
    allocator: std.mem.Allocator,
) !std.ArrayList(i32) {
    var c_bboxes_array = try allocator.alloc(c.Rect, bboxes.len);
    defer allocator.free(c_bboxes_array);
    for (bboxes, 0..) |bbox, i| c_bboxes_array[i] = bbox.toC();
    const c_bboxes_struct = c.Rects{
        .rects = @as([*]c.Rect, @ptrCast(c_bboxes_array.ptr)),
        .length = @as(i32, @intCast(bboxes.len)),
    };

    const c_scores_struct = c.FloatVector{
        .val = @as([*]f32, @ptrCast(scores.ptr)),
        .length = @as(i32, @intCast(scores.len)),
    };

    var indices_vector: c.IntVector = undefined;
    indices_vector.length = @as(i32, @intCast(max_index));

    c.NMSBoxesWithParams(
        c_bboxes_struct,
        c_scores_struct,
        score_threshold,
        nms_threshold,
        &indices_vector,
        eta,
        top_k,
    );
    defer c.IntVector_Close(indices_vector);

    const len = @as(usize, @intCast(indices_vector.length));
    var indices = try std.ArrayList(i32).initCapacity(allocator, len);
    {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            try indices.append(indices_vector.val[i]);
        }
    }
    return indices;
}

test "dnn" {
    _ = @import("dnn/test.zig");
}

//*    implementation done
//*    pub const Net = ?*anyopaque;
//*    pub const Layer = ?*anyopaque;
//*    pub extern fn Net_ReadNet(model: [*c]const u8, config: [*c]const u8) Net;
//*    pub extern fn Net_ReadNetBytes(framework: [*c]const u8, model: struct_ByteArray, config: struct_ByteArray) Net;
//*    pub extern fn Net_ReadNetFromCaffe(prototxt: [*c]const u8, caffeModel: [*c]const u8) Net;
//*    pub extern fn Net_ReadNetFromCaffeBytes(prototxt: struct_ByteArray, caffeModel: struct_ByteArray) Net;
//*    pub extern fn Net_ReadNetFromTensorflow(model: [*c]const u8) Net;
//*    pub extern fn Net_ReadNetFromTensorflowBytes(model: struct_ByteArray) Net;
//*    pub extern fn Net_ReadNetFromTorch(model: [*c]const u8) Net;
//*    pub extern fn Net_ReadNetFromONNX(model: [*c]const u8) Net;
//*    pub extern fn Net_ReadNetFromONNXBytes(model: struct_ByteArray) Net;
//*    pub extern fn Net_BlobFromImage(image: Mat, scalefactor: f64, size: Size, mean: Scalar, swapRB: bool, crop: bool) Mat;
//*    pub extern fn Net_BlobFromImages(images: struct_Mats, blob: Mat, scalefactor: f64, size: Size, mean: Scalar, swapRB: bool, crop: bool, ddepth: c_int) void;
//*    pub extern fn Net_ImagesFromBlob(blob_: Mat, images_: [*c]struct_Mats) void;
//*    pub extern fn Net_Close(net: Net) void;
//*    pub extern fn Net_Empty(net: Net) bool;
//*    pub extern fn Net_SetInput(net: Net, blob: Mat, name: [*c]const u8) void;
//*    pub extern fn Net_Forward(net: Net, outputName: [*c]const u8) Mat;
//*    pub extern fn Net_ForwardLayers(net: Net, outputBlobs: [*c]struct_Mats, outBlobNames: struct_CStrings) void;
//*    pub extern fn Net_SetPreferableBackend(net: Net, backend: c_int) void;
//*    pub extern fn Net_SetPreferableTarget(net: Net, target: c_int) void;
//*    pub extern fn Net_GetPerfProfile(net: Net) i64;
//*    pub extern fn Net_GetUnconnectedOutLayers(net: Net, res: [*c]IntVector) void;
//*    pub extern fn Net_GetLayerNames(net: Net, names: [*c]CStrings) void;
//*    pub extern fn Net_GetBlobChannel(blob: Mat, imgidx: c_int, chnidx: c_int) Mat;
//*    pub extern fn Net_GetBlobSize(blob: Mat) Scalar;
//*    pub extern fn Net_GetLayer(net: Net, layerid: c_int) Layer;
//*    pub extern fn Layer_Close(layer: Layer) void;
//*    pub extern fn Layer_InputNameToIndex(layer: Layer, name: [*c]const u8) c_int;
//*    pub extern fn Layer_OutputNameToIndex(layer: Layer, name: [*c]const u8) c_int;
//*    pub extern fn Layer_GetName(layer: Layer) [*c]const u8;
//*    pub extern fn Layer_GetType(layer: Layer) [*c]const u8;
//*    pub extern fn NMSBoxes(bboxes: struct_Rects, scores: FloatVector, score_threshold: f32, nms_threshold: f32, indices: [*c]IntVector) void;
//*    pub extern fn NMSBoxesWithParams(bboxes: struct_Rects, scores: FloatVector, score_threshold: f32, nms_threshold: f32, indices: [*c]IntVector, eta: f32, top_k: c_int) void;

// TODO: FP16BlobFromImage
