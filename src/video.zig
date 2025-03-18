const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const Mat = core.Mat;
const Rect = core.Rect;
const Rects = core.Rects;
const Size = core.Size;
const TermCriteria = core.TermCriteria;

/// For further details, please see: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ggaaedb1f94e6b143cef163622c531afd88a01106d6d20122b782ff25eaeffe9a5be
pub const Motion = enum(u2) {
    /// cv::MOTION_TRANSLATION = 0,
    translation = 0,
    /// cv::MOTION_EUCLIDEAN = 1,
    euclidean = 1,
    /// cv::MOTION_AFFINE = 2,
    affine = 2,
    /// cv::MOTION_HOMOGRAPHY = 3
    homography = 3,
};

/// BackgroundSubtractorMOG2 is a wrapper around the cv::BackgroundSubtractorMOG2.
const BackgroundSubtractorMOG2 = struct {
    ptr: c.BackgroundSubtractorMOG2,

    const Self = @This();

    /// NewBackgroundSubtractorMOG2 returns a new BackgroundSubtractor algorithm
    /// of type MOG2. MOG2 is a Gaussian Mixture-based Background/Foreground
    /// Segmentation Algorithm.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/de1/group__video__motion.html#ga2beb2dee7a073809ccec60f145b6b29c
    /// https://docs.opencv.org/master/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    ///
    pub fn init() !Self {
        const ptr = c.BackgroundSubtractorMOG2_Create();
        return try initFromC(ptr);
    }

    /// NewBackgroundSubtractorMOG2WithParams returns a new BackgroundSubtractor algorithm
    /// of type MOG2 with customized parameters. MOG2 is a Gaussian Mixture-based Background/Foreground
    /// Segmentation Algorithm.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/de1/group__video__motion.html#ga2beb2dee7a073809ccec60f145b6b29c
    /// https://docs.opencv.org/master/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    ///
    pub fn initWithParams(history: i32, var_threshold: f64, detect_shadows: bool) !Self {
        const ptr = c.BackgroundSubtractorMOG2_CreateWithParams(history, var_threshold, detect_shadows);
        return try initFromC(ptr);
    }

    fn initFromC(ptr: c.BackgroundSubtractorMOG2) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close BackgroundSubtractorMOG2.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        _ = c.BackgroundSubtractorMOG2_Close(self.ptr);
        self.*.ptr = null;
    }

    /// Apply computes a foreground mask using the current BackgroundSubtractorMOG2.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
    ///
    pub fn apply(self: *Self, src: Mat, dst: *Mat) void {
        _ = c.BackgroundSubtractorMOG2_Apply(self.ptr, src.toC(), dst.*.toC());
    }
};

/// BackgroundSubtractorKNN is a wrapper around the cv::BackgroundSubtractorKNN.
pub const BackgroundSubtractorKNN = struct {
    ptr: c.BackgroundSubtractorKNN,

    const Self = @This();

    /// NewBackgroundSubtractorKNN returns a new BackgroundSubtractor algorithm
    /// of type KNN. K-Nearest Neighbors (KNN) uses a Background/Foreground
    /// Segmentation Algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d
    /// https://docs.opencv.org/master/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    ///
    pub fn init() !Self {
        const ptr = c.BackgroundSubtractorKNN_Create();
        return try initFromC(ptr);
    }

    /// NewBackgroundSubtractorKNNWithParams returns a new BackgroundSubtractor algorithm
    /// of type KNN with customized parameters. K-Nearest Neighbors (KNN) uses a Background/Foreground
    /// Segmentation Algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d
    /// https://docs.opencv.org/master/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    ///
    pub fn initWithPatams(history: i32, dist2_threshold: f64, detect_shadows: bool) !Self {
        const ptr = c.BackgroundSubtractorKNN_CreateWithParams(history, dist2_threshold, detect_shadows);
        return try initFromC(ptr);
    }

    fn initFromC(ptr: c.BackgroundSubtractorKNN) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close BackgroundSubtractorKNN.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.BackgroundSubtractorKNN_Close(self.ptr);
        self.*.ptr = null;
    }

    /// Apply computes a foreground mask using the current BackgroundSubtractorKNN.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
    ///
    pub fn apply(self: *Self, src: Mat, dst: *Mat) void {
        _ = c.BackgroundSubtractorKNN_Apply(self.ptr, src.toC(), dst.*.toC());
    }
};

pub const CalcOpticalFlow = struct {
    /// For further details, please see: https://docs.opencv.org/master/dc/d6b/group__video__track.html#gga2c6cc144c9eee043575d5b311ac8af08a9d4430ac75199af0cf6fcdefba30eafe
    pub const Optflow = enum(u9) {
        default = 0,
        /// cv::OPTFLOW_USE_INITIAL_FLOW = 4,
        use_initial_flow = 4,
        /// cv::OPTFLOW_LK_GET_MIN_EIGENVALS = 8,
        lk_get_min_eigenvals = 8,
        /// cv::OPTFLOW_FARNEBACK_GAUSSIAN = 256
        farneback_gaussian = 256,
    };

    /// CalcOpticalFlowFarneback computes a dense optical flow using
    /// Gunnar Farneback's algorithm.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    ///
    pub fn farneback(
        prev_img: Mat,
        next_img: Mat,
        flow: *Mat,
        pyr_scale: f64,
        levels: i32,
        winsize: i32,
        iterations: i32,
        poly_n: i32,
        poly_sigma: f64,
        flags: Optflow,
    ) void {
        _ = c.CalcOpticalFlowFarneback(
            prev_img.ptr,
            next_img.ptr,
            flow.*.ptr,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            @intFromEnum(flags),
        );
    }

    /// CalcOpticalFlowPyrLK calculates an optical flow for a sparse feature set using
    /// the iterative Lucas-Kanade method with pyramids.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    ///
    pub fn pyrLK(
        prev_img: Mat,
        next_img: Mat,
        prev_pts: Mat,
        next_pts: Mat,
        status: *Mat,
        err: *Mat,
    ) void {
        _ = c.CalcOpticalFlowPyrLK(
            prev_img.toC(),
            next_img.toC(),
            prev_pts.toC(),
            next_pts.toC(),
            status.*.toC(),
            err.*.toC(),
        );
    }

    /// CalcOpticalFlowPyrLKWithParams calculates an optical flow for a sparse feature set using
    /// the iterative Lucas-Kanade method with pyramids.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    ///
    pub fn pyrLKWithParams(
        prev_img: Mat,
        next_img: Mat,
        prev_pts: Mat,
        next_pts: Mat,
        status: *Mat,
        err: *Mat,
        win_size: Size,
        max_level: i32,
        criteria: TermCriteria,
        flags: Optflow,
        min_eig_threshold: f64,
    ) void {
        _ = c.CalcOpticalFlowPyrLKWithParams(
            prev_img.toC(),
            next_img.toC(),
            prev_pts.toC(),
            next_pts.toC(),
            status.*.toC(),
            err.*.toC(),
            win_size.toC(),
            max_level,
            criteria.toC(),
            @intFromEnum(flags),
            min_eig_threshold,
        );
    }
};

// FindTransformECC finds the geometric transform (warp) between two images in terms of the ECC criterion.
//
// For futther details, please see:
// https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga1aa357007eaec11e9ed03500ecbcbe47
//
pub fn findTransformECC(
    template_image: Mat,
    input_image: Mat,
    warp_matrix: *Mat,
    motion_type: Motion,
    criteria: TermCriteria,
    input_mask: Mat,
    gauss_filt_size: i32,
) f64 {
    return c.FindTransformECC(
        template_image.toC(),
        input_image.toC(),
        warp_matrix.*.toC(),
        @intFromEnum(motion_type),
        criteria.toC(),
        input_mask.toC(),
        gauss_filt_size,
    );
}

/// TrackerMIL is a Tracker that uses the MIL algorithm. MIL trains a classifier in an online manner
/// to separate the object from the background.
/// Multiple Instance Learning avoids the drift problem for a robust tracking.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d0/d26/classcv_1_1TrackerMIL.html
///
pub const Tracker = struct {
    ptr: ?*anyopaque,
    vtable: VTable,

    const Self = @This();

    const UpdateReturn = struct { box: Rect, success: bool };

    const VTable = struct {
        deinitFn: *const fn (self: *Self) void,

        initializeFn: *const fn (self: *Self, image: Mat, bounding_box: Rect) bool,

        updateFn: *const fn (self: *Self, image: Mat) UpdateReturn,
    };

    pub fn init(object: anytype) !Self {
        const T = @TypeOf(object);
        const T_info = @typeInfo(T);

        if (T_info != .pointer) @compileError("ptr must be a pointer");
        if (T_info.pointer.size != .One) @compileError("ptr must be a single item pointer");
        if (!@hasDecl(T_info.pointer.child, "init")) @compileError("object must have an init function");
        if (!@hasDecl(T_info.pointer.child, "deinit")) @compileError("object must have a deinit function");

        const gen = struct {
            pub fn deinit(self: *Self) void {
                assert(self.ptr != null);
                T_info.pointer.child.deinit(self.ptr);
                self.ptr = null;
            }

            pub fn initialize(self: *Self, image: Mat, bounding_box: Rect) bool {
                return c.Tracker_Init(self.ptr, image.toC(), bounding_box.toC());
            }

            pub fn update(self: *Self, image: Mat) UpdateReturn {
                var c_box: c.Rect = undefined;
                const success = c.Tracker_Update(self.ptr, image.toC(), @ptrCast(&c_box));
                const rect = Rect.initFromC(c_box);
                return UpdateReturn{
                    .box = rect,
                    .success = success,
                };
            }
        };

        const t_ptr = try T_info.pointer.child.init();

        return .{
            .ptr = t_ptr,
            .vtable = .{
                .deinitFn = gen.deinit,
                .initializeFn = gen.initialize,
                .updateFn = gen.update,
            },
        };
    }

    /// Initialize initializes the tracker with a known bounding box that surrounded the target.
    /// Note: this can only be called once. If you lose the object, you have to Close() the instance,
    /// create a new one, and call Init() on it again.
    ///
    /// see: https://docs.opencv.org/master/d0/d0a/classcv_1_1Tracker.html#a4d285747589b1bdd16d2e4f00c3255dc
    ///
    pub fn initialize(self: *Self, image: Mat, bounding_box: Rect) bool {
        return self.vtable.initializeFn(self, image, bounding_box);
    }

    /// Update updates the tracker, returns a new bounding box and a boolean determining whether the tracker lost the target.
    ///
    /// see: https://docs.opencv.org/master/d0/d0a/classcv_1_1Tracker.html#a549159bd0553e6a8de356f3866df1f18
    ///
    pub fn update(self: *Self, image: Mat) UpdateReturn {
        return self.vtable.updateFn(self, image);
    }

    pub fn deinit(self: *Self) void {
        self.vtable.deinitFn(self);
    }
};

/// TrackerMIL is a Tracker that uses the MIL algorithm. MIL trains a classifier in an online manner
/// to separate the object from the background.
/// Multiple Instance Learning avoids the drift problem for a robust tracking.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d0/d26/classcv_1_1TrackerMIL.html
///
pub const TrackerMIL = struct {
    const Self = @This();

    fn init() !c.TrackerMIL {
        const p = c.TrackerMIL_Create();
        return try epnn(p);
    }

    fn deinit(ptr: c.TrackerMIL) void {
        c.TrackerMIL_Close(ptr);
    }

    pub fn tracker(self: *Self) !Tracker {
        return try Tracker.init(self);
    }
};

const testing = std.testing;
const imgcodecs = @import("imgcodecs.zig");
const imgproc = @import("imgproc.zig");
const file_path = "test/images/face.jpg";
test "video BackgroundSubtractorMOG2" {
    var img = try imgcodecs.imRead(file_path, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var mog2 = try BackgroundSubtractorMOG2.init();
    defer mog2.deinit();

    mog2.apply(img, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "video BackgroundSubtractorMOG2 with params" {
    var img = try imgcodecs.imRead(file_path, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var mog2 = try BackgroundSubtractorMOG2.initWithParams(250, 8, false);
    defer mog2.deinit();

    mog2.apply(img, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "video BackgroundSubtractorKNN" {
    var img = try imgcodecs.imRead(file_path, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var knn = try BackgroundSubtractorKNN.init();
    defer knn.deinit();

    knn.apply(img, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "video BackgroundSubtractorKNN with params" {
    var img = try imgcodecs.imRead(file_path, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var knn = try BackgroundSubtractorKNN.initWithPatams(250, 200, false);
    defer knn.deinit();

    knn.apply(img, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "video CalcOpticalFlow.pyrLK" {
    var img1 = try imgcodecs.imRead(file_path, .color);
    defer img1.deinit();
    try testing.expectEqual(false, img1.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.cvtColor(img1, &dst, .bgra_to_gray);

    var img2 = try dst.clone();
    defer img2.deinit();

    var prev_pts = try Mat.init();
    defer prev_pts.deinit();

    var next_pts = try Mat.init();
    defer next_pts.deinit();

    var status = try Mat.init();
    defer status.deinit();

    var err = try Mat.init();
    defer err.deinit();
}

test "video CalcOpticalFlow.pyrLK with params" {
    var img1 = try imgcodecs.imRead(file_path, .color);
    defer img1.deinit();
    try testing.expectEqual(false, img1.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.cvtColor(img1, &dst, .bgra_to_gray);

    var img2 = try dst.clone();
    defer img2.deinit();

    var prev_pts = try Mat.init();
    defer prev_pts.deinit();

    var next_pts = try Mat.init();
    defer next_pts.deinit();

    var status = try Mat.init();
    defer status.deinit();

    var err = try Mat.init();
    defer err.deinit();

    var corners = try Mat.init();
    defer corners.deinit();

    imgproc.goodFeaturesToTrack(dst, &corners, 500, 0.01, 10);
    var tc = try core.TermCriteria.init(.{ .count = true, .eps = true }, 20, 0.03);
    defer tc.deinit();
    imgproc.cornerSubPix(
        dst,
        &corners,
        Size.init(10, 10),
        Size.init(-1, -1),
        tc,
    );

    CalcOpticalFlow.pyrLK(
        dst,
        img2,
        corners,
        next_pts,
        &status,
        &err,
    );

    try testing.expectEqual(false, status.isEmpty());
    try testing.expectEqual(@as(i32, 323), status.rows());
    try testing.expectEqual(@as(i32, 1), status.cols());
}

test "video CalcOpticalFlow.pyrLKWithParams" {
    var img1 = try imgcodecs.imRead(file_path, .color);
    defer img1.deinit();
    try testing.expectEqual(false, img1.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.cvtColor(img1, &dst, .bgra_to_gray);

    var img2 = try dst.clone();
    defer img2.deinit();

    var prev_pts = try Mat.init();
    defer prev_pts.deinit();

    var next_pts = try Mat.init();
    defer next_pts.deinit();

    var status = try Mat.init();
    defer status.deinit();

    var err = try Mat.init();
    defer err.deinit();

    var corners = try Mat.init();
    defer corners.deinit();

    imgproc.goodFeaturesToTrack(dst, &corners, 500, 0.01, 10);
    var tc = try core.TermCriteria.init(.{ .count = true, .eps = true }, 30, 0.03);
    defer tc.deinit();
    imgproc.cornerSubPix(
        dst,
        &corners,
        Size.init(10, 10),
        Size.init(-1, -1),
        tc,
    );

    CalcOpticalFlow.pyrLKWithParams(
        dst,
        img2,
        corners,
        next_pts,
        &status,
        &err,
        Size.init(21, 21),
        3,
        tc,
        .default,
        0.0001,
    );

    try testing.expectEqual(false, status.isEmpty());
    try testing.expectEqual(@as(i32, 323), status.rows());
    try testing.expectEqual(@as(i32, 1), status.cols());
}

test "video findTransformECC" {
    var img1 = try imgcodecs.imRead(file_path, .gray_scale);
    defer img1.deinit();
    try testing.expectEqual(false, img1.isEmpty());

    var test_img = try Mat.init();
    defer test_img.deinit();
    imgproc.resize(img1, &test_img, Size.init(216, 216), 0, 0, .{ .type = .linear });

    var translation_ground = try Mat.initEye(2, 3, .cv32fc1);
    defer translation_ground.deinit();
    translation_ground.set(f32, 0, 2, 1.4159);
    translation_ground.set(f32, 1, 2, 17.1828);

    var warped_img = try Mat.init();
    defer warped_img.deinit();
    imgproc.warpAffineWithParams(
        test_img,
        &warped_img,
        translation_ground,
        Size.init(200, 200),
        .{ .type = .linear, .warp_inverse_map = true },
        .{ .type = .constant },
        core.Color{},
    );

    var map_translation = try Mat.initEye(2, 3, .cv32fc1);
    defer map_translation.deinit();
    const eec_iteration = 50;
    const eec_epsilon = -1;
    const ct = try core.TermCriteria.init(.{ .count = true, .eps = true }, eec_iteration, eec_epsilon);

    var input_mask = try Mat.init();
    defer input_mask.deinit();

    const gauss_filt_size = 5;

    _ = findTransformECC(
        warped_img,
        test_img,
        &map_translation,
        .translation,
        ct,
        input_mask,
        gauss_filt_size,
    );

    const max_rms_ecc = 0.1;

    const rms = b: {
        var rms: f64 = 0.0;
        {
            var i: usize = 0;
            while (i < map_translation.rows()) : (i += 1) {
                var j: usize = 0;
                while (j < map_translation.cols()) : (j += 1) {
                    const diff = map_translation.get(f32, i, j) - translation_ground.get(f32, i, j);
                    rms += diff * diff;
                }
            }
        }
        rms /= @floatFromInt(map_translation.rows() * map_translation.cols());
        break :b rms;
    };
    try testing.expect(rms < max_rms_ecc);
}

test "video tracker MIL" {
    var img = try imgcodecs.imRead(file_path, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var tracker_mil = TrackerMIL{};
    var tracker = try tracker_mil.tracker();
    defer tracker.deinit();

    const rect = Rect.init(250, 150, 200, 250);
    const init = tracker.initialize(img, rect);
    try testing.expectEqual(true, init);
    const res = tracker.update(img);
    try testing.expectEqual(true, res.success);
}

//*    implementation done
//*    pub const BackgroundSubtractorMOG2 = ?*anyopaque;
//*    pub const BackgroundSubtractorKNN = ?*anyopaque;
//*    pub const Tracker = ?*anyopaque;
//*    pub const TrackerMIL = ?*anyopaque;
//*    pub const TrackerGOTURN = ?*anyopaque;
//*    pub extern fn BackgroundSubtractorMOG2_Create(...) BackgroundSubtractorMOG2;
//*    pub extern fn BackgroundSubtractorMOG2_CreateWithParams(history: c_int, varThreshold: f64, detectShadows: bool) BackgroundSubtractorMOG2;
//*    pub extern fn BackgroundSubtractorMOG2_Close(b: BackgroundSubtractorMOG2) void;
//*    pub extern fn BackgroundSubtractorMOG2_Apply(b: BackgroundSubtractorMOG2, src: Mat, dst: Mat) void;
//*    pub extern fn BackgroundSubtractorKNN_Create(...) BackgroundSubtractorKNN;
//*    pub extern fn BackgroundSubtractorKNN_CreateWithParams(history: c_int, dist2Threshold: f64, detectShadows: bool) BackgroundSubtractorKNN;
//*    pub extern fn BackgroundSubtractorKNN_Close(b: BackgroundSubtractorKNN) void;
//*    pub extern fn BackgroundSubtractorKNN_Apply(b: BackgroundSubtractorKNN, src: Mat, dst: Mat) void;
//*    pub extern fn CalcOpticalFlowPyrLK(prevImg: Mat, nextImg: Mat, prevPts: Mat, nextPts: Mat, status: Mat, err: Mat) void;
//*    pub extern fn CalcOpticalFlowPyrLKWithParams(prevImg: Mat, nextImg: Mat, prevPts: Mat, nextPts: Mat, status: Mat, err: Mat, winSize: Size, maxLevel: c_int, criteria: TermCriteria, flags: c_int, minEigThreshold: f64) void;
//*    pub extern fn CalcOpticalFlowFarneback(prevImg: Mat, nextImg: Mat, flow: Mat, pyrScale: f64, levels: c_int, winsize: c_int, iterations: c_int, polyN: c_int, polySigma: f64, flags: c_int) void;
//*    pub extern fn FindTransformECC(templateImage: Mat, inputImage: Mat, warpMatrix: Mat, motionType: c_int, criteria: TermCriteria, inputMask: Mat, gaussFiltSize: c_int) f64;
//*    pub extern fn Tracker_Init(self: Tracker, image: Mat, boundingBox: Rect) bool;
//*    pub extern fn Tracker_Update(self: Tracker, image: Mat, boundingBox: [*c]Rect) bool;
//*    pub extern fn TrackerMIL_Create(...) TrackerMIL;
//*    pub extern fn TrackerMIL_Close(self: TrackerMIL) void;
