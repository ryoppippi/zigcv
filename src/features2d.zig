const std = @import("std");
const c = @import("c_api.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const core = @import("core.zig");
const Mat = core.Mat;
const Scalar = core.Scalar;
const Color = core.Color;
const KeyPoint = core.KeyPoint;
const KeyPoints = core.KeyPoints;
const NormTypes = core.NormTypes;

/// AKAZE is a wrapper around the cv::AKAZE algorithm.
pub const AKAZE = struct {
    ptr: c.AKAZE,

    const Self = @This();

    /// NewAKAZE returns a new AKAZE algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d8/d30/classcv_1_1AKAZE.html
    ///
    pub fn init() !Self {
        const ptr = c.AKAZE_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.AKAZE) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close AKAZE.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.AKAZE_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using AKAZE.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.AKAZE_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }

    /// DetectAndCompute keypoints and compute in an image using AKAZE.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
    ///
    pub fn detectAndCompute(self: Self, src: Mat, mask: Mat, desc: *Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.AKAZE_DetectAndCompute(self.ptr, src.toC(), mask.toC(), desc.*.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// AgastFeatureDetector is a wrapper around the cv::AgastFeatureDetector.
pub const AgastFeatureDetector = struct {
    ptr: c.AgastFeatureDetector,

    const Self = @This();

    /// NewAgastFeatureDetector returns a new AgastFeatureDetector algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/d19/classcv_1_1AgastFeatureDetector.html
    ///
    pub fn init() !Self {
        const ptr = c.AgastFeatureDetector_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.AgastFeatureDetector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close AgastFeatureDetector.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.AgastFeatureDetector_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using AgastFeatureDetector.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.AgastFeatureDetector_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// BRISK is a wrapper around the cv::BRISK algorithm.
pub const BRISK = struct {
    ptr: c.BRISK,

    const Self = @This();

    /// NewBRISK returns a new BRISK algorithm
    pub fn init() !Self {
        const ptr = c.BRISK_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.BRISK) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close BRISK.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.BRISK_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using BRISK.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.BRISK_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }

    /// DetectAndCompute keypoints and compute in an image using BRISK.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
    ///
    pub fn detectAndCompute(self: Self, src: Mat, mask: Mat, desc: *Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.BRISK_DetectAndCompute(self.ptr, src.toC(), mask.toC(), desc.*.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

pub const FastFeatureDetector = struct {
    ptr: c.FastFeatureDetector,

    const Self = @This();

    /// FastFeatureDetectorType defines the detector type
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/df/d74/classcv_1_1FastFeatureDetector.html#a4654f6fb0aa4b8e9123b223bfa0e2a08
    pub const Type = enum(u2) {
        ///FastFeatureDetectorType58 is an alias of FastFeatureDetector::TYPE_5_8
        type58 = 0,
        ///FastFeatureDetectorType712 is an alias of FastFeatureDetector::TYPE_7_12
        type712 = 1,
        ///FastFeatureDetectorType916 is an alias of FastFeatureDetector::TYPE_9_16
        type916 = 2,
    };

    /// NewFastFeatureDetector returns a new FastFeatureDetector algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/df/d74/classcv_1_1FastFeatureDetector.html
    ///
    pub fn init() !Self {
        const ptr = c.FastFeatureDetector_Create();
        return try Self.initFromC(ptr);
    }

    /// Close FastFeatureDetector.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.FastFeatureDetector_Close(self.ptr);
        self.ptr = null;
    }

    /// NewFastFeatureDetectorWithParams returns a new FastFeatureDetector algorithm with parameters
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/df/d74/classcv_1_1FastFeatureDetector.html#ab986f2ff8f8778aab1707e2642bc7f8e
    ///
    pub fn initWithParams(threshold: i32, nonmax_suppression: bool, type_: Type) !Self {
        const ptr = c.FastFeatureDetector_CreateWithParams(threshold, nonmax_suppression, @intFromEnum(type_));
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.FastFeatureDetector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Detect keypoints in an image using FastFeatureDetector.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.FastFeatureDetector_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// GFTTDetector is a wrapper around the cv::GFTTDetector algorithm.
/// https://docs.opencv.org/master/df/d21/classcv_1_1GFTTDetector.html
pub const GFTTDetector = struct {
    ptr: c.GFTTDetector,

    const Self = @This();

    /// NewGFTTDetector returns a new GFTTDetector algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/df/d21/classcv_1_1GFTTDetector.html
    ///
    pub fn init() !Self {
        const ptr = c.GFTTDetector_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.GFTTDetector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close GFTTDetector.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.GFTTDetector_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using GFTTDetector.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.GFTTDetector_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// KAZE is a wrapper around the cv::KAZE algorithm.
pub const KAZE = struct {
    ptr: c.KAZE,

    const Self = @This();

    /// NewKAZE returns a new KAZE algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d3/d61/classcv_1_1KAZE.html
    ///
    pub fn init() !Self {
        const ptr = c.KAZE_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.KAZE) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close KAZE.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.KAZE_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using KAZE.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.KAZE_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }

    /// DetectAndCompute keypoints and compute in an image using KAZE.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
    ///
    pub fn detectAndCompute(self: Self, src: Mat, mask: Mat, dst: *Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.KAZE_DetectAndCompute(self.ptr, src.toC(), mask.toC(), dst.*.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// MSER is a wrapper around the cv::MSER algorithm.
pub const MSER = struct {
    ptr: c.MSER,

    const Self = @This();

    /// NewMSER returns a new MSER algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html
    ///
    pub fn init() !Self {
        const ptr = c.MSER_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.MSER) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close MSER.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.MSER_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using MSER.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.MSER_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// ORB is a wrapper around the cv::ORB.
pub const ORB = struct {
    ptr: c.ORB,

    const Self = @This();

    const Type = enum(u1) {
        harris = 0,
        fast = 1,
    };

    /// NewORB returns a new ORB algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html
    ///
    pub fn init() !Self {
        const ptr = c.ORB_Create();
        return try Self.initFromC(ptr);
    }

    /// NewORBWithParams returns a new ORB algorithm with parameters
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html#aeff0cbe668659b7ca14bb85ff1c4073b
    ///
    pub fn initWithParams(
        n_features: i32,
        scale_factor: f32,
        nlevels: i32,
        edge_threshold: i32,
        firstLevel: i32,
        WTA_K: i32,
        score_type: Type,
        patchSize: i32,
        fast_threshold: i32,
    ) !Self {
        const ptr = c.ORB_CreateWithParams(
            n_features,
            scale_factor,
            nlevels,
            edge_threshold,
            firstLevel,
            WTA_K,
            @intFromEnum(score_type),
            patchSize,
            fast_threshold,
        );
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.ORB) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close ORB.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.ORB_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using ORB.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.ORB_Detect(self.ptr, src.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }

    /// DetectAndCompute detects keypoints and computes from an image using ORB.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
    ///
    pub fn detectAndCompute(self: Self, src: Mat, mask: Mat, dst: *Mat, allocator: std.mem.Allocator) !KeyPoints {
        const kp = c.ORB_DetectAndCompute(self.ptr, src.toC(), mask.toC(), dst.*.toC());
        defer c.KeyPoints_Close(kp);
        return try KeyPoint.toArrayList(kp, allocator);
    }
};

/// SIFT is a wrapper around the cv::SIFT algorithm.
/// Due to the patent having expired, this is now in the main OpenCV code modules.
pub const SIFT = struct {
    ptr: c.SIFT,

    const Self = @This();

    /// NewSIFT returns a new SIFT algorithm.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
    ///
    pub fn init() !Self {
        const ptr = c.SIFT_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.SIFT) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Detect keypoints in an image using SIFT.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.SIFT_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using SIFT.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        return try KeyPoint.toArrayList(c.SIFT_Detect(self.ptr, src.toC()), allocator);
    }

    /// DetectAndCompute detects and computes keypoints in an image using SIFT.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
    ///
    pub fn detectAndCompute(self: Self, src: Mat, mask: Mat, desc: *Mat, allocator: std.mem.Allocator) !KeyPoints {
        const return_keypoints: c.KeyPoints = c.SIFT_DetectAndCompute(self.ptr, src.toC(), mask.toC(), desc.*.toC());
        defer c.KeyPoints_Close(return_keypoints);
        return try KeyPoint.toArrayList(return_keypoints, allocator);
    }
};

/// SimpleBlobDetector is a wrapper around the cv::SimpleBlobDetector.
pub const SimpleBlobDetector = struct {
    ptr: c.SimpleBlobDetector,

    const Self = @This();

    /// NewSimpleBlobDetector returns a new SimpleBlobDetector algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d7a/classcv_1_1SimpleBlobDetector.html
    ///
    pub fn init() !Self {
        const ptr = c.SimpleBlobDetector_Create();
        return try Self.initFromC(ptr);
    }

    /// NewSimpleBlobDetectorWithParams returns a new SimpleBlobDetector with custom parameters
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d7a/classcv_1_1SimpleBlobDetector.html
    ///
    pub const Params = struct {
        blob_color: u8,
        filter_by_area: bool,
        filter_by_circularity: bool,
        filter_by_color: bool,
        filter_by_convexity: bool,
        filter_by_inertia: bool,
        max_area: f32,
        max_circularity: f32,
        max_convexity: f32,
        max_inertia_ratio: f32,
        max_threshold: f32,
        min_area: f32,
        min_circularity: f32,
        min_convexity: f32,
        min_dist_between_blobs: f32,
        min_inertia_ratio: f32,
        min_repeatability: usize,
        min_threshold: f32,
        threshold_step: f32,

        pub fn default() Params {
            const default_params = c.SimpleBlobDetectorParams_Create();
            return .{
                .blob_color = default_params.blobColor,
                .filter_by_area = default_params.filterByArea,
                .filter_by_circularity = default_params.filterByCircularity,
                .filter_by_color = default_params.filterByColor,
                .filter_by_convexity = default_params.filterByConvexity,
                .filter_by_inertia = default_params.filterByInertia,
                .max_area = default_params.maxArea,
                .max_circularity = default_params.maxCircularity,
                .max_convexity = default_params.maxConvexity,
                .max_inertia_ratio = default_params.maxInertiaRatio,
                .max_threshold = default_params.maxThreshold,
                .min_area = default_params.minArea,
                .min_circularity = default_params.minCircularity,
                .min_convexity = default_params.minConvexity,
                .min_dist_between_blobs = default_params.minDistBetweenBlobs,
                .min_inertia_ratio = default_params.minInertiaRatio,
                .min_repeatability = default_params.minRepeatability,
                .min_threshold = default_params.minThreshold,
                .threshold_step = default_params.thresholdStep,
            };
        }

        pub fn toC(params: Params) c.struct_SimpleBlobDetectorParams {
            return .{
                .blobColor = params.blob_color,
                .filterByArea = params.filter_by_area,
                .filterByCircularity = params.filter_by_circularity,
                .filterByColor = params.filter_by_color,
                .filterByConvexity = params.filter_by_convexity,
                .filterByInertia = params.filter_by_inertia,
                .maxArea = params.max_area,
                .maxCircularity = params.max_circularity,
                .maxConvexity = params.max_convexity,
                .maxInertiaRatio = params.max_inertia_ratio,
                .maxThreshold = params.max_threshold,
                .minArea = params.min_area,
                .minCircularity = params.min_circularity,
                .minConvexity = params.min_convexity,
                .minDistBetweenBlobs = params.min_dist_between_blobs,
                .minInertiaRatio = params.min_inertia_ratio,
                .minRepeatability = params.min_repeatability,
                .minThreshold = params.min_threshold,
                .thresholdStep = params.threshold_step,
            };
        }
    };

    pub fn initWithParams(
        params: Self.Params,
    ) !Self {
        const c_params = params.toC();
        const ptr = c.SimpleBlobDetector_Create_WithParams(c_params);
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.SimpleBlobDetector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close SimpleBlobDetector.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.SimpleBlobDetector_Close(self.ptr);
        self.ptr = null;
    }

    /// Detect keypoints in an image using SimpleBlobDetector.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887
    ///
    pub fn detect(self: Self, src: Mat, allocator: std.mem.Allocator) !KeyPoints {
        return try KeyPoint.toArrayList(c.SimpleBlobDetector_Detect(self.ptr, src.toC()), allocator);
    }
};

/// BFMatcher is a wrapper around the the cv::BFMatcher algorithm
pub const BFMatcher = struct {
    ptr: c.BFMatcher,

    const Self = @This();

    /// NewBFMatcher returns a new BFMatcher
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d3/da1/classcv_1_1BFMatcher.html#abe0bb11749b30d97f60d6ade665617bd
    ///
    pub fn init() !Self {
        const ptr = c.BFMatcher_Create();
        return try Self.initFromC(ptr);
    }

    /// NewBFMatcherWithParams creates a new BFMatchers but allows setting parameters
    /// to values other than just the defaults.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d3/da1/classcv_1_1BFMatcher.html#abe0bb11749b30d97f60d6ade665617bd
    ///
    pub fn initWithParams(
        norm_type: NormTypes,
        cross_check: bool,
    ) !Self {
        const ptr = c.BFMatcher_Create_WithParams(@intFromEnum(norm_type), cross_check);
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.BFMatcher) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close BFMatcher.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.BFMatcher_Close(self.ptr);
        self.ptr = null;
    }

    /// KnnMatch Finds the k best matches for each descriptor from a query set.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/db/d39/classcv_1_1DescriptorMatcher.html#aa880f9353cdf185ccf3013e08210483a
    ///
    pub fn knnMatch(
        self: Self,
        query: Mat,
        train: Mat,
        k: i32,
        allocator: std.mem.Allocator,
    ) !MultiDMatches {
        const res: c.MultiDMatches = c.BFMatcher_KnnMatch(self.ptr, query.toC(), train.toC(), k);
        defer c.MultiDMatches_Close(res);
        return try DMatch.toMultiArrayList(res.dmatches[0..@as(usize, @intCast(res.length))], allocator);
    }
};

/// FlannBasedMatcher is a wrapper around the the cv::FlannBasedMatcher algorithm
pub const FlannBasedMatcher = struct {
    ptr: c.FlannBasedMatcher,

    const Self = @This();

    /// NewFlannBasedMatcher returns a new FlannBasedMatcher
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/dc/de2/classcv_1_1FlannBasedMatcher.html#ab9114a6471e364ad221f89068ca21382
    ///
    pub fn init() !Self {
        const ptr = c.FlannBasedMatcher_Create();
        return try Self.initFromC(ptr);
    }

    pub fn initFromC(ptr: c.FlannBasedMatcher) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close FlannBasedMatcher.
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.FlannBasedMatcher_Close(self.ptr);
    }

    /// KnnMatch Finds the k best matches for each descriptor from a query set.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/db/d39/classcv_1_1DescriptorMatcher.html#aa880f9353cdf185ccf3013e08210483a
    ///
    pub fn knnMatch(
        self: Self,
        query: Mat,
        train: Mat,
        k: i32,
        allocator: std.mem.Allocator,
    ) !MultiDMatches {
        const res: c.MultiDMatches = c.FlannBasedMatcher_KnnMatch(self.ptr, query.toC(), train.toC(), k);
        defer c.MultiDMatches_Close(res);
        return try DMatch.toMultiArrayList(res.dmatches[0..@as(usize, @intCast(res.length))], allocator);
    }
};

/// DrawMatchesFlag are the flags setting drawing feature
///
/// For further details please see:
/// https://docs.opencv.org/master/de/d30/structcv_1_1DrawMatchesFlags.html
pub const DrawMatchesFlag = enum(u2) {
    /// DrawDefault creates new image and for each keypoint only the center point will be drawn
    draw_default = 0,
    /// DrawOverOutImg draws matches on existing content of image
    over_out_img = 1,
    /// NotDrawSinglePoints will not draw single points
    not_draw_single_points = 2,
    /// DrawRichKeyPoints draws the circle around each keypoint with keypoint size and orientation
    draw_rich_keypoints = 3,
};

/// DrawKeyPoints draws keypoints
///
/// For further details please see:
/// https://docs.opencv.org/master/d4/d5d/group__features2d__draw.html#gab958f8900dd10f14316521c149a60433
pub fn drawKeyPoints(src: Mat, kp: []KeyPoint, dst: *Mat, color_: Color, flags: DrawMatchesFlag, allocator: std.mem.Allocator) !void {
    var c_keypoints_array = try std.ArrayList(c.KeyPoint).initCapacity(allocator, kp.len);
    defer c_keypoints_array.deinit();
    for (kp) |keypoint| try c_keypoints_array.append(keypoint.toC());
    const c_keypoints = c.KeyPoints{
        .length = @as(i32, @intCast(kp.len)),
        .keypoints = @as([*]c.KeyPoint, @ptrCast(c_keypoints_array.items)),
    };
    c.DrawKeyPoints(src.toC(), c_keypoints, dst.*.toC(), color_.toScalar().toC(), @intFromEnum(flags));
}

/// DrawMatches draws matches on combined train and querry images.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d5d/group__features2d__draw.html#gad8f463ccaf0dc6f61083abd8717c261a
pub fn drawMatches(
    img1: Mat,
    kp1: []KeyPoint,
    img2: Mat,
    kp2: []KeyPoint,
    matches1to2: []DMatch,
    out_img: *Mat,
    matches_color: Color,
    point_color: Color,
    matches_mask: []u8,
    flags: DrawMatchesFlag,
    allocator: std.mem.Allocator,
) !void {
    var c_keypoints_array1 = try allocator.alloc(c.KeyPoint, kp1.len);
    defer allocator.free(c_keypoints_array1);
    for (kp1, 0..) |keypoint, i| c_keypoints_array1[i] = keypoint.toC();
    const c_keypoints1 = c.KeyPoints{
        .length = @as(i32, @intCast(kp1.len)),
        .keypoints = @as([*]c.KeyPoint, @ptrCast(c_keypoints_array1.ptr)),
    };

    var c_keypoints_array2 = try allocator.alloc(c.KeyPoint, kp2.len);
    defer allocator.free(c_keypoints_array2);
    for (kp2, 0..) |keypoint, i| c_keypoints_array2[i] = keypoint.toC();
    const c_keypoints2 = c.KeyPoints{
        .length = @as(i32, @intCast(kp2.len)),
        .keypoints = @as([*]c.KeyPoint, @ptrCast(c_keypoints_array2.ptr)),
    };

    var c_matches1to2_array = try allocator.alloc(c.DMatch, matches1to2.len);
    defer allocator.free(c_matches1to2_array);
    for (matches1to2, 0..) |match, i| c_matches1to2_array[i] = match.toC();
    const c_matches1to2 = c.DMatches{
        .length = @as(i32, @intCast(matches1to2.len)),
        .dmatches = @as([*]c.DMatch, @ptrCast(c_matches1to2_array.ptr)),
    };

    const c_matches_mask = core.toByteArray(matches_mask);
    c.DrawMatches(
        img1.toC(),
        c_keypoints1,
        img2.toC(),
        c_keypoints2,
        c_matches1to2,
        out_img.*.toC(),
        matches_color.toScalar().toC(),
        point_color.toScalar().toC(),
        c_matches_mask,
        @intFromEnum(flags),
    );
}

pub const DMatch = struct {
    query_idx: i32,
    train_idx: i32,
    img_idx: i32,
    distance: f32,

    const Self = @This();
    const CSelf = c.DMatch;

    pub fn init(
        query_idx: i32,
        train_idx: i32,
        img_idx: i32,
        distance: f32,
    ) Self {
        return Self{
            .query_idx = query_idx,
            .train_idx = train_idx,
            .img_idx = img_idx,
            .distance = distance,
        };
    }

    pub fn toC(self: Self) CSelf {
        return CSelf{
            .queryIdx = self.query_idx,
            .trainIdx = self.train_idx,
            .imgIdx = self.img_idx,
            .distance = self.distance,
        };
    }

    pub fn toArrayList(ret: []CSelf, allocator: std.mem.Allocator) !DMathes {
        const len = @as(usize, @intCast(ret.len));
        var result = try std.ArrayList(Self).initCapacity(allocator, len);
        {
            for (ret) |dmatch| {
                try result.append(Self.init(
                    dmatch.queryIdx,
                    dmatch.trainIdx,
                    dmatch.imgIdx,
                    dmatch.distance,
                ));
            }
        }
        return result;
    }

    pub fn toMultiArrayList(ret: []c.DMatches, allocator: std.mem.Allocator) !MultiDMatches {
        const m_len = @as(usize, @intCast(ret.len));
        var result = try std.ArrayList(std.ArrayList(Self)).initCapacity(allocator, m_len);
        for (ret) |dmatches| {
            const len = @as(usize, @intCast(dmatches.length));
            var dmatch_list = try std.ArrayList(Self).initCapacity(allocator, len);
            {
                var i: usize = 0;
                while (i < len) : (i += 1) {
                    const dmatch = dmatches.dmatches[i];
                    try dmatch_list.append(Self.init(
                        dmatch.queryIdx,
                        dmatch.trainIdx,
                        dmatch.imgIdx,
                        dmatch.distance,
                    ));
                }
            }
            try result.append(dmatch_list);
        }
        return .{ .array = result };
    }
};

const DMathes = std.ArrayList(DMatch);
const MultiDMatches = struct {
    array: std.ArrayList(DMathes),

    pub fn deinit(self: MultiDMatches) void {
        for (self.array.items) |m| m.deinit();
        self.array.deinit();
    }
};

const testing = std.testing;
const imgcodecs = @import("imgcodecs.zig");

test "feature2d AKAZE" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var ak = try AKAZE.init();
    defer ak.deinit();

    var kp = try ak.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expect(kp.items.len >= 512);

    var desc = try Mat.init();
    defer desc.deinit();
    var mask = try Mat.init();
    defer mask.deinit();

    var kp2 = try ak.detectAndCompute(img, mask, &desc, std.testing.allocator);
    defer kp2.deinit();

    try testing.expect(kp2.items.len >= 512);
    try testing.expectEqual(kp2.items.len, kp.items.len);
    try testing.expectEqual(false, desc.isEmpty());
}

test "feature2d AgastFeatureDetector" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var ad = try AgastFeatureDetector.init();
    defer ad.deinit();

    var kp = try ad.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expect(kp.items.len >= 2800);
}

test "feature2d BRISK" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var br = try BRISK.init();
    defer br.deinit();

    var kp = try br.detect(img, std.testing.allocator);
    kp.deinit();

    var desc = try Mat.init();
    defer desc.deinit();
    var mask = try Mat.init();
    defer mask.deinit();

    var kp2 = try br.detectAndCompute(img, mask, &desc, std.testing.allocator);
    defer kp2.deinit();

    try testing.expectEqual(@as(usize, 1105), kp2.items.len);
    try testing.expectEqual(false, desc.isEmpty());
}

test "feature2d FastFeatureDetector" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var fd = try FastFeatureDetector.init();
    defer fd.deinit();

    var kp = try fd.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expect(kp.items.len >= 2690);
}

test "feature2d GFTTDetector" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var gft = try GFTTDetector.init();
    defer gft.deinit();

    var kp = try gft.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expect(kp.items.len >= 512);
}

test "feature2d KAZE" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    var kaze = try KAZE.init();
    defer kaze.deinit();

    var kp = try kaze.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expect(kp.items.len >= 512);

    var desc = try Mat.init();
    defer desc.deinit();
    var mask = try Mat.init();
    defer mask.deinit();

    var kp2 = try kaze.detectAndCompute(img, mask, &desc, std.testing.allocator);
    defer kp2.deinit();

    try testing.expect(kp2.items.len >= 512);
    try testing.expectEqual(false, desc.isEmpty());
}

test "feature2d MSER" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var mser = try MSER.init();
    defer mser.deinit();

    var kp = try mser.detect(img, std.testing.allocator);
    defer kp.deinit();

    const len = kp.items.len;
    try testing.expect(len == 232 or len == 234 or len == 261);
}

// test "feature2d ORB" {
// var img = try imgcodecs.imRead("test/images/face.jpg", .color);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var dst = try Mat.init();
// defer dst.deinit();

// var orb = try ORB.init();
// defer orb.deinit();

// var kp = try orb.detect(img, std.testing.allocator);
// defer kp.deinit();

// try testing.expectEqual(@as(usize, 500), kp.items.len);

// var mask = try Mat.init();
// defer mask.deinit();
// var desc = try Mat.init();
// defer desc.deinit();

// var kp2 = try orb.detectAndCompute(img, mask, &desc, std.testing.allocator);
// defer kp2.deinit();

// try testing.expectEqual(@as(usize, 500), kp2.items.len);
// try testing.expectEqual(false, desc.isEmpty());
// }

test "feature2d SimpleBlobDetector" {
    var img = try imgcodecs.imRead("test/images/face.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var sbd = try SimpleBlobDetector.init();
    defer sbd.deinit();

    var kp = try sbd.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expectEqual(@as(usize, 2), kp.items.len);
}

test "feature2d SimpleBlobDetectorWithParams" {
    var img = try imgcodecs.imRead("test/images/circles.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var params = SimpleBlobDetector.Params.default();
    params.max_area = 27500;

    var sbd = try SimpleBlobDetector.initWithParams(params);
    defer sbd.deinit();

    var kp = try sbd.detect(img, std.testing.allocator);
    defer kp.deinit();

    try testing.expectEqual(@as(usize, 4), kp.items.len);
}

test "feature2d BFMatcher" {
    const img_path = "test/images/sift_descriptor.png";
    var desc1 = try imgcodecs.imRead(img_path, .gray_scale);
    defer desc1.deinit();
    try testing.expectEqual(false, desc1.isEmpty());

    var desc2 = try imgcodecs.imRead(img_path, .gray_scale);
    defer desc2.deinit();
    try testing.expectEqual(false, desc2.isEmpty());

    var bfm = try BFMatcher.init();
    defer bfm.deinit();

    const k = 2;
    var matches = try bfm.knnMatch(desc1, desc2, k, std.testing.allocator);
    defer matches.deinit();

    try testing.expect(matches.array.items.len > 0);

    for (matches.array.items) |m| {
        try testing.expectEqual(@as(usize, k), m.items.len);
    }
}

test "feature2d FlannBasedMatcher" {
    const img_path = "test/images/sift_descriptor.png";
    var desc1 = try imgcodecs.imRead(img_path, .gray_scale);
    defer desc1.deinit();
    desc1.convertTo(&desc1, .cv32fc1);

    var desc2 = try imgcodecs.imRead(img_path, .gray_scale);
    defer desc2.deinit();
    desc2.convertTo(&desc2, .cv32fc1);

    var fbm = try FlannBasedMatcher.init();
    defer fbm.deinit();

    const k = 2;

    var matches = try fbm.knnMatch(desc1, desc2, k, std.testing.allocator);
    defer matches.deinit();

    try testing.expect(matches.array.items.len > 0);

    for (matches.array.items) |m| {
        try testing.expectEqual(@as(usize, k), m.items.len);
    }
}

test "feature2d drawPoints" {
    var img = try imgcodecs.imRead("test/images/simple.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var ffd = try FastFeatureDetector.init();
    defer ffd.deinit();

    var kp = try ffd.detect(img, testing.allocator);
    defer kp.deinit();

    var simple_KP = try Mat.init();
    defer simple_KP.deinit();

    try drawKeyPoints(img, kp.items, &simple_KP, Color.init(255, 0, 0, 0), .draw_default, testing.allocator);

    try testing.expectEqual(false, simple_KP.isEmpty());
    try testing.expectEqual(img.rows(), simple_KP.rows());
    try testing.expectEqual(img.cols(), simple_KP.cols());
}

test "feature2d DrawMatches" {
    const query_path = "test/images/box.png";
    const train_path = "test/images/box_in_scene.png";

    var query = try imgcodecs.imRead(query_path, .color);
    defer query.deinit();
    try testing.expectEqual(false, query.isEmpty());

    var train = try imgcodecs.imRead(train_path, .color);
    defer train.deinit();
    try testing.expectEqual(false, train.isEmpty());

    var sift = try SIFT.init();
    defer sift.deinit();

    var m1 = try Mat.init();
    defer m1.deinit();
    var m2 = try Mat.init();
    defer m2.deinit();
    var desc1 = try Mat.init();
    defer desc1.deinit();
    var desc2 = try Mat.init();
    defer desc2.deinit();

    var kp1 = try sift.detectAndCompute(query, m1, &desc1, std.testing.allocator);
    defer kp1.deinit();
    var kp2 = try sift.detectAndCompute(train, m2, &desc2, std.testing.allocator);
    defer kp2.deinit();

    var bf = try BFMatcher.init();
    defer bf.deinit();

    var matches = try bf.knnMatch(desc1, desc2, 2, std.testing.allocator);
    defer matches.deinit();

    try testing.expect(matches.array.items.len > 0);

    var good = std.ArrayList(DMatch).init(std.testing.allocator);
    defer good.deinit();
    for (matches.array.items) |m| {
        if (m.items[0].distance < 0.75 * m.items[1].distance) {
            try good.append(m.items[0]);
        }
    }

    const C = Color.init(255, 0, 0, 0);

    const mask = "";

    var out = try Mat.init();
    defer out.deinit();

    try drawMatches(query, kp1.items, train, kp2.items, good.items, &out, C, C, mask, .draw_default, testing.allocator);

    try testing.expectEqual(false, out.isEmpty());
    try testing.expectEqual(query.cols() + train.cols(), out.cols());
    try testing.expect(train.rows() <= out.cols());
    try testing.expect(query.rows() <= out.cols());

    const mask2 = "";
    var smoke = try Mat.init();
    defer smoke.deinit();

    try drawMatches(query, kp1.items, train, kp2.items, good.items, &smoke, C, C, mask2, .draw_default, testing.allocator);
}

test "feature2d SIFT" {
    var img = try imgcodecs.imRead("test/images/sift_descriptor.png", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var si = try SIFT.init();
    defer si.deinit();

    var kp = try si.detect(img, testing.allocator);
    defer kp.deinit();

    try testing.expect(@as(usize, 512) != kp.items.len);

    var mask = try Mat.init();
    defer mask.deinit();

    var desc = try Mat.init();
    defer desc.deinit();

    var kp2 = try si.detectAndCompute(img, mask, &dst, testing.allocator);
    defer kp2.deinit();

    try testing.expect(@as(usize, 512) != kp2.items.len);
    try testing.expectEqual(false, dst.isEmpty());
}

//*    implementation done
//*    pub const AKAZE = ?*anyopaque;
//*    pub const AgastFeatureDetector = ?*anyopaque;
//*    pub const BRISK = ?*anyopaque;
//*    pub const FastFeatureDetector = ?*anyopaque;
//*    pub const GFTTDetector = ?*anyopaque;
//*    pub const KAZE = ?*anyopaque;
//*    pub const MSER = ?*anyopaque;
//*    pub const ORB = ?*anyopaque;
//*    pub const SimpleBlobDetector = ?*anyopaque;
//*    pub const BFMatcher = ?*anyopaque;
//*    pub const FlannBasedMatcher = ?*anyopaque;
//*    pub const SIFT = ?*anyopaque;
//*    pub extern fn AKAZE_Create(...) AKAZE;
//*    pub extern fn AKAZE_Close(a: AKAZE) void;
//*    pub extern fn AKAZE_Detect(a: AKAZE, src: Mat) struct_KeyPoints;
//*    pub extern fn AKAZE_DetectAndCompute(a: AKAZE, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//*    pub extern fn AgastFeatureDetector_Create(...) AgastFeatureDetector;
//*    pub extern fn AgastFeatureDetector_Close(a: AgastFeatureDetector) void;
//*    pub extern fn AgastFeatureDetector_Detect(a: AgastFeatureDetector, src: Mat) struct_KeyPoints;
//*    pub extern fn BRISK_Create(...) BRISK;
//*    pub extern fn BRISK_Close(b: BRISK) void;
//*    pub extern fn BRISK_Detect(b: BRISK, src: Mat) struct_KeyPoints;
//*    pub extern fn BRISK_DetectAndCompute(b: BRISK, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//*    pub extern fn FastFeatureDetector_Create(...) FastFeatureDetector;
//*    pub extern fn FastFeatureDetector_CreateWithParams(threshold: c_int, nonmaxSuppression: bool, @"type": c_int) FastFeatureDetector;
//*    pub extern fn FastFeatureDetector_Close(f: FastFeatureDetector) void;
//*    pub extern fn FastFeatureDetector_Detect(f: FastFeatureDetector, src: Mat) struct_KeyPoints;
//*    pub extern fn GFTTDetector_Create(...) GFTTDetector;
//*    pub extern fn GFTTDetector_Close(a: GFTTDetector) void;
//*    pub extern fn GFTTDetector_Detect(a: GFTTDetector, src: Mat) struct_KeyPoints;
//*    pub extern fn KAZE_Create(...) KAZE;
//*    pub extern fn KAZE_Close(a: KAZE) void;
//*    pub extern fn KAZE_Detect(a: KAZE, src: Mat) struct_KeyPoints;
//*    pub extern fn KAZE_DetectAndCompute(a: KAZE, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//*    pub extern fn MSER_Create(...) MSER;
//*    pub extern fn MSER_Close(a: MSER) void;
//*    pub extern fn MSER_Detect(a: MSER, src: Mat) struct_KeyPoints;
//*    pub extern fn ORB_Create(...) ORB;
//*    pub extern fn ORB_CreateWithParams(nfeatures: c_int, scaleFactor: f32, nlevels: c_int, edgeThreshold: c_int, firstLevel: c_int, WTA_K: c_int, scoreType: c_int, patchSize: c_int, fastThreshold: c_int) ORB;
//*    pub extern fn ORB_Close(o: ORB) void;
//*    pub extern fn ORB_Detect(o: ORB, src: Mat) struct_KeyPoints;
//*    pub extern fn ORB_DetectAndCompute(o: ORB, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//*    pub extern fn SimpleBlobDetector_Create(...) SimpleBlobDetector;
//*    pub extern fn SimpleBlobDetector_Create_WithParams(params: SimpleBlobDetectorParams) SimpleBlobDetector;
//*    pub extern fn SimpleBlobDetector_Close(b: SimpleBlobDetector) void;
//*    pub extern fn SimpleBlobDetector_Detect(b: SimpleBlobDetector, src: Mat) struct_KeyPoints;
//*    pub extern fn SimpleBlobDetectorParams_Create(...) SimpleBlobDetectorParams;
//*    pub extern fn BFMatcher_Create(...) BFMatcher;
//*    pub extern fn BFMatcher_CreateWithParams(normType: c_int, crossCheck: bool) BFMatcher;
//*    pub extern fn BFMatcher_Close(b: BFMatcher) void;
//*    pub extern fn BFMatcher_KnnMatch(b: BFMatcher, query: Mat, train: Mat, k: c_int) struct_MultiDMatches;
//*    pub extern fn FlannBasedMatcher_Create(...) FlannBasedMatcher;
//*    pub extern fn FlannBasedMatcher_Close(f: FlannBasedMatcher) void;
//*    pub extern fn FlannBasedMatcher_KnnMatch(f: FlannBasedMatcher, query: Mat, train: Mat, k: c_int) struct_MultiDMatches;
//*    pub extern fn DrawKeyPoints(src: Mat, kp: struct_KeyPoints, dst: Mat, s: Scalar, flags: c_int) void;
//*    pub extern fn SIFT_Create(...) SIFT;
//*    pub extern fn SIFT_Close(f: SIFT) void;
//*    pub extern fn SIFT_Detect(f: SIFT, src: Mat) struct_KeyPoints;
//*    pub extern fn SIFT_DetectAndCompute(f: SIFT, src: Mat, mask: Mat, desc: Mat) struct_KeyPoints;
//*    pub extern fn DrawMatches(img1: Mat, kp1: struct_KeyPoints, img2: Mat, kp2: struct_KeyPoints, matches1to2: struct_DMatches, outImg: Mat, matchesColor: Scalar, pointColor: Scalar, matchesMask: struct_ByteArray, flags: c_int) void;
