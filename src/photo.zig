const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const Mat = core.Mat;
const Mats = core.Mats;
const Point = core.Point;

pub const SeamlessCloneFlag = enum(u2) {
    /// NormalClone The power of the method is fully expressed when inserting objects with complex outlines into a new background.
    normal_clone = 0,

    /// MixedClone The classic method, color-based selection and alpha masking might be time consuming and often leaves an undesirable halo. Seamless cloning, even averaged with the original image, is not effective. Mixed seamless cloning based on a loose selection proves effective.
    mixed_clone,

    /// MonochromeTransfer Monochrome transfer allows the user to easily replace certain features of one object by alternative features.
    monochrome_transfer,
};

pub const EdgeFilter = enum(u2) {
    /// RecursFilter Recursive Filtering.
    recurs_filter = 1,

    /// NormconvFilter Normalized Convolution Filtering.
    normconv_filter = 2,
};

/// ColorChange mix two differently colored versions of an image seamlessly.
///
/// For further details, please see:
/// https://docs.opencv.org/master/df/da0/group__photo__clone.html#ga6684f35dc669ff6196a7c340dc73b98e
///
pub fn colorChange(src: Mat, mask: Mat, dst: *Mat, red_mul: f32, green_mul: f32, blue_mul: f32) void {
    _ = c.ColorChange(src.ptr, mask.ptr, dst.*.ptr, red_mul, green_mul, blue_mul);
}

/// SeamlessClone blend two image by Poisson Blending.
///
/// For further details, please see:
/// https://docs.opencv.org/master/df/da0/group__photo__clone.html#ga2bf426e4c93a6b1f21705513dfeca49d
///
pub fn seamlessClone(src: Mat, dst: *Mat, mask: Mat, p: Point, blend: Mat, flags: SeamlessCloneFlag) void {
    _ = c.SeamlessClone(src.ptr, dst.*.ptr, mask.ptr, p.toC(), blend.ptr, @intFromEnum(flags));
}

/// IlluminationChange modifies locally the apparent illumination of an image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/df/da0/group__photo__clone.html#gac5025767cf2febd8029d474278e886c7
///
pub fn illuminationChange(src: Mat, mask: Mat, dst: *Mat, alpha: f32, beta: f32) void {
    _ = c.IlluminationChange(src.ptr, mask.ptr, dst.*.ptr, alpha, beta);
}

/// TextureFlattening washes out the texture of the selected region, giving its contents a flat aspect.
///
/// For further details, please see:
/// https://docs.opencv.org/master/df/da0/group__photo__clone.html#gad55df6aa53797365fa7cc23959a54004
///
pub fn textureFlattening(src: Mat, mask: Mat, dst: *Mat, low_threshold: f32, high_threshold: f32, kernel_size: i32) void {
    _ = c.TextureFlattening(src.ptr, mask.ptr, dst.*.ptr, low_threshold, high_threshold, kernel_size);
}

///     pub extern fn FastNlMeansDenoisingColoredMulti(src: struct_Mats, dst: Mat, imgToDenoiseIndex: c_int, temporalWindowSize: c_int) void;
/// FastNlMeansDenoisingColoredMulti denoises the selected images.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d1/d79/group__photo__denoise.html#gaa501e71f52fb2dc17ff8ca5e7d2d3619
///
pub fn fastNlMeansDenoisingColoredMulti(
    src: []Mat,
    dst: *Mat,
    img_to_denoise_index: i32,
    temporal_window_size: i32,
) !void {
    const c_mats = try Mat.toCStructs(src);
    defer Mat.deinitCStructs(c_mats);
    _ = c.FastNlMeansDenoisingColoredMulti(c_mats, dst.*.ptr, img_to_denoise_index, temporal_window_size);
}

/// FastNlMeansDenoisingColoredMulti denoises the selected images.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d1/d79/group__photo__denoise.html#gaa501e71f52fb2dc17ff8ca5e7d2d3619
///
pub fn fastNlMeansDenoisingColoredMultiWithParams(
    src: []Mat,
    dst: *Mat,
    img_to_denoise_index: i32,
    temporal_window_size: i32,
    h: f32,
    h_color: f32,
    template_window_size: i32,
    search_window_size: i32,
) !void {
    const c_mats = try Mat.toCStructs(src);
    defer Mat.deinitCStructs(c_mats);
    _ = c.FastNlMeansDenoisingColoredMultiWithParams(
        c_mats,
        dst.*.ptr,
        img_to_denoise_index,
        temporal_window_size,
        h,
        h_color,
        template_window_size,
        search_window_size,
    );
}

/// FastNlMeansDenoising performs image denoising using Non-local Means Denoising algorithm
/// http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
///
pub fn fastNlMeansDenoising(src: Mat, dst: *Mat) void {
    _ = c.FastNlMeansDenoising(src.ptr, dst.*.ptr);
}

/// FastNlMeansDenoisingWithParams performs image denoising using Non-local Means Denoising algorithm
/// http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
///
pub fn fastNlMeansDenoisingWithParams(
    src: Mat,
    dst: *Mat,
    h: f32,
    template_window_size: i32,
    search_window_size: i32,
) void {
    _ = c.FastNlMeansDenoisingWithParams(
        src.ptr,
        dst.*.ptr,
        h,
        template_window_size,
        search_window_size,
    );
}

/// FastNlMeansDenoisingColored is a modification of fastNlMeansDenoising function for colored images.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga21abc1c8b0e15f78cd3eff672cb6c476
///
pub fn fastNlMeansDenoisingColored(src: Mat, dst: *Mat) void {
    _ = c.FastNlMeansDenoisingColored(src.ptr, dst.*.ptr);
}

/// FastNlMeansDenoisingColoredWithParams is a modification of fastNlMeansDenoising function for colored images.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga21abc1c8b0e15f78cd3eff672cb6c476
///
pub fn fastNlMeansDenoisingColoredWithParams(
    src: Mat,
    dst: *Mat,
    h: f32,
    h_color: f32,
    template_window_size: i32,
    search_window_size: i32,
) void {
    _ = c.FastNlMeansDenoisingColoredWithParams(
        src.ptr,
        dst.*.ptr,
        h,
        h_color,
        template_window_size,
        search_window_size,
    );
}

/// MergeMertens is a wrapper around the cv::MergeMertens.
pub const MergeMertens = struct {
    ptr: c.MergeMertens,

    const Self = @This();

    /// NewMergeMertens returns returns a new MergeMertens white LDR merge algorithm.
    /// of type MergeMertens with default parameters.
    /// MergeMertens algorithm merge the ldr image should result in a HDR image.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html
    /// https://docs.opencv.org/master/d7/dd6/classcv_1_1MergeMertens.html
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html#ga79d59aa3cb3a7c664e59a4b5acc1ccb6
    ///
    pub fn init() !Self {
        const ptr = c.MergeMertens_Create();
        return try initFromC(ptr);
    }

    /// NewMergeMertensWithParams returns a new MergeMertens white LDR merge algorithm
    /// of type MergeMertens with customized parameters.
    /// MergeMertens algorithm merge the ldr image should result in a HDR image.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html
    /// https://docs.opencv.org/master/d7/dd6/classcv_1_1MergeMertens.html
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html#ga79d59aa3cb3a7c664e59a4b5acc1ccb6
    ///
    pub fn initWithParams(contrast_weight: f32, saturation_weight: f32, exposure_weight: f32) !Self {
        const ptr = c.MergeMertens_CreateWithParams(contrast_weight, saturation_weight, exposure_weight);
        return try initFromC(ptr);
    }

    fn initFromC(ptr: c.MergeMertens) !Self {
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    ///Close MergeMertens
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.MergeMertens_Close(self.ptr);
        self.*.ptr = null;
    }

    /// BalanceWhite computes merge LDR images using the current MergeMertens.
    /// Return a image MAT : 8bits 3 channel image ( RGB 8 bits )
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dd6/classcv_1_1MergeMertens.html#a2d2254b2aab722c16954de13a663644d
    ///
    pub fn process(self: *Self, src: []const Mat, dst: *Mat) !void {
        const c_mats: c.struct_Mats = try Mat.toCStructs(src);
        defer Mat.deinitCStructs(c_mats);
        _ = c.MergeMertens_Process(self.ptr, c_mats, dst.*.ptr);
    }
};

/// AlignMTB is a wrapper around the cv::AlignMTB.
pub const AlignMTB = struct {
    ptr: c.AlignMTB,

    const Self = @This();

    /// NewAlignMTB returns an AlignMTB for converts images to median threshold bitmaps.
    /// of type AlignMTB converts images to median threshold bitmaps (1 for pixels
    /// brighter than median luminance and 0 otherwise) and than aligns the resulting
    /// bitmaps using bit operations.
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html
    /// https://docs.opencv.org/master/d7/db6/classcv_1_1AlignMTB.html
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html#ga2f1fafc885a5d79dbfb3542e08db0244
    ///
    pub fn init() !Self {
        const ptr = c.AlignMTB_Create();
        return try initFromC(ptr);
    }

    /// NewAlignMTBWithParams returns an AlignMTB for converts images to median threshold bitmaps.
    /// of type AlignMTB converts images to median threshold bitmaps (1 for pixels
    /// brighter than median luminance and 0 otherwise) and than aligns the resulting
    /// bitmaps using bit operations.
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html
    /// https://docs.opencv.org/master/d7/db6/classcv_1_1AlignMTB.html
    /// https://docs.opencv.org/master/d6/df5/group__photo__hdr.html#ga2f1fafc885a5d79dbfb3542e08db0244
    ///
    pub fn initWithParams(max_bits: i32, exclude_range: i32, cut: bool) !Self {
        const ptr = c.AlignMTB_CreateWithParams(max_bits, exclude_range, cut);
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.AlignMTB) !Self {
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    ///Close AlignMTB
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.AlignMTB_Close(self.ptr);
        self.*.ptr = null;
    }

    /// Process computes an alignment using the current AlignMTB.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/db6/classcv_1_1AlignMTB.html#a37b3417d844f362d781f34155cbcb201
    ///
    pub fn process(self: Self, src: []const Mat, allocator: std.mem.Allocator) !Mats {
        const c_mats: c.struct_Mats = try Mat.toCStructs(src);
        defer Mat.deinitCStructs(c_mats);
        var c_dst_mats: c.struct_Mats = undefined;
        _ = c.AlignMTB_Process(self.ptr, c_mats, &c_dst_mats);
        return Mat.toArrayList(c_dst_mats, allocator);
    }
};

/// DetailEnhance filter enhances the details of a particular image
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gae5930dd822c713b36f8529b21ddebd0c
///
pub fn detailEnhance(src: Mat, dst: *Mat, sigma_s: f32, sigma_r: f32) void {
    _ = c.DetailEnhance(src.ptr, dst.*.ptr, sigma_s, sigma_r);
}

/// EdgePreservingFilter filtering is the fundamental operation in image and video processing.
/// Edge-preserving smoothing filters are used in many different applications.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gafaee2977597029bc8e35da6e67bd31f7
///
pub fn edgePreservingFilter(src: Mat, dst: *Mat, flags: EdgeFilter, sigma_s: f32, sigma_r: f32) void {
    _ = c.EdgePreservingFilter(src.ptr, dst.*.ptr, @intFromEnum(flags), sigma_s, sigma_r);
}

/// EdgePreservingFilter filtering is the fundamental operation in image and video processing.
/// Edge-preserving smoothing filters are used in many different applications.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gafaee2977597029bc8e35da6e67bd31f7
///
pub fn edgePreservingFilterWithKernel(src: Mat, dst: *Mat, kernel: Mat) void {
    _ = c.EdgePreservingFilterWithKernel(src.ptr, dst.*.ptr, kernel.ptr);
}

/// PencilSketch pencil-like non-photorealistic line drawing.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gae5930dd822c713b36f8529b21ddebd0c
///
pub fn pencilSketch(src: Mat, dst1: *Mat, dst2: *Mat, sigma_s: f32, sigma_r: f32, shade_factor: f32) void {
    _ = c.PencilSketch(src.ptr, dst1.*.ptr, dst2.*.ptr, sigma_s, sigma_r, shade_factor);
}

/// Stylization aims to produce digital imagery with a wide variety of effects
/// not focused on photorealism. Edge-aware filters are ideal for stylization,
/// as they can abstract regions of low contrast while preserving, or enhancing,
/// high-contrast features.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/df/dac/group__photo__render.html#gacb0f7324017df153d7b5d095aed53206
///
pub fn stylization(src: Mat, dst: *Mat, sigma_s: f32, sigma_r: f32) void {
    _ = c.Stylization(src.ptr, dst.*.ptr, sigma_s, sigma_r);
}

const testing = std.testing;
const imgcodecs = @import("imgcodecs.zig");
test "photo colorchange" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var dst = try Mat.initSize(20, 20, .cv8uc3);
    defer dst.deinit();

    var mask = try src.clone();
    defer mask.deinit();

    colorChange(src, mask, &dst, 1.5, 1.5, 1.5);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 20), dst.rows());
    try testing.expectEqual(@as(i32, 20), dst.cols());
}

test "photo seamlessClone" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var dst = try Mat.initSize(20, 20, .cv8uc3);
    defer dst.deinit();

    var mask = try src.clone();
    defer mask.deinit();

    var blend = try Mat.initSize(dst.rows(), dst.cols(), dst.getType());
    defer blend.deinit();

    const center = Point.init(@divExact(dst.rows(), 2), @divExact(dst.cols(), 2));
    seamlessClone(src, &dst, mask, center, blend, .normal_clone);

    try testing.expectEqual(false, blend.isEmpty());
    try testing.expectEqual(@as(i32, 20), dst.rows());
    try testing.expectEqual(@as(i32, 20), dst.cols());
    try testing.expectEqual(@as(i32, 20), blend.rows());
    try testing.expectEqual(@as(i32, 20), blend.cols());
}

test "photo illumination change" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var mask = try src.clone();
    defer mask.deinit();

    var dst = try Mat.initSize(20, 20, .cv8uc3);
    defer dst.deinit();

    illuminationChange(src, mask, &dst, 0.2, 0.4);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 20), dst.rows());
    try testing.expectEqual(@as(i32, 20), dst.cols());
}

test "photo textureFlattening" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var mask = try src.clone();
    defer mask.deinit();

    var dst = try Mat.initSize(20, 20, .cv8uc3);
    defer dst.deinit();

    textureFlattening(src, mask, &dst, 30, 45, 3);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 20), dst.rows());
    try testing.expectEqual(@as(i32, 20), dst.cols());
}

test "photo fastNlMeansDenoisingColoredMulti" {
    var src: [3]Mat = undefined;
    for (&src) |*s| s.* = try Mat.initSize(20, 20, .cv8uc3);
    defer for (&src) |*s| s.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    try fastNlMeansDenoisingColoredMulti(src[0..], &dst, 1, 1);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(src[0].rows(), dst.rows());
    try testing.expectEqual(src[0].cols(), dst.cols());
}

test "photo fastNlMeansDenoisingColoredMultiWithParams" {
    var src: [3]Mat = undefined;
    for (&src) |*s| s.* = try Mat.initSize(20, 20, .cv8uc3);
    defer for (&src) |*s| s.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    try fastNlMeansDenoisingColoredMultiWithParams(src[0..], &dst, 1, 1, 3, 3, 7, 21);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(src[0].rows(), dst.rows());
    try testing.expectEqual(src[0].cols(), dst.cols());
}

test "photo MergeMertens" {
    var src: [3]Mat = undefined;
    for (&src) |*s| s.* = try Mat.initSize(20, 20, .cv32fc3);
    defer for (&src) |*s| s.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    var mertens = try MergeMertens.init();
    defer mertens.deinit();

    try mertens.process(src[0..], &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(src[0].rows(), dst.rows());
    try testing.expectEqual(src[0].cols(), dst.cols());
}

test "photo AlignMTB" {
    var src: [3]Mat = undefined;
    for (&src) |*s| s.* = try Mat.initSize(20, 20, .cv8uc3);
    defer for (&src) |*s| s.deinit();

    var align_mtb = try AlignMTB.init();
    defer align_mtb.deinit();

    var dst = try align_mtb.process(src[0..], testing.allocator);
    defer dst.deinit();

    try testing.expect(dst.list.items.len > 0);

    const dst0 = dst.list.items[0];
    try testing.expectEqual(false, dst0.isEmpty());
    try testing.expectEqual(src[0].rows(), dst0.rows());
    try testing.expectEqual(src[0].cols(), dst0.cols());
}

test "photo fastNlMeansDenoising" {
    var img = try imgcodecs.imRead("test/images/face-detect.jpg", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    fastNlMeansDenoising(img, &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "photo FastNlMeansDenoisingColoredMultiWithParams" {
    var img = try imgcodecs.imRead("test/images/face-detect.jpg", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    fastNlMeansDenoisingWithParams(img, &dst, 3, 7, 21);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "photo fastNlMeansDenoisingColored" {
    var img = try imgcodecs.imRead("test/images/face-detect.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    fastNlMeansDenoisingColored(img, &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "photo fastNlMeansDenoisingColoredWithParams" {
    var img = try imgcodecs.imRead("test/images/face-detect.jpg", .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    fastNlMeansDenoisingColoredWithParams(img, &dst, 3, 3, 7, 21);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "photo detailEnhance" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    detailEnhance(src, &dst, 100, 0.5);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(src.rows(), dst.rows());
    try testing.expectEqual(src.cols(), dst.cols());
}

test "photo edgePreservingFilter" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    edgePreservingFilter(src, &dst, .recurs_filter, 100, 0.5);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(src.rows(), dst.rows());
    try testing.expectEqual(src.cols(), dst.cols());
}

test "photo pencilSketch" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var dst1 = try Mat.init();
    defer dst1.deinit();

    var dst2 = try Mat.init();
    defer dst2.deinit();

    pencilSketch(src, &dst1, &dst2, 100, 0.5, 0.5);

    try testing.expectEqual(false, dst1.isEmpty());
    try testing.expectEqual(src.rows(), dst1.rows());
    try testing.expectEqual(src.cols(), dst1.cols());

    try testing.expectEqual(false, dst2.isEmpty());
    try testing.expectEqual(src.rows(), dst2.rows());
    try testing.expectEqual(src.cols(), dst2.cols());
}

test "photo stylization" {
    var src = try Mat.initSize(20, 20, .cv8uc3);
    defer src.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    stylization(src, &dst, 100, 0.5);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(src.rows(), dst.rows());
    try testing.expectEqual(src.cols(), dst.cols());
}

//*    implementation done
//*    pub const MergeMertens = ?*anyopaque;
//*    pub const AlignMTB = ?*anyopaque;
//*    pub extern fn ColorChange(src: Mat, mask: Mat, dst: Mat, red_mul: f32, green_mul: f32, blue_mul: f32) void;
//*    pub extern fn SeamlessClone(src: Mat, dst: Mat, mask: Mat, p: Point, blend: Mat, flags: c_int) void;
//*    pub extern fn IlluminationChange(src: Mat, mask: Mat, dst: Mat, alpha: f32, beta: f32) void;
//*    pub extern fn TextureFlattening(src: Mat, mask: Mat, dst: Mat, low_threshold: f32, high_threshold: f32, kernel_size: c_int) void;
//*    pub extern fn FastNlMeansDenoisingColoredMulti(src: struct_Mats, dst: Mat, imgToDenoiseIndex: c_int, temporalWindowSize: c_int) void;
//*    pub extern fn FastNlMeansDenoisingColoredMultiWithParams(src: struct_Mats, dst: Mat, imgToDenoiseIndex: c_int, temporalWindowSize: c_int, h: f32, hColor: f32, templateWindowSize: c_int, searchWindowSize: c_int) void;
//*    pub extern fn FastNlMeansDenoising(src: Mat, dst: Mat) void;
//*    pub extern fn FastNlMeansDenoisingWithParams(src: Mat, dst: Mat, h: f32, templateWindowSize: c_int, searchWindowSize: c_int) void;
//*    pub extern fn FastNlMeansDenoisingColored(src: Mat, dst: Mat) void;
//*    pub extern fn FastNlMeansDenoisingColoredWithParams(src: Mat, dst: Mat, h: f32, hColor: f32, templateWindowSize: c_int, searchWindowSize: c_int) void;
//*    pub extern fn MergeMertens_Create(...) MergeMertens;
//*    pub extern fn MergeMertens_CreateWithParams(contrast_weight: f32, saturation_weight: f32, exposure_weight: f32) MergeMertens;
//*    pub extern fn MergeMertens_Process(b: MergeMertens, src: struct_Mats, dst: Mat) void;
//*    pub extern fn MergeMertens_Close(b: MergeMertens) void;
//*    pub extern fn AlignMTB_Create(...) AlignMTB;
//*    pub extern fn AlignMTB_CreateWithParams(max_bits: c_int, exclude_range: c_int, cut: bool) AlignMTB;
//*    pub extern fn AlignMTB_Process(b: AlignMTB, src: struct_Mats, dst: [*c]struct_Mats) void;
//*    pub extern fn AlignMTB_Close(b: AlignMTB) void;
//*    pub extern fn DetailEnhance(src: Mat, dst: Mat, sigma_s: f32, sigma_r: f32) void;
//*    pub extern fn EdgePreservingFilter(src: Mat, dst: Mat, filter: c_int, sigma_s: f32, sigma_r: f32) void;
//*    pub extern fn PencilSketch(src: Mat, dst1: Mat, dst2: Mat, sigma_s: f32, sigma_r: f32, shade_factor: f32) void;
//*    pub extern fn Stylization(src: Mat, dst: Mat, sigma_s: f32, sigma_r: f32) void;
