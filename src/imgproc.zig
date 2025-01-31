const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const Mat = core.Mat;
const Size = core.Size;
const Scalar = core.Scalar;
const Color = core.Color;
const Rect = core.Rect;
const Point = core.Point;
const Point2f = core.Point2f;
const PointVector = core.PointVector;
const PointsVector = core.PointsVector;
const Point2fVector = core.Point2fVector;
const RotatedRect = core.RotatedRect;
const MatType = Mat.MatType;
const TermCriteria = core.TermCriteria;
const ColorConversionCode = @import("imgproc/color_codes.zig").ColorConversionCode;

const BorderType = core.BorderType;

pub const ConnectedComponentsAlgorithmType = enum(u2) {
    /// SAUF algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
    wu = 0,

    /// BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity.
    default = 1,

    /// BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity
    grana = 2,
};

pub const ConnectedComponentsType = enum(u3) {
    ///The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    stat_left = 0,

    ///The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
    stat_top = 1,

    /// The horizontal size of the bounding box.
    stat_width = 2,

    /// The vertical size of the bounding box.
    stat_height = 3,

    /// The total area (in pixels) of the connected component.
    stat_area = 4,

    stat_max = 5,
};

pub const MorphType = enum(u3) {
    /// MorphErode operation
    erode = 0,

    /// MorphDilate operation
    dilate = 1,

    /// MorphOpen operation
    open = 2,

    /// MorphClose operation
    close = 3,

    /// MorphGradient operation
    gradient = 4,

    /// MorphTophat operation
    tophat = 5,

    /// MorphBlackhat operation
    blackhat = 6,

    /// MorphHitmiss operation
    hitmiss = 7,
};

pub const HoughMode = enum(u2) {
    /// HoughStandard is the classical or standard Hough transform.
    standard = 0,
    /// HoughProbabilistic is the probabilistic Hough transform (more efficient
    /// in case if the picture contains a few long linear segments).
    probabilistic = 1,
    /// HoughMultiScale is the multi-scale variant of the classical Hough
    /// transform.
    multi_scale = 2,
    /// HoughGradient is basically 21HT, described in: HK Yuen, John Princen,
    /// John Illingworth, and Josef Kittler. Comparative study of hough
    /// transform methods for circle finding. Image and Vision Computing,
    /// 8(1):71–77, 1990.
    gradient = 3,
};

pub const GrabCutMode = enum(u2) {
    /// GCInitWithRect makes the function initialize the state and the mask using the provided rectangle.
    /// After that it runs the itercount iterations of the algorithm.
    init_with_rect = 0,
    /// GCInitWithMask makes the function initialize the state using the provided mask.
    /// GCInitWithMask and GCInitWithRect can be combined.
    /// Then all the pixels outside of the ROI are automatically initialized with GC_BGD.
    init_with_mask = 1,
    /// GCEval means that the algorithm should just resume.
    eval = 2,
    /// GCEvalFreezeModel means that the algorithm should just run a single iteration of the GrabCut algorithm
    /// with the fixed model
    eval_freeze_model = 3,
};

pub const LineType = enum(i6) {
    /// Filled line
    filled = -1,
    /// Line4 4-connected line
    line4 = 4,
    /// Line8 8-connected line
    line8 = 8,
    /// LineAA antialiased line
    line_aa = 16,
};

pub const HersheyFont = struct {
    /// FontItalic is the flag for italic font.
    italic: bool = false,

    type: enum(u4) {
        /// FontHersheySimplex is normal size sans-serif font.
        simplex = 0,
        /// FontHersheyPlain issmall size sans-serif font.
        plain = 1,
        /// FontHersheyDuplex normal size sans-serif font
        /// (more complex than FontHersheySIMPLEX).
        duplex = 2,
        /// FontHersheyComplex i a normal size serif font.
        complex = 3,
        /// FontHersheyTriplex is a normal size serif font
        /// (more complex than FontHersheyCOMPLEX).
        triplex = 4,
        /// FontHersheyComplexSmall is a smaller version of FontHersheyCOMPLEX.
        complex_small = 5,
        /// FontHersheyScriptSimplex is a hand-writing style font.
        script_simplex = 6,
        /// FontHersheyScriptComplex is a more complex variant of FontHersheyScriptSimplex.
        script_complex = 7,
    },

    pub fn toNum(self: HersheyFont) u5 {
        return @as(u5, @bitCast(packed struct {
            type: u4,
            italic: bool,
        }{
            .type = @intFromEnum(self.type),
            .italic = self.italic,
        }));
    }

    comptime {
        std.debug.assert((HersheyFont{ .type = .simplex }).toNum() == 0);
        std.debug.assert((HersheyFont{ .type = .plain }).toNum() == 1);
        std.debug.assert((HersheyFont{ .type = .simplex, .italic = true }).toNum() == 16);
    }
};

pub const InterpolationFlag = struct {
    /// WarpFillOutliers fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero.
    warp_fill_outliers: bool = false,
    /// WarpInverseMap, inverse transformation.
    warp_inverse_map: bool = false,

    type: enum(u3) {
        /// InterpolationNearestNeighbor is nearest neighbor. (fast but low quality)
        nearest_neighbor = 0,

        /// InterpolationLinear is bilinear interpolation.
        linear = 1,

        /// InterpolationCubic is bicube interpolation.
        cubic = 2,

        /// InterpolationArea uses pixel area relation. It is preferred for image
        /// decimation as it gives moire-free results.
        area = 3,

        /// InterpolationLanczos4 is Lanczos interpolation over 8x8 neighborhood.
        lanczos4 = 4,

        /// InterpolationMax indicates use maximum interpolation.
        max = 7,
    } = .linear,

    pub fn toNum(self: InterpolationFlag) u5 {
        return @as(u5, @bitCast(packed struct {
            type: u3,
            warp_fill_outliers: bool,
            warp_inverse_map: bool,
        }{
            .type = @intFromEnum(self.type),
            .warp_fill_outliers = self.warp_fill_outliers,
            .warp_inverse_map = self.warp_inverse_map,
        }));
    }

    comptime {
        std.debug.assert((InterpolationFlag{ .type = .nearest_neighbor }).toNum() == 0);
        std.debug.assert((InterpolationFlag{ .type = .linear }).toNum() == 1);
        std.debug.assert((InterpolationFlag{ .type = .nearest_neighbor, .warp_fill_outliers = true }).toNum() == 8);
        std.debug.assert((InterpolationFlag{ .type = .nearest_neighbor, .warp_inverse_map = true }).toNum() == 16);
        std.debug.assert((InterpolationFlag{ .type = .linear, .warp_fill_outliers = true, .warp_inverse_map = true }).toNum() == 25);
        std.debug.assert((InterpolationFlag{}).toNum() == 1);
    }
};

pub const ColormapType = enum(u4) {
    autumn = 0,
    bone = 1,
    jet = 2,
    winter = 3,
    rainbow = 4,
    ocean = 5,
    summer = 6,
    spring = 7,
    cool = 8,
    hsv = 9,
    pink = 10,
    hot = 11,
    parula = 12,
};

pub const HomographyMethod = enum(u4) {
    all_points = 0,
    lmeds = 4,
    ransac = 8,
};

/// DistanceType types for Distance Transform and M-estimatorss
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gaa2bfbebbc5c320526897996aafa1d8eb
pub const DistanceType = enum(u3) {
    user = 0,
    l1 = 1,
    l2 = 2,
    c = 3,
    l12 = 4,
    fair = 5,
    welsch = 6,
    huber = 7,
};

pub const DistanceTransformLabelType = enum(u1) {
    c_comp = 0,
    pixel = 1,
};

pub const DistanceTransformMask = enum(i32) {
    mask_3 = 3,
    mask_5 = 5,
    mask_precise = 0,
};

pub const ThresholdType = struct {
    /// ThresholdOtsu threshold type
    otsu: bool = false,
    /// ThresholdTriangle threshold type
    triangle: bool = false,

    type: enum(u3) {
        /// ThresholdBinary threshold type
        binary = 0,

        /// ThresholdBinaryInv threshold type
        binary_inv = 1,

        /// ThresholdTrunc threshold type
        trunc = 2,

        /// ThresholdToZero threshold type
        to_zero = 3,

        /// ThresholdToZeroInv threshold type
        to_zero_inv = 4,

        /// ThresholdMask threshold type
        mask = 7,
    },

    pub fn toNum(self: ThresholdType) u5 {
        return @as(u5, @bitCast(packed struct {
            type: u3,
            otsu: bool,
            triangle: bool,
        }{
            .type = @intFromEnum(self.type),
            .otsu = self.otsu,
            .triangle = self.triangle,
        }));
    }

    comptime {
        std.debug.assert((ThresholdType{ .type = .binary }).toNum() == 0);
        std.debug.assert((ThresholdType{ .type = .binary_inv }).toNum() == 1);
        std.debug.assert((ThresholdType{ .type = .binary, .otsu = true }).toNum() == 8);
        std.debug.assert((ThresholdType{ .type = .binary, .triangle = true }).toNum() == 16);
        std.debug.assert((ThresholdType{ .type = .binary, .otsu = true, .triangle = true }).toNum() == 24);
    }
};

/// AdaptiveThresholdType type of adaptive threshold operation.
pub const AdaptiveThresholdType = enum(u1) {
    //// AdaptiveThresholdMean threshold type
    mean = 0,

    //// AdaptiveThresholdGaussian threshold type
    gaussian = 1,
};

/// RetrievalMode is the mode of the contour retrieval algorithm.
pub const RetrievalMode = enum(u3) {
    /// RetrievalExternal retrieves only the extreme outer contours.
    /// It sets `hierarchy[i][2]=hierarchy[i][3]=-1` for all the contours.
    external = 0,

    /// RetrievalList retrieves all of the contours without establishing
    /// any hierarchical relationships.
    list = 1,

    /// RetrievalCComp retrieves all of the contours and organizes them into
    /// a two-level hierarchy. At the top level, there are external boundaries
    /// of the components. At the second level, there are boundaries of the holes.
    /// If there is another contour inside a hole of a connected component, it
    /// is still put at the top level.
    c_comp = 2,

    /// RetrievalTree retrieves all of the contours and reconstructs a full
    /// hierarchy of nested contours.
    tree = 3,

    /// RetrievalFloodfill lacks a description in the original header.
    floodfill = 4,
};

/// ContourApproximationMode is the mode of the contour approximation algorithm.
pub const ContourApproximationMode = enum(u3) {
    /// ChainApproxNone stores absolutely all the contour points. That is,
    /// any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be
    /// either horizontal, vertical or diagonal neighbors, that is,
    /// max(abs(x1-x2),abs(y2-y1))==1.
    none = 1,

    /// ChainApproxSimple compresses horizontal, vertical, and diagonal segments
    /// and leaves only their end points.
    /// For example, an up-right rectangular contour is encoded with 4 points.
    simple = 2,

    /// ChainApproxTC89L1 applies one of the flavors of the Teh-Chin chain
    /// approximation algorithms.
    tc89L1 = 3,

    /// ChainApproxTC89KCOS applies one of the flavors of the Teh-Chin chain
    /// approximation algorithms.
    tc89kcos = 4,
};

/// MorphShape is the shape of the structuring element used for Morphing operations.
pub const MorphShape = enum(u2) {
    // MorphRect is the rectangular morph shape.
    rect = 0,
    /// MorphCross is the cross morph shape.
    cross = 1,
    /// MorphEllipse is the ellipse morph shape.
    ellipse = 2,
};

/// TemplateMatchMode is the type of the template matching operation.
pub const TemplateMatchMode = enum(u3) {
    /// TmSqdiff maps to TM_SQDIFF
    sq_diff = 0,
    /// TmSqdiffNormed maps to TM_SQDIFF_NORMED
    sq_diff_normed = 1,
    /// TmCcorr maps to TM_CCORR
    ccorr = 2,
    /// TmCcorrNormed maps to TM_CCORR_NORMED
    ccorr_normed = 3,
    /// TmCcoeff maps to TM_CCOEFF
    ccoeff = 4,
    /// TmCcoeffNormed maps to TM_CCOEFF_NORMED
    ccoeff_normed = 5,
};

/// HistCompMethod is the method for Histogram comparison
/// For more information, see https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#ga994f53817d621e2e4228fc646342d386
pub const HistCompMethod = enum(u3) {
    /// HistCmpCorrel calculates the Correlation
    correl = 0,

    /// HistCmpChiSqr calculates the Chi-Square
    chi_sqr = 1,

    /// HistCmpIntersect calculates the Intersection
    intersect = 2,

    /// HistCmpBhattacharya applies the HistCmpBhattacharya by calculating the Bhattacharya distance.
    /// HistCmpHellinger applies the HistCmpBhattacharya comparison. It is a synonym to HistCmpBhattacharya.
    bhattacharya = 3,

    /// HistCmpChiSqrAlt applies the Alternative Chi-Square (regularly used for texture comparsion).
    chi_sqr_alt = 4,

    /// HistCmpKlDiv applies the Kullback-Liebler divergence comparison.
    kl_div = 5,
};

/// CLAHE is a wrapper around the cv::CLAHE algorithm.
pub const CLAHE = struct {
    ptr: c.CLAHE,

    const Self = @This();

    /// NewCLAHE returns a new CLAHE algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/db6/classcv_1_1CLAHE.html
    ///
    pub fn init() !Self {
        const ptr = c.CLAHE_Create();
        return initFromC(ptr);
    }

    /// NewCLAHEWithParams returns a new CLAHE algorithm
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/db6/classcv_1_1CLAHE.html
    ///
    pub fn initWithParams(clip_limit: f64, tile_grid_size: Size) !Self {
        const ptr = c.CLAHE_CreateWithParams(clip_limit, tile_grid_size.toC());
        return initFromC(ptr);
    }

    fn initFromC(ptr: c.CLAHE) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    /// Close the CLAHE algorithm
    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.CLAHE_Close(self.ptr);
        self.ptr = null;
    }

    pub fn toC(self: Self) c.CLANE {
        return self.ptr;
    }

    /// Apply CLAHE.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d6/db6/classcv_1_1CLAHE.html#a4e92e0e427de21be8d1fae8dcd862c5e
    ///
    pub fn apply(self: Self, src: Mat, dst: *Mat) void {
        c.CLAHE_Apply(self.ptr, src.toC(), dst.toC());
    }
};

pub fn arcLength(curve: PointVector, is_closed: bool) f64 {
    return c.ArcLength(curve.toC(), is_closed);
}

pub fn approxPolyDP(curve: PointVector, epsilon: f64, closed: bool) !PointVector {
    const ptr = c.ApproxPolyDP(curve.toC(), epsilon, closed);
    return try PointVector.initFromC(ptr);
}

/// CvtColor converts an image from one color space to another.
/// It converts the src Mat image to the dst Mat using the
/// code param containing the desired ColorConversionCode color space.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0
///
pub fn cvtColor(src: Mat, dst: *Mat, code: ColorConversionCode) void {
    c.CvtColor(src.ptr, dst.*.ptr, @intFromEnum(code));
}

/// EqualizeHist normalizes the brightness and increases the contrast of the image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e
pub fn equalizeHist(src: Mat, dst: *Mat) void {
    return c.EqualizeHist(src.ptr, dst.*.ptr);
}

/// CalcHist Calculates a histogram of a set of images
///
/// For futher details, please see:
/// https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#ga6ca1876785483836f72a77ced8ea759a
pub fn calcHist(mats: []Mat, chans: []i32, mask: Mat, hist: *Mat, sz: []i32, rng: []f32, acc: bool) !void {
    const c_mat = try Mat.toCStructs(mats);
    const c_chans = c.IntVector{
        .val = &chans[0],
        .length = @as(i32, @intCast(chans.len)),
    };
    const c_sz = c.IntVector{
        .val = @as([*]i32, @ptrCast(sz.ptr)),
        .length = @as(i32, @intCast(sz.len)),
    };
    const c_rng = c.FloatVector{
        .val = @as([*]f32, @ptrCast(rng.ptr)),
        .length = @as(i32, @intCast(rng.len)),
    };

    c.CalcHist(c_mat, c_chans, mask.ptr, hist.*.ptr, c_sz, c_rng, acc);
}
/// CompareHist Compares two histograms.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#gaf4190090efa5c47cb367cf97a9a519bd
pub fn compareHist(hist1: Mat, hist2: Mat, method: HistCompMethod) f64 {
    return c.CompareHist(hist1.ptr, hist2.ptr, @intFromEnum(method));
}

pub fn calcBackProject(mats: []Mat, chans: []i32, hist: *Mat, backProject: Mat, rng: []f32, uniform: bool) !void {
    const c_mats = try Mat.toCStructs(mats);
    const c_chans = c.IntVector{
        .val = &chans[0],
        .length = @as(i32, @intCast(chans.len)),
    };
    const c_rng = c.FloatVector{
        .val = @as([*]f32, @ptrCast(rng.ptr)),
        .length = @as(i32, @intCast(rng.len)),
    };
    c.CalcBackProject(c_mats, c_chans, hist.*.ptr, backProject.ptr, c_rng, uniform);
}

/// ConvexHull finds the convex hull of a point set.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656
///
pub fn convexHull(points: PointVector, hull: *Mat, clockwise: bool, return_points: bool) void {
    c.ConvexHull(points.toC(), hull.*.ptr, clockwise, return_points);
}

/// ConvexityDefects finds the convexity defects of a contour.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gada4437098113fd8683c932e0567f47ba
///
pub fn convexityDefects(points: PointVector, hull: Mat, result: *Mat) void {
    c.ConvexityDefects(points.toC(), hull.ptr, result.*.ptr);
}

/// BilateralFilter applies a bilateral filter to an image.
///
/// Bilateral filtering is described here:
/// http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
///
pub fn bilateralFilter(src: Mat, dst: *Mat, d: i32, sc: f64, ss: f64) void {
    c.BilateralFilter(src.ptr, dst.*.ptr, d, sc, ss);
}

/// Blur blurs an image Mat using a normalized box filter.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37
///
pub fn blur(src: Mat, dst: *Mat, ps: Size) void {
    c.Blur(src.ptr, dst.*.ptr, ps.toC());
}

/// BoxFilter blurs an image using the box filter.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gad533230ebf2d42509547d514f7d3fbc3
///
pub fn boxFilter(src: Mat, dst: *Mat, ddepth: i32, ps: Size) void {
    c.BoxFilter(src.ptr, dst.*.ptr, ddepth, ps.toC());
}

/// SqBoxFilter calculates the normalized sum of squares of the pixel values overlapping the filter.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga045028184a9ef65d7d2579e5c4bff6c0
///
pub fn sqBoxFilter(src: Mat, dst: *Mat, ddepth: i32, ps: Size) void {
    c.SqBoxFilter(src.ptr, dst.*.ptr, ddepth, ps.toC());
}

/// Dilate dilates an image by using a specific structuring element.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
///
pub fn dilate(src: Mat, dst: *Mat, kernel: Mat) void {
    c.Dilate(src.ptr, dst.*.ptr, kernel.ptr);
}

/// DilateWithParams dilates an image by using a specific structuring element.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
pub fn dilateWithParams(src: Mat, dst: *Mat, kernel: Mat, anchor: Point, iterations: BorderType, border_type: BorderType, border_value: Color) void {
    c.DilateWithParams(src.ptr, dst.*.ptr, kernel.ptr, anchor.toC(), iterations.toNum(), border_type.toNum(), border_value.toScalar().toC());
}

/// DistanceTransform Calculates the distance to the closest zero pixel for each pixel of the source image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042
///
pub fn distanceTransform(src: Mat, dst: *Mat, labels: *Mat, distance_type: DistanceType, mask_size: DistanceTransformMask, label_type: DistanceTransformLabelType) void {
    c.DistanceTransform(src.ptr, dst.*.ptr, labels.*.ptr, @intFromEnum(distance_type), @intFromEnum(mask_size), @intFromEnum(label_type));
}

/// Erode erodes an image by using a specific structuring element.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
///
pub fn erode(src: Mat, dst: *Mat, kernel: Mat) void {
    c.Erode(src.ptr, dst.*.ptr, kernel.ptr);
}

/// ErodeWithParams erodes an image by using a specific structuring element.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
///
pub fn erodeWithParams(src: Mat, dst: *Mat, kernel: Mat, anchor: Point, iterations: i32, border_type: i32) void {
    c.ErodeWithParams(src.ptr, dst.*.ptr, kernel.ptr, anchor.toC(), iterations, border_type);
}

/// MatchTemplate compares a template against overlapped image regions.
///
/// For further details, please see:
/// https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be
///
pub fn matchTemplate(image: Mat, templ: Mat, result: *Mat, method: TemplateMatchMode, mask: Mat) void {
    c.MatchTemplate(image.ptr, templ.ptr, result.*.ptr, @intFromEnum(method), mask.ptr);
}

/// Moments calculates all of the moments up to the third order of a polygon
/// or rasterized shape.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga556a180f43cab22649c23ada36a8a139
///
pub fn moments(src: Mat, binary_image: bool) c.struct_Moment {
    return c.Moments(src.ptr, binary_image);
}

/// PyrDown blurs an image and downsamples it.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff
///
pub fn pyrDown(src: Mat, dst: *Mat, dstsize: Size, border_type: BorderType) void {
    c.PyrDown(src.ptr, dst.*.ptr, dstsize.toC(), border_type.toNum());
}

/// PyrUp upsamples an image and then blurs it.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gada75b59bdaaca411ed6fee10085eb784
///
pub fn pyrUp(src: Mat, dst: *Mat, dstsize: Size, border_type: BorderType) void {
    c.PyrUp(src.ptr, dst.*.ptr, dstsize.toC(), border_type.toNum());
}

/// BoundingRect calculates the up-right bounding rectangle of a point set.
///
/// For further details, please see:
/// https://docs.opencv.org/3.3.0/d3/dc0/group__imgproc__shape.html#gacb413ddce8e48ff3ca61ed7cf626a366
///
pub fn boundingRect(pts: PointVector) Rect {
    return Rect.initFromC(c.BoundingRect(pts.toC()));
}

/// BoxPoints finds the four vertices of a rotated rect. Useful to draw the rotated rectangle.
///
/// For further Details, please see:
/// https://docs.opencv.org/3.3.0/d3/dc0/group__imgproc__shape.html#gaf78d467e024b4d7936cf9397185d2f5c
///
pub fn boxPoints(rect: RotatedRect, box_pts: *Mat) void {
    c.BoxPoints(rect.toC(), box_pts.*.ptr);
}

/// ContourArea calculates a contour area.
///
/// For further details, please see:
/// https://docs.opencv.org/3.3.0/d3/dc0/group__imgproc__shape.html#ga2c759ed9f497d4a618048a2f56dc97f1
///
pub fn contourArea(pts: PointVector) f64 {
    return c.ContourArea(pts.toC());
}

/// MinAreaRect finds a rotated rectangle of the minimum area enclosing the input 2D point set.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9
///
pub fn minAreaRect(pts: PointVector) RotatedRect {
    return RotatedRect.initFromC(c.MinAreaRect(pts.toC()));
}

/// FitEllipse Fits an ellipse around a set of 2D points.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaf259efaad93098103d6c27b9e4900ffa
///
pub fn fitEllipse(pts: PointVector) RotatedRect {
    return RotatedRect.initFromC(c.FitEllipse(pts.toC()));
}

/// MinEnclosingCircle finds a circle of the minimum area enclosing the input 2D point set.
///
/// For further details, please see:
/// https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga8ce13c24081bbc7151e9326f412190f1
pub fn minEnclosingCircle(pts: PointVector) struct { point: Point2f, radius: f32 } {
    var c_center: c.Point2f = undefined;
    var radius: f32 = undefined;
    c.MinEnclosingCircle(pts.toC(), @as([*]c.Point2f, @ptrCast(&c_center)), @as([*]f32, @ptrCast(&radius)));
    const center: Point2f = Point2f.initFromC(c_center);
    return .{ .point = center, .radius = radius };
}

/// FindContours finds contours in a binary image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b
///
//     pub extern fn FindContours(src: Mat, hierarchy: Mat, mode: c_int, method: c_int) PointsVector;
pub fn findContours(src: Mat, mode: RetrievalMode, method: ContourApproximationMode) !PointsVector {
    var hierarchy = try Mat.init();
    defer hierarchy.deinit();
    return try findContoursWithParams(src, &hierarchy, mode, method);
}

/// FindContoursWithParams finds contours in a binary image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
///
pub fn findContoursWithParams(src: Mat, hierarchy: *Mat, mode: RetrievalMode, method: ContourApproximationMode) !PointsVector {
    return try PointsVector.initFromC(c.FindContours(src.toC(), hierarchy.*.toC(), @intFromEnum(mode), @intFromEnum(method)));
}

/// PointPolygonTest performs a point-in-contour test.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga1a539e8db2135af2566103705d7a5722
///
pub fn pointPolygonTest(pts: PointVector, pt: Point, measure_dist: bool) f64 {
    return c.PointPolygonTest(pts.toC(), pt.toC(), measure_dist);
}

/// ConnectedComponents computes the connected components labeled image of boolean image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
///
pub fn connectedComponents(src: Mat, labels: *Mat) i32 {
    return c.ConnectedComponents(
        src.ptr,
        labels.*.ptr,
        8,
        @intFromEnum(MatType.cv32sc1),
        @intFromEnum(ConnectedComponentsAlgorithmType.default),
    );
}

/// ConnectedComponents computes the connected components labeled image of boolean image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
///
pub fn connectedComponentsWithParams(src: Mat, labels: *Mat, connectivity: i32, ltype: MatType, ccltype: ConnectedComponentsAlgorithmType) i32 {
    return c.ConnectedComponents(src.ptr, labels.*.ptr, connectivity, @intFromEnum(ltype), @intFromEnum(ccltype));
}

/// ConnectedComponentsWithStats computes the connected components labeled image of boolean
/// image and also produces a statistics output for each label.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
///
pub fn connectedComponentsWithStats(src: Mat, labels: *Mat, stats: *Mat, centroids: *Mat) i32 {
    return c.ConnectedComponentsWithStats(
        src.ptr,
        labels.*.ptr,
        stats.*.ptr,
        centroids.*.ptr,
        8,
        @intFromEnum(MatType.cv32sc1),
        @intFromEnum(ConnectedComponentsAlgorithmType.default),
    );
}

/// ConnectedComponentsWithStats computes the connected components labeled image of boolean
/// image and also produces a statistics output for each label.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
///
pub fn connectedComponentsWithStatsWithParams(src: Mat, labels: *Mat, stats: *Mat, centroids: *Mat, connectivity: i32, ltype: MatType, ccltype: ConnectedComponentsAlgorithmType) i32 {
    return c.ConnectedComponentsWithStats(src.ptr, labels.*.ptr, stats.*.ptr, centroids.*.ptr, connectivity, @intFromEnum(ltype), @intFromEnum(ccltype));
}

/// GaussianBlur blurs an image Mat using a Gaussian filter.
/// The function convolves the src Mat image into the dst Mat using
/// the specified Gaussian kernel params.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
///
pub fn gaussianBlur(src: Mat, dst: *Mat, ps: Size, sigma_x: f64, sigma_y: f64, border_type: BorderType) void {
    c.GaussianBlur(src.ptr, dst.*.ptr, ps.toC(), sigma_x, sigma_y, border_type.toNum());
}

/// GetGaussianKernel returns Gaussian filter coefficients.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
pub fn getGaussianKernel(ksize: i32, sigma: f64) !Mat {
    return try Mat.initFromC(c.GetGaussianKernel(ksize, sigma, @intFromEnum(MatType.cv64fc1)));
}

/// GetGaussianKernelWithParams returns Gaussian filter coefficients.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
pub fn getGaussianKernelWithParams(ksize: i32, sigma: f64, ktype: MatType) !Mat {
    return try Mat.initFromC(c.GetGaussianKernel(ksize, sigma, @intFromEnum(ktype)));
}

/// Laplacian calculates the Laplacian of an image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6
///
pub fn laplacian(src: Mat, dst: *Mat, d_depth: MatType, k_size: i32, scale: f64, delta: f64, border_type: BorderType) void {
    c.Laplacian(src.ptr, dst.*.ptr, @intFromEnum(d_depth), k_size, scale, delta, border_type.toNum());
}

/// Scharr calculates the first x- or y- image derivative using Scharr operator.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaa13106761eedf14798f37aa2d60404c9
///
pub fn scharr(src: Mat, dst: *Mat, d_depth: MatType, dx: i32, dy: i32, scale: f64, delta: f64, border_type: BorderType) void {
    c.Scharr(src.ptr, dst.*.ptr, @intFromEnum(d_depth), dx, dy, scale, delta, border_type.toNum());
}

/// GetStructuringElement returns a structuring element of the specified size
/// and shape for morphological operations.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc
///
pub fn getStructuringElement(shape: MorphShape, ksize: Size) !Mat {
    return try Mat.initFromC(c.GetStructuringElement(@intFromEnum(shape), ksize.toC()));
}

/// MorphologyDefaultBorder returns "magic" border value for erosion and dilation.
/// It is automatically transformed to Scalar::all(-DBL_MAX) for dilation.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga94756fad83d9d24d29c9bf478558c40a
///
pub fn morphologyDefaultBorderValue() Scalar {
    return Scalar.initFromC(c.MorphologyDefaultBorderValue());
}

/// MorphologyEx performs advanced morphological transformations.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
pub fn morphologyEx(src: Mat, dst: *Mat, op: MorphType, kernel: Mat) void {
    c.MorphologyEx(src.ptr, dst.*.ptr, @intFromEnum(op), kernel.ptr);
}

/// MorphologyExWithParams performs advanced morphological transformations.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
pub fn morphologyExWithParams(src: Mat, dst: *Mat, op: MorphType, kernel: Mat, iterations: i32, border_type: BorderType) void {
    const c_pt = Point.init(-1, -1).toC();
    c.MorphologyExWithParams(src.ptr, dst.*.ptr, @intFromEnum(op), kernel.ptr, c_pt, iterations, border_type.toNum());
}

/// MedianBlur blurs an image using the median filter.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
pub fn medianBlur(src: Mat, dst: *Mat, ksize: i32) void {
    c.MedianBlur(src.ptr, dst.*.ptr, ksize);
}

/// Canny finds edges in an image using the Canny algorithm.
/// The function finds edges in the input image image and marks
/// them in the output map edges using the Canny algorithm.
/// The smallest value between threshold1 and threshold2 is used
/// for edge linking. The largest value is used to
/// find initial segments of strong edges.
/// See http://en.wikipedia.org/wiki/Canny_edge_detector
///
/// For further details, please see:
/// http://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
pub fn canny(src: Mat, edges: *Mat, t1: f64, t2: f64) void {
    c.Canny(src.ptr, edges.*.ptr, t1, t2);
}

/// CornerSubPix Refines the corner locations. The function iterates to find
/// the sub-pixel accurate location of corners or radial saddle points.
///
/// For further details, please see:
/// https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
///
pub fn cornerSubPix(img: Mat, corners: *Mat, winSize: Size, zeroZone: Size, criteria: TermCriteria) void {
    c.CornerSubPix(img.ptr, corners.*.ptr, winSize.toC(), zeroZone.toC(), criteria.toC());
}

/// GoodFeaturesToTrack determines strong corners on an image. The function
/// finds the most prominent corners in the image or in the specified image region.
///
/// For further details, please see:
/// https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
///
pub fn goodFeaturesToTrack(img: Mat, corners: *Mat, maxCorners: i32, quality: f64, minDist: f64) void {
    c.GoodFeaturesToTrack(img.ptr, corners.*.ptr, maxCorners, quality, minDist);
}

/// Grabcut runs the GrabCut algorithm.
/// The function implements the GrabCut image segmentation algorithm.
/// For further details, please see:
/// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga909c1dda50efcbeaa3ce126be862b37f
pub fn grabCut(img: Mat, mask: *Mat, rect: Rect, bgd_model: *Mat, fgd_model: *Mat, iter_count: i32, mode: GrabCutMode) void {
    c.GrabCut(img.ptr, mask.*.ptr, rect.toC(), bgd_model.*.ptr, fgd_model.*.ptr, iter_count, @intFromEnum(mode));
}

/// HoughCircles finds circles in a grayscale image using the Hough transform.
/// The only "method" currently supported is HoughGradient. If you want to pass
/// more parameters, please see `HoughCirclesWithParams`.
///
/// For further details, please see:
/// https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
pub fn houghCircles(src: Mat, circles: *Mat, method: HoughMode, dp: f64, min_dist: f64) void {
    c.HoughCircles(src.ptr, circles.*.ptr, @intFromEnum(method), dp, min_dist);
}

/// HoughCirclesWithParams finds circles in a grayscale image using the Hough
/// transform. The only "method" currently supported is HoughGradient.
///
/// For further details, please see:
/// https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
pub fn houghCirclesWithParams(src: Mat, circles: *Mat, method: HoughMode, dp: f64, min_dist: f64, param1: f64, param2: f64, min_radius: i32, max_radius: i32) void {
    c.HoughCirclesWithParams(src.ptr, circles.*.ptr, @intFromEnum(method), dp, min_dist, param1, param2, min_radius, max_radius);
}

/// HoughLines implements the standard or standard multi-scale Hough transform
/// algorithm for line detection. For a good explanation of Hough transform, see:
/// http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm
///
/// For further details, please see:
/// http://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
pub fn houghLines(src: Mat, lines: *Mat, rho: f64, theta: f64, threshold_int: i32) void {
    c.HoughLines(src.ptr, lines.*.ptr, rho, theta, threshold_int);
}

/// HoughLinesP implements the probabilistic Hough transform
/// algorithm for line detection. For a good explanation of Hough transform, see:
/// http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm
///
/// For further details, please see:
/// http://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
pub fn houghLinesP(src: Mat, lines: *Mat, rho: f64, theta: f64, threshold_int: i32) void {
    c.HoughLinesP(src.ptr, lines.*.ptr, rho, theta, threshold_int);
}
pub fn houghLinesPWithParams(src: Mat, lines: *Mat, rho: f64, theta: f64, threshold_int: i32, minLineLength: f64, maxLineGap: f64) void {
    c.HoughLinesPWithParams(src.ptr, lines.*.ptr, rho, theta, threshold_int, minLineLength, maxLineGap);
}
// HoughLinesPointSet implements the Hough transform algorithm for line
// detection on a set of points. For a good explanation of Hough transform, see:
// http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm
//
// For further details, please see:
// https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga2858ef61b4e47d1919facac2152a160e
pub fn houghLinesPointSet(
    points: Mat,
    lines: *Mat,
    lines_max: c_int,
    threshold_int: i32,
    min_rho: f64,
    max_rho: f64,
    rho_step: f64,
    min_theta: f64,
    max_theta: f64,
    theta_step: f64,
) void {
    c.HoughLinesPointSet(
        points.ptr,
        lines.*.ptr,
        lines_max,
        threshold_int,
        min_rho,
        max_rho,
        rho_step,
        min_theta,
        max_theta,
        theta_step,
    );
}

/// Integral calculates one or more integral images for the source image.
/// For further details, please see:
/// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga97b87bec26908237e8ba0f6e96d23e28
///
pub fn integral(src: Mat, sum: *Mat, sqsum: *Mat, tilted: *Mat) void {
    c.Integral(src.ptr, sum.*.ptr, sqsum.*.ptr, tilted.*.ptr);
}

/// Threshold applies a fixed-level threshold to each array element.
///
/// For further details, please see:
/// https://docs.opencv.org/3.3.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
///
pub fn threshold(src: Mat, dst: *Mat, thresh: f64, maxvalue: f64, typ: ThresholdType) f64 {
    return c.Threshold(src.ptr, dst.*.ptr, thresh, maxvalue, typ.toNum());
}

pub fn adaptiveThreshold(src: Mat, dst: *Mat, max_value: f64, adaptive_type: AdaptiveThresholdType, type_: ThresholdType, block_size: i32, C: f64) void {
    c.AdaptiveThreshold(src.ptr, dst.*.ptr, max_value, @intFromEnum(adaptive_type), type_.toNum(), block_size, C);
}
// ArrowedLine draws a arrow segment pointing from the first point
// to the second one.
//
// For further details, please see:
// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga0a165a3ca093fd488ac709fdf10c05b2
//
pub fn arrowedLine(img: *Mat, pt1: Point, pt2: Point, color: Color, thickness: i32) void {
    c.ArrowedLine(img.*.ptr, pt1.toC(), pt2.toC(), color.toScalar().toC(), thickness);
}

/// Circle draws a circle.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
///
pub fn circle(img: *Mat, center: Point, radius: i32, color: Color, thickness: i32) void {
    c.Circle(img.*.ptr, center.toC(), radius, color.toScalar().toC(), thickness);
}

/// CircleWithParams draws a circle.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
///
pub fn circleWithParams(img: *Mat, center: Point, radius: i32, color: Color, thickness: i32, line_type: LineType, shift: i32) void {
    c.CircleWithParams(img.*.ptr, center.toC(), radius, color.toScalar().toC(), thickness, @intFromEnum(line_type), shift);
}

/// Ellipse draws a simple or thick elliptic arc or fills an ellipse sector.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
///
pub fn ellipse(img: *Mat, center: Point, axes: Point, angle: f64, start_angle: f64, end_angle: f64, color: Color, thickness: i32) void {
    c.Ellipse(img.*.ptr, center.toC(), axes.toC(), angle, start_angle, end_angle, color.toScalar().toC(), thickness);
}

/// Ellipse draws a simple or thick elliptic arc or fills an ellipse sector.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
pub fn ellipseWithParams(img: *Mat, center: Point, axes: Point, angle: f64, start_angle: f64, end_angle: f64, color: Color, thickness: i32, line_type: LineType, shift: i32) void {
    c.EllipseWithParams(img.*.ptr, center.toC(), axes.toC(), angle, start_angle, end_angle, color.toScalar().toC(), thickness, @intFromEnum(line_type), shift);
}

/// Line draws a line segment connecting two points.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
pub fn line(img: *Mat, pt1: Point, pt2: Point, color: Color, thickness: i32) void {
    c.Line(img.ptr, pt1.toC(), pt2.toC(), color.toScalar().toC(), thickness);
}

/// Rectangle draws a simple, thick, or filled up-right rectangle.
/// It renders a rectangle with the desired characteristics to the target Mat image.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga346ac30b5c74e9b5137576c9ee9e0e8c
///
pub fn rectangle(img: *Mat, rect: Rect, color: Color, thickness: i32) void {
    c.Rectangle(img.*.ptr, rect.toC(), color.toScalar().toC(), thickness);
}

/// RectangleWithParams draws a simple, thick, or filled up-right rectangle.
/// It renders a rectangle with the desired characteristics to the target Mat image.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga346ac30b5c74e9b5137576c9ee9e0e8c
///
pub fn rectangleWithParams(img: *Mat, rect: Rect, color: Color, thickness: i32, line_type: LineType, shift: i32) void {
    c.RectangleWithParams(img.*.ptr, rect.toC(), color.toScalar().toC(), thickness, @intFromEnum(line_type), shift);
}

/// FillPoly fills the area bounded by one or more polygons.
///
/// For more information, see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf30888828337aa4c6b56782b5dfbd4b7
pub fn fillPoly(img: *Mat, points: PointsVector, color: Color) void {
    c.FillPoly(img.*.ptr, points.toC(), color.toScalar().toC());
}

/// FillPolyWithParams fills the area bounded by one or more polygons.
///
/// For more information, see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf30888828337aa4c6b56782b5dfbd4b7
pub fn fillPolyWithParams(img: *Mat, points: PointsVector, color: Color, line_type: LineType, shift: i32, offset: Point) void {
    c.FillPolyWithParams(img.*.ptr, points.toC(), color.toScalar().toC(), @intFromEnum(line_type), shift, offset.toC());
}

/// Polylines draws several polygonal curves.
///
/// For more information, see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga1ea127ffbbb7e0bfc4fd6fd2eb64263c
pub fn polylines(img: *Mat, points: PointsVector, is_closed: bool, color: Color, thickness: i32) void {
    c.Polylines(img.*.ptr, points.toC(), is_closed, color.toScalar().toC(), thickness);
}

/// GetTextSize calculates the width and height of a text string.
/// It returns an image.Point with the size required to draw text using
/// a specific font face, scale, and thickness.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82
///
pub fn getTextSize(text: []const u8, font_face: HersheyFont, font_scale: f64, thickness: i32) Size {
    const c_size = c.GetTextSize(@as([*]const u8, @ptrCast(text)), font_face.toNum(), font_scale, thickness);
    return Size.initFromC(c_size);
}

/// GetTextSizeWithBaseline calculates the width and height of a text string including the basline of the text.
/// It returns an image.Point with the size required to draw text using
/// a specific font face, scale, and thickness as well as its baseline.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82
///
pub fn getTextSizeWithBaseline(text: []const u8, font_face: HersheyFont, font_scale: f64, thickness: i32) struct { size: Size, baseline: i32 } {
    var baseline: i32 = 0;
    const c_size = c.GetTextSizeWithBaseline(@as([*]const u8, @ptrCast(text)), font_face.toNum(), font_scale, thickness, &baseline);
    const size = Size.initFromC(c_size);
    return .{
        .size = size,
        .baseline = baseline,
    };
}

// PutText draws a text string.
// It renders the specified text string into the img Mat at the location
// passed in the "org" param, using the desired font face, font scale,
// color, and line thinkness.
//
// For further details, please see:
// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
//
pub fn putText(img: *Mat, text: []const u8, org: Point, font_face: HersheyFont, font_scale: f64, color: Color, thickness: i32) void {
    c.PutText(img.*.ptr, @as([*]const u8, @ptrCast(text)), org.toC(), font_face.toNum(), font_scale, color.toScalar().toC(), thickness);
}

// PutTextWithParams draws a text string.
// It renders the specified text string into the img Mat at the location
// passed in the "org" param, using the desired font face, font scale,
// color, and line thinkness.
//
// For further details, please see:
// http://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
//
pub fn putTextWithParams(img: *Mat, text: []const u8, org: Point, font_face: HersheyFont, font_scale: f64, color: Color, thickness: i32, line_type: LineType, bottom_left_origin: bool) void {
    c.PutTextWithParams(img.*.ptr, @as([*]const u8, @ptrCast(text)), org.toC(), font_face.toNum(), font_scale, color.toScalar().toC(), thickness, @intFromEnum(line_type), bottom_left_origin);
}

/// Resize resizes an image.
/// It resizes the image src down to or up to the specified size, storing the
/// result in dst. Note that src and dst may be the same image. If you wish to
/// scale by factor, an empty sz may be passed and non-zero fx and fy. Likewise,
/// if you wish to scale to an explicit size, a non-empty sz may be passed with
/// zero for both fx and fy.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
pub fn resize(src: Mat, dst: *Mat, sz: Size, fx: f64, fy: f64, interp: InterpolationFlag) void {
    c.Resize(src.ptr, dst.*.ptr, sz.toC(), fx, fy, interp.toNum());
}

/// GetRectSubPix retrieves a pixel rectangle from an image with sub-pixel accuracy.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga77576d06075c1a4b6ba1a608850cd614
pub fn getRectSubPix(src: Mat, patch_size: Size, center: Point, dst: *Mat) void {
    c.GetRectSubPix(src.ptr, patch_size.toC(), center.toC(), dst.*.ptr);
}

/// GetRotationMatrix2D calculates an affine matrix of 2D rotation.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
pub fn getRotationMatrix2D(center: Point, angle: f64, scale: f64) !Mat {
    return try Mat.initFromC(c.GetRotationMatrix2D(center.toC(), angle, scale));
}

/// WarpAffine applies an affine transformation to an image. For more parameters please check WarpAffineWithParams
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
pub fn warpAffine(src: Mat, dst: *Mat, rot_mat: Mat, dsize: Size) void {
    c.WarpAffine(src.ptr, dst.*.ptr, rot_mat.ptr, dsize.toC());
}

/// WarpAffineWithParams applies an affine transformation to an image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
pub fn warpAffineWithParams(src: Mat, dst: *Mat, rot_mat: Mat, dsize: Size, flags: InterpolationFlag, border_mode: BorderType, border_value: Color) void {
    c.WarpAffineWithParams(src.ptr, dst.*.ptr, rot_mat.ptr, dsize.toC(), flags.toNum(), border_mode.toNum(), border_value.toScalar().toC());
}

/// WarpPerspective applies a perspective transformation to an image.
/// For more parameters please check WarpPerspectiveWithParams.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
pub fn warpPerspective(src: Mat, dst: *Mat, m: Mat, dsize: Size) void {
    c.WarpPerspective(src.ptr, dst.*.ptr, m.ptr, dsize.toC());
}

/// WarpPerspectiveWithParams applies a perspective transformation to an image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
pub fn warpPerspectiveWithParams(src: Mat, dst: *Mat, rot_mat: Mat, dsize: Size, flags: InterpolationFlag, border_mode: BorderType, border_value: Color) void {
    c.WarpPerspectiveWithParams(src.ptr, dst.*.ptr, rot_mat.ptr, dsize.toC(), flags.toNum(), border_mode.toNum(), border_value.toScalar().toC());
}

/// Watershed performs a marker-based image segmentation using the watershed algorithm.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1
pub fn watershed(image: Mat, markers: *Mat) void {
    c.Watershed(image.ptr, markers.*.ptr);
}

/// ApplyColorMap applies a GNU Octave/MATLAB equivalent colormap on a given image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#gadf478a5e5ff49d8aa24e726ea6f65d15
pub fn applyColorMap(src: Mat, dst: *Mat, colormap: ColormapType) void {
    c.ApplyColorMap(src.ptr, dst.*.ptr, @intFromEnum(colormap));
}

/// ApplyCustomColorMap applies a custom defined colormap on a given image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#gacb22288ddccc55f9bd9e6d492b409cae
pub fn applyCustomColorMap(src: Mat, dst: *Mat, colormap: Mat) void {
    c.ApplyCustomColorMap(src.ptr, dst.*.ptr, colormap.ptr);
}

/// GetPerspectiveTransform returns 3x3 perspective transformation for the
/// corresponding 4 point pairs as image.Point.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8c1ae0e3589a9d77fffc962c49b22043
pub fn getPerspectiveTransform(src: PointVector, dst: PointVector) !Mat {
    return try Mat.initFromC(c.GetPerspectiveTransform(src.toC(), dst.toC()));
}

/// GetPerspectiveTransform2f returns 3x3 perspective transformation for the
/// corresponding 4 point pairs as gocv.Point2f.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8c1ae0e3589a9d77fffc962c49b22043
pub fn getPerspectiveTransform2f(src: Point2fVector, dst: Point2fVector) !Mat {
    return try Mat.initFromC(c.GetPerspectiveTransform2f(src.toC(), dst.toC()));
}

/// GetAffineTransform returns a 2x3 affine transformation matrix for the
/// corresponding 3 point pairs as image.Point.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999
pub fn getAffineTransform(src: PointVector, dst: PointVector) !Mat {
    return try Mat.initFromC(c.GetAffineTransform(src.toC(), dst.toC()));
}

/// GetAffineTransform2f returns a 2x3 affine transformation matrix for the
/// corresponding 3 point pairs as gocv.Point2f.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999
pub fn getAffineTransform2f(src: Point2fVector, dst: Point2fVector) !Mat {
    return try Mat.initFromC(c.GetAffineTransform2f(src.toC(), dst.toC()));
}

/// FindHomography finds an optimal homography matrix using 4 or more point pairs (as opposed to GetPerspectiveTransform, which uses exactly 4)
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
///
pub fn findHomography(src: Mat, dst: *Mat, method: HomographyMethod, ransac_reproj_threshold: f64, mask: *Mat, max_iters: i32, confidence: f64) !Mat {
    return try Mat.initFromC(c.FindHomography(src.ptr, dst.*.ptr, @intFromEnum(method), ransac_reproj_threshold, mask.*.ptr, max_iters, confidence));
}

/// DrawContours draws contours outlines or filled contours.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
pub fn drawContours(src: *Mat, contours: PointsVector, contour_idx: i32, color: Color, thickness: i32) void {
    c.DrawContours(src.*.ptr, contours.toC(), contour_idx, color.toScalar().toC(), thickness);
}

/// DrawContoursWithParams draws contours outlines or filled contours.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
pub fn drawContoursWithParams(
    src: *Mat,
    contours: PointsVector,
    contour_idx: i32,
    color: Color,
    thickness: i32,
    line_type: LineType,
    hierarchy: Mat,
    max_level: i32,
    offset: Point,
) void {
    c.DrawContoursWithParams(
        src.*.ptr,
        contours.toC(),
        contour_idx,
        color.toScalar().toC(),
        thickness,
        @intFromEnum(line_type),
        hierarchy.ptr,
        max_level,
        offset.toC(),
    );
}

/// Sobel calculates the first, second, third, or mixed image derivatives using an extended Sobel operator
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
///
pub fn sobel(src: Mat, dst: *Mat, ddepth: MatType, dx: i32, dy: i32, ksize: i32, scale: f64, delta: f64, border_type: BorderType) void {
    c.Sobel(src.ptr, dst.*.ptr, @intFromEnum(ddepth), dx, dy, ksize, scale, delta, border_type.toNum());
}

/// SpatialGradient calculates the first order image derivative in both x and y using a Sobel operator.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga405d03b20c782b65a4daf54d233239a2
///
pub fn spatialGradient(src: Mat, dx: *Mat, dy: *Mat, ksize: MatType, border_type: BorderType) void {
    c.SpatialGradient(src.ptr, dx.*.ptr, dy.*.ptr, @intFromEnum(ksize), border_type.toNum());
}

/// Remap applies a generic geometrical transformation to an image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
pub fn remap(src: Mat, dst: *Mat, map1: Mat, map2: Mat, interpolation: InterpolationFlag, border_mode: BorderType, border_value: Color) void {
    c.Remap(src.ptr, dst.*.ptr, map1.ptr, map2.ptr, interpolation.toNum(), border_mode.toNum(), border_value.toScalar().toC());
}

/// Filter2D applies an arbitrary linear filter to an image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
pub fn filter2D(src: Mat, dst: *Mat, ddepth: i32, kernel: Mat, anchor: Point, delta: f64, border_type: BorderType) void {
    c.Filter2D(src.ptr, dst.*.ptr, ddepth, kernel.ptr, anchor.toC(), delta, border_type.toNum());
}

/// SepFilter2D applies a separable linear filter to the image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga910e29ff7d7b105057d1625a4bf6318d
pub fn sepFilter2D(src: Mat, dst: *Mat, ddepth: i32, kernel_x: Mat, kernel_y: Mat, anchor: Point, delta: f64, border_type: BorderType) void {
    c.SepFilter2D(src.ptr, dst.*.ptr, ddepth, kernel_x.ptr, kernel_y.ptr, anchor.toC(), delta, border_type.toNum());
}

/// LogPolar remaps an image to semilog-polar coordinates space.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaec3a0b126a85b5ca2c667b16e0ae022d
pub fn logPolar(src: Mat, dst: *Mat, center: Point, m: f64, flags: InterpolationFlag) void {
    c.LogPolar(src.ptr, dst.*.ptr, center.toC(), m, flags.toNum());
}

/// FitLine fits a line to a 2D or 3D point set.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaf849da1fdafa67ee84b1e9a23b93f91f
pub fn fitLine(pts: PointVector, line_mat: *Mat, dist_type: DistanceType, param: f64, reps: f64, aeps: f64) void {
    c.FitLine(pts.toC(), line_mat.*.ptr, @intFromEnum(dist_type), param, reps, aeps);
}

/// LinearPolar remaps an image to polar coordinates space.
///
/// For further details, please see:
/// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaa38a6884ac8b6e0b9bed47939b5362f3
///
pub fn linearPolar(src: Mat, dst: *Mat, center: Point, max_radius: f64, flags: InterpolationFlag) void {
    c.LinearPolar(src.ptr, dst.*.ptr, center.toC(), max_radius, flags.toNum());
}

/// ClipLine clips the line against the image rectangle.
/// For further details, please see:
/// https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf483cb46ad6b049bc35ec67052ef1c2c
///
pub fn clipLine(imgSize: Size, pt1: Point, pt2: Point) bool {
    return c.ClipLine(imgSize.toC(), pt1.toC(), pt2.toC());
}

pub fn invertAffineTransform(src: Mat, dst: *Mat) void {
    c.InvertAffineTransform(src.ptr, dst.*.ptr);
}

/// Apply phaseCorrelate.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga552420a2ace9ef3fb053cd630fdb4952
///
pub fn phaseCorrelate(src1: Mat, src2: Mat, window: Mat) struct { point: Point2f, response: f64 } {
    var response: f64 = undefined;
    const p = c.PhaseCorrelate(src1.ptr, src2.ptr, window.ptr, &response);
    return .{ .point = Point2f.initFromC(p), .response = response };
}

/// Adds the square of a source image to the accumulator image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga1a567a79901513811ff3b9976923b199
///
pub fn accumulate(src: Mat, dst: *Mat) void {
    c.Mat_Accumulate(src.ptr, dst.*.ptr);
}

/// Adds an image to the accumulator image with mask.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga1a567a79901513811ff3b9976923b199
///
pub fn accumulateWithMask(src: Mat, dst: *Mat, mask: Mat) void {
    c.Mat_AccumulateWithMask(src.ptr, dst.*.ptr, mask.ptr);
}

/// Adds the square of a source image to the accumulator image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#gacb75e7ffb573227088cef9ceaf80be8c
///
pub fn accumulateSquare(src: Mat, dst: *Mat) void {
    c.Mat_AccumulateSquare(src.ptr, dst.*.ptr);
}

/// Adds the square of a source image to the accumulator image with mask.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#gacb75e7ffb573227088cef9ceaf80be8c
///
pub fn accumulateSquareWithMask(src: Mat, dst: *Mat, mask: Mat) void {
    c.Mat_AccumulateSquareWithMask(src.ptr, dst.*.ptr, mask.ptr);
}

/// Adds the per-element product of two input images to the accumulator image.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga82518a940ecfda49460f66117ac82520
///
pub fn accumulateProduct(src1: Mat, src2: Mat, dst: *Mat) void {
    c.Mat_AccumulateProduct(src1.ptr, src2.ptr, dst.*.ptr);
}

/// Adds the per-element product of two input images to the accumulator image with mask.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga82518a940ecfda49460f66117ac82520
///
pub fn accumulateProductWithMask(src1: Mat, src2: Mat, dst: *Mat, mask: Mat) void {
    c.Mat_AccumulateProductWithMask(src1.ptr, src2.ptr, dst.*.ptr, mask.ptr);
}

/// Updates a running average.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga4f9552b541187f61f6818e8d2d826bc7
///
pub fn accumulatedWeighted(src: Mat, dst: *Mat, alpha: f64) void {
    c.Mat_AccumulatedWeighted(src.ptr, dst.*.ptr, alpha);
}

/// Updates a running average with mask.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html#ga4f9552b541187f61f6818e8d2d826bc7
///
pub fn accumulatedWeightedWithMask(src: Mat, dst: *Mat, alpha: f64, mask: Mat) void {
    c.Mat_AccumulatedWeightedWithMask(src.ptr, dst.*.ptr, alpha, mask.ptr);
}

test "imgproc" {
    _ = @import("imgproc/test.zig");
    _ = HersheyFont;
    _ = InterpolationFlag;
    _ = ThresholdType;
}

//*    implementation done
//*    pub const CLAHE = ?*anyopaque;
//*    pub extern fn ArcLength(curve: PointVector, is_closed: bool) f64;
//*    pub extern fn ApproxPolyDP(curve: PointVector, epsilon: f64, closed: bool) PointVector;
//*    pub extern fn CvtColor(src: Mat, dst: Mat, code: c_int) void;
//*    pub extern fn EqualizeHist(src: Mat, dst: Mat) void;
//*    pub extern fn CalcHist(mats: struct_Mats, chans: IntVector, mask: Mat, hist: Mat, sz: IntVector, rng: FloatVector, acc: bool) void;
//*    pub extern fn CalcBackProject(mats: struct_Mats, chans: IntVector, hist: Mat, backProject: Mat, rng: FloatVector, uniform: bool) void;
//*    pub extern fn CompareHist(hist1: Mat, hist2: Mat, method: c_int) f64;
//*    pub extern fn ConvexHull(points: PointVector, hull: Mat, clockwise: bool, returnPoints: bool) void;
//*    pub extern fn ConvexityDefects(points: PointVector, hull: Mat, result: Mat) void;
//*    pub extern fn BilateralFilter(src: Mat, dst: Mat, d: c_int, sc: f64, ss: f64) void;
//*    pub extern fn Blur(src: Mat, dst: Mat, ps: Size) void;
//*    pub extern fn BoxFilter(src: Mat, dst: Mat, ddepth: c_int, ps: Size) void;
//*    pub extern fn SqBoxFilter(src: Mat, dst: Mat, ddepth: c_int, ps: Size) void;
//*    pub extern fn Dilate(src: Mat, dst: Mat, kernel: Mat) void;
//*    pub extern fn DilateWithParams(src: Mat, dst: Mat, kernel: Mat, anchor: Point, iterations: c_int, borderType: c_int, borderValue: Scalar) void;
//*    pub extern fn DistanceTransform(src: Mat, dst: Mat, labels: Mat, distanceType: c_int, maskSize: c_int, labelType: c_int) void;
//*    pub extern fn Erode(src: Mat, dst: Mat, kernel: Mat) void;
//*    pub extern fn ErodeWithParams(src: Mat, dst: Mat, kernel: Mat, anchor: Point, iterations: c_int, borderType: c_int) void;
//*    pub extern fn MatchTemplate(image: Mat, templ: Mat, result: Mat, method: c_int, mask: Mat) void;
//*    pub extern fn Moments(src: Mat, binaryImage: bool) struct_Moment;
//*    pub extern fn PyrDown(src: Mat, dst: Mat, dstsize: Size, borderType: c_int) void;
//*    pub extern fn PyrUp(src: Mat, dst: Mat, dstsize: Size, borderType: c_int) void;
//*    pub extern fn BoundingRect(pts: PointVector) struct_Rect;
//*    pub extern fn BoxPoints(rect: RotatedRect, boxPts: Mat) void;
//*    pub extern fn ContourArea(pts: PointVector) f64;
//*    pub extern fn MinAreaRect(pts: PointVector) struct_RotatedRect;
//*    pub extern fn FitEllipse(pts: PointVector) struct_RotatedRect;
//*    pub extern fn MinEnclosingCircle(pts: PointVector, center: [*c]Point2f, radius: [*c]f32) void;
//*    pub extern fn FindContours(src: Mat, hierarchy: Mat, mode: c_int, method: c_int) PointsVector;
//*    pub extern fn PointPolygonTest(pts: PointVector, pt: Point, measureDist: bool) f64;
//*    pub extern fn ConnectedComponents(src: Mat, dst: Mat, connectivity: c_int, ltype: c_int, ccltype: c_int) c_int;
//*    pub extern fn ConnectedComponentsWithStats(src: Mat, labels: Mat, stats: Mat, centroids: Mat, connectivity: c_int, ltype: c_int, ccltype: c_int) c_int;
//*    pub extern fn GaussianBlur(src: Mat, dst: Mat, ps: Size, sX: f64, sY: f64, bt: c_int) void;
//*    pub extern fn GetGaussianKernel(ksize: c_int, sigma: f64, ktype: c_int) Mat;
//*    pub extern fn Laplacian(src: Mat, dst: Mat, dDepth: c_int, kSize: c_int, scale: f64, delta: f64, borderType: c_int) void;
//*    pub extern fn Scharr(src: Mat, dst: Mat, dDepth: c_int, dx: c_int, dy: c_int, scale: f64, delta: f64, borderType: c_int) void;
//*    pub extern fn GetStructuringElement(shape: c_int, ksize: Size) Mat;
//*    pub extern fn MorphologyDefaultBorderValue(...) Scalar;
//*    pub extern fn MorphologyEx(src: Mat, dst: Mat, op: c_int, kernel: Mat) void;
//*    pub extern fn MorphologyExWithParams(src: Mat, dst: Mat, op: c_int, kernel: Mat, pt: Point, iterations: c_int, borderType: c_int) void;
//*    pub extern fn MedianBlur(src: Mat, dst: Mat, ksize: c_int) void;
//*    pub extern fn Canny(src: Mat, edges: Mat, t1: f64, t2: f64) void;
//*    pub extern fn CornerSubPix(img: Mat, corners: Mat, winSize: Size, zeroZone: Size, criteria: TermCriteria) void;
//*    pub extern fn GoodFeaturesToTrack(img: Mat, corners: Mat, maxCorners: c_int, quality: f64, minDist: f64) void;
//*    pub extern fn GrabCut(img: Mat, mask: Mat, rect: Rect, bgdModel: Mat, fgdModel: Mat, iterCount: c_int, mode: c_int) void;
//*    pub extern fn HoughCircles(src: Mat, circles: Mat, method: c_int, dp: f64, minDist: f64) void;
//*    pub extern fn HoughCirclesWithParams(src: Mat, circles: Mat, method: c_int, dp: f64, minDist: f64, param1: f64, param2: f64, minRadius: c_int, maxRadius: c_int) void;
//*    pub extern fn HoughLines(src: Mat, lines: Mat, rho: f64, theta: f64, threshold: c_int) void;
//*    pub extern fn HoughLinesP(src: Mat, lines: Mat, rho: f64, theta: f64, threshold: c_int) void;
//*    pub extern fn HoughLinesPWithParams(src: Mat, lines: Mat, rho: f64, theta: f64, threshold: c_int, minLineLength: f64, maxLineGap: f64) void;
//*    pub extern fn HoughLinesPointSet(points: Mat, lines: Mat, lines_max: c_int, threshold: c_int, min_rho: f64, max_rho: f64, rho_step: f64, min_theta: f64, max_theta: f64, theta_step: f64) void;
//*    pub extern fn Integral(src: Mat, sum: Mat, sqsum: Mat, tilted: Mat) void;
//*    pub extern fn Threshold(src: Mat, dst: Mat, thresh: f64, maxvalue: f64, typ: c_int) f64;
//*    pub extern fn AdaptiveThreshold(src: Mat, dst: Mat, maxValue: f64, adaptiveTyp: c_int, typ: c_int, blockSize: c_int, c: f64) void;
//*    pub extern fn ArrowedLine(img: Mat, pt1: Point, pt2: Point, color: Scalar, thickness: c_int) void;
//*    pub extern fn Circle(img: Mat, center: Point, radius: c_int, color: Scalar, thickness: c_int) void;
//*    pub extern fn CircleWithParams(img: Mat, center: Point, radius: c_int, color: Scalar, thickness: c_int, lineType: c_int, shift: c_int) void;
//*    pub extern fn Ellipse(img: Mat, center: Point, axes: Point, angle: f64, startAngle: f64, endAngle: f64, color: Scalar, thickness: c_int) void;
//*    pub extern fn EllipseWithParams(img: Mat, center: Point, axes: Point, angle: f64, startAngle: f64, endAngle: f64, color: Scalar, thickness: c_int, lineType: c_int, shift: c_int) void;
//*    pub extern fn Line(img: Mat, pt1: Point, pt2: Point, color: Scalar, thickness: c_int) void;
//*    pub extern fn Rectangle(img: Mat, rect: Rect, color: Scalar, thickness: c_int) void;
//*    pub extern fn RectangleWithParams(img: Mat, rect: Rect, color: Scalar, thickness: c_int, lineType: c_int, shift: c_int) void;
//*    pub extern fn FillPoly(img: Mat, points: PointsVector, color: Scalar) void;
//*    pub extern fn FillPolyWithParams(img: Mat, points: PointsVector, color: Scalar, lineType: c_int, shift: c_int, offset: Point) void;
//*    pub extern fn Polylines(img: Mat, points: PointsVector, isClosed: bool, color: Scalar, thickness: c_int) void;
//*    pub extern fn GetTextSize(text: [*c]const u8, fontFace: c_int, fontScale: f64, thickness: c_int) struct_Size;
//*    pub extern fn GetTextSizeWithBaseline(text: [*c]const u8, fontFace: c_int, fontScale: f64, thickness: c_int, baseline: [*c]c_int) struct_Size;
//*    pub extern fn PutText(img: Mat, text: [*c]const u8, org: Point, fontFace: c_int, fontScale: f64, color: Scalar, thickness: c_int) void;
//*    pub extern fn PutTextWithParams(img: Mat, text: [*c]const u8, org: Point, fontFace: c_int, fontScale: f64, color: Scalar, thickness: c_int, lineType: c_int, bottomLeftOrigin: bool) void;
//*    pub extern fn Resize(src: Mat, dst: Mat, sz: Size, fx: f64, fy: f64, interp: c_int) void;
//*    pub extern fn GetRectSubPix(src: Mat, patchSize: Size, center: Point, dst: Mat) void;
//*    pub extern fn GetRotationMatrix2D(center: Point, angle: f64, scale: f64) Mat;
//*    pub extern fn WarpAffine(src: Mat, dst: Mat, rot_mat: Mat, dsize: Size) void;
//*    pub extern fn WarpAffineWithParams(src: Mat, dst: Mat, rot_mat: Mat, dsize: Size, flags: c_int, borderMode: c_int, borderValue: Scalar) void;
//*    pub extern fn WarpPerspective(src: Mat, dst: Mat, m: Mat, dsize: Size) void;
//*    pub extern fn WarpPerspectiveWithParams(src: Mat, dst: Mat, rot_mat: Mat, dsize: Size, flags: c_int, borderMode: c_int, borderValue: Scalar) void;
//*    pub extern fn Watershed(image: Mat, markers: Mat) void;
//*    pub extern fn ApplyColorMap(src: Mat, dst: Mat, colormap: c_int) void;
//*    pub extern fn ApplyCustomColorMap(src: Mat, dst: Mat, colormap: Mat) void;
//*    pub extern fn GetPerspectiveTransform(src: PointVector, dst: PointVector) Mat;
//*    pub extern fn GetPerspectiveTransform2f(src: Point2fVector, dst: Point2fVector) Mat;
//*    pub extern fn GetAffineTransform(src: PointVector, dst: PointVector) Mat;
//*    pub extern fn GetAffineTransform2f(src: Point2fVector, dst: Point2fVector) Mat;
//*    pub extern fn FindHomography(src: Mat, dst: Mat, method: c_int, ransacReprojThreshold: f64, mask: Mat, maxIters: c_int, confidence: f64) Mat;
//*    pub extern fn DrawContours(src: Mat, contours: PointsVector, contourIdx: c_int, color: Scalar, thickness: c_int) void;
//*    pub extern fn DrawContoursWithParams(src: Mat, contours: PointsVector, contourIdx: c_int, color: Scalar, thickness: c_int, lineType: c_int, hierarchy: Mat, maxLevel: c_int, offset: Point) void;
//*    pub extern fn Sobel(src: Mat, dst: Mat, ddepth: c_int, dx: c_int, dy: c_int, ksize: c_int, scale: f64, delta: f64, borderType: c_int) void;
//*    pub extern fn SpatialGradient(src: Mat, dx: Mat, dy: Mat, ksize: c_int, borderType: c_int) void;
//*    pub extern fn Remap(src: Mat, dst: Mat, map1: Mat, map2: Mat, interpolation: c_int, borderMode: c_int, borderValue: Scalar) void;
//*    pub extern fn Filter2D(src: Mat, dst: Mat, ddepth: c_int, kernel: Mat, anchor: Point, delta: f64, borderType: c_int) void;
//*    pub extern fn SepFilter2D(src: Mat, dst: Mat, ddepth: c_int, kernelX: Mat, kernelY: Mat, anchor: Point, delta: f64, borderType: c_int) void;
//*    pub extern fn LogPolar(src: Mat, dst: Mat, center: Point, m: f64, flags: c_int) void;
//*    pub extern fn FitLine(pts: PointVector, line: Mat, distType: c_int, param: f64, reps: f64, aeps: f64) void;
//*    pub extern fn LinearPolar(src: Mat, dst: Mat, center: Point, maxRadius: f64, flags: c_int) void;
//*    pub extern fn ClipLine(imgSize: Size, pt1: Point, pt2: Point) bool;
//*    pub extern fn CLAHE_Create(...) CLAHE;
//*    pub extern fn CLAHE_CreateWithParams(clipLimit: f64, tileGridSize: Size) CLAHE;
//*    pub extern fn CLAHE_Close(c: CLAHE) void;
//*    pub extern fn CLAHE_Apply(c: CLAHE, src: Mat, dst: Mat) void;
//*    pub extern fn InvertAffineTransform(src: Mat, dst: Mat) void;
//*    pub extern fn PhaseCorrelate(src1: Mat, src2: Mat, window: Mat, response: [*c]f64) Point2f;
//*    pub extern fn Mat_Accumulate(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_AccumulateWithMask(src: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_AccumulateSquare(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_AccumulateSquareWithMask(src: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_AccumulateProduct(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_AccumulateProductWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_AccumulatedWeighted(src: Mat, dst: Mat, alpha: f64) void;
//*    pub extern fn Mat_AccumulatedWeightedWithMask(src: Mat, dst: Mat, alpha: f64, mask: Mat) void;
// TODO: ToImg
// TODO: ToImageYUV
