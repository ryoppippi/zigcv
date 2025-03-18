const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const Mat = core.Mat;
const Size = core.Size;
const Rect = core.Rect;
const Point2f = core.Point2f;
const Point2fVector = core.Point2fVector;
const Points2fVector = core.Points2fVector;
const Point3f = core.Point3f;
const Point3fVector = core.Point3fVector;
const Points3fVector = core.Points3fVector;

/// Calib is a wrapper around OpenCV's "Camera Calibration and 3D Reconstruction" of
/// Fisheye Camera model
///
/// For more details, please see:
/// https://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html
/// CalibFlag value for calibration
pub const CalibFlags = packed struct(i32) {
    /// CalibUseIntrinsicGuess indicates that cameraMatrix contains valid initial values
    /// of fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially
    /// set to the image center ( imageSize is used), and focal distances are computed
    /// in a least-squares fashion.
    use_intrinsic_guess: bool = false,

    /// CalibRecomputeExtrinsic indicates that extrinsic will be recomputed after each
    /// iteration of intrinsic optimization.
    recompute_extrinsic: bool = false,

    /// CalibCheckCond indicates that the functions will check validity of condition number
    check_cond: bool = false,

    /// CalibFixSkew indicates that skew coefficient (alpha) is set to zero and stay zero
    fix_skew: bool = false,

    /// CalibFixK1 indicates that selected distortion coefficients are set to zeros and stay zero
    fix_k1: bool = false,

    /// CalibFixK2 indicates that selected distortion coefficients are set to zeros and stay zero
    fix_k2: bool = false,

    /// CalibFixK3 indicates that selected distortion coefficients are set to zeros and stay zero
    fix_k3: bool = false,

    /// CalibFixK4 indicates that selected distortion coefficients are set to zeros and stay zero
    fix_k4: bool = false,

    /// CalibFixIntrinsic indicates that fix K1, K2? and D1, D2? so that only R, T matrices are estimated
    fix_intrinsic: bool = false,

    /// CalibFixPrincipalPoint indicates that the principal point is not changed during the global optimization.
    /// It stays at the center or at a different location specified when CalibUseIntrinsicGuess is set too.
    fix_principal_point: bool = false,

    zero_disparity: bool = false,

    _pading: u21 = 0,

    const Self = @This();

    pub fn toNum(self: Self) i32 {
        return @as(i32, @bitCast(self));
    }

    comptime {
        std.debug.assert(@sizeOf(Self) == @sizeOf(i32));
        std.debug.assert((Self{ .use_intrinsic_guess = true }).toNum() == 1 << 0);
        std.debug.assert((Self{}).toNum() == 0);
    }
};

/// CalibCBFlag value for chessboard calibration
/// For more details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
pub const CalibCBFlags = packed struct(i32) {
    /// Various operation flags that can be zero or a combination of the following values:
    ///  Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness).
    adaptive_thresh: bool = false,
    ///  Normalize the image gamma with equalizeHist before applying fixed or adaptive thresholding.
    normalize_image: bool = false,
    ///  Use additional criteria (like contour area, perimeter, square-like shape) to filter out false quads extracted at the contour retrieval stage.
    filter_quads: bool = false,
    ///  Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed.
    fast_check: bool = false,
    ///  Run an exhaustive search to improve detection rate.
    exhaustive: bool = false,
    ///  Up sample input image to improve sub-pixel accuracy due to aliasing effects.
    accuracy: bool = false,
    ///  The detected pattern is allowed to be larger than patternSize (see description).
    larger: bool = false,
    ///  The detected pattern must have a marker (see description). This should be used if an accurate camera calibration is required.
    marker: bool = false,

    _pading: u24 = 0,

    const Self = @This();

    pub fn toNum(self: Self) i32 {
        return @as(i32, @bitCast(self));
    }

    comptime {
        std.debug.assert(@sizeOf(Self) == @sizeOf(i32));
        std.debug.assert((Self{ .adaptive_thresh = true }).toNum() == 1 << 0);
        std.debug.assert((Self{}).toNum() == 0);
    }
};

pub const Fisheye = struct {
    pub fn undistortImage(image: Mat, undistorted: *Mat, k: Mat, d: Mat) void {
        _ = c.Fisheye_UndistortImage(
            image.toC(),
            undistorted.*.toC(),
            k.toC(),
            d.toC(),
        );
    }

    pub fn undistortImageWithParams(distorted: Mat, undistorted: *Mat, k: Mat, d: Mat, k_new: Mat, size: Size) void {
        _ = c.Fisheye_UndistortImageWithParams(distorted.toC(), undistorted.*.toC(), k.toC(), d.toC(), k_new.toC(), size.toC());
    }

    pub fn undistortPoints(distorted: Mat, undistorted: *Mat, k: Mat, d: Mat, r: Mat, p: Mat) void {
        _ = c.Fisheye_UndistortPoints(distorted.toC(), undistorted.*.toC(), k.toC(), d.toC(), r.toC(), p.toC());
    }

    pub fn estimateNewCameraMatrixForUndistortRectify(
        k: Mat,
        d: Mat,
        img_size: Size,
        r: Mat,
        p: *Mat,
        balance: f64,
        new_size: Size,
        fov_scale: f64,
    ) !Mat {
        const mat = try Mat.init();
        _ = c.Fisheye_EstimateNewCameraMatrixForUndistortRectify(
            k.toC(),
            d.toC(),
            img_size.toC(),
            r.toC(),
            p.*.toC(),
            balance,
            new_size.toC(),
            fov_scale,
        );
        return mat;
    }
};

/// InitUndistortRectifyMap computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
///
pub fn initUndistortRectifyMap(
    camera_matrix: Mat,
    dist_coeffs: Mat,
    r: Mat,
    new_camera_matrix: Mat,
    size: Size,
    m1type: core.Mat.MatType,
    map1: Mat,
    map2: Mat,
) void {
    _ = c.InitUndistortRectifyMap(
        camera_matrix.toC(),
        dist_coeffs.toC(),
        r.toC(),
        new_camera_matrix.toC(),
        size.toC(),
        @intFromEnum(m1type),
        map1.toC(),
        map2.toC(),
    );
}

/// GetOptimalNewCameraMatrixWithParams computes and returns the optimal new camera matrix based on the free scaling parameter.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1
///
//     pub extern fn GetOptimalNewCameraMatrixWithParams(cameraMatrix: Mat, distCoeffs: Mat, size: Size, alpha: f64, newImgSize: Size, validPixROI: [*c]Rect, centerPrincipalPoint: bool) Mat;
pub fn getOptimalNewCameraMatrixWithParams(
    camera_matrix: Mat,
    dist_coeffs: Mat,
    size: Size,
    alpha: f64,
    new_img_size: Size,
    center_principal_point: bool,
) !struct { mat: Mat, roi: Rect } {
    var c_rect: c.Rect = undefined;
    const mat_ptr = c.GetOptimalNewCameraMatrixWithParams(
        camera_matrix.toC(),
        dist_coeffs.toC(),
        size.toC(),
        alpha,
        new_img_size.toC(),
        &c_rect,
        center_principal_point,
    );
    const mat = try Mat.initFromC(mat_ptr);
    const rect = Rect.initFromC(c_rect);
    return .{ .mat = mat, .roi = rect };
}

/// CalibrateCamera finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
///
//     pub extern fn CalibrateCamera(objectPoints: Points3fVector, imagePoints: Points2fVector, imageSize: Size, cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, flag: c_int) f64;
pub fn calibrateCamera(
    object_points: Points3fVector,
    image_points: Points2fVector,
    image_size: Size,
    camera_matrix: *Mat,
    dist_coeffs: *Mat,
    rvecs: *Mat,
    tvecs: *Mat,
    flag: CalibFlags,
) f64 {
    return c.CalibrateCamera(
        object_points.toC(),
        image_points.toC(),
        image_size.toC(),
        camera_matrix.*.toC(),
        dist_coeffs.*.toC(),
        rvecs.*.toC(),
        tvecs.*.toC(),
        flag.toNum(),
    );
}

pub fn undistort(src: Mat, dst: *Mat, camera_matrix: Mat, dist_coeffs: Mat, new_camera_matrix: Mat) void {
    _ = c.Undistort(src.toC(), dst.*.toC(), camera_matrix.toC(), dist_coeffs.toC(), new_camera_matrix.toC());
}

/// UndistortPoints transforms points to compensate for lens distortion
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga55c716492470bfe86b0ee9bf3a1f0f7e
pub fn undistortPoints(src: Mat, dst: *Mat, camera_matrix: Mat, dist_coeffs: Mat, r: Mat, p: Mat) void {
    _ = c.UndistortPoints(src.toC(), dst.*.toC(), camera_matrix.toC(), dist_coeffs.toC(), r.toC(), p.toC());
}

/// FindChessboardCorners finds the positions of internal corners of the chessboard.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
///
pub fn findChessboardCorners(
    image: Mat,
    pattern_size: Size,
    corners: *Mat,
    flags: CalibCBFlags,
) bool {
    return c.FindChessboardCorners(image.toC(), pattern_size.toC(), corners.*.toC(), flags.toNum());
}

/// FindChessboardCorners finds the positions of internal corners of the chessboard using a sector based approach.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#gadc5bcb05cb21cf1e50963df26986d7c9
///
pub fn findChessboardCornersSB(
    image: Mat,
    pattern_size: Size,
    corners: *Mat,
    flags: CalibCBFlags,
) bool {
    return c.FindChessboardCornersSB(image.toC(), pattern_size.toC(), corners.*.toC(), flags.toNum());
}

/// FindChessboardCornersSBWithMeta finds the positions of internal corners of the chessboard using a sector based approach.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
///
pub fn findChessboardCornersSBWithMeta(
    image: Mat,
    pattern_size: Size,
    corners: *Mat,
    flags: CalibCBFlags,
    meta: *Mat,
) bool {
    return c.FindChessboardCornersSBWithMeta(image.toC(), pattern_size.toC(), corners.*.toC(), flags.toNum(), meta.*.toC());
}

/// DrawChessboardCorners renders the detected chessboard corners.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga6a10b0bb120c4907e5eabbcd22319022
///
pub fn drawChessboardCorners(
    image: *Mat,
    pattern_size: Size,
    corners: Mat,
    pattern_was_found: bool,
) void {
    _ = c.DrawChessboardCorners(image.toC(), pattern_size.toC(), corners.toC(), pattern_was_found);
}

/// EstimateAffinePartial2D computes an optimal limited affine transformation
/// with 4 degrees of freedom between two 2D point sets.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d
pub fn estimateAffinePartial2D(from: Point2fVector, to: Point2fVector) !Mat {
    const ptr = c.EstimateAffinePartial2D(from.toC(), to.toC());
    return try Mat.initFromC(ptr);
}

/// EstimateAffine2D Computes an optimal affine transformation between two 2D point sets.
///
/// For further details, please see:
/// https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#ga27865b1d26bac9ce91efaee83e94d4dd
pub fn estimateAffine2D(from: Point2fVector, to: Point2fVector) !Mat {
    const ptr = c.EstimateAffine2D(from.toC(), to.toC());
    return try Mat.initFromC(ptr);
}

pub const estimateAffine2DMethod = enum(i32) {
    ransac = 8,
    lmeds = 4,
};
/// EstimateAffine2DWithParams Computes an optimal affine transformation between two 2D point sets
/// with additional optional parameters.
///
/// For further details, please see:
/// https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#ga27865b1d26bac9ce91efaee83e94d4dd
pub fn estimateAffine2DWithParams(
    from: Point2fVector,
    to: Point2fVector,
    inliers: Mat,
    method: estimateAffine2DMethod,
    ransac_reproj_threshold: f64,
    max_iters: usize,
    confidence: f64,
    refine_iters: usize,
) !Mat {
    const ptr = c.EstimateAffine2DWithParams(
        from.toC(),
        to.toC(),
        inliers.toC(),
        @intFromEnum(method),
        ransac_reproj_threshold,
        max_iters,
        confidence,
        refine_iters,
    );
    return try Mat.initFromC(ptr);
}

const testing = std.testing;
const imgcodecs = @import("imgcodecs.zig");
const imgproc = @import("imgproc.zig");
const img_dir = "test/images/";
const cache_dir = "./.zig-cache/tmp/";
test "calib3d fisheye undistortImage" {
    var img = try imgcodecs.imRead(img_dir ++ "fisheye_sample.jpg", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var k = try Mat.initSize(3, 3, .cv64fc1);
    defer k.deinit();

    k.set(f64, 0, 0, 689.21);
    k.set(f64, 0, 1, 0);
    k.set(f64, 0, 2, 1295.56);

    k.set(f64, 1, 0, 0);
    k.set(f64, 1, 1, 690.48);
    k.set(f64, 1, 2, 942.17);

    k.set(f64, 2, 0, 0);
    k.set(f64, 2, 1, 0);
    k.set(f64, 2, 2, 1);

    var d = try Mat.initSize(1, 4, .cv64fc1);
    defer d.deinit();

    d.set(f64, 0, 0, 0);
    d.set(f64, 0, 1, 0);
    d.set(f64, 0, 2, 0);
    d.set(f64, 0, 3, 0);

    Fisheye.undistortImage(img, &dst, k, d);
    try testing.expectEqual(false, dst.isEmpty());
    try imgcodecs.imWrite(cache_dir ++ "fisheye_sample_undistort.jpg", dst);
}

test "calib3d fisheye undistortImageWithParams" {
    var img = try imgcodecs.imRead(img_dir ++ "fisheye_sample.jpg", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var k = try Mat.initSize(3, 3, .cv64fc1);
    defer k.deinit();

    k.set(f64, 0, 0, 689.21);
    k.set(f64, 0, 1, 0);
    k.set(f64, 0, 2, 1295.56);

    k.set(f64, 1, 0, 0);
    k.set(f64, 1, 1, 690.48);
    k.set(f64, 1, 2, 942.17);

    k.set(f64, 2, 0, 0);
    k.set(f64, 2, 1, 0);
    k.set(f64, 2, 2, 1);

    var d = try Mat.initSize(1, 4, .cv64fc1);
    defer d.deinit();

    d.set(f64, 0, 0, 0);
    d.set(f64, 0, 1, 0);
    d.set(f64, 0, 2, 0);
    d.set(f64, 0, 3, 0);

    var k_new = try Mat.initSize(3, 3, .cv64fc1);
    defer k_new.deinit();

    k.copyTo(&k_new);

    k_new.set(f64, 0, 0, 0.4 * k.get(f64, 0, 0));
    k_new.set(f64, 1, 1, 0.4 * k.get(f64, 1, 1));

    const size = core.Size.init(dst.rows(), dst.cols());

    Fisheye.undistortImageWithParams(img, &dst, k, d, k_new, size);
    try testing.expectEqual(false, dst.isEmpty());
    try imgcodecs.imWrite(cache_dir ++ "fisheye_sample_undistort_with_params.jpg", dst);
}

test "calib3d initUndistortRectifyMap getOptimalNewCameraMatrixWithParams" {
    var img = try imgcodecs.imRead(img_dir ++ "fisheye_sample.jpg", .unchanged);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var k = try Mat.initSize(3, 3, .cv64fc1);
    defer k.deinit();

    k.set(f64, 0, 0, 842.0261028);
    k.set(f64, 0, 1, 0);
    k.set(f64, 0, 2, 667.7569792);

    k.set(f64, 1, 0, 0);
    k.set(f64, 1, 1, 707.3668897);
    k.set(f64, 1, 2, 385.56476464);

    k.set(f64, 2, 0, 0);
    k.set(f64, 2, 1, 0);
    k.set(f64, 2, 2, 1);

    var d = try Mat.initSize(1, 5, .cv64fc1);
    defer d.deinit();

    d.set(f64, 0, 0, -3.65584802e-01);
    d.set(f64, 0, 1, 1.41555815e-01);
    d.set(f64, 0, 2, -2.62985819e-03);
    d.set(f64, 0, 3, 2.05841873e-04);
    d.set(f64, 0, 4, -2.35021914e-02);
    const res = try getOptimalNewCameraMatrixWithParams(
        k,
        d,
        Size.init(img.rows(), img.cols()),
        1,
        Size.init(img.rows(), img.cols()),
        false,
    );
    var new_c = res.mat;
    defer new_c.deinit();
    try testing.expectEqual(false, new_c.isEmpty());
    var r = try Mat.init();
    defer r.deinit();
    var mapx = try Mat.init();
    defer mapx.deinit();
    var mapy = try Mat.init();
    defer mapy.deinit();
    initUndistortRectifyMap(
        k,
        d,
        r,
        new_c,
        Size.init(img.rows(), img.cols()),
        .cv32fc1,
        mapx,
        mapy,
    );
    imgproc.remap(img, &dst, mapx, mapy, .{ .type = .linear }, .{ .type = .constant }, core.Color.init(0, 0, 0, 0));

    try imgcodecs.imWrite(cache_dir ++ "fisheye_sample_RectifyMap.jpg", dst);
}

test "calib3d undistort" {
    var img = try imgcodecs.imRead(img_dir ++ "distortion.jpg", .unchanged);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var k = try Mat.initSize(3, 3, .cv64fc1);
    defer k.deinit();

    k.set(f64, 0, 0, 689.21);
    k.set(f64, 0, 1, 0);
    k.set(f64, 0, 2, 1295.56);

    k.set(f64, 1, 0, 0);
    k.set(f64, 1, 1, 690.48);
    k.set(f64, 1, 2, 942.17);

    k.set(f64, 2, 0, 0);
    k.set(f64, 2, 1, 0);
    k.set(f64, 2, 2, 1);

    var d = try Mat.initSize(1, 4, .cv64fc1);
    defer d.deinit();

    d.set(f64, 0, 0, 0);
    d.set(f64, 0, 1, 0);
    d.set(f64, 0, 2, 0);
    d.set(f64, 0, 3, 0);

    var k_new = try Mat.init();
    defer k_new.deinit();

    k.copyTo(&k_new);

    k_new.set(f64, 0, 0, 0.5 * k.get(f64, 0, 0));
    k_new.set(f64, 1, 1, 0.5 * k.get(f64, 1, 1));

    undistort(img, &dst, k, d, k_new);
    try testing.expectEqual(false, dst.isEmpty());
    try imgcodecs.imWrite(cache_dir ++ "distortion_up.jpg", dst);
}

test "calib3d undistortPoint" {
    var k = try Mat.initSize(3, 3, .cv64fc1);
    defer k.deinit();

    k.set(f64, 0, 0, 1094.7249578198823);
    k.set(f64, 0, 1, 0);
    k.set(f64, 0, 2, 959.4907612030962);

    k.set(f64, 1, 0, 0);
    k.set(f64, 1, 1, 1094.9945708128778);
    k.set(f64, 1, 2, 536.4566143451868);

    k.set(f64, 2, 0, 0);
    k.set(f64, 2, 1, 0);
    k.set(f64, 2, 2, 1);

    var d = try Mat.initSize(1, 4, .cv64fc1);
    defer d.deinit();

    d.set(f64, 0, 0, -0.05207412392075069);
    d.set(f64, 0, 1, -0.089168300192224);
    d.set(f64, 0, 2, 0.10465607695792184);
    d.set(f64, 0, 3, -0.045693446831115585);

    var r = try Mat.init();
    defer r.deinit();

    var src = try Mat.initSize(3, 1, .cv64fc2);
    defer src.deinit();
    var dst = try Mat.initSize(3, 1, .cv64fc2);
    defer dst.deinit();

    // This camera matrix is 1920x1080. Points where x < 960 and y < 540 should move toward the top left (x and y get smaller)
    // The centre point should be mostly unchanged
    // Points where x > 960 and y > 540 should move toward the bottom right (x and y get bigger)

    // The index being used for col here is actually the channel (i.e. the point's x/y dimensions)
    // (since there's only 1 column so the formula: (colNumber * numChannels + channelNumber) reduces to
    // (0 * 2) + channelNumber
    // so col = 0 is the x coordinate and col = 1 is the y coordinate

    src.set(f64, 0, 0, 480);
    src.set(f64, 0, 1, 270);

    src.set(f64, 1, 0, 960);
    src.set(f64, 1, 1, 540);

    src.set(f64, 2, 0, 1440);
    src.set(f64, 2, 1, 810);

    var k_new = try Mat.init();
    defer k_new.deinit();

    k.copyTo(&k_new);

    k_new.set(f64, 0, 0, 0.4 * k.get(f64, 0, 0));
    k_new.set(f64, 1, 1, 0.4 * k.get(f64, 1, 1));

    const img_size = Size.init(1920, 1080);

    _ = try Fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, img_size, r, &k_new, 1, img_size, 1);

    _ = Fisheye.undistortPoints(src, &dst, k, d, r, k_new);

    try testing.expect(0 != dst.get(f64, 0, 0));
}

test "calib3d find and draw chessboard" {
    var img = try imgcodecs.imRead(img_dir ++ "chessboard_4x6.png", .unchanged);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var corners = try Mat.init();
    defer corners.deinit();

    const found = findChessboardCorners(img, Size.init(4, 6), &corners, .{});
    try testing.expectEqual(true, found);
    try testing.expectEqual(false, corners.isEmpty());

    var img2 = try Mat.initSize(150, 150, .cv8uc1);
    defer img2.deinit();

    drawChessboardCorners(&img2, Size.init(4, 6), corners, true);

    try testing.expectEqual(false, img2.isEmpty());
    try imgcodecs.imWrite(cache_dir ++ "chessboard_4x6_result.png", img2);
}

// test "calib3d find and draw chessboardSB" {
// var img = try imgcodecs.imRead(img_dir ++ "chessboard_4x6.png", .unchanged);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var corners = try Mat.init();
// defer corners.deinit();

// const found = findChessboardCornersSB(img, Size.init(4, 6), &corners, .{});
// try testing.expectEqual(true, found);
// try testing.expectEqual(false, corners.isEmpty());

// var img2 = try Mat.initSize(150, 150, .cv8uc1);
// defer img2.deinit();

// drawChessboardCorners(&img2, Size.init(4, 6), corners, true);

// try testing.expectEqual(false, img2.isEmpty());
// try imgcodecs.imWrite(cache_dir ++ "chessboard_4x6_sb_result.png", img2);
// }

// test "calib3d find and draw chessboardSBWithMeta" {
// var img = try imgcodecs.imRead(img_dir ++ "chessboard_4x6.png", .unchanged);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var corners = try Mat.init();
// defer corners.deinit();

// var meta = try Mat.init();
// defer meta.deinit();

// const found = findChessboardCornersSBWithMeta(
// img,
// Size.init(4, 6),
// &corners,
// .{},
// &meta,
// );
// try testing.expectEqual(true, found);
// try testing.expectEqual(false, corners.isEmpty());

// var img2 = try Mat.initSize(150, 150, .cv8uc1);
// defer img2.deinit();

// drawChessboardCorners(&img2, Size.init(4, 6), corners, true);

// try testing.expectEqual(false, img2.isEmpty());
// try imgcodecs.imWrite(cache_dir ++ "chessboard_4x6_sb_meta_result.png", img2);
// }

test "calib3d calibrateCamera" {
    var img = try imgcodecs.imRead(img_dir ++ "chessboard_4x6_distort.png", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var corners = try Mat.init();
    defer corners.deinit();

    const size = Size.init(4, 6);
    const found = findChessboardCorners(img, size, &corners, .{});
    try testing.expectEqual(true, found);
    try testing.expectEqual(false, corners.isEmpty());

    var img_points = try Point2fVector.initFromMat(corners);
    defer img_points.deinit();

    var obj_points = try Point3fVector.init();
    defer obj_points.deinit();

    {
        var j: f32 = 0;
        while (j < @as(f32, @floatFromInt(size.height))) : (j += 1) {
            var i: f32 = 0;
            while (i < @as(f32, @floatFromInt(size.width))) : (i += 1) {
                obj_points.append(Point3f.init(
                    100 * i,
                    100 * j,
                    0,
                ));
            }
        }
    }

    var camera_matrix = try Mat.init();
    defer camera_matrix.deinit();
    var dist_coeffs = try Mat.init();
    defer dist_coeffs.deinit();
    var rvecs = try Mat.init();
    defer rvecs.deinit();
    var tvecs = try Mat.init();
    defer tvecs.deinit();

    var obj_points_vec = try Points3fVector.init();
    defer obj_points_vec.deinit();
    obj_points_vec.append(obj_points);

    var img_points_vec = try Points2fVector.init();
    defer img_points_vec.deinit();
    img_points_vec.append(img_points);

    _ = calibrateCamera(
        obj_points_vec,
        img_points_vec,
        Size.init(img.cols(), img.rows()),
        &camera_matrix,
        &dist_coeffs,
        &rvecs,
        &tvecs,
        .{},
    );

    var dest = try Mat.init();
    defer dest.deinit();
    undistort(img, &dest, camera_matrix, dist_coeffs, camera_matrix);

    var target = try imgcodecs.imRead(img_dir ++ "chessboard_4x6_distort_correct.png", .gray_scale);
    defer target.deinit();
    try testing.expectEqual(false, target.isEmpty());

    var xor = try Mat.init();
    defer xor.deinit();

    Mat.bitwiseXor(dest, target, &xor);
    const different_pixels_number = xor.sum().val1;
    const max_different_pixels_number: f32 = @as(f32, @floatFromInt(img.cols())) * @as(f32, @floatFromInt(img.rows())) * 0.005;
    try testing.expect(max_different_pixels_number >= different_pixels_number);
}

test "calib3d estimateAffine2D" {
    const src = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(10, 5),
        Point2f.init(10, 10),
        Point2f.init(5, 10),
    };

    const dst = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(10, 0),
        Point2f.init(10, 10),
        Point2f.init(0, 10),
    };

    var pvsrc = try Point2fVector.initFromPoints(&src, testing.allocator);
    defer pvsrc.deinit();

    var pvdst = try Point2fVector.initFromPoints(&dst, testing.allocator);
    defer pvdst.deinit();

    var m = try estimateAffine2D(pvsrc, pvdst);
    defer m.deinit();

    try testing.expectEqual(@as(i32, 2), m.rows());
    try testing.expectEqual(@as(i32, 3), m.cols());
}

test "calib3d estimateAffine2DWithParams" {
    const src = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(10, 5),
        Point2f.init(10, 10),
        Point2f.init(5, 10),
    };

    const dst = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(10, 0),
        Point2f.init(10, 10),
        Point2f.init(0, 10),
    };

    var pvsrc = try Point2fVector.initFromPoints(&src, testing.allocator);
    defer pvsrc.deinit();

    var pvdst = try Point2fVector.initFromPoints(&dst, testing.allocator);
    defer pvdst.deinit();

    var inliers = try Mat.init();
    defer inliers.deinit();

    var m = try estimateAffine2DWithParams(
        pvsrc,
        pvdst,
        inliers,
        .ransac,
        3,
        2000,
        0.99,
        10,
    );
    defer m.deinit();

    try testing.expectEqual(@as(i32, 2), m.rows());
    try testing.expectEqual(@as(i32, 3), m.cols());
}

//*    implementation done
//*    pub extern fn Fisheye_UndistortImage(distorted: Mat, undistorted: Mat, k: Mat, d: Mat) void;
//*    pub extern fn Fisheye_UndistortImageWithParams(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, knew: Mat, size: Size) void;
//*    pub extern fn Fisheye_UndistortPoints(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, R: Mat, P: Mat) void;
//*    pub extern fn Fisheye_EstimateNewCameraMatrixForUndistortRectify(k: Mat, d: Mat, imgSize: Size, r: Mat, p: Mat, balance: f64, newSize: Size, fovScale: f64) void;
//*    pub extern fn InitUndistortRectifyMap(cameraMatrix: Mat, distCoeffs: Mat, r: Mat, newCameraMatrix: Mat, size: Size, m1type: c_int, map1: Mat, map2: Mat) void;
//*    pub extern fn GetOptimalNewCameraMatrixWithParams(cameraMatrix: Mat, distCoeffs: Mat, size: Size, alpha: f64, newImgSize: Size, validPixROI: [*c]Rect, centerPrincipalPoint: bool) Mat;
//*    pub extern fn CalibrateCamera(objectPoints: Points3fVector, imagePoints: Points2fVector, imageSize: Size, cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, flag: c_int) f64;
//*    pub extern fn Undistort(src: Mat, dst: Mat, cameraMatrix: Mat, distCoeffs: Mat, newCameraMatrix: Mat) void;
//*    pub extern fn UndistortPoints(distorted: Mat, undistorted: Mat, k: Mat, d: Mat, r: Mat, p: Mat) void;
//*    pub extern fn FindChessboardCorners(image: Mat, patternSize: Size, corners: Mat, flags: c_int) bool;
//*    pub extern fn FindChessboardCornersSB(image: Mat, patternSize: Size, corners: Mat, flags: c_int) bool;
//*    pub extern fn FindChessboardCornersSBWithMeta(image: Mat, patternSize: Size, corners: Mat, flags: c_int, meta: Mat) bool;
//*    pub extern fn DrawChessboardCorners(image: Mat, patternSize: Size, corners: Mat, patternWasFound: bool) void;
//*    pub extern fn EstimateAffinePartial2D(from: Point2fVector, to: Point2fVector) Mat;
//*    pub extern fn EstimateAffine2D(from: Point2fVector, to: Point2fVector) Mat;
//*    pub extern fn EstimateAffine2DWithParams(from: Point2fVector, to: Point2fVector, inliers: Mat, method: c_int, ransacReprojThreshold: f64, maxIters: usize, confidence: f64, refineIters: usize) Mat;
