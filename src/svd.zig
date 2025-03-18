const c = @import("c_api.zig");
const core = @import("core.zig");
const Mat = core.Mat;

/// SVDCompute decomposes matrix and stores the results to user-provided matrices
///
/// https://docs.opencv.org/4.1.2/df/df7/classcv_1_1SVD.html#a76f0b2044df458160292045a3d3714c6
pub fn svdCompute(src: Mat, w: *Mat, u: *Mat, vt: *Mat) void {
    c.SVD_Compute(src.toC(), w.*.toC(), u.*.toC(), vt.*.toC());
}

// test "svd" {
// const std = @import("std");
// const testing = std.testing;

// const result_w = [2]f32{ 6.167493, 3.8214223 };
// const result_u = [4]f32{ -0.1346676, -0.99089086, 0.9908908, -0.1346676 };
// const result_vt = [4]f32{ 0.01964448, 0.999807, -0.999807, 0.01964448 };

// var src = try Mat.initSize(2, 2, .cv32fc1);
// src.set(f32, 0, 0, 3.76956568);
// src.set(f32, 0, 1, -0.90478725);
// src.set(f32, 1, 0, 0.634576);
// src.set(f32, 1, 1, 6.10002347);

// var w = try Mat.init();
// defer w.deinit();

// var u = try Mat.init();
// defer u.deinit();

// var vt = try Mat.init();
// defer vt.deinit();

// svdCompute(src, &w, &u, &vt);

// const data_w = try w.dataPtr(f32);
// const data_u = try u.dataPtr(f32);
// const data_vt = try vt.dataPtr(f32);

// try testing.expectEqualSlices(f32, result_w[0..], data_w[0..]);
// try testing.expectEqualSlices(f32, result_u[0..], data_u[0..]);
// try testing.expectEqualSlices(f32, result_vt[0..], data_vt[0..]);
// }

//*    implementation done
//*    pub extern fn SVD_Compute(src: Mat, w: Mat, u: Mat, vt: Mat) void;
