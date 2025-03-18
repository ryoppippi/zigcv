const std = @import("std");
const testing = std.testing;
const allocator = std.testing.allocator;
const core = @import("../core.zig");
const Mat = core.Mat;
const Scalar = core.Scalar;

test "core mat" {
    var mat = try Mat.init();
    defer mat.deinit();
    try testing.expectEqual(true, mat.isEmpty());
}

test "core mat size" {
    const size: i32 = 10;
    const mat_type = Mat.MatType.cv8sc1;
    var mat = try Mat.initSize(size, size, mat_type);
    defer mat.deinit();

    const mat_size = mat.size();
    try testing.expectEqual(size, mat_size[0]);
    try testing.expectEqual(size, mat_size[1]);
    try testing.expectEqual(@as(usize, 2), mat_size.len);

    try testing.expectEqual(size, mat.rows());
    try testing.expectEqual(size, mat.cols());
    try testing.expectEqual(mat_type, mat.getType());
}

test "core mat sizes" {
    var sizes = [3]i32{ 10, 20, 30 };
    const mat_type = Mat.MatType.cv8sc1;
    var mat = try Mat.initSizes(&sizes, mat_type);
    defer mat.deinit();

    const mat_size = mat.size();

    for (mat_size, 0..) |size, i| {
        try testing.expectEqual(sizes[i], @as(i32, @intCast(size)));
    }

    try testing.expectEqual(@as(i32, 10 * 20 * 30), mat.total());
    try testing.expectEqual(@as(i32, -1), mat.rows());
    try testing.expectEqual(@as(i32, -1), mat.cols());
    try testing.expectEqual(mat_type, mat.getType());
}

test "core mat channnel" {
    var mat = try Mat.initSize(1, 1, .cv8uc1);
    defer mat.deinit();

    try testing.expectEqual(@as(i32, 1), mat.channels());
}

test "core mat type" {
    var mat = try Mat.initSize(1, 1, .cv8uc1);
    defer mat.deinit();

    try testing.expectEqual(Mat.MatType.cv8uc1, mat.getType());

    var mat2 = try Mat.initSize(1, 1, .cv16sc2);
    defer mat2.deinit();

    try testing.expectEqual(Mat.MatType.cv16sc2, mat2.getType());
}

test "core mat eye" {
    var mat = try Mat.initEye(3, 3, .cv8sc1);
    defer mat.deinit();
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                if (i == j) {
                    try testing.expectEqual(@as(u8, 1), mat.get(u8, i, j));
                } else {
                    try testing.expectEqual(@as(u8, 0), mat.get(u8, i, j));
                }
            }
        }
    }
}

test "core mat zeros" {
    var mat = try Mat.initZeros(3, 3, .cv8sc1);
    defer mat.deinit();
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                try testing.expectEqual(@as(u8, 0), mat.get(u8, i, j));
            }
        }
    }
}

test "core mat ones" {
    var mat = try Mat.initOnes(3, 3, .cv8sc1);
    defer mat.deinit();
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            var j: usize = 0;
            while (j < 3) : (j += 1) {
                try testing.expectEqual(@as(u8, 1), mat.get(u8, i, j));
            }
        }
    }
}

test "core mat initFromMat" {
    var mat = try Mat.initSize(101, 102, .cv8sc1);
    defer mat.deinit();

    var pmat = try mat.initFromMat(11, 12, .cv8uc1, 10, 10);
    defer pmat.deinit();

    try testing.expectEqual(@as(i32, 11), pmat.rows());
    try testing.expectEqual(@as(i32, 12), pmat.cols());
}

// test "core mat initSizeFromScalar" {
// const s = Scalar.init(255, 105, 180, 0);
// var mat = try Mat.initSizeFromScalar(s, 2, 3, .cv8uc3);
// defer mat.deinit();
// try testing.expectEqual(false, mat.isEmpty());
// try testing.expectEqual(@as(i32, 2), mat.rows());
// try testing.expectEqual(@as(i32, 3), mat.cols());
// try testing.expectEqual(@as(i32, 3), mat.channels());
// try testing.expectEqual(Mat.MatType.cv8uc3, mat.getType());
// try testing.expectEqual(@as(i32, 2 * 3), mat.total());
// try testing.expectEqual(@as(i32, 3 * 3), mat.step());
// }

// test "core mat copyTo" {
// var mat = try Mat.initOnes(100, 102, .cv8sc1);
// defer mat.deinit();
// var mat2 = try Mat.init();
// defer mat2.deinit();
// mat.copyTo(&mat2);

// try testing.expectEqual(mat.rows(), mat2.rows());
// try testing.expectEqual(mat.cols(), mat2.cols());
// try testing.expectEqual(mat.channels(), mat2.channels());
// try testing.expectEqual(mat.getType(), mat2.getType());
// {
// var i: usize = 0;
// while (i < mat.rows()) : (i += 1) {
// var j: usize = 0;
// while (j < mat.cols()) : (j += 1) {
// try testing.expectEqual(mat.get(u8, i, j), mat2.get(u8, i, j));
// }
// }
// }
// }

// test "core mat copyToWithMask" {
// var mat = try Mat.initSize(101, 102, .cv8uc1);
// defer mat.deinit();
// var diff = try Mat.init();
// defer diff.deinit();
// var mask = try Mat.initSize(101, 102, .cv8uc1);
// defer mask.deinit();

// mat.set(u8, 0, 0, 255);
// mat.set(u8, 0, 1, 255);

// mask.set(u8, 0, 0, 255);

// var copy = try Mat.init();
// defer copy.deinit();

// mat.copyToWithMask(&copy, mask);

// try testing.expectEqual(mat.rows(), copy.rows());
// try testing.expectEqual(mat.cols(), copy.cols());

// try testing.expectEqual(@as(u8, 255), copy.get(u8, 0, 0));
// try testing.expectEqual(@as(u8, 0), copy.get(u8, 0, 1));
// }

// test "core mat clone" {
// var mat = try Mat.initOnes(100, 102, .cv8sc1);
// defer mat.deinit();

// mat.set(i8, 0, 0, 3);

// var clone = try mat.clone();
// defer clone.deinit();

// try testing.expectEqual(mat.rows(), clone.rows());
// try testing.expectEqual(mat.cols(), clone.cols());
// try testing.expectEqual(mat.channels(), clone.channels());
// try testing.expectEqual(mat.getType(), clone.getType());

// {
// var i: usize = 0;
// while (i < mat.rows()) : (i += 1) {
// var j: usize = 0;
// while (j < mat.cols()) : (j += 1) {
// try testing.expectEqual(mat.get(u8, i, j), clone.get(u8, i, j));
// }
// }
// }
// }
