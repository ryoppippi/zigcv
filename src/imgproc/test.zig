const std = @import("std");
const testing = std.testing;
const core = @import("../core.zig");
const imgcodecs = @import("../imgcodecs.zig");
const imgproc = @import("../imgproc.zig");
const Mat = core.Mat;
const Size = core.Size;
const Color = core.Color;
const Scalar = core.Scalar;
const Point = core.Point;
const Point2f = core.Point2f;
const Rect = core.Rect;
const PointVector = core.PointVector;
const PointsVector = core.PointsVector;
const Point2fVector = core.Point2fVector;

const img_dir = "test/images/";
const face_detect_filepath = img_dir ++ "face-detect.jpg";

test "imgproc CLAHE" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();

    var src = try Mat.init();
    defer src.deinit();

    img.convertTo(&src, .cv8uc1);

    var dst = try Mat.init();
    defer dst.deinit();

    var c = try imgproc.CLAHE.init();
    defer c.deinit();

    c.apply(src, &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc CLAHE WithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();

    var src = try Mat.init();
    defer src.deinit();

    img.convertTo(&src, .cv8uc1);

    var dst = try Mat.init();
    defer dst.deinit();

    var c = try imgproc.CLAHE.initWithParams(2, Size.init(10, 10));
    defer c.deinit();

    c.apply(src, &dst);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc approxPolyDP" {
    var img = try Mat.initSize(100, 200, .cv8uc1);
    defer img.deinit();

    const white = Color.init(255, 255, 255, 255);

    // Draw a triangle
    imgproc.line(&img, Point.init(25, 25), Point.init(25, 75), white, 1);
    imgproc.line(&img, Point.init(25, 75), Point.init(75, 50), white, 1);
    imgproc.line(&img, Point.init(75, 50), Point.init(25, 25), white, 1);
    // Draw rectangle
    imgproc.rectangle(&img, Rect.init(125, 25, 50, 50), white, 1);

    var contours = try imgproc.findContours(img, .external, .simple);
    defer contours.deinit();

    const c0 = try contours.at(0);
    const triangle_perimeter = imgproc.arcLength(c0, true);
    var trianle_contour = try imgproc.approxPolyDP(c0, 0.04 * triangle_perimeter, true);
    defer trianle_contour.deinit();

    const expected_triangle_contour = [_]Point{
        Point.init(25, 25),
        Point.init(25, 75),
        Point.init(75, 50),
    };

    var actual_triangle_contour = try trianle_contour.toPoints(testing.allocator);
    defer actual_triangle_contour.deinit();

    try testing.expectEqual(expected_triangle_contour.len, actual_triangle_contour.items.len);
    for (expected_triangle_contour, 0..) |expected_point, i| {
        try testing.expectEqual(expected_point, actual_triangle_contour.items[i]);
    }

    const c1 = try contours.at(1);
    const rectangle_perimeter = imgproc.arcLength(c1, true);
    var rectangle_contour = try imgproc.approxPolyDP(c1, 0.04 * rectangle_perimeter, true);
    defer rectangle_contour.deinit();

    const expected_rectangle_contour = [_]Point{
        Point.init(125, 24),
        Point.init(124, 75),
        Point.init(175, 76),
        Point.init(176, 25),
    };

    var actual_rectangle_contour = try rectangle_contour.toPoints(testing.allocator);
    defer actual_rectangle_contour.deinit();

    try testing.expectEqual(expected_rectangle_contour.len, actual_rectangle_contour.items.len);
    for (expected_rectangle_contour, 0..) |expected_point, i| {
        try testing.expectEqual(expected_point, actual_rectangle_contour.items[i]);
    }
}

test "imgproc convexity" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var res = try imgproc.findContours(img, .external, .simple);
    defer res.deinit();
    try testing.expect(1 <= res.size());

    const area = imgproc.contourArea(try res.at(0));
    try testing.expectEqual(@as(f64, 127280), area);

    var hull = try Mat.init();
    defer hull.deinit();

    imgproc.convexHull(try res.at(0), &hull, true, false);
    try testing.expectEqual(false, hull.isEmpty());

    var defects = try Mat.init();
    defer defects.deinit();

    imgproc.convexityDefects(try res.at(0), hull, &defects);
    try testing.expectEqual(false, defects.isEmpty());
}

test "imgproc min enclosing circle" {
    const epsilon = 0.001;
    const expected_radius = 2.0;
    const expected_x = 0.0;
    const expected_y = 0.0;
    const pts = [_]Point{
        Point.init(0, 2),
        Point.init(2, 0),
        Point.init(0, -2),
        Point.init(-2, 0),
        Point.init(1, -1),
    };

    var pv = try core.PointVector.initFromPoints(pts[0..], testing.allocator);
    defer pv.deinit();

    const res = imgproc.minEnclosingCircle(pv);
    const radius = res.radius;
    const point = res.point;

    try testing.expect(@abs(@as(f64, radius) - expected_radius) <= epsilon);
    try testing.expect(@abs(@as(f64, point.x) - expected_x) <= epsilon);
    try testing.expect(@abs(@as(f64, point.y) - expected_y) <= epsilon);
}

test "imgproc cvtColor" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.cvtColor(img, &dst, .bgra_to_gray);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc bilateral filter" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.bilateralFilter(img, &dst, 1, 2.0, 3.0);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc blur" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.blur(img, &dst, Size.init(3, 3));

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc sobel" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.sobel(img, &dst, .cv16sc1, 0, 1, 3, 1, 0, .{});

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc spatialGradient" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dx = try Mat.init();
    defer dx.deinit();

    var dy = try Mat.init();
    defer dy.deinit();

    imgproc.spatialGradient(img, &dx, &dy, .cv16sc1, .{});

    try testing.expectEqual(false, dx.isEmpty());
    try testing.expectEqual(img.rows(), dx.rows());
    try testing.expectEqual(img.cols(), dx.cols());
    try testing.expectEqual(false, dy.isEmpty());
    try testing.expectEqual(img.rows(), dy.rows());
    try testing.expectEqual(img.cols(), dy.cols());
}

test "imgproc boxfilter" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.boxFilter(img, &dst, -1, Size.init(3, 3));

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc sqBoxfilter" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.sqBoxFilter(img, &dst, -1, Size.init(3, 3));

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc dilate" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.dilate(img, &dst, kernel);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc dilateWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.dilateWithParams(img, &dst, kernel, Point.init(0, 0), .{ .type = .wrap }, .{}, Color{});

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc distanceTransform" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var gray = try Mat.init();
    defer gray.deinit();
    imgproc.cvtColor(img, &gray, .bgra_to_gray);

    var thresh = try Mat.init();
    defer thresh.deinit();
    _ = imgproc.threshold(gray, &thresh, 25, 255, .{ .type = .binary });

    var dst = try Mat.init();
    defer dst.deinit();

    var labels = try Mat.init();
    defer labels.deinit();

    imgproc.distanceTransform(thresh, &dst, &labels, .l2, .mask_3, .c_comp);

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc matchTemplate" {
    var img_scene = try imgcodecs.imRead(img_dir ++ "face.jpg", .gray_scale);
    defer img_scene.deinit();
    try testing.expectEqual(false, img_scene.isEmpty());

    var img_template = try imgcodecs.imRead(img_dir ++ "toy.jpg", .gray_scale);
    defer img_template.deinit();
    try testing.expectEqual(false, img_template.isEmpty());

    var result = try Mat.init();
    defer result.deinit();

    var m = try Mat.init();
    imgproc.matchTemplate(img_scene, img_template, &result, .ccoeff_normed, m);
    m.deinit();

    const res = Mat.minMaxLoc(result);
    const max_confidence = res.max_val;
    try testing.expect(max_confidence >= 0.95);
}

test "imgproc moments" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    const result = imgproc.moments(img, true);
    _ = result;
}

test "imgproc pyrdown" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.pyrDown(img, &dst, Size.init(dst.cols(), dst.rows()), .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expect(@abs(@as(f64, @floatFromInt((img.cols() - 2 * dst.cols())))) < 2.0);
    try testing.expect(@abs(@as(f64, @floatFromInt((img.rows() - 2 * dst.rows())))) < 2.0);
}

test "imgproc pyrup" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.pyrUp(img, &dst, Size.init(dst.cols(), dst.rows()), .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expect(@abs(@as(f64, @floatFromInt((2 * img.cols() - dst.cols())))) < 2.0);
    try testing.expect(@abs(@as(f64, @floatFromInt((2 * img.rows() - dst.rows())))) < 2.0);
}

test "imgproc boxPoints" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();

    var thresh = try Mat.init();
    defer thresh.deinit();

    _ = imgproc.threshold(img, &thresh, 25, 255, .{ .type = .binary });

    var contours = try imgproc.findContours(thresh, .external, .simple);
    defer contours.deinit();

    var contour = try contours.at(0);

    var hull = try Mat.init();
    defer hull.deinit();

    imgproc.convexHull(try contours.at(0), &hull, false, false);

    var hull_points = std.ArrayList(Point).init(testing.allocator);
    defer hull_points.deinit();

    {
        var i: usize = 0;
        while (i < hull.cols()) : (i += 1) {
            var j: usize = 0;
            while (j < hull.rows()) : (j += 1) {
                const p = hull.get(i32, j, i);
                try hull_points.append(contour.at(p));
            }
        }
    }

    var pvhp = try PointVector.initFromPoints(hull_points.items, testing.allocator);
    defer pvhp.deinit();

    const rect = imgproc.minAreaRect(pvhp);
    var pts = try Mat.init();
    defer pts.deinit();

    imgproc.boxPoints(rect, &pts);

    try testing.expectEqual(false, pts.isEmpty());
    try testing.expectEqual(@as(i32, 4), pts.rows());
    try testing.expectEqual(@as(i32, 2), pts.cols());
}

test "imgproc areaRect" {
    var src = [_]Point{
        Point.init(0, 2),
        Point.init(2, 0),
        Point.init(4, 2),
        Point.init(2, 4),
    };

    var pv = try PointVector.initFromPoints(src[0..], testing.allocator);
    defer pv.deinit();

    const m = imgproc.minAreaRect(pv);

    try testing.expectEqual(@as(i32, 2.0), m.center.x);
    try testing.expectEqual(@as(i32, 2.0), m.center.y);
    try testing.expectEqual(@as(f64, 45.0), m.angle);
}

test "imgproc fitEllipse" {
    var src = [_]Point{
        Point.init(1, 1),
        Point.init(0, 1),
        Point.init(0, 2),
        Point.init(1, 3),
        Point.init(2, 3),
        Point.init(4, 2),
        Point.init(4, 1),
        Point.init(0, 3),
        Point.init(0, 2),
    };
    var pv = try PointVector.initFromPoints(src[0..], testing.allocator);
    defer pv.deinit();

    const rect = imgproc.fitEllipse(pv);

    try testing.expectEqual(@as(i32, 2.0), rect.center.x);
    try testing.expectEqual(@as(i32, 2.0), rect.center.y);
    try testing.expectEqual(@as(f64, 78.60807800292969), rect.angle);
}

test "imgproc findContours" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var res = try imgproc.findContours(img, .external, .simple);
    defer res.deinit();

    try testing.expect(res.size() > 0);

    const r0 = try res.at(0);
    const area = imgproc.contourArea(r0);
    try testing.expectEqual(@as(f64, 127280.0), area);

    const r = imgproc.boundingRect(r0);
    try testing.expectEqual(@as(i32, 0), r.x);
    try testing.expectEqual(@as(i32, 0), r.y);
    try testing.expectEqual(@as(i32, 400), r.width);
    try testing.expectEqual(@as(i32, 320), r.height);

    var length = imgproc.arcLength(r0, true);
    try testing.expectEqual(@as(i32, 1436), @as(i32, @intFromFloat(length)));

    length = imgproc.arcLength(r0, false);
    try testing.expectEqual(@as(i32, 1037), @as(i32, @intFromFloat(length)));
}

test "imgproc findContoursWithParams" {
    var img = try imgcodecs.imRead(img_dir ++ "contours.png", .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var hierarchy = try Mat.init();
    defer hierarchy.deinit();

    var res = try imgproc.findContoursWithParams(img, &hierarchy, .tree, .none);
    defer res.deinit();

    try testing.expectEqual(@as(i32, 4), res.size());
    try testing.expectEqual(hierarchy.cols(), res.size());
}

test "imgproc polygonTest" {
    const tests = [_]struct {
        name: []const u8, // name of the testcase
        thickness: i32, // thickness of the polygon
        point: Point, // point to be checked
        result: f64, // expected result; either distance or -1, 0, 1 based on measure parameter
        measure: bool, // enable distance measurement, if true
    }{
        .{ .name = "Outside the polygon - measure=false", .thickness = 1, .point = Point.init(5, 15), .result = -1.0, .measure = false },
        .{ .name = "On the polygon - measure=false", .thickness = 1, .point = Point.init(10, 10), .result = 0.0, .measure = false },
        .{ .name = "Inside the polygon - measure=true", .thickness = 1, .point = Point.init(20, 30), .result = 10.0, .measure = true },
        .{ .name = "Outside the polygon - measure=true", .thickness = 1, .point = Point.init(5, 15), .result = -5.0, .measure = true },
        .{ .name = "On the polygon - measure=true", .thickness = 1, .point = Point.init(10, 10), .result = 0.0, .measure = true },
    };

    var pts = [_]Point{
        Point.init(10, 10),
        Point.init(10, 80),
        Point.init(80, 80),
        Point.init(80, 10),
    };

    var ctr = try PointVector.initFromPoints(pts[0..], testing.allocator);
    defer ctr.deinit();

    for (tests) |t| {
        const result = imgproc.pointPolygonTest(ctr, t.point, t.measure);
        testing.expectEqual(t.result, result) catch |e| {
            std.debug.print("Testcase: {s}\n", .{t.name});
            return e;
        };
    }
}

test "imgproc connectedComponents" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    const res = imgproc.connectedComponents(img, &labels);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
}

test "imgproc connectedComponentsWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    const res = imgproc.connectedComponentsWithParams(img, &labels, 8, .cv32sc1, .default);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
}

test "imgproc connectedComponentsWithStats" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    var stats = try Mat.init();
    defer stats.deinit();

    var centroids = try Mat.init();
    defer centroids.deinit();

    const res = imgproc.connectedComponentsWithStats(img, &labels, &stats, &centroids);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
    try testing.expectEqual(false, stats.isEmpty());
    try testing.expectEqual(false, centroids.isEmpty());
}

test "imgproc connectedComponentsWithStatsWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var labels = try Mat.init();
    defer labels.deinit();

    var stats = try Mat.init();
    defer stats.deinit();

    var centroids = try Mat.init();
    defer centroids.deinit();

    const res = imgproc.connectedComponentsWithStatsWithParams(img, &labels, &stats, &centroids, 8, .cv32sc1, .default);
    try testing.expect(res > 0);
    try testing.expectEqual(false, labels.isEmpty());
    try testing.expectEqual(false, stats.isEmpty());
    try testing.expectEqual(false, centroids.isEmpty());
}

test "imgproc erode" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.erode(img, &dest, kernel);
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc erodeWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.erodeWithParams(img, &dest, kernel, Point.init(-1, -1), 3, 0);
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc morphologyDefaultBorderValue" {
    var morphologyDefaultBorderValue = imgproc.morphologyDefaultBorderValue();
    for (morphologyDefaultBorderValue.toArray()) |s| {
        try testing.expect(s != 0);
    }
}

test "imgproc morphologyEx" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.morphologyEx(img, &dest, .open, kernel);
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc morphologyExWithParams" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dest = try Mat.init();
    defer dest.deinit();

    var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernel.deinit();

    imgproc.morphologyExWithParams(img, &dest, .open, kernel, 2, .{ .type = .constant });
    try testing.expectEqual(false, dest.isEmpty());
    try testing.expectEqual(img.rows(), dest.rows());
    try testing.expectEqual(img.cols(), dest.cols());
}

test "imgproc gaussianBlur" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.gaussianBlur(img, &dst, Size.init(23, 23), 30, 50, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc getGaussianKernel" {
    var kernel = try imgproc.getGaussianKernel(1, 0.5);
    defer kernel.deinit();
    try testing.expectEqual(false, kernel.isEmpty());
}

test "imgproc getGaussianKernelWithParams" {
    var kernel = try imgproc.getGaussianKernelWithParams(1, 0.5, .cv64fc1);
    defer kernel.deinit();
    try testing.expectEqual(false, kernel.isEmpty());
}

test "imgproc laplacian" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.laplacian(img, &dst, .cv16sc1, 1, 1, 0, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc scharr" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.scharr(img, &dst, .cv16sc1, 1, 0, 0, 0, .{});
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc medianBlur" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.medianBlur(img, &dst, 3);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc canny" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.canny(img, &dst, 50, 150);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc goodFeaturesToTrack & CornerSubPix" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();

    var corners = try Mat.init();
    defer corners.deinit();

    imgproc.goodFeaturesToTrack(img, &corners, 500, 0.01, 10);
    try testing.expectEqual(false, corners.isEmpty());
    try testing.expectEqual(@as(i32, 205), corners.rows());
    try testing.expectEqual(@as(i32, 1), corners.cols());

    var tc = try core.TermCriteria.init(.{ .count = true, .eps = true }, 20, 0.03);
    defer tc.deinit();

    imgproc.cornerSubPix(
        img,
        &corners,
        Size.init(10, 10),
        Size.init(-1, -1),
        tc,
    );
    try testing.expectEqual(false, corners.isEmpty());
    try testing.expectEqual(@as(i32, 205), corners.rows());
    try testing.expectEqual(@as(i32, 1), corners.cols());
}

test "imgproc grabcut" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var src = try Mat.init();
    defer src.deinit();
    imgproc.cvtColor(img, &img, .rgba_to_bgr);
    img.convertTo(&src, .cv8uc3);

    var mask = try Mat.initSize(img.rows(), img.cols(), .cv8uc1);
    defer mask.deinit();

    var bgd_model = try Mat.init();
    defer bgd_model.deinit();
    var fgd_model = try Mat.init();
    defer fgd_model.deinit();

    const r = Rect.init(0, 0, 50, 50);

    imgproc.grabCut(src, &mask, r, &bgd_model, &fgd_model, 1, .eval);
    try testing.expectEqual(false, bgd_model.isEmpty());
    try testing.expectEqual(false, fgd_model.isEmpty());
}

test "imgproc houghCircles" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var circles = try Mat.init();
    defer circles.deinit();

    imgproc.houghCircles(img, &circles, .gradient, 5.0, 5.0);
    try testing.expectEqual(false, circles.isEmpty());
    try testing.expectEqual(@as(i32, 1), circles.rows());
    try testing.expectEqual(@as(i32, 317), circles.cols());
}

// test "imgproc HoughCirclesWithParams" {
// var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var circles = try Mat.init();
// defer circles.deinit();

// imgproc.houghCirclesWithParams(img, &circles, .gradient, 5.0, 5.0, 100, 100, 0, 0);

// try testing.expectEqual(false, circles.isEmpty());
// try testing.expectEqual(@as(i32, 1), circles.rows());
// try testing.expect(circles.cols() >= 317);
// }

// test "imgproc houghLines" {
// var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var dst = try Mat.init();
// defer dst.deinit();

// imgproc.houghLines(img, &dst, 1, std.math.pi / 180.0, 50);

// try testing.expectEqual(false, dst.isEmpty());
// try testing.expectEqual(@as(i32, 6465), dst.rows());
// try testing.expectEqual(@as(i32, 1), dst.cols());

// try testing.expectEqual(@as(f32, 226), dst.get(f32, 0, 0));
// try testing.expectEqual(@as(f32, 0.7853982), dst.get(f32, 0, 1));

// try testing.expectEqual(@as(f32, 228), dst.get(f32, 1, 0));
// try testing.expectEqual(@as(f32, 0.7853982), dst.get(f32, 1, 1));

// try testing.expectEqual(@as(f32, 23), dst.get(f32, 6463, 0));
// try testing.expectEqual(@as(f32, 0.75049156), dst.get(f32, 6463, 1));

// try testing.expectEqual(@as(f32, 23), dst.get(f32, 6464, 0));
// try testing.expectEqual(@as(f32, 0.82030475), dst.get(f32, 6464, 1));
// }

// test "imgproc houghLinesP" {
// var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var dst = try Mat.init();
// defer dst.deinit();

// imgproc.houghLinesP(img, &dst, 1, std.math.pi / 180.0, 50);

// try testing.expectEqual(false, dst.isEmpty());
// try testing.expectEqual(@as(i32, 4356), dst.rows());
// try testing.expectEqual(@as(i32, 1), dst.cols());

// try testing.expectEqual(@as(i32, 46), dst.get(i32, 0, 0));
// try testing.expectEqual(@as(i32, 0), dst.get(i32, 0, 1));
// try testing.expectEqual(@as(i32, 365), dst.get(i32, 0, 2));
// try testing.expectEqual(@as(i32, 319), dst.get(i32, 0, 3));

// try testing.expectEqual(@as(i32, 86), dst.get(i32, 1, 0));
// try testing.expectEqual(@as(i32, 319), dst.get(i32, 1, 1));
// try testing.expectEqual(@as(i32, 399), dst.get(i32, 1, 2));
// try testing.expectEqual(@as(i32, 6), dst.get(i32, 1, 3));

// try testing.expectEqual(@as(i32, 96), dst.get(i32, 433, 0));
// try testing.expectEqual(@as(i32, 319), dst.get(i32, 433, 1));
// try testing.expectEqual(@as(i32, 108), dst.get(i32, 433, 2));
// try testing.expectEqual(@as(i32, 316), dst.get(i32, 433, 3));

// try testing.expectEqual(@as(i32, 39), dst.get(i32, 434, 0));
// try testing.expectEqual(@as(i32, 280), dst.get(i32, 434, 1));
// try testing.expectEqual(@as(i32, 89), dst.get(i32, 434, 2));
// try testing.expectEqual(@as(i32, 227), dst.get(i32, 434, 3));
// }

// test "imgproc houghLinesPWithParams" {
// var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var dst = try Mat.init();
// defer dst.deinit();

// imgproc.houghLinesPWithParams(img, &dst, 1, std.math.pi / 180.0, 50, 1, 1);

// try testing.expectEqual(false, dst.isEmpty());
// try testing.expectEqual(@as(i32, 514), dst.rows());
// try testing.expectEqual(@as(i32, 1), dst.cols());

// try testing.expectEqual(@as(i32, 46), dst.get(i32, 0, 0));
// try testing.expectEqual(@as(i32, 0), dst.get(i32, 0, 1));
// try testing.expectEqual(@as(i32, 365), dst.get(i32, 0, 2));
// try testing.expectEqual(@as(i32, 319), dst.get(i32, 0, 3));

// try testing.expectEqual(@as(i32, 86), dst.get(i32, 1, 0));
// try testing.expectEqual(@as(i32, 319), dst.get(i32, 1, 1));
// try testing.expectEqual(@as(i32, 399), dst.get(i32, 1, 2));
// try testing.expectEqual(@as(i32, 6), dst.get(i32, 1, 3));

// try testing.expectEqual(@as(i32, 0), dst.get(i32, 433, 0));
// try testing.expectEqual(@as(i32, 126), dst.get(i32, 433, 1));
// try testing.expectEqual(@as(i32, 71), dst.get(i32, 433, 2));
// try testing.expectEqual(@as(i32, 57), dst.get(i32, 433, 3));

// try testing.expectEqual(@as(i32, 309), dst.get(i32, 434, 0));
// try testing.expectEqual(@as(i32, 319), dst.get(i32, 434, 1));
// try testing.expectEqual(@as(i32, 399), dst.get(i32, 434, 2));
// try testing.expectEqual(@as(i32, 229), dst.get(i32, 434, 3));
// }

test "imgproc houghLinesPointSet" {
    const points = [_][2]f32{
        .{ 0, 369 },   .{ 10, 364 },  .{ 20, 358 },  .{ 30, 352 },
        .{ 40, 346 },  .{ 50, 341 },  .{ 60, 335 },  .{ 70, 329 },
        .{ 80, 323 },  .{ 90, 318 },  .{ 100, 312 }, .{ 110, 306 },
        .{ 120, 300 }, .{ 130, 295 }, .{ 140, 289 }, .{ 150, 284 },
        .{ 160, 277 }, .{ 170, 271 }, .{ 180, 266 }, .{ 190, 260 },
    };

    var img = try Mat.initSize(points.len, 1, .cv32fc2);
    defer img.deinit();

    for (points, 0..) |p, i| {
        img.set(f32, i, 0, p[0]);
        img.set(f32, i, 1, p[1]);
    }

    var dst = try Mat.init();
    defer dst.deinit();

    const rho_min = 0.0;
    const rho_max = 360.0;
    const rho_step = 1.0;
    const theta_min = 0.0;
    const theta_max = std.math.pi / 2.0;
    const theta_step = std.math.pi / 180.0;

    imgproc.houghLinesPointSet(
        img,
        &dst,
        20,
        1,
        rho_min,
        rho_max,
        rho_step,
        theta_min,
        theta_max,
        theta_step,
    );

    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 20), dst.rows());
    try testing.expectEqual(@as(i32, 1), dst.cols());
    try testing.expectEqual(@as(f64, 19), dst.get(f64, 0, 0));
    try testing.expectEqual(@as(f64, 320), dst.get(f64, 0, 1));
    try testing.expectEqual(@as(f64, 1.0471975803375244), dst.get(f64, 0, 2));

    try testing.expectEqual(@as(f64, 7), dst.get(f64, 1, 0));
    try testing.expectEqual(@as(f64, 321), dst.get(f64, 1, 1));
    try testing.expectEqual(@as(f64, 1.0646508932113647), dst.get(f64, 1, 2));

    try testing.expectEqual(@as(f64, 2), dst.get(f64, 18, 0));
    try testing.expectEqual(@as(f64, 317), dst.get(f64, 18, 1));
    try testing.expectEqual(@as(f64, 1.1344640254974365), dst.get(f64, 18, 2));

    try testing.expectEqual(@as(f64, 2), dst.get(f64, 19, 0));
    try testing.expectEqual(@as(f64, 330), dst.get(f64, 19, 1));
    try testing.expectEqual(@as(f64, 1.1344640254974365), dst.get(f64, 19, 2));
}

test "imgproc integral" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var sum = try Mat.init();
    defer sum.deinit();
    var sqsum = try Mat.init();
    defer sqsum.deinit();
    var tilted = try Mat.init();
    defer tilted.deinit();

    imgproc.integral(img, &sum, &sqsum, &tilted);
    try testing.expectEqual(false, sum.isEmpty());
    try testing.expectEqual(false, sqsum.isEmpty());
    try testing.expectEqual(false, tilted.isEmpty());
}

test "imgproc threshold" {
    var img = try imgcodecs.imRead(face_detect_filepath, .color);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    _ = imgproc.threshold(img, &dst, 100, 255, .{ .type = .binary });
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc adaptiveThreshold" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    _ = imgproc.adaptiveThreshold(img, &dst, 255, .gaussian, .{ .type = .binary }, 11, 2);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc circle" {
    const tests = [_]struct {
        thickness: i32,
        point: Point,
        result: u8,
    }{
        .{
            .thickness = 3,
            .point = .{ .x = 80, .y = 89 },
            .result = 255,
        },
        .{
            .thickness = -1,
            .point = .{ .x = 60, .y = 60 },
            .result = 255,
        },
    };
    for (tests) |t| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();
        const white = Color.init(255, 255, 255, 0);
        imgproc.circle(&img, Point.init(70, 70), 20, white, t.thickness);
        try testing.expectEqual(t.result, img.get(u8, @as(usize, @intCast(t.point.x)), @as(usize, @intCast(t.point.y))));
    }
}

test "imgproc circleWithParams" {
    const tests = [_]struct {
        thickness: i32,
        shift: i32,
        point: Point,
        result: u8,
    }{
        .{
            .thickness = 3,
            .shift = 0,
            .point = .{ .x = 80, .y = 89 },
            .result = 255,
        },
        .{
            .thickness = -1,
            .shift = 0,
            .point = .{ .x = 60, .y = 60 },
            .result = 255,
        },
        .{
            .thickness = 3,
            .shift = 1,
            .point = .{ .x = 47, .y = 38 },
            .result = 255,
        },
        .{
            .thickness = 3,
            .shift = 1,
            .point = .{ .x = 48, .y = 38 },
            .result = 0,
        },
    };
    for (tests) |t| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();
        const white = Color.init(255, 255, 255, 0);
        imgproc.circleWithParams(&img, Point.init(70, 70), 20, white, t.thickness, .line4, t.shift);
        try testing.expectEqual(t.result, img.get(u8, @as(usize, @intCast(t.point.x)), @as(usize, @intCast(t.point.y))));
    }
}

test "imgproc rectangle" {
    const tests = [_]struct {
        thickness: i32,
        point: Point,
    }{
        .{
            .thickness = 1,
            .point = .{ .x = 10, .y = 60 },
        },
        .{
            .thickness = -1,
            .point = .{ .x = 30, .y = 30 },
        },
    };
    for (tests) |t| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();
        const white = Color.init(255, 255, 255, 0);
        imgproc.rectangle(&img, Rect.init(10, 10, 70, 70), white, t.thickness);
        try testing.expect(img.get(u8, @as(usize, @intCast(t.point.x)), @as(usize, @intCast(t.point.y))) >= 50);
    }
}

test "imgproc rectangleWithParams" {
    const tests = [_]struct {
        thickness: i32,
        shift: i32,
        point: Point,
    }{
        .{
            .thickness = 1,
            .shift = 0,
            .point = .{ .x = 10, .y = 60 },
        },
        .{
            .thickness = -1,
            .shift = 0,
            .point = .{ .x = 30, .y = 30 },
        },
        .{
            .thickness = 1,
            .shift = 1,
            .point = .{ .x = 5, .y = 5 },
        },
    };
    for (tests) |t| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();
        const white = Color.init(255, 255, 255, 0);
        imgproc.rectangleWithParams(&img, Rect.init(10, 10, 70, 70), white, t.thickness, .line4, t.shift);
        try testing.expectEqual(@as(u8, 255), img.get(u8, @as(usize, @intCast(t.point.x)), @as(usize, @intCast(t.point.y))));
    }
}

test "imgproc equlizeHist" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.equalizeHist(img, &dst);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(img.rows(), dst.rows());
    try testing.expectEqual(img.cols(), dst.cols());
}

test "imgproc calcHist" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var dst = try Mat.init();
    defer dst.deinit();

    var mask = try Mat.init();
    defer mask.deinit();
    var m = [1]Mat{img};
    var chans = [1]i32{0};
    var size = [1]i32{256};
    var rng = [2]f32{ 0.0, 256.0 };
    try imgproc.calcHist(&m, &chans, mask, &dst, &size, &rng, false);
    try testing.expectEqual(false, dst.isEmpty());
    try testing.expectEqual(@as(i32, 256), dst.rows());
    try testing.expectEqual(@as(i32, 1), dst.cols());
}

test "imgproc compareHist" {
    var img = try imgcodecs.imRead(face_detect_filepath, .gray_scale);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    var hist1 = try Mat.init();
    defer hist1.deinit();

    var hist2 = try Mat.init();
    defer hist2.deinit();

    var mask = try Mat.init();
    defer mask.deinit();

    var m = [1]Mat{img};
    var chans = [1]i32{0};
    var size = [1]i32{256};
    var rng = [2]f32{ 0.0, 256.0 };
    try imgproc.calcHist(&m, &chans, mask, &hist1, &size, &rng, false);
    try testing.expectEqual(false, hist1.isEmpty());
    try imgproc.calcHist(&m, &chans, mask, &hist2, &size, &rng, false);
    try testing.expectEqual(false, hist2.isEmpty());

    const dist = imgproc.compareHist(hist1, hist2, .correl);
    try testing.expectEqual(@as(f64, 1), dist);
}

test "imgproc drawing" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();

    const blue = Color{ .b = 255 };

    imgproc.arrowedLine(&img, Point.init(50, 50), Point.init(75, 75), blue, 3);
    imgproc.circle(&img, Point.init(60, 60), 20, blue, 3);
    imgproc.rectangle(&img, Rect.init(50, 50, 25, 25), blue, 3);
    imgproc.line(&img, Point.init(50, 50), Point.init(75, 75), blue, 3);

    try testing.expectEqual(false, img.isEmpty());
}

test "imgproc getTextSize" {
    const size = imgproc.getTextSize("test", .{ .type = .simplex }, 1.2, 1);

    try testing.expectEqual(@as(i32, 72), size.width);
    try testing.expectEqual(@as(i32, 26), size.height);

    const res = imgproc.getTextSizeWithBaseline("text", .{ .type = .simplex }, 1.2, 1);

    try testing.expectEqual(@as(i32, 72), res.size.width);
    try testing.expectEqual(@as(i32, 26), res.size.height);
    try testing.expectEqual(@as(i32, 11), res.baseline);
}

test "imgproc putText" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    imgproc.putText(&img, "Testing", Point.init(10, 10), .{ .type = .plain }, 1.2, Color.init(0, 0, 255, 0), 2);

    try testing.expectEqual(false, img.isEmpty());
}

test "imgproc putTextWithParams" {
    var img = try Mat.initSize(150, 150, .cv8uc1);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    imgproc.putTextWithParams(&img, "Testing", Point.init(10, 10), .{ .type = .plain }, 1.2, Color.init(0, 0, 255, 0), 2, .line_aa, false);

    try testing.expectEqual(false, img.isEmpty());
}

// test "imgproc resize" {
// var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
// defer img.deinit();

// var dst = try Mat.init();
// defer dst.deinit();

// imgproc.resize(img, &dst, Size.init(0, 0), 0.5, 0.5, .{});
// try testing.expectEqual(false, dst.isEmpty());
// try testing.expectEqual(@as(i32, 172), dst.rows());
// try testing.expectEqual(@as(i32, 200), dst.cols());

// imgproc.resize(img, &dst, Size.init(440, 377), 0, 0, .{});
// try testing.expectEqual(false, dst.isEmpty());
// try testing.expectEqual(@as(i32, 377), dst.rows());
// try testing.expectEqual(@as(i32, 440), dst.cols());
// }

// test "imgproc getRectSubPix" {
// var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
// defer img.deinit();

// var dst = try Mat.init();
// defer dst.deinit();

// imgproc.getRectSubPix(img, Size.init(100, 100), Point.init(100, 100), &dst);
// try testing.expectEqual(false, dst.isEmpty());
// try testing.expectEqual(@as(i32, 100), dst.rows());
// try testing.expectEqual(@as(i32, 100), dst.cols());
// }

test "imgprc getRotationMatrix2D" {
    const args = struct {
        center: Point,
        angle: f64,
        scale: f64,
    };
    const tests = [_]struct {
        args: args,
        want: [2][3]f64,
    }{
        .{
            .args = .{
                .center = Point.init(0, 0),
                .angle = 90,
                .scale = 1,
            },
            .want = .{
                .{ 6.123233995736766e-17, 1, 0 },
                .{ -1, 6.123233995736766e-17, 0 },
            },
        },
        .{ .args = .{
            .center = Point.init(0, 0),
            .angle = 45,
            .scale = 1,
        }, .want = .{
            .{ 0.7071067811865476, 0.7071067811865475, 0 },
            .{ -0.7071067811865475, 0.7071067811865476, 0 },
        } },
        .{ .args = .{
            .center = Point.init(0, 0),
            .angle = 0,
            .scale = 1,
        }, .want = .{
            .{ 1, 0, 0 },
            .{ 0, 1, 0 },
        } },
    };
    for (tests) |tt| {
        var got = try imgproc.getRotationMatrix2D(tt.args.center, tt.args.angle, tt.args.scale);
        defer got.deinit();
        {
            var i: usize = 0;
            while (i < got.rows()) : (i += 1) {
                var j: usize = 0;
                while (j < got.cols()) : (j += 1) {
                    try testing.expectEqual(tt.want[i][j], got.get(f64, i, j));
                }
            }
        }
    }
}

test "imgproc wrapaffine" {
    var src = try Mat.initSize(256, 256, .cv8uc1);
    defer src.deinit();
    var rot = try imgproc.getRotationMatrix2D(Point.init(0, 0), 1, 1);
    defer rot.deinit();
    var dst = try src.clone();
    defer dst.deinit();

    imgproc.warpAffine(src, &dst, rot, Size.init(256, 256));

    const result = dst.norm(.l2);
    try testing.expectEqual(@as(f64, 0), result);
}

test "imgproc wrapaffineWithParams" {
    var src = try Mat.initSize(256, 256, .cv8uc1);
    defer src.deinit();
    var rot = try imgproc.getRotationMatrix2D(Point.init(0, 0), 1, 1);
    defer rot.deinit();
    var dst = try src.clone();
    defer dst.deinit();

    imgproc.warpAffineWithParams(
        src,
        &dst,
        rot,
        Size.init(256, 256),
        .{ .type = .linear },
        .{ .type = .constant },
        Color{},
    );

    const result = dst.norm(.l2);
    try testing.expectEqual(@as(f64, 0), result);
}

test "imgproc clipLine" {
    const pt1 = Point.init(5, 5);
    const pt2 = Point.init(5, 5);
    const rect = Size.init(20, 20);
    const ok = imgproc.clipLine(rect, pt1, pt2);
    try testing.expectEqual(true, ok);
}

test "imgproc watershed" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer src.deinit();

    var gray = try Mat.init();
    defer gray.deinit();
    imgproc.cvtColor(src, &gray, .bgr_to_gray);

    var img_thresh = try Mat.init();
    defer img_thresh.deinit();
    _ = imgproc.threshold(gray, &img_thresh, 0, 255, .{ .type = .binary, .otsu = true });

    var markers = try Mat.init();
    defer markers.deinit();

    _ = imgproc.connectedComponents(img_thresh, &markers);

    imgproc.watershed(src, &markers);
    try testing.expectEqual(false, markers.isEmpty());
    try testing.expectEqual(src.rows(), markers.rows());
    try testing.expectEqual(src.cols(), markers.cols());
}

test "imgproc applyColorMap" {
    const args = struct {
        colormap: imgproc.ColormapType,
        want: f64,
    };
    const tests = [_]args{
        .{ .colormap = .autumn, .want = 118090.29593069873 },
        .{ .colormap = .bone, .want = 122067.44213343704 },
        .{ .colormap = .jet, .want = 98220.64722857409 },
        .{ .colormap = .winter, .want = 94279.52859449394 },
        .{ .colormap = .rainbow, .want = 92591.40608069411 },
        .{ .colormap = .ocean, .want = 106444.16919681415 },
        .{ .colormap = .summer, .want = 114434.44957703952 },
        .{ .colormap = .spring, .want = 123557.60209715953 },
        .{ .colormap = .cool, .want = 123557.60209715953 },
        .{ .colormap = .hsv, .want = 107679.25179903508 },
        .{ .colormap = .pink, .want = 136043.97287274434 },
        .{ .colormap = .hot, .want = 124941.02475968412 },
        .{ .colormap = .parula, .want = 111483.33555738274 },
    };

    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .gray_scale);
    defer src.deinit();

    for (tests) |tt| {
        var dst = try src.clone();
        defer dst.deinit();
        imgproc.applyColorMap(src, &dst, tt.colormap);
        const result = dst.norm(.l2);
        try testing.expectEqual(tt.want, result);
    }
}

test "imgproc applyCustomColorMap" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .gray_scale);
    defer src.deinit();

    var applyCustomColorMap = try Mat.initSize(256, 1, .cv8uc1);
    defer applyCustomColorMap.deinit();

    var dst = try src.clone();
    defer dst.deinit();
    imgproc.applyCustomColorMap(src, &dst, applyCustomColorMap);
    const result = dst.norm(.l2);
    try testing.expectEqual(@as(f64, 0), result);
}

test "imgproc getPerspectiveTransform" {
    var src = [_]Point{
        Point.init(0, 0),
        Point.init(10, 5),
        Point.init(10, 10),
        Point.init(5, 10),
    };
    var pvsrc = try PointVector.initFromPoints(src[0..], testing.allocator);
    defer pvsrc.deinit();

    var dst = [_]Point{
        Point.init(0, 0),
        Point.init(10, 0),
        Point.init(10, 10),
        Point.init(0, 10),
    };
    var pvdst = try PointVector.initFromPoints(dst[0..], testing.allocator);
    defer pvdst.deinit();

    var m = try imgproc.getPerspectiveTransform(pvsrc, pvdst);
    defer m.deinit();

    try testing.expectEqual(@as(i32, 3), m.cols());
    try testing.expectEqual(@as(i32, 3), m.rows());
}

test "imgproc getPerspectiveTransform2f" {
    var src = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(10.5, 5.5),
        Point2f.init(10.5, 10.5),
        Point2f.init(5.5, 10.5),
    };
    var pvsrc = try Point2fVector.initFromPoints(src[0..], testing.allocator);
    defer pvsrc.deinit();

    var dst = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(590.20, 24.12),
        Point2f.init(100.12, 150.21),
        Point2f.init(0, 10),
    };
    var pvdst = try Point2fVector.initFromPoints(dst[0..], testing.allocator);
    defer pvdst.deinit();

    var m = try imgproc.getPerspectiveTransform2f(pvsrc, pvdst);
    defer m.deinit();

    try testing.expectEqual(@as(i32, 3), m.cols());
    try testing.expectEqual(@as(i32, 3), m.rows());
}

test "imgproc getAffineTransform2f" {
    var src = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(10.5, 5.5),
        Point2f.init(10.5, 10.5),
    };
    var pvsrc = try Point2fVector.initFromPoints(src[0..], testing.allocator);
    defer pvsrc.deinit();

    var dst = [_]Point2f{
        Point2f.init(0, 0),
        Point2f.init(590.20, 24.12),
        Point2f.init(100.12, 150.21),
    };
    var pvdst = try Point2fVector.initFromPoints(dst[0..], testing.allocator);
    defer pvdst.deinit();

    var m = try imgproc.getAffineTransform2f(pvsrc, pvdst);
    defer m.deinit();

    try testing.expectEqual(@as(i32, 3), m.cols());
    try testing.expectEqual(@as(i32, 2), m.rows());
}

test "imgproc getAffineTransform2f 2" {
    var src = [_]Point{
        Point.init(0, 0),
        Point.init(10, 5),
        Point.init(10, 10),
    };
    var pvsrc = try PointVector.initFromPoints(src[0..], testing.allocator);
    defer pvsrc.deinit();

    var dst = [_]Point{
        Point.init(0, 0),
        Point.init(590, 24),
        Point.init(100, 150),
    };
    var pvdst = try PointVector.initFromPoints(dst[0..], testing.allocator);
    defer pvdst.deinit();

    var m = try imgproc.getAffineTransform(pvsrc, pvdst);
    defer m.deinit();

    try testing.expectEqual(@as(i32, 3), m.cols());
    try testing.expectEqual(@as(i32, 2), m.rows());
}

test "imgproc findHomography" {
    var src = try Mat.initSize(4, 1, .cv64fc2);
    defer src.deinit();

    var dst = try Mat.initSize(4, 1, .cv64fc2);
    defer dst.deinit();

    var srcPoints = [_]Point2f{
        Point2f.init(193, 932),
        Point2f.init(191, 378),
        Point2f.init(1497, 183),
        Point2f.init(1889, 681),
    };

    var dstPoints = [_]Point2f{
        Point2f.init(51.51206544281359, -0.10425475260813055),
        Point2f.init(51.51211051314331, -0.10437947532732306),
        Point2f.init(51.512222354139325, -0.10437679311830816),
        Point2f.init(51.51214828037607, -0.1042212249954444),
    };

    for (srcPoints, 0..) |point, i| {
        src.set(f64, i, 0, @as(f64, point.x));
        src.set(f64, i, 1, @as(f64, point.y));
    }

    for (dstPoints, 0..) |point, i| {
        dst.set(f64, i, 0, @as(f64, point.x));
        dst.set(f64, i, 1, @as(f64, point.y));
    }

    var mask = try Mat.init();
    defer mask.deinit();

    var m = try imgproc.findHomography(src, &dst, .all_points, 3, &mask, 2000, 0.995);
    defer m.deinit();

    var pvsrc = try Point2fVector.initFromPoints(srcPoints[0..], testing.allocator);
    defer pvsrc.deinit();

    var pvdst = try Point2fVector.initFromPoints(dstPoints[0..], testing.allocator);
    defer pvdst.deinit();

    var m2 = try imgproc.getPerspectiveTransform2f(pvsrc, pvdst);
    defer m2.deinit();

    {
        var row: usize = 0;
        while (row < 3) : (row += 1) {
            var col: usize = 0;
            while (col < 3) : (col += 1) {
                if (@abs(m.get(f64, row, col) - m2.get(f64, row, col)) > 0.002) {
                    try testing.expectEqual(@as(f64, 0), @abs((m.get(f64, row, col) - m2.get(f64, row, col))));
                }
            }
        }
    }
}

test "imgproc warpPerspective" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer img.deinit();

    const w = img.cols();
    const h = img.rows();

    var s = [_]Point{
        Point.init(0, 0),
        Point.init(10, 5),
        Point.init(10, 10),
        Point.init(5, 10),
    };

    var pvs = try PointVector.initFromPoints(s[0..], testing.allocator);
    defer pvs.deinit();

    var d = [_]Point{
        Point.init(0, 0),
        Point.init(10, 0),
        Point.init(10, 10),
        Point.init(0, 10),
    };

    var pvd = try PointVector.initFromPoints(d[0..], testing.allocator);
    defer pvd.deinit();

    var m = try imgproc.getPerspectiveTransform(pvs, pvd);
    defer m.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.warpPerspective(img, &dst, m, Size.init(w, h));

    try testing.expectEqual(w, dst.cols());
    try testing.expectEqual(h, dst.rows());
}

test "imgproc wrapPerspectiveWithParams" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer img.deinit();
    try testing.expectEqual(false, img.isEmpty());

    const w = img.cols();
    const h = img.rows();

    var s = [_]Point{
        Point.init(0, 0),
        Point.init(10, 5),
        Point.init(10, 10),
        Point.init(5, 10),
    };

    var pvs = try PointVector.initFromPoints(s[0..], testing.allocator);
    defer pvs.deinit();

    var d = [_]Point{
        Point.init(0, 0),
        Point.init(10, 0),
        Point.init(10, 10),
        Point.init(0, 10),
    };

    var pvd = try PointVector.initFromPoints(d[0..], testing.allocator);
    defer pvd.deinit();

    var m = try imgproc.getPerspectiveTransform(pvs, pvd);
    defer m.deinit();

    var dst = try Mat.init();
    defer dst.deinit();

    imgproc.warpPerspectiveWithParams(img, &dst, m, Size.init(w, h), .{ .type = .linear }, .{ .type = .constant }, Color{});

    try testing.expectEqual(w, dst.cols());
    try testing.expectEqual(h, dst.rows());
}

test "imgproc drawContours" {
    var img = try Mat.initSize(100, 200, .cv8uc1);
    defer img.deinit();

    // Draw rectangle
    const white = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
    imgproc.rectangle(&img, Rect.init(125, 25, 175, 75), white, 1);

    var contours = try imgproc.findContours(img, .external, .simple);
    defer contours.deinit();

    try testing.expectEqual(@as(u8, 0), img.get(u8, 23, 123));
    try testing.expectEqual(@as(u8, 206), img.get(u8, 25, 125));

    imgproc.drawContours(&img, contours, -1, white, 2);

    // contour should be drawn with thickness = 2
    try testing.expectEqual(@as(u8, 255), img.get(u8, 24, 124));
    try testing.expectEqual(@as(u8, 255), img.get(u8, 25, 125));
}

test "imgproc drawContoursWithParams" {
    var img = try Mat.initSize(200, 200, .cv8uc1);
    defer img.deinit();

    // Draw circle
    const white = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
    const black = Color{ .r = 0, .g = 0, .b = 0, .a = 255 };
    imgproc.circle(&img, Point.init(100, 100), 80, white, -1);
    imgproc.circle(&img, Point.init(100, 100), 55, black, -1);
    imgproc.circle(&img, Point.init(100, 100), 30, white, -1);

    var hierarchy = try Mat.init();
    defer hierarchy.deinit();
    var contours = try imgproc.findContoursWithParams(img, &hierarchy, .tree, .simple);
    defer contours.deinit();

    // Draw contours by different line-type and assert value
    const cases = [_]struct {
        line_type: imgproc.LineType,
        expect_uchar: u8,
    }{
        .{
            .line_type = .line4,
            .expect_uchar = 255,
        },
        .{
            .line_type = .line8,
            .expect_uchar = 0,
        },
        .{
            .line_type = .line_aa,
            .expect_uchar = 68,
        },
    };
    for (cases) |c| {
        var bg = try Mat.initSize(img.rows(), img.cols(), .cv8uc1);
        defer bg.deinit();

        imgproc.drawContoursWithParams(&bg, contours, -1, white, 1, c.line_type, hierarchy, 0, Point.init(0, 0));

        if (bg.get(u8, 22, 88) != c.expect_uchar) {
            try testing.expectEqual(c.expect_uchar, bg.get(u8, 22, 88));
        }
    }
}

test "imgproc ellipse" {
    const tests = [_]struct {
        name: []const u8,
        thickness: i32,
        point: Point,
    }{
        .{
            .name = "Without filling",
            .thickness = 2,
            .point = Point.init(24, 50),
        },
        .{
            .name = "With filling",
            .thickness = -1,
            .point = Point.init(55, 47),
        },
    };

    for (tests) |tc| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();

        const white = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
        imgproc.ellipse(&img, Point.init(50, 50), Point.init(25, 25), 0, 0, 360, white, tc.thickness);

        try testing.expectEqual(@as(u8, 255), img.get(u8, @as(usize, @intCast(tc.point.x)), @as(usize, @intCast(tc.point.y))));
    }
}

test "imgproc ellipseWithParams" {
    const tests = [_]struct {
        name: []const u8,
        thickness: i32,
        linetype: imgproc.LineType,
        shift: i32 = 0,
        point: Point,
        want: ?i32 = 255,
    }{
        .{
            .name = "Without filling and shift, line = Line8",
            .thickness = 2,
            .linetype = .line8,
            .point = Point.init(24, 50),
        },
        .{
            .name = "With filling, without shift, line = Line8",
            .thickness = -1,
            .linetype = .line8,
            .point = Point.init(55, 47),
        },
        .{
            .name = "Without filling, with shift 2, line = Line8",
            .thickness = 2,
            .linetype = .line8,
            .shift = 2,
            .point = Point.init(6, 12),
        },
        .{
            .name = "Without filling, with shift 2, line = Line8",
            .thickness = 2,
            .linetype = .line8,
            .shift = 2,
            .point = Point.init(19, 13),
        },

        .{
            .name = "Without filling and shift, line = LineAA",
            .thickness = 2,
            .linetype = .line_aa,
            .point = Point.init(77, 54),
            .want = null,
        },
    };

    for (tests) |tc| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();

        const white = Color{ .r = 255, .g = 255, .b = 255, .a = 0 };
        imgproc.ellipseWithParams(
            &img,
            Point.init(50, 50),
            Point.init(25, 25),
            0,
            0,
            360,
            white,
            tc.thickness,
            tc.linetype,
            tc.shift,
        );
        const r = img.get(u8, @as(usize, @intCast(tc.point.x)), @as(usize, @intCast(tc.point.y)));
        if (tc.want) |want| {
            testing.expectEqual(want, r) catch |e| {
                std.debug.print("test: {s}\n", .{tc.name});
                std.debug.print("{any}\n", .{e});
                return error.TestExpectedEqual;
            };
        } else {
            try testing.expect(r > 10);
            try testing.expect(r < 220);
        }
    }
}

test "imgproc fillPoly" {
    var img = try Mat.initSize(100, 100, .cv8uc1);
    defer img.deinit();

    const white = Color{ .r = 255, .g = 255, .b = 255, .a = 0 };
    var pts = [_][4]Point{
        .{
            .{ .x = 10, .y = 10 },
            .{ .x = 10, .y = 20 },
            .{ .x = 20, .y = 20 },
            .{ .x = 20, .y = 10 },
        },
    };
    var pv = try PointsVector.initFromPoints(&pts, testing.allocator);
    defer pv.deinit();

    imgproc.fillPoly(&img, pv, white);

    try testing.expectEqual(@as(u8, 255), img.get(u8, 10, 10));
}

test "imgproc fillPolyWithParams" {
    const tests = [_]struct {
        name: []const u8,
        offset: Point,
        point: Point,
        result: u8,
    }{
        .{
            .name = "No offset",
            .offset = Point.init(0, 0),
            .point = Point.init(10, 10),
            .result = 255,
        },
        .{
            .name = "Offset of 2",
            .offset = Point.init(2, 2),
            .point = Point.init(12, 12),
            .result = 255,
        },
    };

    const white = Color{ .r = 255, .g = 255, .b = 255, .a = 0 };
    var pts = [_][4]Point{
        .{
            .{ .x = 10, .y = 10 },
            .{ .x = 10, .y = 20 },
            .{ .x = 20, .y = 20 },
            .{ .x = 20, .y = 10 },
        },
    };
    var pv = try PointsVector.initFromPoints(&pts, testing.allocator);
    defer pv.deinit();

    for (tests) |tc| {
        var img = try Mat.initSize(100, 100, .cv8uc1);
        defer img.deinit();

        imgproc.fillPolyWithParams(&img, pv, white, .line4, 0, tc.offset);

        try testing.expectEqual(tc.result, img.get(u8, @as(usize, @intCast(tc.point.x)), @as(usize, @intCast(tc.point.y))));
    }
}

test "imgproc polylines" {
    var img = try Mat.initSize(100, 100, .cv8uc1);
    defer img.deinit();

    const white = Color{ .r = 255, .g = 255, .b = 255, .a = 0 };
    var pts = [_][4]Point{
        .{
            .{ .x = 10, .y = 10 },
            .{ .x = 10, .y = 20 },
            .{ .x = 20, .y = 20 },
            .{ .x = 20, .y = 10 },
        },
    };
    var pv = try PointsVector.initFromPoints(&pts, testing.allocator);
    defer pv.deinit();

    imgproc.polylines(&img, pv, true, white, 1);

    try testing.expectEqual(@as(u8, 255), img.get(u8, 10, 10));
}

// test "imgproc remap" {
// var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
// defer img.deinit();

// var dst = try Mat.init();
// defer dst.deinit();
// try testing.expectEqual(true, dst.isEmpty());

// var map1 = try Mat.initSize(256, 256, .cv32fc2);
// defer map1.deinit();
// map1.set(f32, 50, 50, 25.4);
// var map2 = try Mat.init();
// defer map2.deinit();

// imgproc.remap(img, &dst, map1, map2, .{}, .{ .type = .constant }, Color{});
// try testing.expectEqual(false, dst.isEmpty());
// }

// test "imgproc filter2D" {
// var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
// defer img.deinit();

// var dst = try img.clone();
// defer dst.deinit();

// var kernel = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
// defer kernel.deinit();

// imgproc.filter2D(img, &dst, -1, kernel, Point.init(-1, -1), 0, .{});
// try testing.expectEqual(false, dst.isEmpty());
// }

test "imgproc sepFilter2D" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer img.deinit();

    var dst = try img.clone();
    defer dst.deinit();

    var kernelX = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernelX.deinit();
    var kernelY = try imgproc.getStructuringElement(.rect, Size.init(1, 1));
    defer kernelY.deinit();

    imgproc.sepFilter2D(img, &dst, -1, kernelX, kernelY, Point.init(-1, -1), 0, .{});
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc logPolar" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer img.deinit();

    var dst = try img.clone();
    defer dst.deinit();

    imgproc.logPolar(img, &dst, Point.init(22, 22), 1, .{});
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc linearPolar" {
    var img = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .color);
    defer img.deinit();

    var dst = try img.clone();
    defer dst.deinit();

    imgproc.linearPolar(img, &dst, Point.init(22, 22), 1, .{});
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc fitLine" {
    var points = [_]Point{
        .{ .x = 125, .y = 24 },
        .{ .x = 124, .y = 75 },
        .{ .x = 175, .y = 76 },
        .{ .x = 176, .y = 25 },
    };
    var pv = try PointVector.initFromPoints(&points, testing.allocator);
    defer pv.deinit();

    var line = try Mat.init();
    defer line.deinit();

    imgproc.fitLine(pv, &line, .l2, 0, 0.01, 0.01);
    try testing.expectEqual(false, line.isEmpty());
}

test "imgproc invertAffineTransform" {
    var src = try Mat.initSize(2, 3, .cv32fc1);
    defer src.deinit();

    var dst = try Mat.initSize(2, 3, .cv32fc1);
    defer dst.deinit();

    imgproc.invertAffineTransform(src, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

// test "imgproc phaseCorrelate" {
// var template = try imgcodecs.imRead(img_dir ++ "simple.jpg", .gray_scale);
// defer template.deinit();

// var matched = try imgcodecs.imRead(img_dir ++ "simple-translated.jpg", .gray_scale);
// defer matched.deinit();

// var notMatchedOrig = try imgcodecs.imRead(img_dir ++ "space_shuttle.jpg", .gray_scale);
// defer notMatchedOrig.deinit();

// var notMatched = try Mat.init();
// defer notMatched.deinit();

// imgproc.resize(notMatchedOrig, &notMatched, Size.init(matched.size()[0], matched.size()[1]), 0, 0, .{ .type = .linear });

// var template32FC1 = try Mat.init();
// defer template32FC1.deinit();
// var matched32FC1 = try Mat.init();
// defer matched32FC1.deinit();
// var notMatched32FC1 = try Mat.init();
// defer notMatched32FC1.deinit();

// template.convertTo(&template32FC1, .cv32fc1);
// matched.convertTo(&matched32FC1, .cv32fc1);
// notMatched.convertTo(&notMatched32FC1, .cv32fc1);

// var window = try Mat.init();
// defer window.deinit();

// const shiftTranslated = imgproc.phaseCorrelate(template32FC1, matched32FC1, window);
// const responseTranslated = imgproc.phaseCorrelate(template32FC1, matched32FC1, window);

// try testing.expect(shiftTranslated.point.x < 15);
// try testing.expect(shiftTranslated.point.y < 15);

// try testing.expect(responseTranslated.response > 0.85);

// const responseDifferent = imgproc.phaseCorrelate(template32FC1, notMatched32FC1, window);
// try testing.expect(responseDifferent.response < 0.05);
// }

test "imgproc accumulate" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var dst = try Mat.initSize(src.rows(), src.cols(), .cv64fc3);
    defer dst.deinit();

    imgproc.accumulate(src, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc AccumulateWithMask" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var dst = try Mat.initSize(src.rows(), src.cols(), .cv64fc3);
    defer dst.deinit();

    var mask = try Mat.init();
    defer mask.deinit();
    imgproc.accumulateWithMask(src, &dst, mask);

    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc accumulateSquare" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var dst = try Mat.initSize(src.rows(), src.cols(), .cv64fc3);
    defer dst.deinit();

    imgproc.accumulateSquare(src, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc AccumulateSquareWithMask" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var dst = try Mat.initSize(src.rows(), src.cols(), .cv64fc3);
    defer dst.deinit();

    var mask = try Mat.init();
    defer mask.deinit();
    imgproc.accumulateSquareWithMask(src, &dst, mask);

    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc AccumulateProduct" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var src2 = try src.clone();
    defer src2.deinit();

    var dst = try Mat.initSize(src.size()[0], src.size()[1], .cv64fc3);
    defer dst.deinit();

    imgproc.accumulateProduct(src, src2, &dst);
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc AccumulateProductWithMask" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var src2 = try src.clone();
    defer src2.deinit();

    var dst = try Mat.initSize(src.size()[0], src.size()[1], .cv64fc3);
    defer dst.deinit();

    var mask = try Mat.init();
    defer mask.deinit();
    imgproc.accumulateProductWithMask(src, src2, &dst, mask);

    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc AccumulatedWeighted" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var dst = try Mat.initSize(src.size()[0], src.size()[1], .cv64fc3);
    defer dst.deinit();

    imgproc.accumulatedWeighted(src, &dst, 0.5);
    try testing.expectEqual(false, dst.isEmpty());
}

test "imgproc AccumulatedWeightedWithMask" {
    var src = try imgcodecs.imRead(img_dir ++ "gocvlogo.jpg", .unchanged);
    defer src.deinit();

    var dst = try Mat.initSize(src.size()[0], src.size()[1], .cv64fc3);
    defer dst.deinit();

    var mask = try Mat.init();
    defer mask.deinit();
    imgproc.accumulatedWeightedWithMask(src, &dst, 0.5, mask);

    try testing.expectEqual(false, dst.isEmpty());
}
