const c = @import("c_api.zig");
const std = @import("std");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;

pub const Mat = @import("core/mat.zig");
pub const Mats = Mat.Mats;

pub const BorderType = struct {
    /// BorderIsolated border type
    isolate: bool = false,

    type: enum(u4) {
        /// BorderConstant border type
        constant = 0,

        /// BorderReplicate border type
        replicate = 1,

        /// BorderReflect border type
        reflect = 2,

        /// BorderWrap border type
        wrap = 3,

        /// BorderReflect101 border type
        reflect101 = 4,

        /// BorderTransparent border type
        transparent = 5,
    } = .reflect101,

    pub fn toNum(self: BorderType) u5 {
        return @as(u5, @bitCast(packed struct {
            type: u4,
            isolate: bool,
        }{
            .type = @intFromEnum(self.type),
            .isolate = self.isolate,
        }));
    }

    comptime {
        std.debug.assert((BorderType{ .type = .constant }).toNum() == 0);
        std.debug.assert((BorderType{ .type = .replicate }).toNum() == 1);
        std.debug.assert((BorderType{ .type = .constant, .isolate = true }).toNum() == 16);
        std.debug.assert((BorderType{}).type == .reflect101);
    }
};

pub const NormType = enum(u8) {
    /// NormInf indicates use infinite normalization.
    inf = 1,

    /// NormL1 indicates use L1 normalization.
    l1 = 2,

    /// NormL2 indicates use L2 normalization.
    l2 = 4,

    /// NormL2Sqr indicates use L2 squared normalization.
    l2_sqr = 5,

    /// NormHamming indicates use Hamming normalization.
    hamming = 6,

    /// NormHamming2 indicates use Hamming 2-bit normalization.
    hamming2 = 7,

    /// NormRelative indicates use relative normalization.
    relative = 8,

    /// NormMinMax indicates use min/max normalization.
    min_max = 32,
};

pub const Point = struct {
    x: i32,
    y: i32,

    const Self = @This();

    pub fn init(x: i32, y: i32) Self {
        return .{ .x = x, .y = y };
    }

    pub fn initFromC(p: c.Point) Self {
        return .{ .x = p.x, .y = p.y };
    }

    pub fn toC(self: Self) c.Point {
        return .{ .x = self.x, .y = self.y };
    }
};

pub const PointVector = struct {
    ptr: c.PointVector,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.PointVector_New();
        return try initFromC(ptr);
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.PointVector_NewFromMat(mat_ptr);
        return try initFromC(ptr);
    }

    pub fn initFromPoints(points: []const Point, allocator: std.mem.Allocator) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();
        const len = @as(usize, @intCast(points.len));

        var c_point_array = try arena_allocator.alloc(c.Point, len);
        for (points, 0..) |point, i| c_point_array[i] = point.toC();
        const contour: c.Contour = .{
            .length = @as(i32, @intCast(points.len)),
            .points = @as([*]c.Point, @ptrCast(c_point_array.ptr)),
        };
        const ptr = c.PointVector_NewFromPoints(contour);
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.PointVector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.PointVector_Close(self.ptr);
        self.ptr = null;
    }

    pub fn at(self: Self, idx: i32) Point {
        const p = c.PointVector_At(self.ptr, idx);
        return Point.initFromC(p);
    }

    pub fn size(self: Self) i32 {
        return c.PointVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point) void {
        c.PointVector_Append(self.ptr, p.toC());
    }

    pub fn toC(self: Self) c.PointVector {
        return self.ptr;
    }

    pub fn toPoints(self: Self, allocator: std.mem.Allocator) !std.ArrayList(Point) {
        const size_ = self.size();
        var array = try std.ArrayList(Point).initCapacity(allocator, @as(usize, @intCast(size_)));
        {
            var i: i32 = 0;
            while (i < size_) : (i += 1) {
                try array.append(self.at(i));
            }
        }
        return array;
    }
};

pub const PointsVector = struct {
    ptr: c.PointsVector,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.PointsVector_New();
        return try initFromC(ptr);
    }

    pub fn initFromPoints(points: anytype, allocator: std.mem.Allocator) !Self {
        if (points.len == 0) return try init();

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();

        var c_points_array = try arena_allocator.alloc(c.Points, points.len);
        for (points, 0..) |point, i| {
            var c_point_array = try arena_allocator.alloc(c.Point, point.len);
            for (point, 0..) |p, j| c_point_array[j] = p.toC();
            c_points_array[i] = .{
                .length = @as(i32, @intCast(point.len)),
                .points = @as([*]c.Point, @ptrCast(c_point_array.ptr)),
            };
        }

        const c_points = c.struct_Contours{
            .length = @as(i32, @intCast(points.len)),
            .contours = @as([*]c.Contour, @ptrCast(c_points_array.ptr)),
        };
        const ptr = c.PointsVector_NewFromPoints(c_points);
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.PointsVector) !Self {
        const nn_ptr = try epnn(ptr);
        return .{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.PointsVector_Close(self.ptr);
        self.ptr = null;
    }

    pub fn at(self: Self, idx: i32) !PointVector {
        const ptr = c.PointsVector_At(self.ptr, idx);
        return try PointVector.initFromC(ptr);
    }

    pub fn size(self: Self) i32 {
        return c.PointsVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: PointVector) void {
        c.PointsVector_Append(self.ptr, p.ptr);
    }

    pub fn toC(self: Self) c.PointsVector {
        return self.ptr;
    }
};

pub const Point2f = struct {
    x: f32,
    y: f32,

    const Self = @This();

    pub fn init(x: f32, y: f32) Self {
        return .{ .x = x, .y = y };
    }

    pub fn initFromC(p: c.Point2f) Self {
        return .{ .x = p.x, .y = p.y };
    }

    pub fn toC(self: Self) c.Point2f {
        return .{ .x = self.x, .y = self.y };
    }
};

pub const Point2fVector = struct {
    ptr: c.Point2fVector,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.Point2fVector_New();
        return try initFromC(ptr);
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.Point2fVector_NewFromMat(mat_ptr);
        return try initFromC(ptr);
    }

    pub fn initFromPoints(points: []const Point2f, allocator: std.mem.Allocator) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();

        var c_point_array = try arena_allocator.alloc(c.Point2f, points.len);
        for (points, 0..) |point, i| c_point_array[i] = point.toC();
        return .{
            .ptr = c.Point2fVector_NewFromPoints(
                c.Points2f{
                    .length = @as(i32, @intCast(points.len)),
                    .points = @as([*]c.Point2f, @ptrCast(c_point_array.ptr)),
                },
            ),
        };
    }

    pub fn initFromC(ptr: c.Point2fVector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.Point2fVector_Close(self.ptr);
        self.ptr = null;
    }

    pub fn at(self: Self, idx: i32) Point {
        return c.Point2fVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) i32 {
        return c.Point2fVector_Size(self.ptr);
    }

    pub fn toC(self: Self) c.Point2fVector {
        return self.ptr;
    }

    pub fn toArrayList(self: Self, allocator: std.mem.Allocator) !std.ArrayList(Point2f) {
        return try utils.fromCStructsToArrayList(self.ptr, self.size(), Point2f, allocator);
    }
};

pub const Points2fVector = struct {
    ptr: c.Points2fVector,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.Points2fVector_New();
        return try initFromC(ptr);
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.Points2fVector_NewFromMat(mat_ptr);
        return try initFromC(ptr);
    }

    pub fn initFromPoints(points: anytype, allocator: std.mem.Allocator) !Self {
        if (points.len == 0) return try init();

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();

        var c_points_array = try arena_allocator.alloc(c.Points2f, points.len);
        for (points, 0..) |point, i| {
            var c_point_array = try arena_allocator.alloc(c.Point2f, point.len);
            for (point, 0..) |p, j| c_point_array[j] = p.toC();
            c_points_array[i] = .{
                .length = @as(i32, @intCast(point.len)),
                .points = @as([*]c.Point, @ptrCast(c_point_array.ptr)),
            };
        }

        const c_points = c.struct_Contours{
            .length = @as(i32, @intCast(points.len)),
            .contours = @as([*]c.Contour, @ptrCast(c_points_array.ptr)),
        };
        const ptr = c.Points2fVector_NewFromPoints(c_points);
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.Points2fVector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.Points2fVector_Close(self.ptr);
        self.ptr = null;
    }

    pub fn at(self: Self, idx: i32) Point {
        return c.Points2fVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) i32 {
        return c.Points2fVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point2fVector) void {
        c.Points2fVector_Append(self.ptr, p.ptr);
    }

    pub fn toC(self: Self) c.Point2fVector {
        return self.ptr;
    }
};

pub const Point3f = struct {
    x: f32,
    y: f32,
    z: f32,

    const Self = @This();

    pub fn init(x: f32, y: f32, z: f32) Self {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn initFromC(p: c.Point3f) Self {
        return .{ .x = p.x, .y = p.y, .z = p.z };
    }

    pub fn toC(self: Self) c.Point3f {
        return .{ .x = self.x, .y = self.y, .z = self.z };
    }
};

pub const Point3fVector = struct {
    ptr: c.Point3fVector,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.Point3fVector_New();
        return try initFromC(ptr);
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.Point3fVector_NewFromMat(mat_ptr);
        return try initFromC(ptr);
    }

    pub fn initFromPoints(points: []const Point3f, allocator: std.mem.Allocator) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();
        const len = @as(usize, @intCast(points.len));

        var c_point_array = try arena_allocator.alloc(c.Point3f, len);
        for (points, 0..) |point, i| c_point_array[i] = point.toC();
        return .{
            .ptr = c.Point3fVector_NewFromPoints(
                c.Points3f{
                    .length = @as(i32, @intCast(points.len)),
                    .points = @as([*]c.Point3f, @ptrCast(c_point_array.ptr)),
                },
            ),
        };
    }

    pub fn initFromC(ptr: c.Point3fVector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.Point3fVector_Close(self.ptr);
        self.ptr = null;
    }

    pub fn at(self: Self, idx: i32) Point {
        return c.Point3fVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) i32 {
        return c.Point3fVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point3f) void {
        c.Point3fVector_Append(self.ptr, p.toC());
    }

    pub fn toC(self: Self) c.Point3fVector {
        return self.ptr;
    }
};

pub const Points3fVector = struct {
    ptr: c.Points3fVector,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.Points3fVector_New();
        return try initFromC(ptr);
    }

    pub fn initFromMat(mat: Mat) !Self {
        const mat_ptr = try epnn(mat.ptr);
        const ptr = c.Points3fVector_NewFromMat(mat_ptr);
        return try initFromC(ptr);
    }

    pub fn initFromPoints(points: anytype, allocator: std.mem.Allocator) !Self {
        if (points.len == 0) return try init();

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var arena_allocator = arena.allocator();

        var c_points_array = try arena_allocator.alloc(c.Points3f, points.len);
        for (points, 0..) |point, i| {
            var c_point_array = try arena_allocator.alloc(c.Point3f, point.len);
            for (point, 0..) |p, j| c_point_array[j] = p.toC();
            c_points_array[i] = .{
                .length = @as(i32, @intCast(point.len)),
                .points = @as([*]c.Point, @ptrCast(c_point_array.ptr)),
            };
        }

        const c_points = c.struct_Contours{
            .length = @as(i32, @intCast(points.len)),
            .contours = @as([*]c.Contour, @ptrCast(c_points_array.ptr)),
        };
        const ptr = c.Points3fVector_NewFromPoints(c_points);
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.Points3fVector) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.Points3fVector_Close(self.ptr);
        self.ptr = null;
    }

    pub fn at(self: Self, idx: i32) Point {
        return c.Points3fVector_At(self.ptr, idx);
    }

    pub fn size(self: Self) i32 {
        return c.Points3fVector_Size(self.ptr);
    }

    pub fn append(self: *Self, p: Point3fVector) void {
        c.Points3fVector_Append(self.ptr, p.ptr);
    }

    pub fn toC(self: Self) c.Point3fVector {
        return self.ptr;
    }
};

pub const Scalar = struct {
    val1: f64,
    val2: f64,
    val3: f64,
    val4: f64,

    const Self = @This();

    pub fn init(
        val1: f64,
        val2: f64,
        val3: f64,
        val4: f64,
    ) Self {
        return .{
            .val1 = val1,
            .val2 = val2,
            .val3 = val3,
            .val4 = val4,
        };
    }

    pub fn initFromC(s: c.Scalar) Self {
        return .{
            .val1 = s.val1,
            .val2 = s.val2,
            .val3 = s.val3,
            .val4 = s.val4,
        };
    }

    pub fn toC(self: Self) c.Scalar {
        return .{
            .val1 = self.val1,
            .val2 = self.val2,
            .val3 = self.val3,
            .val4 = self.val4,
        };
    }

    pub fn toArray(self: Self) [4]f64 {
        return .{
            self.val1,
            self.val2,
            self.val3,
            self.val4,
        };
    }
};

pub const Color = struct {
    r: f64 = 0,
    g: f64 = 0,
    b: f64 = 0,
    a: f64 = 0,

    const Self = @This();

    pub fn init(
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) Self {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn toScalar(self: Self) Scalar {
        return .{
            .val1 = self.b,
            .val2 = self.g,
            .val3 = self.r,
            .val4 = self.a,
        };
    }
};

pub const KeyPoint = struct {
    x: f64,
    y: f64,
    size: f64,
    angle: f64,
    response: f64,
    octave: i32,
    class_id: i32,

    const Self = @This();

    pub fn init(
        x: f64,
        y: f64,
        size: f64,
        angle: f64,
        response: f64,
        octave: i32,
        class_id: i32,
    ) Self {
        return .{
            .x = x,
            .y = y,
            .size = size,
            .angle = angle,
            .response = response,
            .octave = octave,
            .class_id = class_id,
        };
    }

    pub fn initFromC(kp: c.KeyPoint) Self {
        return .{
            .x = kp.x,
            .y = kp.y,
            .size = kp.size,
            .angle = kp.angle,
            .response = kp.response,
            .octave = kp.octave,
            .class_id = kp.classID,
        };
    }

    pub fn toC(self: Self) c.KeyPoint {
        return .{
            .x = self.x,
            .y = self.y,
            .size = self.size,
            .angle = self.angle,
            .response = self.response,
            .octave = self.octave,
            .classID = self.class_id,
        };
    }

    pub fn toArrayList(c_kps: c.KeyPoints, allocator: std.mem.Allocator) !KeyPoints {
        return try utils.fromCStructsToArrayList(c_kps.keypoints, c_kps.length, Self, allocator);
    }
};

pub const KeyPoints = std.ArrayList(KeyPoint);

pub const Rect = struct {
    x: i32,
    y: i32,
    width: i32,
    height: i32,

    const Self = @This();

    pub fn init(
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    ) Self {
        return .{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
        };
    }

    pub fn initFromC(r: c.Rect) Self {
        return .{
            .x = r.x,
            .y = r.y,
            .width = r.width,
            .height = r.height,
        };
    }

    pub fn toC(self: Self) c.Rect {
        return .{
            .x = self.x,
            .y = self.y,
            .width = self.width,
            .height = self.height,
        };
    }

    pub fn toArrayList(c_rects: c.Rects, allocator: std.mem.Allocator) !Rects {
        return try utils.fromCStructsToArrayList(c_rects.rects, c_rects.length, Self, allocator);
    }

    pub fn deinitRects(rects: c.Rects) void {
        c.Rects_Close(rects);
    }
};

pub const Rects = std.ArrayList(Rect);

pub const RotatedRect = struct {
    pts: c.Points,
    bounding_rect: Rect,
    center: Point,
    size: Size,
    angle: f64,

    const Self = @This();

    pub fn init(
        pts: c.Points,
        bounding_rect: Rect,
        center: Point,
        size: Size,
        angle: f64,
    ) Self {
        return .{
            .pts = pts,
            .bounding_rect = bounding_rect,
            .center = center,
            .size = size,
            .angle = angle,
        };
    }

    pub fn initFromC(r: c.RotatedRect) Self {
        return .{
            .pts = r.pts,
            .bounding_rect = Rect.initFromC(r.boundingRect),
            .center = Point.initFromC(r.center),
            .size = Size.initFromC(r.size),
            .angle = r.angle,
        };
    }

    pub fn toC(self: Self) c.RotatedRect {
        return .{
            .pts = self.pts,
            .boundingRect = self.bounding_rect.toC(),
            .center = self.center.toC(),
            .size = self.size.toC(),
            .angle = self.angle,
        };
    }
};

pub const Size = struct {
    width: i32,
    height: i32,

    const Self = @This();

    pub fn init(width: i32, height: i32) Self {
        return .{
            .width = width,
            .height = height,
        };
    }

    pub fn initFromC(r: c.Size) Self {
        return .{
            .width = r.width,
            .height = r.height,
        };
    }

    pub fn toC(self: Self) c.Size {
        return .{
            .width = @as(c_int, @intCast(self.width)),
            .height = @as(c_int, @intCast(self.height)),
        };
    }
};

pub const RNG = struct {
    ptr: c.RNG,

    const Self = @This();

    // TheRNG Returns the default random number generator.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga75843061d150ad6564b5447e38e57722
    //
    pub fn init() Self {
        .{ .ptr = c.TheRNG() };
    }

    pub fn initFromC(ptr: c.RNG) Self {
        return .{ .ptr = ptr };
    }

    pub fn toC(self: Self) c.RNG {
        return self.ptr;
    }

    // TheRNG Sets state of default random number generator.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga757e657c037410d9e19e819569e7de0f
    //
    pub fn setRNGSeed(self: Self, seed: i32) void {
        _ = self;
        c.SetRNGSeed(seed);
    }

    // Fill Fills arrays with random numbers.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#ad26f2b09d9868cf108e84c9814aa682d
    //
    pub fn fill(self: *Self, mat: *Mat, dist_type: i32, a: f64, b: f64, saturate_range: bool) void {
        _ = c.RNG_Fill(self.ptr, mat.*.ptr, dist_type, a, b, saturate_range);
    }

    // Gaussian Returns the next random number sampled from
    // the Gaussian distribution.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#a8df8ce4dc7d15916cee743e5a884639d
    //
    pub fn gaussian(self: *Self, sigma: f64) f64 {
        return c.RNG_Gaussian(self.ptr, sigma);
    }

    // Gaussian Returns the next random number sampled from
    // the Gaussian distribution.
    //
    // For further details, please see:
    // https://docs.opencv.org/master/d1/dd6/classcv_1_1RNG.html#a8df8ce4dc7d15916cee743e5a884639d
    //
    pub fn next(self: *Self) c_uint {
        return c.RNG_Next(self.ptr);
    }
};

pub const STDVector = packed struct {
    value: usize,
    before: usize,
    after: usize,

    const Self = @This();
    pub fn init(ptr: *Self) void {
        c.StdByteVectorInitialize(ptr);
    }

    pub fn deinit(ptr: *Self) void {
        c.StdByteVectorFree(ptr);
    }

    pub fn len(ptr: *Self) usize {
        return c.StdByteVectorLen(ptr);
    }

    pub fn data(ptr: *Self) [*c]u8 {
        return c.StdByteVectorData(ptr);
    }
};

pub const TermCriteria = struct {
    ptr: c.TermCriteria,

    const Self = @This();

    /// TermCriteriaType for TermCriteria.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d9/d5d/classcv_1_1TermCriteria.html#a56fecdc291ccaba8aad27d67ccf72c57
    ///
    pub const Type = packed struct(u2) {
        /// Count is the maximum number of iterations or elements to compute.
        // MaxIter is the maximum number of iterations or elements to compute.
        count: bool = false,

        /// EPS is the desired accuracy or change in parameters at which the
        /// iterative algorithm stops.
        eps: bool = false,

        pub fn toNum(self: @This()) u2 {
            return @as(u2, @bitCast(self));
        }
    };

    pub fn init(type_: Type, max_count: i32, epsilon: f64) !Self {
        const ptr = c.TermCriteria_New(type_.toNum(), max_count, epsilon);
        return try initFromC(ptr);
    }

    fn initFromC(ptr: c.TermCriteria) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        self.ptr = null;
    }

    pub fn toC(self: Self) c.TermCriteria {
        return self.ptr;
    }
};

pub inline fn toByteArray(s: []u8) c.ByteArray {
    return c.ByteArray{
        .data = @as([*c]u8, @ptrCast(s.ptr)),
        .length = @as(i32, @intCast(s.len)),
    };
}

/// GetTickCount returns the number of ticks.
///
/// For further details, please see:
/// https://docs.opencv.org/master/db/de0/group__core__utils.html#gae73f58000611a1af25dd36d496bf4487
///
pub fn getCVTickCount() i64 {
    return c.GetCVTickCount();
}

/// GetTickFrequency returns the number of ticks per second.
///
/// For further details, please see:
/// https://docs.opencv.org/master/db/de0/group__core__utils.html#ga705441a9ef01f47acdc55d87fbe5090c
///
pub fn getTickFrequency() f64 {
    return c.GetTickFrequency();
}

test "core" {
    std.testing.refAllDecls(@This());
    _ = BorderType;
    _ = Point;
    _ = PointVector;
    _ = PointsVector;
    _ = Point2f;
    _ = Point2fVector;
    _ = Points2fVector;
    _ = Point3f;
    _ = Point3fVector;
    _ = Points3fVector;
    _ = Rect;
    _ = Size;
    _ = TermCriteria;
}

//*    implementation done ("i" is internal function so we don't write zig wrappers for them)
//i    pub extern fn MultiDMatches_get(mds: struct_MultiDMatches, index: c_int) struct_DMatches;
//i    pub extern fn toByteArray(buf: [*c]const u8, len: c_int) struct_ByteArray;
//i    pub extern fn ByteArray_Release(buf: struct_ByteArray) void;
//i    pub extern fn Contours_Close(cs: struct_Contours) void;
//i    pub extern fn KeyPoints_Close(ks: struct_KeyPoints) void;
//*    pub extern fn Rects_Close(rs: struct_Rects) void;
//i    pub extern fn Point_Close(p: struct_Point) void;
//i    pub extern fn Points_Close(ps: struct_Points) void;
//i    pub extern fn DMatches_Close(ds: struct_DMatches) void;
//i    pub extern fn MultiDMatches_Close(mds: struct_MultiDMatches) void;
//*    pub extern fn TermCriteria_New(typ: c_int, maxCount: c_int, epsilon: f64) TermCriteria;
//*    pub extern fn GetCVTickCount(...) i64;
//*    pub extern fn GetTickFrequency(...) f64;
//*    pub extern fn PointVector_New(...) PointVector;
//*    pub extern fn PointVector_NewFromPoints(points: Contour) PointVector;
//*    pub extern fn PointVector_NewFromMat(mat: Mat) PointVector;
//*    pub extern fn PointVector_At(pv: PointVector, idx: c_int) Point;
//*    pub extern fn PointVector_Append(pv: PointVector, p: Point) void;
//*    pub extern fn PointVector_Size(pv: PointVector) c_int;
//*    pub extern fn PointVector_Close(pv: PointVector) void;
//*    pub extern fn PointsVector_New(...) PointsVector;
//*    pub extern fn PointsVector_NewFromPoints(points: Contours) PointsVector;
//*    pub extern fn PointsVector_At(psv: PointsVector, idx: c_int) PointVector;
//*    pub extern fn PointsVector_Append(psv: PointsVector, pv: PointVector) void;
//*    pub extern fn PointsVector_Size(psv: PointsVector) c_int;
//*    pub extern fn PointsVector_Close(psv: PointsVector) void;
//*    pub extern fn Point2fVector_New(...) Point2fVector;
//*    pub extern fn Point2fVector_Close(pfv: Point2fVector) void;
//*    pub extern fn Point2fVector_NewFromPoints(pts: Contour2f) Point2fVector;
//*    pub extern fn Point2fVector_NewFromMat(mat: Mat) Point2fVector;
//*    pub extern fn Point2fVector_At(pfv: Point2fVector, idx: c_int) Point2f;
//*    pub extern fn Point2fVector_Size(pfv: Point2fVector) c_int;
//i    pub extern fn IntVector_Close(ivec: struct_IntVector) void;
//i    pub extern fn CStrings_Close(cstrs: struct_CStrings) void;
//*    pub extern fn TheRNG(...) RNG;
//*    pub extern fn SetRNGSeed(seed: c_int) void;
//*    pub extern fn RNG_Fill(rng: RNG, mat: Mat, distType: c_int, a: f64, b: f64, saturateRange: bool) void;
//*    pub extern fn RNG_Gaussian(rng: RNG, sigma: f64) f64;
//*    pub extern fn RNG_Next(rng: RNG) c_uint;
//*    pub extern fn RandN(mat: Mat, mean: Scalar, stddev: Scalar) void;
//*    pub extern fn RandShuffle(mat: Mat) void;
//*    pub extern fn RandShuffleWithParams(mat: Mat, iterFactor: f64, rng: RNG) void;
//*    pub extern fn RandU(mat: Mat, low: Scalar, high: Scalar) void;
//i    pub extern fn copyPointVectorToPoint2fVector(src: PointVector, dest: Point2fVector) void;
//*    pub extern fn StdByteVectorInitialize(data: ?*anyopaque) void;
//*    pub extern fn StdByteVectorFree(data: ?*anyopaque) void;
//*    pub extern fn StdByteVectorLen(data: ?*anyopaque) usize;
//*    pub extern fn StdByteVectorData(data: ?*anyopaque) [*c]u8;
//*    pub extern fn Points2fVector_New(...) Points2fVector;
//*    pub extern fn Points2fVector_NewFromPoints(points: Contours2f) Points2fVector;
//*    pub extern fn Points2fVector_Size(ps: Points2fVector) c_int;
//*    pub extern fn Points2fVector_At(ps: Points2fVector, idx: c_int) Point2fVector;
//*    pub extern fn Points2fVector_Append(psv: Points2fVector, pv: Point2fVector) void;
//*    pub extern fn Points2fVector_Close(ps: Points2fVector) void;
//*    pub extern fn Point3fVector_New(...) Point3fVector;
//*    pub extern fn Point3fVector_NewFromPoints(points: Contour3f) Point3fVector;
//*    pub extern fn Point3fVector_NewFromMat(mat: Mat) Point3fVector;
//*    pub extern fn Point3fVector_Append(pfv: Point3fVector, point: Point3f) void;
//*    pub extern fn Point3fVector_At(pfv: Point3fVector, idx: c_int) Point3f;
//*    pub extern fn Point3fVector_Size(pfv: Point3fVector) c_int;
//*    pub extern fn Point3fVector_Close(pv: Point3fVector) void;
//*    pub extern fn Points3fVector_New(...) Points3fVector;
//*    pub extern fn Points3fVector_NewFromPoints(points: Contours3f) Points3fVector;
//*    pub extern fn Points3fVector_Size(ps: Points3fVector) c_int;
//*    pub extern fn Points3fVector_At(ps: Points3fVector, idx: c_int) Point3fVector;
//*    pub extern fn Points3fVector_Append(psv: Points3fVector, pv: Point3fVector) void;
//*    pub extern fn Points3fVector_Close(ps: Points3fVector) void;
