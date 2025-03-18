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

/// CascadeClassifier is a cascade classifier class for object detection.
///
/// For further details, please see:
/// http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html
///
pub const CascadeClassifier = struct {
    ptr: c.CascadeClassifier,

    const Self = @This();

    pub fn init() !Self {
        var ptr = c.CascadeClassifier_New();
        ptr = try epnn(ptr);
        return Self{ .ptr = ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        _ = c.CascadeClassifier_Close(self.ptr);
        self.*.ptr = null;
    }

    /// Load cascade classifier from a file.
    ///
    /// For further details, please see:
    /// http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#a1a5884c8cc749422f9eb77c2471958bc
    pub fn load(self: *Self, name: []const u8) !void {
        const result = c.CascadeClassifier_Load(self.ptr, @as([*]const u8, @ptrCast(name)));
        if (result == 0) {
            return error.CascadeClassifierLoadFailed;
        }
    }

    /// DetectMultiScale detects objects of different sizes in the input Mat image.
    /// The detected objects are returned as a slice of image.Rectangle structs.
    ///
    /// For further details, please see:
    /// http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    ///
    pub fn detectMultiScale(self: Self, img: Mat, allocator: std.mem.Allocator) !Rects {
        const rec: c.struct_Rects = c.CascadeClassifier_DetectMultiScale(self.ptr, img.ptr);
        defer Rect.deinitRects(rec);
        return try Rect.toArrayList(rec, allocator);
    }

    /// DetectMultiScaleWithParams calls DetectMultiScale but allows setting parameters
    /// to values other than just the defaults.
    ///
    /// For further details, please see:
    /// http://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    ///
    pub fn detectMultiScaleWithParams(
        self: Self,
        img: Mat,
        scale: f64,
        min_neighbors: i32,
        flags: i32,
        min_size: Size,
        max_size: Size,
        allocator: std.mem.Allocator,
    ) !Rects {
        const rec: c.struct_Rects = c.CascadeClassifier_DetectMultiScaleWithParams(
            self.ptr,
            img.ptr,
            scale,
            min_neighbors,
            flags,
            min_size.toC(),
            max_size.toC(),
        );
        defer Rect.deinitRects(rec);
        return try Rect.toArrayList(rec, allocator);
    }
};

/// HOGDescriptor is a Histogram Of Gradiants (HOG) for object detection.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a723b95b709cfd3f95cf9e616de988fc8
///
pub const HOGDescriptor = struct {
    ptr: c.HOGDescriptor,

    const Self = @This();

    pub fn init() !Self {
        var ptr = c.HOGDescriptor_New();
        ptr = try epnn(ptr);
        return .{ .ptr = ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.HOGDescriptor_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn load(self: *Self, name: []const u8) !void {
        const result = c.HOGDescriptor_Load(self.ptr, utils.castZigU8ToC(name));
        if (result == 0) {
            return error.HOGDescriptorLoadFailed;
        }
    }

    /// DetectMultiScale detects objects in the input Mat image.
    /// The detected objects are returned as a slice of image.Rectangle structs.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a660e5cd036fd5ddf0f5767b352acd948
    ///
    pub fn detectMultiScale(self: Self, img: Mat, allocator: std.mem.Allocator) !Rects {
        const rec: c.struct_Rects = c.HOGDescriptor_DetectMultiScale(self.ptr, img.ptr);
        defer Rect.deinitRects(rec);
        return try Rect.toArrayList(rec, allocator);
    }

    /// DetectMultiScaleWithParams calls DetectMultiScale but allows setting parameters
    /// to values other than just the defaults.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a660e5cd036fd5ddf0f5767b352acd948
    ///
    pub fn detectMultiScaleWithParams(
        self: Self,
        img: Mat,
        hit_thresh: f64,
        win_stride: Size,
        padding: Size,
        scale: f64,
        final_threshold: f64,
        use_meanshift_grouping: bool,
        allocator: std.mem.Allocator,
    ) !Rects {
        const rec: c.struct_Rects = c.HOGDescriptor_DetectMultiScaleWithParams(
            self.ptr,
            img.ptr,
            hit_thresh,
            win_stride.toC(),
            padding.toC(),
            scale,
            final_threshold,
            use_meanshift_grouping,
        );
        defer Rect.deinitRects(rec);
        return try Rect.toArrayList(rec, allocator);
    }

    /// HOGDefaultPeopleDetector returns a new Mat with the HOG DefaultPeopleDetector.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a660e5cd036fd5ddf0f5767b352acd948
    ///
    pub fn getDefaultPeopleDetector() !Mat {
        return Mat.initFromC(c.HOG_GetDefaultPeopleDetector());
    }

    /// SetSVMDetector sets the data for the HOGDescriptor.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a09e354ad701f56f9c550dc0385dc36f1
    ///
    pub fn setSVMDetector(self: Self, det: Mat) void {
        _ = c.HOGDescriptor_SetSVMDetector(self.ptr, det.ptr);
    }
};

/// GroupRectangles groups the object candidate rectangles.
///
/// For further details, please see:
/// https://docs.opencv.org/4.x/de/de1/group__objdetect__common.html#ga3dba897ade8aa8227edda66508e16ab9
///
pub fn groupRectangles(rects: []const Rect, group_threshold: i32, eps: f64, allocator: std.mem.Allocator) !Rects {
    if (group_threshold < -1) return error.InvalidGroupThreshold;

    var c_rects_array = try allocator.alloc(c.Rect, rects.len);
    defer allocator.free(c_rects_array);
    for (rects, 0..) |rect, i| c_rects_array[i] = rect.toC();
    const c_rects = c.Rects{
        .rects = @as([*]c.Rect, @ptrCast(c_rects_array.ptr)),
        .length = @as(i32, @intCast(c_rects_array.len)),
    };

    const result_c_rects = c.GroupRectangles(c_rects, group_threshold, eps);
    defer Rect.deinitRects(result_c_rects);

    return Rect.toArrayList(result_c_rects, allocator);
}

/// QRCodeDetector groups the object candidate rectangles.
///
/// For further details, please see:
/// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html
///
pub const QRCodeDetector = struct {
    ptr: c.QRCodeDetector,

    const Self = @This();

    pub fn init() !Self {
        var ptr = c.QRCodeDetector_New();
        ptr = try epnn(ptr);
        return .{ .ptr = ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.QRCodeDetector_Close(self.ptr);
        self.*.ptr = null;
    }

    /// DetectAndDecode Both detects and decodes QR code.
    ///
    /// Returns true as long as some QR code was detected even in case where the decoding failed
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a7290bd6a5d59b14a37979c3a14fbf394
    ///
    pub fn detectAndDecode(self: Self, input: Mat, points: *Mat, straight_qrcode: *Mat) []const u8 {
        const result = c.QRCodeDetector_DetectAndDecode(self.ptr, input.ptr, points.*.ptr, straight_qrcode.*.ptr);
        return std.mem.span(result);
    }

    /// Detect detects QR code in image and returns the quadrangle containing the code.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a64373f7d877d27473f64fe04bb57d22b
    ///
    pub fn detect(self: Self, input: Mat, points: *Mat) bool {
        return c.QRCodeDetector_Detect(self.ptr, input.ptr, points.*.ptr);
    }

    /// Decode decodes QR code in image once it's found by the detect() method. Returns UTF8-encoded output string or empty string if the code cannot be decoded.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a4172c2eb4825c844fb1b0ae67202d329
    ///
    pub fn decode(self: Self, input: Mat, points: Mat, straight_qrcode: *Mat) []const u8 {
        const result = c.QRCodeDetector_Decode(self.ptr, input.ptr, points.ptr, straight_qrcode.*.ptr);
        return std.mem.span(result);
    }

    /// Detects QR codes in image and finds of the quadrangles containing the codes.
    ///
    /// Each quadrangle would be returned as a row in the `points` Mat and each point is a Vecf.
    /// Returns true if QR code was detected
    /// For usage please see TestQRCodeDetector
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#aaf2b6b2115b8e8fbc9acf3a8f68872b6
    pub fn detectMulti(self: Self, input: Mat, points: *Mat) bool {
        return c.QRCodeDetector_DetectMulti(self.ptr, input.ptr, points.*.ptr);
    }

    /// Decode decodes QR code in image once it's found by the detect() method. Returns UTF8-encoded output string or empty string if the code cannot be decoded.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/de/dc3/classcv_1_1QRCodeDetector.html#a4172c2eb4825c844fb1b0ae67202d329
    ///
    pub fn detectAndDecodeMulti(
        self: Self,
        input: Mat,
        allocator: std.mem.Allocator,
    ) !struct {
        is_detected: bool,
        decoded: []const []const u8,
        qr_codes: []Mat,
        points: Mat,
        arena: std.heap.ArenaAllocator,

        pub fn deinit(self_: *@This()) void {
            self_.arena.deinit();
        }
    } {
        var arena = std.heap.ArenaAllocator.init(allocator);
        var arena_allocator = arena.allocator();

        var c_decoded: c.CStrings = undefined;
        var c_qr_codes: c.Mats = undefined;

        defer c.CStrings_Close(c_decoded);

        var points = try Mat.init();

        const result = c.QRCodeDetector_DetectAndDecodeMulti(
            self.ptr,
            input.toC(),
            &c_decoded,
            points.toC(),
            &c_qr_codes,
        );

        const decoded = try arena_allocator.alloc([]const u8, @as(usize, @intCast(c_decoded.length)));
        const qr_codes = try arena_allocator.alloc(Mat, @as(usize, @intCast(c_qr_codes.length)));

        if (result) {
            for (decoded, 0..) |*item, i| {
                item.* = try arena_allocator.dupe(u8, std.mem.span(c_decoded.strs[i]));
            }

            for (qr_codes, 0..) |*item, i| {
                item.* = try Mat.initFromC(c_qr_codes.mats[i]);
            }
        }

        return .{
            .is_detected = result,
            .decoded = decoded,
            .qr_codes = qr_codes,
            .points = points,
            .arena = arena,
        };
    }
};

const testing = std.testing;
const imgcodecs = @import("imgcodecs.zig");

// test "objdetect CascadeClassifier" {
// var img = try imgcodecs.imRead("test/images/face.jpg", .color);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var classifier = try CascadeClassifier.init();
// defer classifier.deinit();

// try classifier.load("test/data/haarcascade_frontalface_default.xml");

// var rects = try classifier.detectMultiScale(img, testing.allocator);
// defer rects.deinit();

// try testing.expectEqual(@as(usize, 1), rects.items.len);
// }

// test "objdetect CascadeClassifierWithParams" {
// var img = try imgcodecs.imRead("test/images/face.jpg", .color);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var classifier = try CascadeClassifier.init();
// defer classifier.deinit();

// try classifier.load("test/data/haarcascade_frontalface_default.xml");

// var rects = try classifier.detectMultiScaleWithParams(img, 1.1, 3, 0, Size.init(0, 0), Size.init(0, 0), testing.allocator);
// defer rects.deinit();

// try testing.expectEqual(@as(usize, 1), rects.items.len);
// }

// test "objdetect HOGDescriptor" {
// var img = try imgcodecs.imRead("test/images/face.jpg", .color);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var hog = try HOGDescriptor.init();
// defer hog.deinit();

// var d: Mat = try HOGDescriptor.getDefaultPeopleDetector();
// defer d.deinit();
// hog.setSVMDetector(d);

// var rects = try hog.detectMultiScale(img, testing.allocator);
// defer rects.deinit();

// try testing.expectEqual(@as(usize, 1), rects.items.len);
// }

// test "objdetect HOGDescriptorWithParams" {
// var img = try imgcodecs.imRead("test/images/face.jpg", .color);
// defer img.deinit();
// try testing.expectEqual(false, img.isEmpty());

// var hog = try HOGDescriptor.init();
// defer hog.deinit();

// var d: Mat = try HOGDescriptor.getDefaultPeopleDetector();
// defer d.deinit();
// hog.setSVMDetector(d);

// var rects = try hog.detectMultiScaleWithParams(
// img,
// 0,
// Size.init(0, 0),
// Size.init(0, 0),
// 1.05,
// 2.0,
// false,
// testing.allocator,
// );
// defer rects.deinit();

// try testing.expectEqual(@as(usize, 1), rects.items.len);
// }

test "objdetect groupRectangles" {
    var rects = [_]Rect{
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 20, 20),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
        Rect.init(10, 10, 25, 25),
    };

    var results = try groupRectangles(rects[0..], 1, 0.1, testing.allocator);
    defer results.deinit();
    try testing.expectEqual(@as(usize, 2), results.items.len);
}

// test "objdetect QRCodeDetector" {
// var img = try imgcodecs.imRead("test/images/qrcode.png", .color);
// try testing.expectEqual(false, img.isEmpty());
// defer img.deinit();

// var detector = try QRCodeDetector.init();
// defer detector.deinit();

// var bbox = try Mat.init();
// defer bbox.deinit();
// var qr = try Mat.init();
// defer qr.deinit();

// const res = detector.detect(img, &bbox);
// try testing.expectEqual(true, res);

// const res2 = detector.decode(img, bbox, &qr);
// const res3 = detector.detectAndDecode(img, &bbox, &qr);

// try testing.expectEqualStrings(res2, res3);
// }

// test "objdetect Multi QRCodeDetector" {
// var img = try imgcodecs.imRead("test/images/multi_qrcodes.png", .color);
// try testing.expectEqual(false, img.isEmpty());
// defer img.deinit();

// var detector = try QRCodeDetector.init();
// defer detector.deinit();

// var mbox = try Mat.init();
// defer mbox.deinit();
// var qr = try Mat.init();
// defer qr.deinit();

// const res = detector.detectMulti(img, &mbox);
// try testing.expectEqual(true, res);
// try testing.expectEqual(@as(i32, 2), mbox.rows());

// var res2 = try detector.detectAndDecodeMulti(img, testing.allocator);
// defer res2.deinit();
// try testing.expectEqual(true, res2.is_detected);
// try testing.expectEqual(@as(usize, 2), res2.decoded.len);

// testing.expectEqualStrings("foo", res2.decoded[0]) catch {
// try testing.expectEqualStrings("bar", res2.decoded[0]);
// try testing.expectEqualStrings("foo", res2.decoded[1]);
// return;
// };
// try testing.expectEqualStrings("bar", res2.decoded[1]);
// }

//*    implementation done
//*    pub const CascadeClassifier = ?*anyopaque;
//*    pub const HOGDescriptor = ?*anyopaque;
//*    pub const QRCodeDetector = ?*anyopaque;
//*    pub extern fn CascadeClassifier_New(...) CascadeClassifier;
//*    pub extern fn CascadeClassifier_Close(cs: CascadeClassifier) void;
//*    pub extern fn CascadeClassifier_Load(cs: CascadeClassifier, name: [*c]const u8) c_int;
//*    pub extern fn CascadeClassifier_DetectMultiScale(cs: CascadeClassifier, img: Mat) struct_Rects;
//*    pub extern fn CascadeClassifier_DetectMultiScaleWithParams(cs: CascadeClassifier, img: Mat, scale: f64, minNeighbors: c_int, flags: c_int, minSize: Size, maxSize: Size) struct_Rects;
//*    pub extern fn HOGDescriptor_New(...) HOGDescriptor;
//*    pub extern fn HOGDescriptor_Close(hog: HOGDescriptor) void;
//*    pub extern fn HOGDescriptor_Load(hog: HOGDescriptor, name: [*c]const u8) c_int;
//*    pub extern fn HOGDescriptor_DetectMultiScale(hog: HOGDescriptor, img: Mat) struct_Rects;
//*    pub extern fn HOGDescriptor_DetectMultiScaleWithParams(hog: HOGDescriptor, img: Mat, hitThresh: f64, winStride: Size, padding: Size, scale: f64, finalThreshold: f64, useMeanshiftGrouping: bool) struct_Rects;
//*    pub extern fn HOG_GetDefaultPeopleDetector(...) Mat;
//*    pub extern fn HOGDescriptor_SetSVMDetector(hog: HOGDescriptor, det: Mat) void;
//*    pub extern fn GroupRectangles(rects: struct_Rects, groupThreshold: c_int, eps: f64) struct_Rects;
//*    pub extern fn QRCodeDetector_New(...) QRCodeDetector;
//*    pub extern fn QRCodeDetector_DetectAndDecode(qr: QRCodeDetector, input: Mat, points: Mat, straight_qrcode: Mat) [*c]const u8;
//*    pub extern fn QRCodeDetector_Detect(qr: QRCodeDetector, input: Mat, points: Mat) bool;
//*    pub extern fn QRCodeDetector_Decode(qr: QRCodeDetector, input: Mat, inputPoints: Mat, straight_qrcode: Mat) [*c]const u8;
//*    pub extern fn QRCodeDetector_Close(qr: QRCodeDetector) void;
//*    pub extern fn QRCodeDetector_DetectMulti(qr: QRCodeDetector, input: Mat, points: Mat) bool;
//*    pub extern fn QRCodeDetector_DetectAndDecodeMulti(qr: QRCodeDetector, input: Mat, decoded: [*c]CStrings, points: Mat, mats: [*c]struct_Mats) bool;
