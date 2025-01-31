const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const Mat = core.Mat;

pub const VideoCapture = struct {
    ptr: c.VideoCapture,

    const Self = @This();

    /// Select preferred API for a capture object.
    /// Note: Backends are available only if they have been built with your OpenCV binaries
    pub const API = enum(u30) {
        /// Auto detect == 0
        any = 0,

        /// V4L/V4L2 capturing support
        v4l = 200,

        /// IEEE 1394/Firewire drivers
        firewire = 300,

        /// QuickTime (obsolete, removed)
        qt = 500,

        /// Unicap drivers (obsolete, removed)
        unicap = 600,

        /// DirectShow (via videoInput)
        d_show = 700,

        /// PvAPI, Prosilica GigE SDK
        pv_api = 800,

        /// OpenNI (for Kinect)
        open_ni = 900,

        /// OpenNI (for Asus Xtion)
        open_ni_asus = 910,

        /// Android - not used
        android = 1000,

        /// XIMEA Camera API
        xi_api = 1100,

        /// AVFoundation framework for iOS (OS X Lion will have the same API)
        av_foundation = 1200,

        /// Smartek Giganetix GigEVisionSDK
        gigabetix = 1300,

        /// Microsoft Media Foundation (via videoInput)
        msmf = 1400,

        /// Microsoft Windows Runtime using Media Foundation
        win_rt = 1410,

        /// RealSense (former Intel Perceptual Computing SDK)
        intel_perc = 1500,

        /// OpenNI2 (for Kinect)
        open_ni2 = 1600,

        /// OpenNI2 (for Asus Xtion and Occipital Structure sensors)
        open_ni2_asus = 1610,

        /// gPhoto2 connection
        g_photo2 = 1700,

        /// GStreamer
        g_streamer = 1800,

        /// Open and record video file or stream using the FFMPEG library
        ffmpeg = 1900,

        /// OpenCV Image Sequence (e.g. img_%02d.jpg)
        images = 2000,

        /// Aravis SDK
        aravis = 2100,

        /// Built-in OpenCV MotionJPEG codec
        opencv_mjpeg = 2200,

        /// Intel MediaSDK
        intel_mfx = 2300,

        /// XINE engine (Linux)
        xine = 2400,
    };

    /// VideoCaptureProperties are the properties used for VideoCapture operations.
    pub const Properties = enum(u6) {
        /// VideoCapturePosMsec contains current position of the
        /// video file in milliseconds.
        pos_msec = 0,

        /// VideoCapturePosFrames 0-based index of the frame to be
        /// decoded/captured next.
        pos_frames = 1,

        /// VideoCapturePosAVIRatio relative position of the video file:
        /// 0=start of the film, 1=end of the film.
        pos_avi_ratio = 2,

        /// VideoCaptureFrameWidth is width of the frames in the video stream.
        frame_width = 3,

        /// VideoCaptureFrameHeight controls height of frames in the video stream.
        frame_height = 4,

        /// VideoCaptureFPS controls capture frame rate.
        fps = 5,

        /// VideoCaptureFOURCC contains the 4-character code of codec.
        /// see VideoWriter::fourcc for details.
        fourcc = 6,

        /// VideoCaptureFrameCount contains number of frames in the video file.
        frame_count = 7,

        /// VideoCaptureFormat format of the Mat objects returned by
        /// VideoCapture::retrieve().
        format = 8,

        /// VideoCaptureMode contains backend-specific value indicating
        /// the current capture mode.
        mode = 9,

        /// VideoCaptureBrightness is brightness of the image
        /// (only for those cameras that support).
        brightness = 10,

        /// VideoCaptureContrast is contrast of the image
        /// (only for cameras that support it).
        contrast = 11,

        /// VideoCaptureSaturation saturation of the image
        /// (only for cameras that support).
        saturation = 12,

        /// VideoCaptureHue hue of the image (only for cameras that support).
        hue = 13,

        /// VideoCaptureGain is the gain of the capture image.
        /// (only for those cameras that support).
        gain = 14,

        /// VideoCaptureExposure is the exposure of the capture image.
        /// (only for those cameras that support).
        exposure = 15,

        /// VideoCaptureConvertRGB is a boolean flags indicating whether
        /// images should be converted to RGB.
        convert_rgb = 16,

        /// VideoCaptureWhiteBalanceBlueU is currently unsupported.
        white_balance_blue_u = 17,

        /// VideoCaptureRectification is the rectification flag for stereo cameras.
        /// Note: only supported by DC1394 v 2.x backend currently.
        rectification = 18,

        /// VideoCaptureMonochrome indicates whether images should be
        /// converted to monochrome.
        monochrome = 19,

        /// VideoCaptureSharpness controls image capture sharpness.
        sharpness = 20,

        /// VideoCaptureAutoExposure controls the DC1394 exposure control
        // done by camera, user can adjust reference level using this feature.
        auto_exposure = 21,

        /// VideoCaptureGamma controls video capture gamma.
        gamma = 22,

        /// VideoCaptureTemperature controls video capture temperature.
        temperature = 23,

        /// VideoCaptureTrigger controls video capture trigger.
        trigger = 24,

        /// VideoCaptureTriggerDelay controls video capture trigger delay.
        trigger_delay = 25,

        /// VideoCaptureWhiteBalanceRedV controls video capture setting for
        // white balance.
        white_balance_red_v = 26,

        /// VideoCaptureZoom controls video capture zoom.
        zoom = 27,

        /// VideoCaptureFocus controls video capture focus.
        focus = 28,

        /// VideoCaptureGUID controls video capture GUID.
        guid = 29,

        /// VideoCaptureISOSpeed controls video capture ISO speed.
        iso_speed = 30,

        /// VideoCaptureBacklight controls video capture backlight.
        backlight = 32,

        /// VideoCapturePan controls video capture pan.
        pan = 33,

        /// VideoCaptureTilt controls video capture tilt.
        tilt = 34,

        /// VideoCaptureRoll controls video capture roll.
        roll = 35,

        /// VideoCaptureIris controls video capture iris.
        iris = 36,

        /// VideoCaptureSettings is the pop up video/camera filter dialog. Note:
        /// only supported by DSHOW backend currently. The property value is ignored.
        settings = 37,

        /// VideoCaptureBufferSize controls video capture buffer size.
        buffer_size = 38,

        /// VideoCaptureAutoFocus controls video capture auto focus..
        auto_focus = 39,

        /// VideoCaptureSarNumerator controls the sample aspect ratio: num/den (num)
        sar_numerator = 40,

        /// VideoCaptureSarDenominator controls the sample aspect ratio: num/den (den)
        sar_denominator = 41,

        /// VideoCaptureBackend is the current api backend (VideoCaptureAPI). Read-only property.
        backend = 42,

        /// VideoCaptureChannel controls the video input or channel number (only for those cameras that support).
        channel = 43,

        /// VideoCaptureAutoWB controls the auto white-balance.
        auto_wb = 44,

        /// VideoCaptureWBTemperature controls the white-balance color temperature
        wb_temperature = 45,

        /// VideoCaptureCodecPixelFormat shows the the codec's pixel format (4-character code). Read-only property.
        /// Subset of AV_PIX_FMT_* or -1 if unknown.
        codec_pixel_format = 46,

        /// VideoCaptureBitrate displays the video bitrate in kbits/s. Read-only property.
        bitrate = 47,
    };

    pub fn init() !Self {
        const ptr = c.VideoCapture_New();
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.VideoCapture) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.VideoCapture_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn captureFile(self: *Self, uri: []const u8) !void {
        if (!c.VideoCapture_Open(self.ptr, @ptrCast(uri))) {
            return error.VideoCaptureOpenFileError;
        }
    }

    pub fn captureFileWithAPI(self: *Self, uri: []const u8, api_preference: API) !void {
        if (!c.VideoCapture_OpenWithAPI(self.ptr, @ptrCast(uri), @intFromEnum(api_preference))) {
            return error.VideoCaptureOpenFileError;
        }
    }

    pub fn openDevice(self: *Self, device: i32) !void {
        if (!c.VideoCapture_OpenDevice(self.ptr, device)) {
            return error.VideoCaptureOpenDeviceError;
        }
    }

    pub fn openDeviceWithAPI(self: *Self, device: i32, api_preference: API) !void {
        if (!c.VideoCapture_OpenDeviceWithAPI(self.ptr, device, @intFromEnum(api_preference))) {
            return error.VideoCaptureOpenDeviceError;
        }
    }

    pub fn get(self: Self, prop: Properties) f64 {
        return c.VideoCapture_Get(self.ptr, @intFromEnum(prop));
    }

    pub fn set(self: *Self, prop: Properties, param: f64) void {
        return c.VideoCapture_Set(self.ptr, @intFromEnum(prop), param);
    }

    pub fn grab(self: Self, skip: i32) void {
        c.VideoCapture_Grab(self.ptr, skip);
    }

    pub fn read(self: Self, buf: *Mat) !void {
        if (c.VideoCapture_Read(self.ptr, buf.*.ptr) == 0) {
            return error.VideCaptureError;
        }
    }

    const ConvertStruct = packed struct {
        c0: u8,
        c1: u8,
        c2: u8,
        c3: u8,
    };

    /// returns a string representation of FourCC bytes, i.e. the name of a codec
    pub fn getCodecString(self: Self) []const u8 {
        const fourcc_f = get(self, .fourcc);
        const fourcc = @as(u32, @intFromFloat(fourcc_f));
        const ps_fourcc = @as(ConvertStruct, @bitCast(fourcc));
        const result =
            [_]u8{
            ps_fourcc.c0,
            ps_fourcc.c1,
            ps_fourcc.c2,
            ps_fourcc.c3,
        };
        return &result;
    }

    /// ToCodec returns an float64 representation of FourCC bytes
    pub fn toCodec(self: Self, codec: []const u8) !f64 {
        _ = self;
        if (codec.len != 4) {
            return error.InvalidCodec;
        }
        const ps_fourcc = ConvertStruct{
            .c0 = codec[0],
            .c1 = codec[1],
            .c2 = codec[2],
            .c3 = codec[3],
        };
        const u_fourcc = @as(u32, @bitCast(ps_fourcc));
        return @as(f64, @floatFromInt(u_fourcc));
    }

    pub fn isOpened(self: Self) bool {
        return c.VideoCapture_IsOpened(self.ptr) != 0;
    }
};

pub const VideoWriter = struct {
    ptr: c.VideoWriter,

    const Self = @This();

    pub fn init() !Self {
        const ptr = c.VideoWriter_New();
        return try initFromC(ptr);
    }

    pub fn initFromC(ptr: c.VideoWriter) !Self {
        const nn_ptr = try epnn(ptr);
        return Self{ .ptr = nn_ptr };
    }

    pub fn deinit(self: *Self) void {
        assert(self.ptr != null);
        c.VideoWriter_Close(self.ptr);
        self.*.ptr = null;
    }

    pub fn open(
        self: *Self,
        name: []const u8,
        codec: []const u8,
        fps: f64,
        width: i32,
        height: i32,
        is_color: bool,
    ) void {
        const c_name = @as([*]const u8, @ptrCast(name));
        const c_codec = @as([*]const u8, @ptrCast(codec));
        _ = c.VideoWriter_Open(
            self.ptr,
            c_name,
            c_codec,
            fps,
            width,
            height,
            is_color,
        );
    }

    pub fn write(self: *Self, img: *Mat) !void {
        _ = try epnn(img.*.ptr);
        _ = c.VideoWriter_Write(self.ptr, img.*.toC());
    }

    pub fn isOpened(self: *Self) bool {
        return c.VideoWriter_IsOpened(self.ptr) != 0;
    }
};

const video_path = "test/images/small.mp4";
const imgcodecs = @import("imgcodecs.zig");
var out_buffer: [std.fs.max_path_bytes]u8 = undefined;

test "videoio VideoCapture captureFile" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    try vc.captureFile(video_path);
    try std.testing.expectEqual(true, vc.isOpened());

    try std.testing.expectEqual(@as(f64, 560), vc.get(.frame_width));
    try std.testing.expectEqual(@as(f64, 320), vc.get(.frame_height));

    vc.grab(10);
    vc.set(.brightness, 100);

    var img = try Mat.init();
    defer img.deinit();

    try vc.read(&img);
    try std.testing.expectEqual(false, img.isEmpty());
}

test "videoio VideoCapture captureFileWithAPI" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    try vc.captureFileWithAPI(video_path, .any);

    const backend = vc.get(.backend);
    try std.testing.expect(@as(f64, @intFromEnum(VideoCapture.API.any)) != backend);
}

test "videoio VideoCapture captureFile invalid file" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    const e = vc.captureFile("not-exist-path/" ++ video_path);
    try std.testing.expectError(error.VideoCaptureOpenFileError, e);
}

test "videoio VideoCapture captureFileWithAPI invalid file" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    const e = vc.captureFileWithAPI("not-exist-path/" ++ video_path, .any);
    try std.testing.expectError(error.VideoCaptureOpenFileError, e);
}

test "videoio VideoCapture openDevice unknown error" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    const e = vc.openDevice(std.math.maxInt(i32));
    try std.testing.expectError(error.VideoCaptureOpenDeviceError, e);
}

test "videoio VideoCapture openDeviceWithAPI unknown error" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    const e = vc.openDeviceWithAPI(std.math.maxInt(i32), .any);
    try std.testing.expectError(error.VideoCaptureOpenDeviceError, e);
}

test "videoio VideoCapture getCodecString" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    try vc.captureFile(video_path);
    const res = vc.getCodecString();
    try std.testing.expect(!std.mem.eql(u8, "", res));
}

test "videoio VideoCapture toCodec" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    try vc.captureFile(video_path);
    const codec = vc.getCodecString();
    const r_codec = try vc.toCodec(codec);
    try std.testing.expectEqual(vc.get(.fourcc), r_codec);
}

test "videoio VideoCapture toCodec failed" {
    var vc = try VideoCapture.init();
    defer vc.deinit();
    const e = vc.toCodec("123");
    try std.testing.expectError(error.InvalidCodec, e);
}

test "videoio VideoWriter" {
    var img = try imgcodecs.imRead("test/images/face-detect.jpg", .color);
    defer img.deinit();
    try std.testing.expectEqual(false, img.isEmpty());

    var vw = try VideoWriter.init();
    vw.open(".zig-cache/tmp/test_video_write.avi", "MJPG", 25, img.cols(), img.rows(), true);
    defer vw.deinit();

    try std.testing.expectEqual(true, vw.isOpened());

    try vw.write(&img);
}

//*    implementation done
//*    pub const VideoCapture = ?*anyopaque;
//*    pub const VideoWriter = ?*anyopaque;
//*    pub extern fn VideoCapture_New(...) VideoCapture;
//*    pub extern fn VideoCapture_Close(v: VideoCapture) void;
//*    pub extern fn VideoCapture_Open(v: VideoCapture, uri: [*c]const u8) bool;
//*    pub extern fn VideoCapture_OpenWithAPI(v: VideoCapture, uri: [*c]const u8, apiPreference: c_int) bool;
//*    pub extern fn VideoCapture_OpenDevice(v: VideoCapture, device: c_int) bool;
//*    pub extern fn VideoCapture_OpenDeviceWithAPI(v: VideoCapture, device: c_int, apiPreference: c_int) bool;
//*    pub extern fn VideoCapture_Set(v: VideoCapture, prop: c_int, param: f64) void;
//*    pub extern fn VideoCapture_Get(v: VideoCapture, prop: c_int) f64;
//*    pub extern fn VideoCapture_IsOpened(v: VideoCapture) c_int;
//*    pub extern fn VideoCapture_Read(v: VideoCapture, buf: Mat) c_int;
//*    pub extern fn VideoCapture_Grab(v: VideoCapture, skip: c_int) void;
//*    pub extern fn VideoWriter_New(...) VideoWriter;
//*    pub extern fn VideoWriter_Close(vw: VideoWriter) void;
//*    pub extern fn VideoWriter_Open(vw: VideoWriter, name: [*c]const u8, codec: [*c]const u8, fps: f64, width: c_int, height: c_int, isColor: bool) void;
//*    pub extern fn VideoWriter_IsOpened(vw: VideoWriter) c_int;
//*    pub extern fn VideoWriter_Write(vw: VideoWriter, img: Mat) void;
