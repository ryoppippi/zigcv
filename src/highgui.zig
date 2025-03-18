const std = @import("std");
const c = @import("c_api.zig");
const core = @import("core.zig");
const utils = @import("utils.zig");
const Mat = core.Mat;
const Rect = core.Rect;
const Rects = core.Rects;

pub const Window = struct {
    name: []const u8,
    open: bool,
    trackbar: ?Trackbar,

    const Self = @This();

    pub const Flag = enum {
        /// WindowNormal indicates a normal window.
        normal,

        /// WindowAutosize indicates a window sized based on the contents.
        autosize,

        /// WindowFullscreen indicates a full-screen window.
        fullscreen,

        /// WindowFreeRatio indicates allow the user to resize without maintaining aspect ratio.
        free_ratio,

        /// WindowKeepRatio indicates always maintain an aspect ratio that matches the contents.
        keep_ratio,

        fn toNum(wp: PropertyFlag, wf: Flag, comptime T: type) T {
            if ((wp == .fullscreen and !(wf == .normal or wf == .fullscreen)) or
                (wp == .autosize and !(wf == .normal or wf == .autosize)) or
                (wp == .aspect_ratio and !(wf == .free_ratio or wf == .keep_ratio)))
            {
                @panic("invalid window property flag and window flag combination");
            }
            return switch (wf) {
                .normal => 0x00000000,
                .autosize => 0x00000001,
                .fullscreen => 1,
                .free_ratio => 0x00000100,
                .keep_ratio => 0x00000000,
            };
        }

        fn toEnum(wp: PropertyFlag, wf: anytype) Flag {
            const f: u32 = switch (@typeInfo(@TypeOf(wf))) {
                .int, .comptime_int => @intCast(wf),
                .float, .comptime_float => @intFromFloat(wf),
                else => unreachable,
            };
            return switch (f) {
                0x00000000 => blk: {
                    break :blk switch (wp) {
                        .fullscreen => .normal,
                        .autosize => .normal,
                        .aspect_ratio => .keep_ratio,
                        else => unreachable,
                    };
                },
                1 => blk: {
                    break :blk switch (wp) {
                        .fullscreen => .fullscreen,
                        .autosize => .autosize,
                        else => unreachable,
                    };
                },
                0x00000100 => .free_ratio,
                else => @panic("invalid number"),
            };
        }
    };

    pub const PropertyFlag = enum(u3) {
        // WindowPropertyFullscreen fullscreen property
        // (can be WINDOW_NORMAL or WINDOW_FULLSCREEN).
        fullscreen = 0,

        // WindowPropertyAutosize is autosize property
        // (can be WINDOW_NORMAL or WINDOW_AUTOSIZE).
        autosize = 1,

        // WindowPropertyAspectRatio window's aspect ration
        // (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO).
        aspect_ratio = 2,

        // WindowPropertyOpenGL opengl support.
        opengl = 3,

        // WindowPropertyVisible or not.
        visible = 4,

        // WindowPropertyTopMost status bar and tool bar
        top_most = 5,

        // WindowPropertyVSYNC enables or disables VSYNC (in OpenGL mode)
        vsync = 6,
    };

    fn getCWindowName(self: Self) [*]const u8 {
        return @as([*]const u8, @ptrCast(self.name));
    }

    pub fn init(window_name: []const u8) !Self {
        if (window_name.len == 0) return error.EmptyWindowName;
        c.Window_New(@as([*]const u8, @ptrCast(window_name)), 0);
        return Self{
            .name = window_name,
            .open = true,
            .trackbar = null,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = c.Window_Close(self.getCWindowName());
        self.open = false;
    }

    pub fn isOpened(self: Self) bool {
        return self.open;
    }

    /// SetWindowProperty changes parameters of a window dynamically.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga66e4a6db4d4e06148bcdfe0d70a5df27
    ///
    pub fn setProperty(self: *Self, flag: PropertyFlag, value: Flag) void {
        _ = c.Window_SetProperty(
            self.getCWindowName(),
            @intFromEnum(flag),
            Flag.toNum(flag, value, f64),
        );
    }

    /// GetWindowProperty returns properties of a window.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#gaaf9504b8f9cf19024d9d44a14e461656
    ///
    pub fn getProperty(self: Self, comptime flag: PropertyFlag) Flag {
        const wf: f64 = c.Window_GetProperty(
            self.getCWindowName(),
            @intFromEnum(flag),
        );
        const wpf = flag;
        return Flag.toEnum(wpf, wf);
    }

    /// SetWindowTitle updates window title.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga56f8849295fd10d0c319724ddb773d96
    ///
    pub fn setTitle(self: *Self, title: []const u8) void {
        _ = c.Window_SetTitle(self.getCWindowName(), @as([*]const u8, @ptrCast(title)));
        self.name = title;
    }

    /// WaitKey waits for a pressed key.
    /// This function is the only method in OpenCV's HighGUI that can fetch
    /// and handle events, so it needs to be called periodically
    /// for normal event processing
    ///
    /// For further details, please see:
    /// http://docs.opencv.org/master/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7
    ///
    pub fn waitKey(self: Self, delay: i32) i32 {
        _ = self;
        return c.Window_WaitKey(delay);
    }

    /// IMShow displays an image Mat in the specified window.
    /// This function should be followed by the WaitKey function which displays
    /// the image for specified milliseconds. Otherwise, it won't display the image.
    ///
    /// For further details, please see:
    /// http://docs.opencv.org/master/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563
    ///
    pub fn imShow(self: *Self, mat: core.Mat) void {
        _ = c.Window_IMShow(self.getCWindowName(), mat.ptr);
    }

    /// MoveWindow moves window to the specified position.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga8d86b207f7211250dbe6e28f76307ffb
    ///
    pub fn move(self: *Self, x: i32, y: i32) void {
        _ = c.Window_Move(self.getCWindowName(), x, y);
    }

    /// ResizeWindow resizes window to the specified size.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga9e80e080f7ef33f897e415358aee7f7e
    ///
    pub fn resize(self: *Self, width: i32, height: i32) void {
        _ = c.Window_Resize(self.getCWindowName(), width, height);
    }

    /// SelectROI selects a Region Of Interest (ROI) on the given image.
    /// It creates a window and allows user to select a ROI using mouse.
    ///
    /// Controls:
    /// use space or enter to finish selection,
    /// use key c to cancel selection (function will return a zero Rect).
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga8daf4730d3adf7035b6de9be4c469af5
    ///
    pub fn selectROI(self: *Self, img: Mat) Rect {
        const c_rects = c.Window_SelectROI(self.getCWindowName(), img.ptr);
        defer c.Rects_Close(c_rects);
        return Rect.fromC(c_rects);
    }

    /// SelectROIs selects multiple Regions Of Interest (ROI) on the given image.
    /// It creates a window and allows user to select ROIs using mouse.
    ///
    /// Controls:
    /// use space or enter to finish current selection and start a new one
    /// use esc to terminate multiple ROI selection process
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga0f11fad74a6432b8055fb21621a0f893
    ///
    pub fn selectROIs(self: *Self, img: Mat, allocator: std.mem.Allocator) !Rects {
        const c_rects: c.Rects = c.Window_SelectROIs(self.getCWindowName(), img.ptr);
        defer c.Rects_Close(c_rects);
        return try Rect.toArrayList(c_rects, allocator);
    }

    /// CreateTrackbar creates a trackbar and attaches it to the specified window.
    ///
    /// For further details, please see:
    /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
    ///
    pub fn createTrackbar(self: *Self, trackbar_name: []const u8, max: i32) void {
        self.trackbar = Trackbar.init(self.name, trackbar_name, max);
    }

    pub const Trackbar = struct {
        trackbar_name: []const u8,
        window_name: []const u8,

        /// CreateTrackbar creates a trackbar and attaches it to the specified window.
        ///
        /// For further details, please see:
        /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
        ///
        pub fn init(window_name: []const u8, trackbar_name: []const u8, max: i32) Trackbar {
            _ = c.Trackbar_Create(@as([*]const u8, @ptrCast(window_name)), @as([*]const u8, @ptrCast(trackbar_name)), max);
            return .{ .trackbar_name = trackbar_name, .window_name = window_name };
        }

        /// GetPos returns the trackbar position.
        ///
        /// For further details, please see:
        /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga122632e9e91b9ec06943472c55d9cda8
        ///
        pub fn getPos(self: Trackbar) i32 {
            return c.Trackbar_GetPos(@as([*]const u8, @ptrCast(self.window_name)), @as([*]const u8, @ptrCast(self.trackbar_name)));
        }

        /// SetPos sets the trackbar position.
        ///
        /// For further details, please see:
        /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga67d73c4c9430f13481fd58410d01bd8d
        ///
        pub fn setPos(self: *Trackbar, pos: i32) void {
            _ = c.Trackbar_SetPos(@as([*]const u8, @ptrCast(self.window_name)), @as([*]const u8, @ptrCast(self.trackbar_name)), pos);
        }

        /// SetMin sets the trackbar minimum position.
        ///
        /// For further details, please see:
        /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#gabe26ffe8d2b60cc678895595a581b7aa
        ///
        pub fn setMin(self: *Trackbar, pos: i32) void {
            _ = c.Trackbar_SetMin(@as([*]const u8, @ptrCast(self.window_name)), @as([*]const u8, @ptrCast(self.trackbar_name)), pos);
        }

        /// SetMax sets the trackbar maximum position.
        ///
        /// For further details, please see:
        /// https://docs.opencv.org/master/d7/dfc/group__highgui.html#ga7e5437ccba37f1154b65210902fc4480
        ///
        pub fn setMax(self: *Trackbar, pos: i32) void {
            _ = c.Trackbar_SetMax(@as([*]const u8, @ptrCast(self.window_name)), @as([*]const u8, @ptrCast(self.trackbar_name)), pos);
        }
    };
};

const testing = @import("std").testing;
const imgcodecs = @import("imgcodecs.zig");

fn hasDisplay() bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const env_map = arena.allocator().create(std.process.EnvMap) catch return false;
    env_map.* = std.process.getEnvMap(arena.allocator()) catch return false;

    const display = env_map.get("DISPLAY") orelse "";

    return display.len > 0;
}

// test "highgui window" {
// if (!hasDisplay()) return error.SkipZigTest;

// var window = try Window.init("test");
// try testing.expectEqualStrings("test", window.name);

// const val = window.waitKey(1);
// try testing.expectEqual(@as(i32, -1), val);

// try testing.expectEqual(true, window.isOpened());

// window.setProperty(.fullscreen, .fullscreen);

// const window_flag = window.getProperty(.fullscreen);
// try testing.expectEqual(Window.Flag.fullscreen, window_flag);

// window.setTitle("test2");
// try testing.expectEqualStrings("test2", window.name);

// window.move(100, 100);

// window.resize(100, 100);

// window.deinit();
// try testing.expectEqual(false, window.isOpened());
// }

test "highgui window imshow" {
    if (!hasDisplay()) return error.SkipZigTest;

    var window = try Window.init("imshow");
    defer window.deinit();

    var img = try imgcodecs.imRead("test/images/face-detect.jpg", .unchanged);
    defer img.deinit();
    window.imShow(img);
}

test "highgui window selectROI" {
    if (!hasDisplay()) return error.SkipZigTest;

    // TODO
}

test "highgui window selectROIs" {
    if (!hasDisplay()) return error.SkipZigTest;

    // TODO
}

test "highgui window trackbar" {
    if (!hasDisplay()) return error.SkipZigTest;

    var window = try Window.init("testtrackbar");
    defer window.deinit();

    window.createTrackbar("trackme", 100);

    if (window.trackbar != null) {
        var t = window.trackbar.?;
        try testing.expectEqual(@as(i32, 0), t.getPos());

        t.setMin(10);
        t.setMax(150);
        t.setPos(50);
        try testing.expectEqual(@as(i32, 50), t.getPos());
    } else @panic("trackbar not found");
}

//*    implementation done
//*    pub extern fn Window_New(winname: [*c]const u8, flags: c_int) void;
//*    pub extern fn Window_Close(winname: [*c]const u8) void;
//*    pub extern fn Window_IMShow(winname: [*c]const u8, mat: Mat) void;
//*    pub extern fn Window_GetProperty(winname: [*c]const u8, flag: c_int) f64;
//*    pub extern fn Window_SetProperty(winname: [*c]const u8, flag: c_int, value: f64) void;
//*    pub extern fn Window_SetTitle(winname: [*c]const u8, title: [*c]const u8) void;
//*    pub extern fn Window_WaitKey(c_int) c_int;
//*    pub extern fn Window_Move(winname: [*c]const u8, x: c_int, y: c_int) void;
//*    pub extern fn Window_Resize(winname: [*c]const u8, width: c_int, height: c_int) void;
//*    pub extern fn Window_SelectROI(winname: [*c]const u8, img: Mat) struct_Rect;
//*    pub extern fn Window_SelectROIs(winname: [*c]const u8, img: Mat) struct_Rects;
//*    pub extern fn Trackbar_Create(winname: [*c]const u8, trackname: [*c]const u8, max: c_int) void;
//     pub extern fn Trackbar_CreateWithValue(winname: [*c]const u8, trackname: [*c]const u8, value: [*c]c_int, max: c_int) void; // deprecated
//*    pub extern fn Trackbar_GetPos(winname: [*c]const u8, trackname: [*c]const u8) c_int;
//*    pub extern fn Trackbar_SetPos(winname: [*c]const u8, trackname: [*c]const u8, pos: c_int) void;
//*    pub extern fn Trackbar_SetMin(winname: [*c]const u8, trackname: [*c]const u8, pos: c_int) void;
//*    pub extern fn Trackbar_SetMax(winname: [*c]const u8, trackname: [*c]const u8, pos: c_int) void;
