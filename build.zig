const std = @import("std");
const Build = std.Build;
const LazyPath = std.build.LazyPath;
const zigcv = @import("libs.zig");

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardOptimizeOption(.{});

    const examples = [_]Program{
        .{
            .name = "hello",
            .path = "examples/hello/main.zig",
            .desc = "Show Webcam",
        },
        .{
            .name = "version",
            .path = "examples/version/main.zig",
            .desc = "Print OpenCV Version",
        },
        .{
            .name = "show_image",
            .path = "examples/showimage/main.zig",
            .desc = "Show Image Demo",
        },
        .{
            .name = "face_detection",
            .path = "examples/facedetect/main.zig",
            .desc = "Face Detection Demo",
        },
        .{
            .name = "face_blur",
            .path = "examples/faceblur/main.zig",
            .desc = "Face Detection and Blur Demo",
        },
        .{
            .name = "dnn_detection",
            .path = "examples/dnndetection/main.zig",
            .desc = "DNN Detection Demo",
        },
        .{
            .name = "saveimage",
            .path = "examples/saveimage/main.zig",
            .desc = "Save Image Demo",
        },
        .{
            .name = "detail_enhance",
            .path = "examples/detail_enhance/main.zig",
            .desc = "Detail Enhanced Image Demo",
        },
    };

    const examples_step = b.step("examples", "Builds all the examples");

    for (examples) |ex| {
        const exe = b.addExecutable(.{
            .name = ex.name,
            .root_source_file = b.path(ex.path),
            .target = target,
            .optimize = mode,
        });
        const exe_step = &exe.step;

        b.installArtifact(exe);

        zigcv.link(b, exe);
        zigcv.addAsPackage(exe);

        const run_cmd = b.addRunArtifact(exe);
        const run_step = b.step(ex.name, ex.desc);
        const artifact_step = &b.addInstallArtifact(exe, .{}).step;
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }
        run_step.dependOn(artifact_step);
        run_step.dependOn(&run_cmd.step);
        examples_step.dependOn(exe_step);
        examples_step.dependOn(artifact_step);
    }

    // var tmp_dir = try std.fs.cwd().createTempDir(.{}, "zig-temp");
    defer tmp_dir.cleanup();

    const test_filter = b.option([]const u8, "test-filter", "Skip tests that do not match filter") orelse null;
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = mode,
        .filter = test_filter,
    });
    zigcv.link(b, unit_tests);
    zigcv.addAsPackage(unit_tests);

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // const emit_docs = b.option(bool, "docs", "Generate Docs");
    // if (emit_docs) |d| {
    //     if (d) exe_tests.emit_docs = .emit;
    // }
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}

const Program = struct {
    name: []const u8,
    path: []const u8,
    desc: []const u8,
    fstage1: bool = false,
};
