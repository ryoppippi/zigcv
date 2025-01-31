const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const examples = [_]Program{
        .{
            .name = "hello",
            .path = "hello/main.zig",
            .desc = "Show Webcam",
        },
        .{
            .name = "version",
            .path = "version/main.zig",
            .desc = "Print OpenCV Version",
        },
        .{
            .name = "show_image",
            .path = "showimage/main.zig",
            .desc = "Show Image Demo",
        },
        .{
            .name = "face_detection",
            .path = "facedetect/main.zig",
            .desc = "Face Detection Demo",
        },
        .{
            .name = "face_blur",
            .path = "faceblur/main.zig",
            .desc = "Face Detection and Blur Demo",
        },
        .{
            .name = "dnn_detection",
            .path = "dnndetection/main.zig",
            .desc = "DNN Detection Demo",
        },
        .{
            .name = "saveimage",
            .path = "saveimage/main.zig",
            .desc = "Save Image Demo",
        },
        .{
            .name = "detail_enhance",
            .path = "detail_enhance/main.zig",
            .desc = "Detail Enhanced Image Demo",
        },
    };

    const examples_step = b.step("examples", "Builds all the examples");

    for (examples) |ex| {
        const exe = b.addExecutable(.{
            .name = ex.name,
            .root_source_file = b.path(ex.path),
            .target = target,
            .optimize = optimize,
        });

        b.installArtifact(exe);

        const zigcv = b.dependency("zigcv", .{});
        exe.root_module.addImport("zigcv", zigcv.module("root"));
        exe.linkLibrary(zigcv.artifact("zigcv"));

        const run_cmd = b.addRunArtifact(exe);
        const run_step = b.step(ex.name, ex.desc);
        const artifact_step = &b.addInstallArtifact(exe, .{}).step;
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }
        run_step.dependOn(artifact_step);
        run_step.dependOn(&run_cmd.step);
        examples_step.dependOn(&exe.step);
        examples_step.dependOn(artifact_step);
    }
}

const Program = struct {
    name: []const u8,
    path: []const u8,
    desc: []const u8,
    fstage1: bool = false,
};
