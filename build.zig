const std = @import("std");

const c_build_options: []const []const u8 = &.{
    "-Wall",
    "-Wextra",
    "--std=c++11",
};

const zig_src_dir = "src/";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gocv_dep = b.dependency("gocv", .{});
    const gocv_src_dir = gocv_dep.path("");
    const gocv_contrib_dir = gocv_dep.path("contrib");

    const opencv_libs, const opencv4_headers_dir, const opencv4_libraries_dir = buildOpenCVStep(b);

    var zigcv = b.addModule("root", .{
        .root_source_file = b.path("src/zigcv.zig"),
        .link_libcpp = true,
    });
    zigcv.addIncludePath(gocv_src_dir); // OpenCV C bindings base
    zigcv.addIncludePath(gocv_contrib_dir); // OpenCV contrib C bindings
    zigcv.addIncludePath(b.path(zig_src_dir)); // Our glue header
    zigcv.addIncludePath(opencv4_headers_dir); // Include the opencv4 headers
    zigcv.addLibraryPath(opencv4_libraries_dir);

    const zigcv_lib = b.addSharedLibrary(.{
        .name = "zigcv",
        .target = target,
        .optimize = optimize,
    });
    zigcv_lib.step.dependOn(&opencv_libs.step);

    zigcv_lib.addIncludePath(gocv_src_dir); // OpenCV C bindings base
    zigcv_lib.addIncludePath(gocv_contrib_dir); // OpenCV contrib C bindings
    zigcv_lib.addIncludePath(b.path(zig_src_dir)); // Our glue header
    zigcv_lib.addIncludePath(opencv4_headers_dir); // Include the opencv4 headers

    zigcv_lib.addCSourceFile(.{
        .file = b.path("src/core/zig_core.cpp"),
        .flags = c_build_options,
    });

    zigcv_lib.addCSourceFiles(.{
        .files = &.{
            "aruco.cpp",
            "asyncarray.cpp",
            "calib3d.cpp",
            "core.cpp",
            "dnn.cpp",
            "features2d.cpp",
            "highgui.cpp",
            "imgcodecs.cpp",
            "imgproc.cpp",
            "objdetect.cpp",
            "persistence_filenode.cpp",
            "persistence_filestorage.cpp",
            "photo.cpp",
            "svd.cpp",
            "version.cpp",
            "video.cpp",
            "videoio.cpp",
        },
        .root = gocv_dep.path(""),
        .flags = c_build_options,
    });

    zigcv_lib.addCSourceFiles(.{
        .files = &.{
            "bgsegm.cpp",
            "face.cpp",
            "freetype.cpp",
            "img_hash.cpp",
            "tracking.cpp",
            "wechat_qrcode.cpp",
            "xfeatures2d.cpp",
            "ximgproc.cpp",
            "xphoto.cpp",
        },
        .root = gocv_contrib_dir,
        .flags = c_build_options,
    });

    zigcv_lib.linkLibCpp();
    zigcv_lib.linkSystemLibrary("z");

    zigcv_lib.addLibraryPath(opencv4_libraries_dir);
    zigcv_lib.linkSystemLibrary("opencv_bioinspired");
    zigcv_lib.linkSystemLibrary("opencv_calib3d");
    zigcv_lib.linkSystemLibrary("opencv_ccalib");
    zigcv_lib.linkSystemLibrary("opencv_core");
    zigcv_lib.linkSystemLibrary("opencv_datasets");
    zigcv_lib.linkSystemLibrary("opencv_dnn");
    zigcv_lib.linkSystemLibrary("opencv_dnn_objdetect");
    zigcv_lib.linkSystemLibrary("opencv_dnn_superres");
    zigcv_lib.linkSystemLibrary("opencv_dpm");
    zigcv_lib.linkSystemLibrary("opencv_features2d");
    zigcv_lib.linkSystemLibrary("opencv_flann");
    zigcv_lib.linkSystemLibrary("opencv_freetype");
    zigcv_lib.linkSystemLibrary("opencv_fuzzy");
    zigcv_lib.linkSystemLibrary("opencv_gapi");
    zigcv_lib.linkSystemLibrary("opencv_hfs");
    zigcv_lib.linkSystemLibrary("opencv_highgui");
    zigcv_lib.linkSystemLibrary("opencv_imgcodecs");
    zigcv_lib.linkSystemLibrary("opencv_imgproc");
    zigcv_lib.linkSystemLibrary("opencv_intensity_transform");
    zigcv_lib.linkSystemLibrary("opencv_line_descriptor");
    zigcv_lib.linkSystemLibrary("opencv_mcc");
    zigcv_lib.linkSystemLibrary("opencv_ml");
    zigcv_lib.linkSystemLibrary("opencv_objdetect");
    zigcv_lib.linkSystemLibrary("opencv_optflow");
    zigcv_lib.linkSystemLibrary("opencv_phase_unwrapping");
    zigcv_lib.linkSystemLibrary("opencv_photo");
    zigcv_lib.linkSystemLibrary("opencv_plot");
    zigcv_lib.linkSystemLibrary("opencv_quality");
    zigcv_lib.linkSystemLibrary("opencv_rapid");
    zigcv_lib.linkSystemLibrary("opencv_reg");
    zigcv_lib.linkSystemLibrary("opencv_rgbd");
    zigcv_lib.linkSystemLibrary("opencv_saliency");
    zigcv_lib.linkSystemLibrary("opencv_shape");
    zigcv_lib.linkSystemLibrary("opencv_signal");
    zigcv_lib.linkSystemLibrary("opencv_stereo");
    zigcv_lib.linkSystemLibrary("opencv_stitching");
    zigcv_lib.linkSystemLibrary("opencv_structured_light");
    zigcv_lib.linkSystemLibrary("opencv_superres");
    zigcv_lib.linkSystemLibrary("opencv_surface_matching");
    zigcv_lib.linkSystemLibrary("opencv_text");
    zigcv_lib.linkSystemLibrary("opencv_video");
    zigcv_lib.linkSystemLibrary("opencv_videoio");
    zigcv_lib.linkSystemLibrary("opencv_videostab");
    zigcv_lib.linkSystemLibrary("opencv_xobjdetect");

    zigcv_lib.linkSystemLibrary("opencv_aruco");
    zigcv_lib.linkSystemLibrary("opencv_bgsegm");
    zigcv_lib.linkSystemLibrary("opencv_face");
    zigcv_lib.linkSystemLibrary("opencv_img_hash");
    zigcv_lib.linkSystemLibrary("opencv_tracking");
    zigcv_lib.linkSystemLibrary("opencv_wechat_qrcode");
    zigcv_lib.linkSystemLibrary("opencv_xfeatures2d");
    zigcv_lib.linkSystemLibrary("opencv_ximgproc");
    zigcv_lib.linkSystemLibrary("opencv_xphoto");

    b.installArtifact(zigcv_lib);

    // ---- Unit tests ---------------------------------------------------------
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/zigcv.zig"),
        .target = target,
        .optimize = optimize,
    });
    unit_tests.addIncludePath(gocv_src_dir); // OpenCV C bindings base
    unit_tests.addIncludePath(gocv_contrib_dir); // OpenCV contrib C bindings
    unit_tests.addIncludePath(b.path(zig_src_dir)); // Our glue header
    unit_tests.addIncludePath(opencv4_headers_dir); // Include the opencv4 headers
    unit_tests.addLibraryPath(opencv4_libraries_dir);
    unit_tests.linkLibrary(zigcv_lib);
    unit_tests.linkLibCpp();
    b.installArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);
}

fn buildOpenCVStep(b: *std.Build) struct {
    *std.Build.Step.Run,
    std.Build.LazyPath,
    std.Build.LazyPath,
} {
    const opencv_dep = b.dependency("opencv", .{});
    const opencv_contrib_dep = b.dependency("opencv_contrib", .{});

    std.log.info("Building OpenCV4 from source with `zig cc` and `zig c++`\n", .{});

    const cmake_bin = b.findProgram(&.{"cmake"}, &.{}) catch @panic("Could not find cmake");

    const configure_cmd = b.addSystemCommand(&.{ cmake_bin, "-B" });
    const build_work_dir = configure_cmd.addOutputDirectoryArg("build_work");
    configure_cmd.setEnvironmentVariable("CC", "zig cc");
    configure_cmd.setEnvironmentVariable("CXX", "zig c++");
    configure_cmd.addArgs(&.{
        "-D",
        "CMAKE_BUILD_TYPE=RELEASE",
        "-D",
        "WITH_IPP=OFF",
        "-D",
    });
    const opencv4_build_dir = configure_cmd.addPrefixedOutputDirectoryArg("CMAKE_INSTALL_PREFIX=", "zig_opencv4_build");
    configure_cmd.addArgs(&.{
        "-D",
    });
    configure_cmd.addPrefixedDirectoryArg("OPENCV_EXTRA_MODULES_PATH=", opencv_contrib_dep.path("modules"));
    configure_cmd.addArgs(&.{
        "-D",
        "OPENCV_ENABLE_NONFREE=ON",
        "-D",
        "WITH_JASPER=OFF",
        "-D",
        "WITH_TBB=ON",
        "-D",
        "BUILD_DOCS=OFF",
        "-D",
        "BUILD_EXAMPLES=OFF",
        "-D",
        "BUILD_TESTS=OFF",
        "-D",
        "BUILD_PERF_TESTS=OFF",
        "-D",
        "BUILD_opencv_java=NO",
        "-D",
        "BUILD_opencv_python=NO",
        "-D",
        "BUILD_opencv_python2=NO",
        "-D",
        "BUILD_opencv_python3=NO",
        "-D",
        "OPENCV_GENERATE_PKGCONFIG=OFF",
    });
    configure_cmd.addDirectoryArg(opencv_dep.path(""));

    const nproc_bin = b.findProgram(&.{"nproc"}, &.{}) catch @panic("Couldn't find nproc");
    const num_cores = std.mem.trim(u8, b.run(&.{nproc_bin}), "\n");

    const build_cmd = b.addSystemCommand(&.{ cmake_bin, "--build" });
    build_cmd.addDirectoryArg(build_work_dir);
    build_cmd.addArgs(&.{ "-j", num_cores });
    build_cmd.step.dependOn(&configure_cmd.step);

    const install_cmd = b.addSystemCommand(&.{ cmake_bin, "--install" });
    install_cmd.addDirectoryArg(build_work_dir);
    install_cmd.step.dependOn(&build_cmd.step);

    return .{
        install_cmd,
        opencv4_build_dir.path(b, "include/opencv4"),
        opencv4_build_dir.path(b, "lib64"),
    };
}
