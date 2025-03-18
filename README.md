# ZigCV

<div align="center">
  <img src="./logo/zigcv.png" width="50%" />
</div>

The ZigCV library provides Zig language bindings for the
[OpenCV 4](http://opencv.org/) computer vision library.

The ZigCV library supports the same nominated version of zig as
[Mach](https://machengine.org/) and the
[zig-gamedev](https://github.com/zig-gamedev/) libraries.

The ZigCV library currently supports
[OpenCV v4.11.0](https://github.com/opencv/opencv/tree/4.11.0).

It uses [GoCV 0.40.0](https://github.com/hybridgroup/gocv/tree/v0.40.0) for its
C bindings to OpenCV.

**Caution**

Under development. The Zig APIs may be missing or change.

## Install

Add to your project's dependencies:

```sh
zig fetch --save 'git+https://codeberg.org/glitchcake/zigcv'
```

Add to your `build.zig`:

```zig
const zigcv = b.dependency("zigcv", .{});
exe.root_module.addImport("zigcv", zigcv.module("root"));
exe.linkLibrary(zigcv.artifact("zigcv"));
```

## Usage

Once added to your project, you may import and use.

```zig
const cv = @import("zigcv");
```

You can also call C bindings directly via the `c` struct on the import.

```zig
cv.c
```

### Example

Here is a minimal program:

```zig
const std = @import("std");
const cv = @import("zigcv");

pub fn main() !void {
    std.debug.print("version via zig binding:\t{s}\n", .{cv.openCVVersion()});
    std.debug.print("version via c api directly:\t{s}\n", .{cv.c.openCVVersion()});
}
```

## More Examples

There are a handful of sample programs in the `examples/` directory.

You can build them by running `zig build` there:

```sh
cd examples && zig build; popd; ./examples/zig-out/bin/hello
```

## Demo

```
./examples/zig-out/bin/face_detection 0
```

<div align="center">
  <img width="400" alt="face detection" src="https://user-images.githubusercontent.com/1560508/188515175-4d344660-5680-43e7-9b74-3bad92507430.gif">
</div>

## Technical restrictions

Due to zig being a relatively new language it does
[not have full C ABI support](https://github.com/ziglang/zig/issues/1481) at the
moment. For use that mainly means we can't use any functions that return structs
that are less than 16 bytes large on x86, and passing structs to any functions
may cause memory error on arm.

## Todo

- [ ] Get all examples working
- [ ] Fix all commented out tests
- [ ] Add cuda and openvino back

## License

MIT

## Authors

Ryotaro "Justin" Kimura (a.k.a. ryoppippi)

[glitchcake](https://codeberg.org/glitchcake/)
