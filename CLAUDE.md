# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZIGCV is a Zig language binding library for OpenCV 4. It provides Zig developers with access to computer vision functionality through idiomatic Zig APIs while wrapping the GoCV C++ interface.

## Essential Commands

### Building
```bash
# Build the library
zig build

# Build all examples
zig build examples

# Build and run a specific example
zig build showimage
zig build facedetect
```

### Testing
```bash
# Run all tests
zig build test

# Download required ML models for tests (required before running tests)
devenv shell -- download-models
# or manually:
bun run scripts/download_models.ts
```

### Development with devenv
```bash
# Enter devenv shell
devenv shell

# Build using devenv (inside the shell)
build

# Run tests (inside the shell)
test
```

## Architecture

The codebase follows a modular structure mirroring OpenCV's organization:

1. **Core Bindings** (`src/c_api.zig`): Direct C bindings to the GoCV wrapper
2. **Module Files** (`src/*.zig`): Each OpenCV module has a corresponding Zig file that wraps C APIs with idiomatic Zig interfaces
3. **Main Export** (`src/main.zig`): Central entry point that re-exports all modules
4. **GoCV Dependency** (`libs/gocv/`): Git submodule containing the C++ wrapper layer

### Key Patterns

- **Memory Management**: All objects follow Zig's init/deinit pattern
- **Error Handling**: Extensive use of error unions for safety
- **Resource Management**: Use `defer obj.deinit()` for RAII-style cleanup
- **Testing**: Tests are embedded in source files using Zig's built-in test framework

### Module Structure Example
```zig
// Typical module pattern
const cv = @import("c_api.zig");

pub const MyType = struct {
    ptr: *cv.C_MyType,
    
    pub fn init() !MyType {
        const ptr = cv.C_MyType_New();
        if (ptr == null) return error.InitFailed;
        return .{ .ptr = ptr };
    }
    
    pub fn deinit(self: *MyType) void {
        cv.C_MyType_Close(self.ptr);
    }
};
```

## Current Limitations

- Zig 0.11.0 required (0.12.0 support in progress)
- Limited by Zig's C ABI capabilities for struct passing
- Some OpenCV modules not yet implemented (see `src/README.md` for status)

## Testing Requirements

DNN tests require pre-downloaded models. Run `devenv shell -- download-models` before testing to fetch:
- Caffe models for face detection
- TensorFlow models for object detection
- Test data files
