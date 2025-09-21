{ pkgs, lib, config, inputs, ... }:

let
  zigVersion = "0.11.0";

  zigOverlayPkgs =
    if builtins.hasAttr pkgs.system inputs."zig-overlay".packages then
      builtins.getAttr pkgs.system inputs."zig-overlay".packages
    else
      throw "zig-overlay does not provide packages for this system";

  zigPackage =
    if builtins.hasAttr zigVersion zigOverlayPkgs then
      builtins.getAttr zigVersion zigOverlayPkgs
    else
      throw "zig-overlay does not provide Zig ${zigVersion} for this system";

  zlsPackages =
    if inputs ? zls && inputs.zls ? packages && builtins.hasAttr pkgs.system inputs.zls.packages then
      builtins.getAttr pkgs.system inputs.zls.packages
    else
      throw "zigtools/zls does not provide packages for this system";

  zlsPackage =
    if zlsPackages ? default then zlsPackages.default
    else if zlsPackages ? zls then zlsPackages.zls
    else throw "zigtools/zls flake did not expose a default package";

  opencvPkg =
    (import inputs."nixpkgs-opencv-4-6" {
      system = pkgs.system;
      config = { allowUnfree = false; };
    }).opencv;

  models = import ./nix/models.nix { inherit pkgs; };

  modelSync = ''
    set -euo pipefail
    : "''${ZIGCV_MODEL_DIR:?ZIGCV_MODEL_DIR must point to the staged model directory}"
    mkdir -p zig-cache/tmp
    cp -f "''${ZIGCV_MODEL_DIR}"/* zig-cache/tmp/
    touch zig-cache/tmp/.models_synced
  '';

  basePackages = [
    zigPackage
    zlsPackage
    opencvPkg
    pkgs.pkg-config
    models
  ];

in
{
  name = "zigcv";

  env = {
    ZIG_VERSION = zigVersion;
    ZLS_VERSION = if builtins.hasAttr "version" zlsPackage then zlsPackage.version else "";
    ZIGCV_MODEL_DIR = "${models}";
  };

  cachix.enable = false;

  packages = basePackages;

  overlays = [
    (final: prev:
      if prev.stdenv.isDarwin then {
        apple-sdk = prev.darwin.apple_sdk.sdkRoot;
        apple-sdk_11 = prev.darwin.apple_sdk_11_0.sdkRoot;
      } else {
      })
  ];

  # devenv 1.9's tasks helper requires nightly Cargo; stub it out to avoid the build.
  task.package = pkgs.writeShellScriptBin "devenv-tasks" ''
    exit 0
  '';

  scripts = {
    version.exec = "zig version";
    download-models.exec = modelSync;
    build.exec = "zig build --verbose";
    test.exec = "zig build test --verbose";
    fmt.exec = "zig fmt ./**/*.zig";
    "fmt-check".exec = "zig fmt --check ./**/*.zig";
  };

  enterShell = ''
    zig version
    pkg-config --modversion opencv4 || true
    if [ ! -f zig-cache/tmp/.models_synced ]; then
      ${modelSync}
    fi
  '';

  enterTest = ''
    zig version | grep "${zigVersion}"
    pkg-config --modversion opencv4
  '';
}
