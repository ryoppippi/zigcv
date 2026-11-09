{ pkgs, lib, config, inputs, ... }:

let
  zigVersion = "0.11.0";

  zigPackage =
    (import inputs."nixpkgs-zig-0-11" {
      system = pkgs.stdenv.hostPlatform.system;
    }).zig_0_11;

  zlsPackages =
    if inputs ? zls && inputs.zls ? packages && builtins.hasAttr pkgs.stdenv.hostPlatform.system inputs.zls.packages then
      builtins.getAttr pkgs.stdenv.hostPlatform.system inputs.zls.packages
    else
      throw "zigtools/zls does not provide packages for this system";

  zlsPackage =
    if zlsPackages ? default then zlsPackages.default
    else if zlsPackages ? zls then zlsPackages.zls
    else throw "zigtools/zls flake did not expose a default package";

  opencvPkg =
    (import inputs."nixpkgs-opencv-4-6" {
      system = pkgs.stdenv.hostPlatform.system;
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

  scripts = {
    version.exec = "zig version";
    download-models.exec = modelSync;
    build.exec = "zig build --verbose";
    "build-release".exec = "zig build -Doptimize=ReleaseFast";
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
