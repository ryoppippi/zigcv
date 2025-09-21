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

  zlsVersion = "0.14.0";
  zlsReleases = {
    "aarch64-darwin" = {
      url = "https://github.com/zigtools/zls/releases/download/${zlsVersion}/zls-aarch64-macos.tar.xz";
      hash = "sha256-37Yn4flgNYNnj1UtgDWhLc6HghXApQezLW8bnQdNbE0=";
    };
    "x86_64-darwin" = {
      url = "https://github.com/zigtools/zls/releases/download/${zlsVersion}/zls-x86_64-macos.tar.xz";
      hash = "sha256-uu5p5GRd7sy0KXC0oB9XNZIgncHPcuMok8WcoGr1Edw=";
    };
    "aarch64-linux" = {
      url = "https://github.com/zigtools/zls/releases/download/${zlsVersion}/zls-aarch64-linux.tar.xz";
      hash = "sha256-2F9Gea85YdsUnq2KNV6rRlLD5zjuyq1pF0yrXxoRlsw=";
    };
    "x86_64-linux" = {
      url = "https://github.com/zigtools/zls/releases/download/${zlsVersion}/zls-x86_64-linux.tar.xz";
      hash = "sha256-Zh+NQCuj3JsEtum8MCZJW+e4ONLxjRSNsr2YvWmcE2A=";
    };
  };

  zlsPackage =
    if builtins.hasAttr pkgs.system zlsReleases then
      let
        release = zlsReleases.${pkgs.system};
      in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "zls";
          version = zlsVersion;
          src = pkgs.fetchurl release;

          dontConfigure = true;
          dontBuild = true;

          unpackPhase = ''
            tar xf "$src"
          '';

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin" "$out/share/doc/zls"
            install -m755 zls "$out/bin/zls"
            if [ -f LICENSE ]; then
              install -m644 LICENSE "$out/share/doc/zls/LICENSE"
            fi
            if [ -f README.md ]; then
              install -m644 README.md "$out/share/doc/zls/README.md"
            fi
            runHook postInstall
          '';

          meta = {
            description = "Zig Language Server pre-built binary";
            homepage = "https://github.com/zigtools/zls";
            license = lib.licenses.mit;
            maintainers = [ ];
            platforms = lib.platforms.unix;
            mainProgram = "zls";
          };
        }
    else pkgs.zls;

  opencvPkg =
    (import inputs."nixpkgs-opencv-4-6" {
      system = pkgs.system;
      config = { allowUnfree = false; };
    }).opencv;

  basePackages = [
    zigPackage
    zlsPackage
    opencvPkg
    pkgs.pkg-config
    pkgs.unzip
    pkgs.deno
  ];

in
{
  name = "zigcv";

  env = {
    ZIG_VERSION = zigVersion;
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
    download-models.exec = "deno run -A ./scripts/download_models.ts";
    build.exec = "zig build --verbose";
    test.exec = "zig build test --verbose";
    fmt.exec = "zig fmt ./**/*.zig";
    "fmt-check".exec = "zig fmt --check ./**/*.zig";
  };

  enterShell = ''
    zig version
    pkg-config --modversion opencv4 || true
  '';

  enterTest = ''
    zig version | grep "${zigVersion}"
    pkg-config --modversion opencv4
  '';
}
