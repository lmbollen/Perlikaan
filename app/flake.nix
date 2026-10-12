{
  description = "Perudo for Android: everything outside Python that the Buildozer build needs";

  inputs.nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.xz";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;                 # the Android SDK
          android_sdk.accept_license = true;
        };
      };
      android = pkgs.androidenv.composeAndroidPackages {
        platformVersions = [ "33" ];
        buildToolsVersions = [ "33.0.2" ];
        includeNDK = true;
        ndkVersions = [ "25.2.9519653" ];
        includeEmulator = false;
      };
      sdk = "${android.androidsdk}/libexec/android-sdk";
      ndk = "${sdk}/ndk/25.2.9519653";
      python = pkgs.python312;
      tools = with pkgs; [
        jdk17
        python
        uv
        git zip unzip patch which file rsync
        autoconf automake libtool pkg-config cmake gnumake ninja ccache
        zlib libffi openssl
      ];
      common = {
        JAVA_HOME = "${pkgs.jdk17}";
        ANDROIDAPI = "33";
        ANDROIDMINAPI = "24";
        # Python and every Python dependency (Kivy, edge-tts, Buildozer, Cython)
        # come from uv: pyproject.toml and uv.lock in this folder.
        UV_PYTHON = "${python}/bin/python3";
      };
    in {
      devShells.${system} = {
        # The usual shell: Nix fetches the Android SDK and NDK (from dl.google.com).
        default = pkgs.mkShell (common // {
          packages = tools ++ [ android.androidsdk ];
          ANDROIDSDK = sdk;
          ANDROIDNDK = ndk;
          # Buildozer reads these to find the tools it would otherwise download.
          ANDROID_SDK_ROOT = sdk;
          ANDROID_NDK_HOME = ndk;
          shellHook = ''
            export PATH="${sdk}/build-tools/33.0.2:${sdk}/platform-tools:$PATH"
            uv sync --quiet
            export PATH="$PWD/.venv/bin:$PATH"
            echo "Perudo Android build shell. ./build.sh builds the APK, uv run python main.py runs the app here."
          '';
        });
        # For networks where dl.google.com is blocked: the same tools, with an
        # SDK and NDK you already have in ANDROIDSDK and ANDROIDNDK (the SDK
        # must hold the platform of android.api in buildozer.spec and build
        # tools; python-for-android accepts NDK 25 to 28).
        prebuilt = pkgs.mkShell (common // {
          packages = tools;
          shellHook = ''
            : "''${ANDROIDSDK:?set ANDROIDSDK to your Android SDK}"
            : "''${ANDROIDNDK:?set ANDROIDNDK to your Android NDK}"
            export ANDROID_SDK_ROOT="$ANDROIDSDK" ANDROID_NDK_HOME="$ANDROIDNDK"
            export PATH="$ANDROIDSDK/platform-tools:$PATH"
            uv sync --quiet
            export PATH="$PWD/.venv/bin:$PATH"
            echo "Perudo Android build shell (prebuilt SDK at $ANDROIDSDK, NDK at $ANDROIDNDK). ./build.sh builds the APK."
          '';
        });
      };
    };
}
