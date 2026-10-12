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
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          android.androidsdk
          jdk17
          python
          uv
          git zip unzip patch which file rsync
          autoconf automake libtool pkg-config cmake gnumake ninja ccache
          zlib libffi openssl
        ];
        JAVA_HOME = "${pkgs.jdk17}";
        ANDROIDSDK = sdk;
        ANDROIDNDK = ndk;
        ANDROIDAPI = "33";
        ANDROIDMINAPI = "24";
        # Buildozer reads these to find the tools it would otherwise download.
        ANDROID_SDK_ROOT = sdk;
        ANDROID_NDK_HOME = ndk;
        # Python and every Python dependency (Kivy, edge-tts, Buildozer, Cython)
        # come from uv: pyproject.toml and uv.lock in this folder.
        UV_PYTHON = "${python}/bin/python3";
        shellHook = ''
          export PATH="${sdk}/build-tools/33.0.2:${sdk}/platform-tools:$PATH"
          uv sync --quiet
          export PATH="$PWD/.venv/bin:$PATH"
          echo "Perudo Android build shell. ./build.sh builds the APK, uv run python main.py runs the app here."
        '';
      };
    };
}
