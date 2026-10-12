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
          # the desktop version of the app, for trying it out: python main.py
          python312Packages.kivy
        ];
        JAVA_HOME = "${pkgs.jdk17}";
        ANDROIDSDK = sdk;
        ANDROIDNDK = ndk;
        ANDROIDAPI = "33";
        ANDROIDMINAPI = "24";
        # Buildozer reads these to find the tools it would otherwise download.
        ANDROID_SDK_ROOT = sdk;
        ANDROID_NDK_HOME = ndk;
        shellHook = ''
          export PATH="${sdk}/build-tools/33.0.2:${sdk}/platform-tools:$PATH"
          if [ ! -d .venv ]; then
            echo "Creating the Python environment for Buildozer..."
            uv venv -q .venv --python ${python}/bin/python3
            uv pip install -q --python .venv/bin/python buildozer "cython<3.1" kivy==2.3.1 edge-tts pytest
          fi
          export PATH="$PWD/.venv/bin:$PATH"
          echo "Perudo Android build shell. Run ./build.sh to build the APK."
        '';
      };
    };
}
