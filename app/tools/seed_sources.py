"""
Seed the Android build for restricted networks.

python-for-android downloads its sources from python.org, sqlite.org,
openssl.org and GitHub source archives, and Buildozer clones
python-for-android from GitHub. On networks where those are blocked but
PyPI, GitHub *release assets* and Ubuntu's package pool are reachable, this
script fetches the same sources from there, repacks them under the file
names python-for-android expects, and drops them into its download cache
with the markers that make it skip the download. It also fetches the
python-for-android release itself from PyPI, so Buildozer needs no git.

    cd app
    uv run python tools/seed_sources.py          # then ./build.sh

What is seeded, and from where:

    python-for-android   PyPI sdist              -> app/.p4a-src (p4a.source_dir)
    python3, hostpython3 Ubuntu pool orig tarball -> packages/python3, hostpython3
    sqlite3              Ubuntu pool orig tarball -> packages/sqlite3
    openssl, libffi      GitHub release assets   -> packages/openssl, libffi
    kivy, pyjnius        PyPI sdists             -> packages/kivy, pyjnius

SDL2 and its libraries come from GitHub release assets, which python-for-
android fetches itself; aiohttp and the rest come from PyPI. The Android
SDK and NDK are not covered: they only exist on dl.google.com (or inside
container images); see README.md.

The versions below are the ones python-for-android 2026.5.9 pins, except
Python and SQLite, which are the nearest Ubuntu carries. Those two differ
from the recipes' versions, so the script writes `.p4a-src/requirements`
(the spec's list with `python3==...` pins) for build.sh to pass as
APP_REQUIREMENTS, and reshapes the one CPython file that python-for-android
patches so its patch still applies.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ARCHS = "arm64-v8a"
# Buildozer's storage dir for the arch (build-arm64-v8a; older Buildozers used
# build-arm64_v8a, which is linked to it below).
PACKAGES = APP / ".buildozer" / "android" / "platform" / f"build-{ARCHS}" / "packages"
_OLD_STORAGE = PACKAGES.parent.parent / f"build-{ARCHS.replace('-', '_')}"
P4A_SRC = APP / ".p4a-src"
ANDROID_PYPROJECT = """\
[build-system]
requires = [
    "setuptools",
    "wheel",
    "Cython>=0.29,<3.1"
]
build-backend = "setuptools.build_meta"
"""

P4A_VERSION = "2026.5.9"
PYTHON = "3.14.4"        # python-for-android 2026.5.9 pins 3.14.2; Ubuntu carries 3.14.4
SQLITE = "3.53.4"        # pinned 3.50.4; Ubuntu carries 3.53.4 (autoconf bundle, same configure)
OPENSSL = "3.3.1"
LIBFFI = "3.4.2"
KIVY = "2.3.1"
PYJNIUS = "1.7.0"
UBUNTU = "http://archive.ubuntu.com/ubuntu/pool/main"


def fetch(url: str) -> bytes:
    print(f"  {url}")
    with urllib.request.urlopen(url, timeout=900) as r:
        return r.read()


def pypi_sdist(name: str, version: str) -> bytes:
    info = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
    url = next(u["url"] for u in info["urls"] if u["packagetype"] == "sdist")
    return fetch(url)


def retar(data: bytes, top: str, edit=None) -> bytes:
    """A .tar.gz of the archive with its top-level directory renamed to `top`.

    `edit(path, content) -> content` may rewrite files (path below `top`)."""
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf, tarfile.open(fileobj=out, mode="w:gz") as of:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            m.name = top + ("/" + parts[1] if len(parts) > 1 else "")
            if m.isfile():
                content = tf.extractfile(m).read()
                if edit is not None and len(parts) > 1:
                    content = edit(parts[1], content)
                m.size = len(content)
                of.addfile(m, io.BytesIO(content))
            else:
                of.addfile(m)
    return out.getvalue()


# CPython 3.14.3 added the "arm" and "armv8l" entries that python-for-android's
# 3.14_armv7l_fix.patch adds itself, and the patch no longer applies. Give the
# dict its 3.14.2 shape so the patch applies (it puts "arm" back; "armv8l" is
# only consulted on 32-bit ARM devices, and this app is built for arm64-v8a).
_SYSCONFIG_NEW = '''        machine = {
            "aarch64": "arm64_v8a",
            "arm": "armeabi_v7a",
            "armv7l": "armeabi_v7a",
            "armv8l": "armeabi_v7a",
            "i686": "x86",
            "x86_64": "x86_64",
        }[machine]
'''
_SYSCONFIG_OLD = '''        machine = {
            "x86_64": "x86_64",
            "i686": "x86",
            "aarch64": "arm64_v8a",
            "armv7l": "armeabi_v7a",
        }[machine]
'''


def edit_cpython(path: str, content: bytes) -> bytes:
    if path == "Lib/sysconfig/__init__.py":
        text = content.decode()
        if _SYSCONFIG_NEW not in text:
            raise SystemExit(f"CPython {PYTHON}: Lib/sysconfig/__init__.py has changed; update edit_cpython()")
        return text.replace(_SYSCONFIG_NEW, _SYSCONFIG_OLD).encode()
    return content


def pinned_requirements() -> str:
    """The spec's requirement list with the seeded Python and SQLite versions."""
    for line in (APP / "buildozer.spec").read_text().splitlines():
        if line.startswith("requirements ="):
            items = [r.strip() for r in line.split("=", 1)[1].split(",")]
            break
    else:
        raise SystemExit("buildozer.spec has no requirements line")
    pins = {"python3": PYTHON, "hostpython3": PYTHON, "sqlite3": SQLITE}
    return ",".join(f"{r}=={pins[r]}" if r in pins else r for r in items)


_SOURCE_EXTS = (".py", ".pyx", ".pxd", ".pxi", ".c", ".h", ".cfg", ".toml", ".txt", ".in")


def rezip(data: bytes, top: str) -> bytes:
    """A .zip of the tarball's files with the top-level directory renamed to `top`.

    Some sdists ship sources with Windows line endings (Kivy 2.3.1's setup.py),
    where the GitHub archive python-for-android expects has Unix ones, and its
    patches would not apply: those files get Unix line endings."""
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in tf.getmembers():
            if m.isfile():
                content = tf.extractfile(m).read()
                if m.name.endswith(_SOURCE_EXTS) and b"\r\n" in content:
                    content = content.replace(b"\r\n", b"\n")
                zf.writestr(top + "/" + m.name.split("/", 1)[1], content)
    return out.getvalue()


def seed(recipe: str, filename: str, data: bytes) -> None:
    d = PACKAGES / recipe
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_bytes(data)
    (d / f".mark-{filename}").touch()
    print(f"  -> {d / filename} ({len(data) / 1e6:.1f} MB)")


def main() -> None:
    PACKAGES.parent.mkdir(parents=True, exist_ok=True)
    if not _OLD_STORAGE.exists():
        _OLD_STORAGE.symlink_to(PACKAGES.parent.name)
    print("python-for-android from PyPI")
    sdist = pypi_sdist("python-for-android", P4A_VERSION)
    P4A_SRC.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(sdist), mode="r:*") as tf:
        top = tf.getmembers()[0].name.split("/", 1)[0]
        tf.extractall(P4A_SRC, filter="data")
    src = P4A_SRC / top
    # The sdist's MANIFEST.in leaves out the android recipe's pyproject.toml,
    # so its build runs isolated without Cython and fails. Put it back, as it
    # is in the release's git tag.
    (src / "pythonforandroid/recipes/android/src/pyproject.toml").write_text(ANDROID_PYPROJECT)
    print(f"  -> {src}")

    print("CPython from Ubuntu's pool")
    py = fetch(f"{UBUNTU}/p/python3.{PYTHON.split('.')[1]}/python3.{PYTHON.split('.')[1]}_{PYTHON}.orig.tar.xz")
    data = retar(py, f"cpython-{PYTHON}", edit_cpython)
    seed("python3", f"v{PYTHON}.tar.gz", data)
    seed("hostpython3", f"v{PYTHON}.tar.gz", data)

    print("SQLite from Ubuntu's pool")
    sq = fetch(f"{UBUNTU}/s/sqlite3/sqlite3_{SQLITE}.orig.tar.xz")
    seed("sqlite3", f"version-{SQLITE}.tar.gz", retar(sq, f"sqlite-version-{SQLITE}"))

    print("OpenSSL and libffi from GitHub release assets")
    seed("openssl", f"openssl-{OPENSSL}.tar.gz",
         fetch(f"https://github.com/openssl/openssl/releases/download/openssl-{OPENSSL}/openssl-{OPENSSL}.tar.gz"))
    seed("libffi", f"v{LIBFFI}.tar.gz",
         fetch(f"https://github.com/libffi/libffi/releases/download/v{LIBFFI}/libffi-{LIBFFI}.tar.gz"))

    print("Kivy and pyjnius from PyPI")
    seed("kivy", f"{KIVY}.zip", rezip(pypi_sdist("kivy", KIVY), f"kivy-{KIVY}"))
    seed("pyjnius", f"{PYJNIUS}.zip", rezip(pypi_sdist("pyjnius", PYJNIUS), f"pyjnius-{PYJNIUS}"))

    (P4A_SRC / "requirements").write_text(pinned_requirements() + "\n")
    print(f"  -> {P4A_SRC / 'requirements'}")

    print("\nSeeded. ./build.sh now uses this python-for-android and these versions.")


if __name__ == "__main__":
    sys.exit(main())
