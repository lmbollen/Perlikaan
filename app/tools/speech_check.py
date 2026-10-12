"""
Does Microsoft's read-aloud service still accept edge-tts? Synthesises one
Dutch line with edge-tts as installed, then with newer browser versions in
its handshake, and reports which work. Run where the service is reachable
(the "Speech check" workflow runs it on GitHub's runners):

    cd app
    uv run python tools/speech_check.py
"""

from __future__ import annotations

import asyncio
import sys

import edge_tts
from edge_tts import communicate, constants

LINE = "Dudo! Ik geloof er niets van."
VOICE = "nl-NL-MaartenNeural"


def use_version(full: str) -> None:
    """Make edge-tts present itself as Edge `full` (e.g. 143.0.3650.75)."""
    major = full.split(".", 1)[0]
    ua = (f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
          f" Chrome/{major}.0.0.0 Safari/537.36 Edg/{major}.0.0.0")
    for module in (constants, communicate):
        if hasattr(module, "SEC_MS_GEC_VERSION"):
            module.SEC_MS_GEC_VERSION = f"1-{full}"
    for headers in (constants.BASE_HEADERS, constants.WSS_HEADERS, constants.VOICE_HEADERS):
        headers["User-Agent"] = ua
    if "Sec-CH-UA" in constants.VOICE_HEADERS:
        constants.VOICE_HEADERS["Sec-CH-UA"] = (f'" Not;A Brand";v="99", "Microsoft Edge";v="{major}",'
                                                f' "Chromium";v="{major}"')


async def synthesise() -> int:
    size = 0
    async for chunk in edge_tts.Communicate(LINE, VOICE).stream():
        if chunk["type"] == "audio":
            size += len(chunk["data"])
    return size


def attempt(label: str) -> bool:
    try:
        size = asyncio.run(synthesise())
        print(f"{label}: OK, {size} bytes of audio")
        return size > 0
    except Exception as exc:
        print(f"{label}: FAILED, {type(exc).__name__}: {str(exc)[:200]}")
        return False


def main() -> int:
    print(f"edge-tts {getattr(edge_tts, '__version__', '?')}, Sec-MS-GEC-Version {constants.SEC_MS_GEC_VERSION}")
    ok = attempt("as installed")
    for full in ("146.0.3800.58", "148.0.3967.54", "150.0.4050.0", "152.0.4200.0"):
        use_version(full)
        attempt(f"as Edge {full}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
