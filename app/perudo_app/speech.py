"""
Speech for the app: edge-tts (Microsoft's online neural voices) renders a
line to an mp3 in the cache, Kivy plays it. Lines are queued and spoken one
after another in a worker thread, so the game never waits for the network.
Every bot keeps its own voice, rate and pitch (seeded by its name), like
perudo.tts does on the desktop.
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
import zlib
from pathlib import Path
from typing import Callable, Optional

from perudo.tts import _EDGE_VOICES, spoken


class Speaker:
    def __init__(self, cache_dir: str, play: Callable[[str], None], enabled: bool = False):
        self.cache = Path(cache_dir)
        self.play = play                 # called with the mp3 path, from the worker thread
        self.enabled = enabled
        self.error: Optional[str] = None
        self._queue: "queue.Queue[tuple[str, str, str]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="perudo-speech", daemon=True)
        self._thread.start()

    def say(self, text: str, lang: str = "nl", voice_seed: str = "") -> None:
        if self.enabled and self._queue.qsize() < 8:
            self._queue.put((text, lang, voice_seed))

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        while True:
            text, lang, seed_text = self._queue.get()
            if not self.enabled:
                continue
            try:
                path = self._render(spoken(text, lang), lang, zlib.crc32(seed_text.encode("utf-8")))
                if path is not None:
                    self.play(str(path))
            except Exception as exc:       # no network, no audio: keep playing silently
                self.error = f"{type(exc).__name__}: {exc}"

    def _render(self, text: str, lang: str, seed: int) -> Optional[Path]:
        voices = _EDGE_VOICES.get(lang.lower().split("-", 1)[0], _EDGE_VOICES["en"])
        voice = voices[seed % len(voices)]
        rate = f"{(seed >> 8) % 26 - 5:+d}%"
        pitch = f"{(seed >> 16) % 21 - 10:+d}Hz"
        key = hashlib.sha256(repr((text, voice, rate, pitch)).encode()).hexdigest()
        path = self.cache / f"{key}.mp3"
        if path.is_file():
            return path
        import edge_tts

        async def render():
            audio = b""
            async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return audio

        audio = asyncio.run(render())
        if not audio:
            return None
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(audio)
        tmp.replace(path)
        return path
