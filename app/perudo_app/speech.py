"""
Speech for the app. Lines are queued and spoken one after another in a
worker thread, so the game never waits for speech.

Two engines:

- "system": the phone's own text-to-speech (Android's TextToSpeech, the
  voices installed on the device). Works offline. Every bot gets its own
  voice (when several are installed), pitch and speed, seeded by its name.
- "edge": Microsoft's online neural voices through edge-tts, rendered to an
  mp3 in the cache and played by Kivy, like perudo.tts on the desktop.
  Microsoft changes what that service accepts from time to time (since
  October 2026 it refuses edge-tts with a 403); when it fails, the line is
  spoken by the system voices instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, List, Optional

from perudo.tts import _EDGE_VOICES, spoken

ENGINES = ("system", "edge")
_LOCALES = {"nl": "nl-NL", "en": "en-US"}


def voice_params(seed: int):
    """(voice index, pitch, speech rate) for a bot, from its seed."""
    pitch = 0.8 + ((seed >> 4) % 11) * 0.06          # 0.80 .. 1.40
    rate = 0.95 + ((seed >> 12) % 6) * 0.05          # 0.95 .. 1.20
    return seed, round(pitch, 2), round(rate, 2)


class AndroidVoice:
    """The device's TextToSpeech engine, through pyjnius. Created lazily in
    the speech thread; `jnius` and `activity` can be stand-ins in tests."""

    def __init__(self, jnius: Any, activity: Any, init_timeout: float = 6.0):
        self.jnius = jnius
        self.activity = activity
        self.init_timeout = init_timeout
        self._tts = None
        self._listener = None            # keep a reference: Java holds it weakly
        self._voices: dict = {}
        self._count = 0

    @classmethod
    def create(cls) -> Optional["AndroidVoice"]:
        try:
            import jnius   # type: ignore
            from android import mActivity   # type: ignore
        except Exception:
            return None
        return cls(jnius, mActivity)

    def _engine(self):
        if self._tts is not None:
            return self._tts
        autoclass = self.jnius.autoclass
        TextToSpeech = autoclass("android.speech.tts.TextToSpeech")
        ready = threading.Event()
        status = {"code": None}
        listener = None
        if hasattr(self.jnius, "PythonJavaClass"):
            base, java_method = self.jnius.PythonJavaClass, self.jnius.java_method

            class Init(base):
                __javainterfaces__ = ["android/speech/tts/TextToSpeech$OnInitListener"]
                __javacontext__ = "app"

                @java_method("(I)V")
                def onInit(self, code):
                    status["code"] = code
                    ready.set()

            listener = Init()
        self._listener = listener
        tts = TextToSpeech(self.activity, listener)
        if listener is not None and not ready.wait(self.init_timeout):
            raise RuntimeError("the phone's text-to-speech engine did not start")
        if status["code"] not in (None, TextToSpeech.SUCCESS):
            raise RuntimeError("the phone has no text-to-speech engine (install one in Android's settings)")
        self._tts = tts
        return tts

    def _voices_for(self, lang: str) -> List[Any]:
        if lang in self._voices:
            return self._voices[lang]
        found = []
        try:
            for voice in self._tts.getVoices().toArray():
                if voice.getLocale().getLanguage() != lang:
                    continue
                features = voice.getFeatures()
                if features is not None and features.contains("notInstalled"):
                    continue
                found.append(voice)
            offline = [v for v in found if not v.isNetworkConnectionRequired()]
            found = sorted(offline or found, key=lambda v: v.getName())
        except Exception:
            found = []
        self._voices[lang] = found
        return found

    def say(self, text: str, lang: str, seed: int) -> None:
        tts = self._engine()
        TextToSpeech = self.jnius.autoclass("android.speech.tts.TextToSpeech")
        key = lang.lower().split("-", 1)[0]
        voices = self._voices_for(key)
        index, pitch, rate = voice_params(seed)
        if voices:
            tts.setVoice(voices[index % len(voices)])
        else:
            Locale = self.jnius.autoclass("java.util.Locale")
            if tts.setLanguage(Locale.forLanguageTag(_LOCALES.get(key, "en-US"))) < 0:
                raise RuntimeError(f"no {key.upper()} voice installed on the phone "
                                   "(Android settings: text-to-speech, install voice data)")
        tts.setPitch(pitch)
        tts.setSpeechRate(rate)
        self._count += 1
        # speak(CharSequence, int, Bundle, String): pyjnius does not turn a Python str into a
        # CharSequence by itself, so hand it a java.lang.String
        JString = self.jnius.autoclass("java.lang.String")
        if tts.speak(JString(text), TextToSpeech.QUEUE_ADD, None, f"perudo-{self._count}") != TextToSpeech.SUCCESS:
            raise RuntimeError("the phone's text-to-speech refused the line")
        # wait for this line, so the next bot's voice settings do not apply to it
        time.sleep(0.2)
        end = time.monotonic() + 30
        while tts.isSpeaking() and time.monotonic() < end:
            time.sleep(0.05)

    def stop(self) -> None:
        if self._tts is not None:
            try:
                self._tts.stop()
            except Exception:
                pass


class Speaker:
    def __init__(self, cache_dir: str, play: Callable[[str], None], enabled: bool = False,
                 engine: str = "system", system: Any = "auto"):
        self.cache = Path(cache_dir)
        self.play = play                 # called with the mp3 path, from the worker thread
        self.enabled = enabled
        self.engine = engine if engine in ENGINES else "system"
        self.system = AndroidVoice.create() if system == "auto" else system
        self.error: Optional[str] = None     # the last problem, for the settings screen
        self.note: Optional[str] = None      # e.g. that the system voices stood in
        self._queue: "queue.Queue[tuple[str, str, str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0                    # lines queued or being spoken
        self._thread = threading.Thread(target=self._run, name="perudo-speech", daemon=True)
        self._thread.start()

    @property
    def has_system_voices(self) -> bool:
        return self.system is not None

    def say(self, text: str, lang: str = "nl", voice_seed: str = "") -> None:
        if self.enabled and self._queue.qsize() < 8:
            with self._lock:
                self._pending += 1
            self._queue.put((text, lang, voice_seed))

    @property
    def idle(self) -> bool:
        """Nothing queued and nothing being spoken: a game may go on."""
        return self._pending == 0

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending -= 1
        if self.system is not None and hasattr(self.system, "stop"):
            self.system.stop()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while not self.idle and time.monotonic() < end:
            time.sleep(0.01)
        return self.idle

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self.speak_now(*item)
            except Exception as exc:             # never let the speech thread die
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self._pending -= 1

    def speak_now(self, text: str, lang: str, seed_text: str) -> None:
        if not self.enabled:
            return
        line = spoken(text, lang)
        seed = zlib.crc32(seed_text.encode("utf-8"))
        if self.engine == "edge":
            try:
                path = self._render(line, lang, seed)
                if path is not None:
                    self.play(str(path))
                self.error = self.note = None
                return
            except Exception as exc:       # Microsoft refused, or no network: use the phone's voices
                self.error = f"Microsoft's voices: {type(exc).__name__}: {str(exc)[:160]}"
                if self.system is None:
                    return
                self.note = "Microsoft's voices failed; the phone's own voices are speaking instead."
        if self.system is None:
            self.error = "The built-in voices are only available on Android."
            return
        try:
            self.system.say(line, lang, seed)
            if self.engine == "system":
                self.error = self.note = None
        except Exception as exc:
            self.error = f"The phone's voices: {exc}"

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
