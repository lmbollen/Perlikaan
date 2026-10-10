"""
Optional text-to-speech for bots that like to talk.

Zet TTS=true in je environment om pratende bots hardop te laten spreken.
Met TTS_BACKEND kies je de stemmen:

  edge   (standaard) Microsofts online neurale stemmen via edge-tts. Klinkt
         het best, maar heeft internet nodig. Afspelen via ffplay (of ffmpeg
         + aplay/paplay). Gegenereerde audio wordt bewaard in EDGE_CACHE_DIR
         (standaard ~/.cache/perudo/edge-tts), dus elke zin gaat maar één
         keer over het netwerk.
  piper  Lokaal en offline. De stemmodellen worden bij het eerste gebruik
         gedownload naar PIPER_VOICES_DIR (standaard
         ~/.cache/perudo/piper-voices). Afspelen via aplay, paplay of ffplay.

`uv sync` installeert beide. Werkt de spraak niet, dan krijg je één
waarschuwing en speelt het spel gewoon (stil) verder.

Gebruik vanuit een bot::

    from perudo.tts import speak, tts_enabled
    speak("Dudo op 3x 5!", lang="nl", voice_seed=self.name)  # "Dudo op 3 vijven!"
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import zlib
from pathlib import Path

# Per language a pool of models; `voice_seed` picks one (and a speaker, for
# multi-speaker models), so every bot keeps its own stable voice.
# Avoid the Dutch "mls" models: they produce gibberish.
_MODELS = {
    "en": ["en_US-libritts_r-medium"],
    "nl": ["nl_NL-ronnie-medium", "nl_NL-pim-medium", "nl_NL-alex-medium"],
}

# Edge voices per language. Every bot gets one of these plus its own pitch
# and speed, so bots that share a voice still sound different.
_EDGE_VOICES = {
    "en": ["en-US-AndrewNeural", "en-US-AvaNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural"],
    "nl": ["nl-NL-MaartenNeural", "nl-NL-ColetteNeural", "nl-NL-FennaNeural",
           "nl-BE-ArnaudNeural", "nl-BE-DenaNeural"],
}

# (singular, plural) per face, so a bid can be spoken as "3 vijven".
_FACE_WORDS = {
    "nl": {1: ("Paco", "Paco's"), 2: ("twee", "tweeën"), 3: ("drie", "drieën"),
           4: ("vier", "vieren"), 5: ("vijf", "vijven"), 6: ("zes", "zessen")},
    "en": {1: ("Paco", "Pacos"), 2: ("two", "twos"), 3: ("three", "threes"),
           4: ("four", "fours"), 5: ("five", "fives"), 6: ("six", "sixes")},
}
_BID_RE = re.compile(r"\b(\d+)x (paco|[1-6])\b", re.IGNORECASE)

_voices: dict[str, object] = {}
_failed = False


def tts_enabled() -> bool:
    return os.environ.get("TTS", "").strip().lower() in ("1", "true", "yes", "on")


def _backend() -> str:
    return os.environ.get("TTS_BACKEND", "edge").strip().lower()


def _lang_key(lang: str) -> str:
    return lang.lower().split("-", 1)[0]


def face_name(face: int, quantity: int = 1, lang: str = "nl") -> str:
    """How to say a face out loud: face_name(5, 3) == "vijven"."""
    singular, plural = _FACE_WORDS.get(_lang_key(lang), _FACE_WORDS["en"])[face]
    return singular if quantity == 1 else plural


def spoken(text: str, lang: str = "nl") -> str:
    """Rewrite bids like "3x 5" to "3 vijven", so TTS doesn't read the x as a letter."""

    def bid_words(m: re.Match) -> str:
        quantity = int(m.group(1))
        face = 1 if m.group(2).lower() == "paco" else int(m.group(2))
        return f"{quantity} {face_name(face, quantity, lang)}"

    return _BID_RE.sub(bid_words, text)


def _voices_dir() -> Path:
    default = Path.home() / ".cache" / "perudo" / "piper-voices"
    return Path(os.environ.get("PIPER_VOICES_DIR", default)).expanduser()


def _edge_cache_dir() -> Path:
    default = Path.home() / ".cache" / "perudo" / "edge-tts"
    return Path(os.environ.get("EDGE_CACHE_DIR", default)).expanduser()


def _get_voice(lang: str, seed: int = 0):
    lang = _lang_key(lang)
    models = _MODELS.get(lang, _MODELS["en"])
    model = models[seed % len(models)]
    if model not in _voices:
        from piper import PiperVoice
        from piper.download_voices import download_voice

        voices_dir = _voices_dir()
        voices_dir.mkdir(parents=True, exist_ok=True)
        model_path = voices_dir / f"{model}.onnx"
        if not model_path.exists():
            print(f"[TTS] Stem {model} downloaden naar {voices_dir}...", flush=True)
            download_voice(model, voices_dir)
        _voices[model] = PiperVoice.load(model_path)
    return _voices[model]


def _player_cmd(sample_rate: int) -> list[str]:
    if shutil.which("aplay"):
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(sample_rate), "-"]
    if shutil.which("paplay"):
        return ["paplay", "--raw", "--format=s16le", "--channels=1", f"--rate={sample_rate}"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-nostats", "-loglevel", "quiet",
                "-f", "s16le", "-ar", str(sample_rate), "-ch_layout", "mono", "-"]
    raise RuntimeError("geen audiospeler gevonden (aplay, paplay of ffplay)")


def _speak_piper(text: str, lang: str, seed: int) -> None:
    from piper import SynthesisConfig

    voice = _get_voice(lang, seed)
    config = SynthesisConfig(
        speaker_id=(seed >> 16) % max(voice.config.num_speakers, 1),
        length_scale=0.85 + (seed >> 8) % 30 / 100,  # 0.85..1.14, lower is faster
    )
    chunks = voice.synthesize(text, syn_config=config)
    audio = b"".join(chunk.audio_int16_bytes for chunk in chunks)
    _play(_player_cmd(voice.config.sample_rate), audio)


def _play(cmd: list[str], audio: bytes) -> None:
    # Players like ffplay write status output (e.g. a trailing newline) even
    # when told to be quiet; keep it out of the game's terminal.
    subprocess.run(cmd, input=audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


async def _edge_mp3(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts

    audio = b""
    async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio


def _play_mp3(audio: bytes) -> None:
    if shutil.which("ffplay"):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-nostats", "-loglevel", "quiet", "-"]
        _play(cmd, audio)
    elif shutil.which("ffmpeg"):
        decode = ["ffmpeg", "-loglevel", "quiet", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", "24000", "-"]
        pcm = subprocess.run(decode, input=audio, capture_output=True, check=True).stdout
        _play(_player_cmd(24000), pcm)
    else:
        raise RuntimeError("edge-tts levert mp3; installeer ffmpeg (ffplay) om af te spelen")


def _speak_edge(text: str, lang: str, seed: int) -> None:
    voices = _EDGE_VOICES.get(_lang_key(lang), _EDGE_VOICES["en"])
    voice = voices[seed % len(voices)]
    rate = f"{(seed >> 8) % 26 - 5:+d}%"     # -5%..+20%
    pitch = f"{(seed >> 16) % 21 - 10:+d}Hz"  # -10Hz..+10Hz
    key = (text, voice, rate, pitch)
    path = _edge_cache_dir() / f"{hashlib.sha256(repr(key).encode()).hexdigest()}.mp3"
    if path.is_file():
        _play_mp3(path.read_bytes())
        return
    audio = asyncio.run(_edge_mp3(text, voice, rate, pitch))
    if audio:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(audio)
            tmp.replace(path)  # atomic, so a crash never leaves a half mp3
        except OSError:
            pass  # cache is best-effort; still play what we fetched
    _play_mp3(audio)


def speak(text: str, lang: str = "nl", voice_seed: str = "") -> None:
    """
    Say `text` out loud (blocking) if TTS is enabled. `voice_seed` (e.g. the
    bot's name) picks a stable voice, pitch and speaking rate, so each bot
    keeps sounding the same.
    """
    global _failed
    if not tts_enabled() or _failed:
        return
    seed = zlib.crc32(voice_seed.encode("utf-8"))
    text = spoken(text, lang)
    try:
        if _backend() == "piper":
            _speak_piper(text, lang, seed)
        else:
            _speak_edge(text, lang, seed)
    except Exception as exc:  # missing module, no internet, no audio...
        _failed = True
        print(f"[TTS] Spraak uitgeschakeld ({_backend()}): {exc}", flush=True)
