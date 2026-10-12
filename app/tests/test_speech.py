"""Speech without a phone: Android's TextToSpeech and edge-tts are stand-ins."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from perudo_app.speech import AndroidVoice, Speaker, voice_params   # noqa: E402


class FakeVoice:
    def __init__(self, name, lang, installed=True, network=False):
        self.name, self.lang, self.installed, self.network = name, lang, installed, network

    def getName(self):
        return self.name

    def getLocale(self):
        lang = self.lang
        return type("L", (), {"getLanguage": lambda _self: lang})()

    def getFeatures(self):
        installed = self.installed
        return type("F", (), {"contains": lambda _self, f: f == "notInstalled" and not installed})()

    def isNetworkConnectionRequired(self):
        return self.network


class FakeJString(str):
    """java.lang.String: the only thing pyjnius passes as a CharSequence."""


class FakeTTS:
    SUCCESS, QUEUE_ADD = 0, 1
    instances = []
    voices = []
    languages_ok = True

    def __init__(self, activity, listener):
        self.calls = []
        FakeTTS.instances.append(self)
        if listener is not None:
            listener.onInit(0)

    def getVoices(self):
        voices = FakeTTS.voices
        return type("S", (), {"toArray": lambda _self: list(voices)})()

    def setVoice(self, v):
        self.calls.append(("voice", v.getName()))

    def setLanguage(self, locale):
        self.calls.append(("language", locale))
        return 0 if FakeTTS.languages_ok else -2

    def setPitch(self, p):
        self.calls.append(("pitch", p))

    def setSpeechRate(self, r):
        self.calls.append(("rate", r))

    def speak(self, text, mode, params, utterance):
        if not isinstance(text, FakeJString):     # what pyjnius does with a plain str
            raise TypeError("No methods called speak in android/speech/tts/TextToSpeech matching your arguments")
        self.calls.append(("speak", str(text), mode))
        return 0

    def isSpeaking(self):
        return False

    def stop(self):
        pass


class FakeJnius:
    class PythonJavaClass:
        pass

    @staticmethod
    def java_method(sig):
        return lambda f: f

    @staticmethod
    def autoclass(name):
        if name == "android.speech.tts.TextToSpeech":
            return FakeTTS
        if name == "java.lang.String":
            return FakeJString
        if name == "java.util.Locale":
            return type("Locale", (), {"forLanguageTag": staticmethod(lambda tag: tag)})
        raise KeyError(name)


def android_voice():
    FakeTTS.instances.clear()
    return AndroidVoice(FakeJnius, activity=object(), init_timeout=1)


def test_each_bot_gets_its_own_voice_pitch_and_speed():
    FakeTTS.voices = [FakeVoice("nl-a", "nl"), FakeVoice("nl-b", "nl"), FakeVoice("en-a", "en"),
                      FakeVoice("nl-c", "nl", installed=False)]
    voice = android_voice()
    voice.say("Dudo!", "nl", seed=1)
    voice.say("Dudo!", "nl", seed=2)
    calls = FakeTTS.instances[0].calls
    picked = [c[1] for c in calls if c[0] == "voice"]
    assert picked == ["nl-b", "nl-a"]                       # installed Dutch voices only, by seed
    assert [c for c in calls if c[0] == "speak"] == [("speak", "Dudo!", FakeTTS.QUEUE_ADD)] * 2
    assert voice_params(1) != voice_params(2)


def test_without_voices_for_the_language_it_says_what_to_install():
    FakeTTS.voices = []
    FakeTTS.languages_ok = False
    try:
        voice = android_voice()
        speaker = Speaker("/nonexistent", play=lambda p: None, enabled=True, engine="system", system=voice)
        speaker.speak_now("Calza!", "nl", "Sniper")
        assert "no NL voice installed" in speaker.error
    finally:
        FakeTTS.languages_ok = True


def test_microsoft_refusing_falls_back_to_the_phone(tmp_path):
    FakeTTS.voices = [FakeVoice("nl-a", "nl")]
    voice = android_voice()
    speaker = Speaker(str(tmp_path), play=lambda p: None, enabled=True, engine="edge", system=voice)

    def refused(*_a):
        raise ConnectionError("403, message='Invalid response status'")

    speaker._render = refused
    speaker.speak_now("Ik geloof er niets van", "nl", "Trihard")
    assert "403" in speaker.error and "phone's own voices" in speaker.note
    assert ("speak", "Ik geloof er niets van", FakeTTS.QUEUE_ADD) in FakeTTS.instances[0].calls


def test_the_system_engine_off_android_explains_itself(tmp_path):
    speaker = Speaker(str(tmp_path), play=lambda p: None, enabled=True, engine="system", system=None)
    speaker.speak_now("Dudo!", "nl", "Sniper")
    assert "only available on Android" in speaker.error
    assert not speaker.has_system_voices


def test_lines_are_spoken_in_words():
    FakeTTS.voices = [FakeVoice("nl-a", "nl")]
    voice = android_voice()
    speaker = Speaker("/nonexistent", play=lambda p: None, enabled=True, engine="system", system=voice)
    speaker.say("4x 5", "nl", "Sniper")
    speaker.wait_idle()
    spoken = [c[1] for c in FakeTTS.instances[0].calls if c[0] == "speak"]
    assert spoken and "x" not in spoken[0]                  # "vier vijven", not "4x 5"
