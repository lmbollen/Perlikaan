"""The Android document picker, with stand-ins for Android's classes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from perudo_app.core import BotRegistry                     # noqa: E402
from perudo_app.picker import REQUEST_CODE, RESULT_OK, AndroidPicker   # noqa: E402

BOT = '''from perudo.models import Dudo


class PickedBot:
    name = "Picked"

    def play(self, state):
        return state.legal_actions[0]
'''


class FakeIntent:
    ACTION_OPEN_DOCUMENT = "open"
    CATEGORY_OPENABLE = "openable"

    def __init__(self, action):
        self.action, self.categories, self.type = action, [], None

    def addCategory(self, c):
        self.categories.append(c)

    def setType(self, t):
        self.type = t


class Reader:
    def __init__(self, stream, *_):
        self.lines = stream.text.splitlines()          # as Java's readLine gives them

    def readLine(self):
        return self.lines.pop(0) if self.lines else None


class Stream:
    def __init__(self, text):
        self.text, self.closed = text, False

    def close(self):
        self.closed = True


class Cursor:
    def __init__(self, name):
        self.name = name

    def getColumnIndex(self, column):
        return 0 if column == "_display_name" else -1

    def moveToFirst(self):
        return True

    def getString(self, _i):
        return self.name

    def close(self):
        pass


class Resolver:
    def __init__(self, name, text):
        self.name, self.stream = name, Stream(text)

    def query(self, *_):
        return Cursor(self.name)

    def openInputStream(self, _uri):
        return self.stream


class Folder:
    def __init__(self, path):
        self.path = path

    def getAbsolutePath(self):
        return str(self.path)


class Context:
    def __init__(self, tmp, name, text):
        self.tmp, self.resolver, self.started = tmp, Resolver(name, text), []

    def getContentResolver(self):
        return self.resolver

    def getCacheDir(self):
        return Folder(self.tmp)

    def startActivityForResult(self, intent, code):
        self.started.append((intent, code))


class Activity:
    def __init__(self):
        self.callbacks = []

    def bind(self, on_activity_result):
        self.callbacks.append(on_activity_result)


class Jnius:
    @staticmethod
    def autoclass(name):
        return {"android.content.Intent": FakeIntent, "java.io.BufferedReader": lambda r: r,
                "java.io.InputStreamReader": Reader}[name]


class Result:
    def __init__(self, uri):
        self.uri = uri

    def getData(self):
        return self.uri


def make(tmp_path, name="clever_bot.py", text=BOT):
    activity, context = Activity(), Context(tmp_path, name, text)
    return AndroidPicker(Jnius, activity, context), activity, context


def test_a_picked_file_is_copied_and_imports(tmp_path):
    picker, activity, context = make(tmp_path)
    got = []
    picker.pick(lambda path, error: got.append((path, error)))
    intent, code = context.started[0]
    assert intent.action == "open" and intent.categories == ["openable"] and intent.type == "*/*"
    activity.callbacks[0](code, RESULT_OK, Result("content://downloads/42"))
    path, error = got[0]
    assert error is None and Path(path).name == "clever_bot.py"
    assert Path(path).read_text() == BOT
    assert context.resolver.stream.closed
    reg = BotRegistry(str(tmp_path / "bots"))
    added = reg.import_file(path)
    assert [e.class_name for e in added] == ["PickedBot"]
    assert (tmp_path / "bots" / "clever_bot.py").exists()     # kept in the app


def test_cancelling_or_another_request_changes_nothing(tmp_path):
    picker, activity, context = make(tmp_path)
    got = []
    picker.pick(lambda path, error: got.append((path, error)))
    activity.callbacks[0](REQUEST_CODE + 1, RESULT_OK, Result("content://x"))   # not ours
    assert got == []
    activity.callbacks[0](REQUEST_CODE, 0, None)                                 # cancelled
    assert got == [(None, None)]


def test_a_name_without_py_gets_it_and_a_failing_read_says_why(tmp_path):
    picker, activity, _ = make(tmp_path, name="mybot")
    got = []
    picker.pick(lambda path, error: got.append((path, error)))
    activity.callbacks[0](REQUEST_CODE, RESULT_OK, Result("content://y"))
    assert Path(got[0][0]).name == "mybot.py"

    picker, activity, context = make(tmp_path)
    context.resolver.openInputStream = lambda _uri: (_ for _ in ()).throw(IOError("permission denied"))
    got = []
    picker.pick(lambda path, error: got.append((path, error)))
    activity.callbacks[0](REQUEST_CODE, RESULT_OK, Result("content://z"))
    assert got[0][0] is None and "permission denied" in got[0][1]
