"""
Picking a bot file to import.

On Android the app cannot look around the phone's storage itself: since
Android 13 (and with scoped storage before that) another app's files, such
as a .py in Downloads, are invisible to it, whatever permission it asks
for. So it asks Android's own document picker (Downloads, Drive, ...),
which hands over just the file the person chose; the app copies it into its
cache and imports that copy.

Elsewhere (the desktop, the iPhone) the app's own file browser does: on an
iPhone that is the app's folder, where the Files app puts files.

`jnius`, `activity` (android.activity, for the result) and `context` (the
Android activity) can be stand-ins in tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

REQUEST_CODE = 0x5EB0           # ours among the activity results
RESULT_OK = -1                  # android.app.Activity.RESULT_OK


class AndroidPicker:
    """Android's document picker: pick(done) shows it; done(path, error)
    follows with the picked file copied into the app's cache (path None
    when nothing was picked or it could not be read)."""

    def __init__(self, jnius: Any, activity: Any, context: Any):
        self.jnius = jnius
        self.activity = activity
        self.context = context
        self._done: Optional[Callable[[Optional[str], Optional[str]], None]] = None
        self.activity.bind(on_activity_result=self._on_result)

    @classmethod
    def create(cls) -> Optional["AndroidPicker"]:
        try:
            import jnius   # type: ignore
            from android import activity, mActivity   # type: ignore
        except Exception:
            return None
        return cls(jnius, activity, mActivity)

    def pick(self, done: Callable[[Optional[str], Optional[str]], None]) -> None:
        Intent = self.jnius.autoclass("android.content.Intent")
        intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.setType("*/*")              # a .py has no type Android knows reliably
        self._done = done
        self.context.startActivityForResult(intent, REQUEST_CODE)

    def _on_result(self, request_code, result_code, intent) -> None:
        if request_code != REQUEST_CODE:
            return
        done, self._done = self._done, None
        if done is None:
            return
        if result_code != RESULT_OK or intent is None or intent.getData() is None:
            done(None, None)                # cancelled
            return
        try:
            done(self._copy(intent.getData()), None)
        except Exception as exc:            # a file Android could not hand over
            done(None, f"{type(exc).__name__}: {exc}")

    def _copy(self, uri) -> str:
        """Read the picked document and keep it in the app's cache, under
        the name it had."""
        resolver = self.context.getContentResolver()
        name = self._display_name(resolver, uri) or "bot.py"
        if not name.endswith(".py"):
            name += ".py"
        InputStreamReader = self.jnius.autoclass("java.io.InputStreamReader")
        BufferedReader = self.jnius.autoclass("java.io.BufferedReader")
        stream = resolver.openInputStream(uri)
        try:
            reader = BufferedReader(InputStreamReader(stream, "UTF-8"))
            lines = []
            while True:
                line = reader.readLine()
                if line is None:
                    break
                lines.append(line)
        finally:
            stream.close()
        folder = Path(self.context.getCacheDir().getAbsolutePath()) / "picked"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / Path(name).name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    @staticmethod
    def _display_name(resolver, uri) -> Optional[str]:
        cursor = resolver.query(uri, None, None, None, None)
        if cursor is None:
            return None
        try:
            column = cursor.getColumnIndex("_display_name")      # OpenableColumns.DISPLAY_NAME
            if column >= 0 and cursor.moveToFirst():
                return cursor.getString(column)
        finally:
            cursor.close()
        return None
