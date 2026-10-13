"""
Playing together over the internet, through Ably (a hosted message relay),
so nobody needs the same Wi-Fi and nobody runs a server.

Every lobby has a random four-character code and a few Ably channels:

  lobby:<code>:host      everything the guests send to the host
  lobby:<code>:<token>   everything the host sends to one guest (the token is
                         random, so only that guest knows the channel)

A side subscribes to its own channel (a server-sent event stream) and
publishes to the other side's with a plain HTTPS POST. The messages are the
same JSON objects as on the local network (see perudo_app.multiplayer); a
Link batches whatever is queued into one Ably message, so a burst of game
events costs one request. Both sides send a ping every few seconds; a side
that stays silent too long has left.

The host stays in charge of the game, exactly as on the local network: Ably
only carries the messages.

The Ably key comes from the ABLY_KEY environment variable, or from
perudo_app/online_key.py, which the build writes (it is not in the
repository). Give the key only the publish and subscribe capabilities on
"lobby:*".
"""

from __future__ import annotations

import base64
import json
import os
import queue
import secrets
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from perudo_app import net

PREFIX = "lobby:"
PING_EVERY = 4.0          # seconds between pings while nothing else is sent
GONE_AFTER = 20.0         # a side silent this long has left
MAX_BATCH = 48 * 1024     # bytes of JSON per Ably message (Ably allows 64 KiB)
REST = "https://rest.ably.io"
REALTIME = "https://realtime.ably.io"


def ably_key() -> Optional[str]:
    key = os.environ.get("ABLY_KEY", "").strip()
    if not key:
        try:
            from perudo_app.online_key import KEY   # type: ignore  # written by build.sh
            key = (KEY or "").strip()
        except Exception:
            key = ""
    return key if ":" in key else None


def new_code() -> str:
    return "".join(secrets.choice(net.ALPHABET) for _ in range(4))


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile:
        try:
            import certifi                       # the phone has no system CA bundle for Python
            cafile = certifi.where()
        except Exception:
            cafile = None
    return ssl.create_default_context(cafile=cafile)


# ---------------------------------------------------------------------------
# backends: where messages actually go
# ---------------------------------------------------------------------------

class Subscription:
    def close(self) -> None:
        raise NotImplementedError


class AblyBackend:
    """Ably over HTTPS: publish with the REST API, receive with server-sent events."""

    def __init__(self, key: str):
        self.key = key
        self._auth = "Basic " + base64.b64encode(key.encode()).decode()
        self._ssl = _ssl_context()

    def publish(self, channel: str, name: str, data: str) -> None:
        url = f"{REST}/channels/{urllib.parse.quote(channel, safe='')}/messages"
        body = json.dumps({"name": name, "data": data}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": self._auth, "Content-Type": "application/json"})
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10, context=self._ssl) as resp:
                    resp.read()
                return
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403, 404):
                    raise ConnectionError(_ably_error(exc)) from exc
                last = exc
            except OSError as exc:
                last = exc
            time.sleep(0.5 * (attempt + 1))
        raise ConnectionError(f"could not reach the relay ({last})")

    def subscribe(self, channel: str, on_message: Callable[[str, str], None],
                  on_error: Callable[[str], None]) -> Subscription:
        return _SseSubscription(self, channel, on_message, on_error)


def _ably_error(exc: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(exc.read().decode()).get("error", {}).get("message", "")
    except Exception:
        detail = ""
    if exc.code == 401:
        return "The online key was refused" + (f" ({detail})" if detail else "")
    return f"The relay said {exc.code}" + (f": {detail}" if detail else "")


class _SseSubscription(Subscription):
    """One channel as a stream of server-sent events, reconnecting (and
    resuming after the last event seen) when the stream drops."""

    def __init__(self, backend: AblyBackend, channel: str, on_message, on_error):
        self.backend = backend
        self.channel = channel
        self.on_message = on_message
        self.on_error = on_error
        self.closed = False
        self._resp = None
        self._last_id: Optional[str] = None
        self.ready = threading.Event()
        threading.Thread(target=self._run, name="perudo-relay-in", daemon=True).start()

    def close(self) -> None:
        self.closed = True
        resp = self._resp
        if resp is not None:
            # The reading thread sits in readline() holding the response's
            # lock, so resp.close() would wait until Ably next sends
            # something (up to a minute). Shutting the socket down wakes the
            # reader at once; closing the response is left to it.
            try:
                resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                threading.Thread(target=resp.close, daemon=True).start()

    def _run(self) -> None:
        failures = 0
        while not self.closed:
            query = urllib.parse.urlencode({"v": "1.2", "channels": self.channel, "enveloped": "true"})
            headers = {"Authorization": self.backend._auth, "Accept": "text/event-stream"}
            if self._last_id:
                headers["Last-Event-ID"] = self._last_id
            req = urllib.request.Request(f"{REALTIME}/sse?{query}", headers=headers)
            try:
                self._resp = urllib.request.urlopen(req, timeout=60, context=self.backend._ssl)
                self.ready.set()
                failures = 0
                self._read(self._resp)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403):
                    self.on_error(_ably_error(exc))
                    return
                failures += 1
            except Exception:
                failures += 1
            finally:
                resp, self._resp = self._resp, None
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            if not self.closed:
                if failures >= 6:
                    self.on_error("Lost the connection to the relay")
                    return
                time.sleep(min(5.0, 0.5 * (failures + 1)))

    def _read(self, resp) -> None:
        data: List[str] = []
        event_id = None
        while not self.closed:
            raw = resp.readline()
            if not raw:
                return
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data:
                    self._dispatch("\n".join(data), event_id)
                data, event_id = [], None
            elif line.startswith(":"):
                continue                                  # a keep-alive comment
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif line.startswith("id:"):
                event_id = line[3:].strip()

    def _dispatch(self, payload: str, event_id: Optional[str]) -> None:
        try:
            msg = json.loads(payload)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        if event_id:
            self._last_id = event_id
        if "action" in msg and "data" not in msg:         # a protocol message, not a message
            return
        body = msg.get("data")                            # we only publish strings
        if isinstance(body, str):
            self.on_message(str(msg.get("name") or ""), body)


class MemoryBackend:
    """An in-process relay with the same behaviour, for tests."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self._subs: Dict[str, List["_MemorySubscription"]] = {}
        self._lock = threading.Lock()
        self.published = 0

    def publish(self, channel: str, name: str, data: str) -> None:
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.published += 1
            subs = list(self._subs.get(channel, []))
        for sub in subs:
            sub.queue.put((name, data))

    def subscribe(self, channel: str, on_message, on_error) -> Subscription:
        sub = _MemorySubscription(self, channel, on_message)
        with self._lock:
            self._subs.setdefault(channel, []).append(sub)
        return sub


class _MemorySubscription(Subscription):
    def __init__(self, backend: MemoryBackend, channel: str, on_message):
        self.backend, self.channel, self.on_message = backend, channel, on_message
        self.queue: "queue.Queue" = queue.Queue()
        self.closed = False
        self.ready = threading.Event()
        self.ready.set()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.closed:
            item = self.queue.get()
            if item is None:
                return
            self.on_message(*item)

    def close(self) -> None:
        self.closed = True
        with self.backend._lock:
            subs = self.backend._subs.get(self.channel, [])
            if self in subs:
                subs.remove(self)
        self.queue.put(None)


# ---------------------------------------------------------------------------
# a link: one side's view of the other, shaped like net.Connection
# ---------------------------------------------------------------------------

class Link:
    """Sends to one channel, under this side's name; messages from the other
    side are handed in with receive(). The same interface as net.Connection
    (send, close, closed, member_id), so the lobby code does not care."""

    def __init__(self, backend, channel: str, sender: str,
                 on_message: Callable[["Link", dict], None], on_close: Callable[["Link"], None]):
        self.backend = backend
        self.channel = channel
        self.sender = sender
        self.on_message = on_message
        self.on_close = on_close
        self.member_id: Optional[int] = None
        self.closed = False
        self.error: Optional[str] = None
        self.last_heard = time.monotonic()
        self._out: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._closed_lock = threading.Lock()
        threading.Thread(target=self._send_loop, name="perudo-relay-out", daemon=True).start()

    def start(self) -> "Link":
        return self

    def send(self, msg: Dict[str, Any]) -> bool:
        if self.closed:
            return False
        self._out.put(msg)
        return True

    def close(self, say_bye: bool = True) -> None:
        with self._closed_lock:
            if self.closed:
                return
            self.closed = True
        self._out.put({"t": "bye"} if say_bye else None)
        try:
            self.on_close(self)
        except Exception:
            pass

    def receive(self, batch: List[dict]) -> None:
        self.last_heard = time.monotonic()
        for msg in batch:
            if self.closed:
                return
            kind = msg.get("t") if isinstance(msg, dict) else None
            if kind == "ping":
                continue
            if kind == "bye":
                self.close(say_bye=False)
                return
            self.on_message(self, msg)

    def _send_loop(self) -> None:
        last_sent = time.monotonic()
        while True:
            try:
                first = self._out.get(timeout=PING_EVERY)
            except queue.Empty:
                if self.closed:
                    return
                first = {"t": "ping"}
            batch = [first] if first is not None else []
            while True:                                   # take whatever else is waiting
                try:
                    batch.append(self._out.get_nowait())
                except queue.Empty:
                    break
            stop = None in batch or (self.closed and batch and batch[-1] == {"t": "bye"})
            batch = [m for m in batch if m is not None]
            for chunk in _chunks(batch):
                try:
                    self.backend.publish(self.channel, self.sender, chunk)
                    last_sent = time.monotonic()
                except Exception as exc:
                    if not self.closed:
                        self.error = str(exc)
                        self.close(say_bye=False)
                    return
            if stop or (self.closed and self._out.empty()):
                return


def _chunks(batch: List[dict]) -> List[str]:
    out, current, size = [], [], 2
    for msg in batch:
        text = json.dumps(msg, separators=(",", ":"))
        if current and size + len(text) + 1 > MAX_BATCH:
            out.append("[" + ",".join(current) + "]")
            current, size = [], 2
        current.append(text)
        size += len(text) + 1
    if current:
        out.append("[" + ",".join(current) + "]")
    return out


def _parse(data: str) -> List[dict]:
    try:
        batch = json.loads(data)
    except ValueError:
        return []
    return [m for m in batch if isinstance(m, dict)] if isinstance(batch, list) else []


def _wait_ready(sub: Subscription, timeout: float) -> None:
    ready = getattr(sub, "ready", None)
    if ready is not None and not ready.wait(timeout):
        raise ConnectionError("could not reach the relay")


# ---------------------------------------------------------------------------
# the host's side
# ---------------------------------------------------------------------------

class RelayHost:
    """Listens on the lobby's host channel; every guest becomes a Link."""

    def __init__(self, backend, on_message: Callable[[Link, dict], None], on_close: Callable[[Link], None],
                 code: Optional[str] = None, claim_wait: float = 1.5):
        self.backend = backend
        self.on_message = on_message
        self.on_close = on_close
        self.closed = False
        self.error: Optional[str] = None
        self.host_id = secrets.token_hex(6)
        self._links: Dict[str, Link] = {}
        self._lock = threading.Lock()
        self._sub: Optional[Subscription] = None
        for _attempt in range(5):
            self.code = code or new_code()
            if self._claim(claim_wait):
                break
            if self._sub is not None:
                self._sub.close()
            if code is not None:
                raise ConnectionError("that lobby code is taken")
        else:
            raise ConnectionError("could not find a free lobby code")
        threading.Thread(target=self._watch, name="perudo-relay-watch", daemon=True).start()

    @property
    def channel(self) -> str:
        return f"{PREFIX}{self.code}:host"

    def _claim(self, wait: float) -> bool:
        """Listen on the code's host channel and say we have it: our own
        claim coming back means we are listening; another host's answer
        means the code is in use."""
        self._claimed = threading.Event()
        self._taken = False
        self._sub = self.backend.subscribe(self.channel, self._on_raw, self._on_error)
        _wait_ready(self._sub, 15)
        deadline = time.monotonic() + 15
        while not self._claimed.is_set() and time.monotonic() < deadline:
            self.backend.publish(self.channel, "host:" + self.host_id,
                                 json.dumps([{"t": "claim", "host": self.host_id}]))
            self._claimed.wait(1.0)
        if not self._claimed.is_set():
            raise ConnectionError("the relay does not answer")
        time.sleep(wait)
        return not self._taken

    def _on_error(self, message: str) -> None:
        self.error = message
        self.close()

    def _on_raw(self, name: str, data: str) -> None:
        if self.closed:
            return
        batch = _parse(data)
        if name.startswith("host:"):                       # hosts: claims and answers
            for msg in batch:
                if msg.get("t") == "claim":
                    if msg.get("host") == self.host_id:
                        self._claimed.set()
                    elif self._claimed.is_set():
                        self.backend.publish(self.channel, "host:" + self.host_id,
                                             json.dumps([{"t": "taken", "for": msg.get("host")}]))
                elif msg.get("t") == "taken" and msg.get("for") == self.host_id:
                    self._taken = True
            return
        if not name or not self._claimed.is_set():
            return
        with self._lock:
            link = self._links.get(name)
            if link is None or link.closed:
                if not batch or batch[0].get("t") != "hello":
                    return                                 # a guest that already left
                link = Link(self.backend, f"{PREFIX}{self.code}:{name}", "host", self.on_message,
                            self._closed_link)
                self._links[name] = link
        link.receive(batch)

    def _closed_link(self, link: Link) -> None:
        with self._lock:
            for token, other in list(self._links.items()):
                if other is link:
                    del self._links[token]
        self.on_close(link)

    def _watch(self) -> None:
        while not self.closed:
            time.sleep(1.0)
            now = time.monotonic()
            with self._lock:
                silent = [l for l in self._links.values() if now - l.last_heard > GONE_AFTER]
            for link in silent:
                link.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self._lock:
            links = list(self._links.values())
        for link in links:
            link.close()
        if self._sub is not None:
            self._sub.close()


# ---------------------------------------------------------------------------
# a guest's side
# ---------------------------------------------------------------------------

class RelayGuest:
    """A guest's link to the host of `code`, plus the subscription to the
    guest's own channel."""

    def __init__(self, backend, code: str, on_message: Callable[[Link, dict], None],
                 on_close: Callable[[Link], None]):
        self.backend = backend
        self.code = net.normalise_code(code)
        self.token = secrets.token_hex(8)
        self.link = Link(backend, f"{PREFIX}{self.code}:host", self.token, on_message, self._closed)
        self._on_close = on_close
        self._ready = threading.Event()
        self._sub = backend.subscribe(f"{PREFIX}{self.code}:{self.token}", self._on_raw,
                                      self._on_error)
        _wait_ready(self._sub, 15)
        # Our own probe coming back means the subscription is live.
        deadline = time.monotonic() + 15
        while not self._ready.is_set() and time.monotonic() < deadline:
            backend.publish(f"{PREFIX}{self.code}:{self.token}", "probe", "[]")
            self._ready.wait(1.0)
        if not self._ready.is_set():
            self.close()
            raise ConnectionError("the relay does not answer")
        threading.Thread(target=self._watch, name="perudo-relay-watch", daemon=True).start()

    def _on_raw(self, name: str, data: str) -> None:
        if name == "probe":
            self._ready.set()
        elif name == "host":
            self.link.receive(_parse(data))

    def _on_error(self, message: str) -> None:
        self.link.error = message
        self.link.close(say_bye=False)

    def _closed(self, link: Link) -> None:
        self._sub.close()
        self._on_close(link)

    def _watch(self) -> None:
        while not self.link.closed:
            time.sleep(1.0)
            if time.monotonic() - self.link.last_heard > GONE_AFTER:
                self.link.close()

    def close(self) -> None:
        self.link.close()
        self._sub.close()
