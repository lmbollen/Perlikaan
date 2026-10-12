"""
Networking for games on one local network (the same Wi-Fi, or a phone's
hotspot): lobby codes, connections that carry JSON messages, and the
encoding of the engine's types.

A lobby code is four characters that say where the host is: the last two
parts of its IPv4 address and which of 16 ports it listens on, scrambled so
codes look random. A joining phone combines them with the first two parts
of its own address (phones on one network share those), so nobody types an
address and no discovery broadcast (and no extra permission) is needed.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

from perudo import models

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # 32 symbols, no 0/O or 1/I
BASE_PORT = 47360
SLOTS = 16
_SCRAMBLE = 0x5A3C9                              # 20 bits
PROTOCOL = 1
MAX_LINE = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# lobby codes
# ---------------------------------------------------------------------------

def make_code(ip: str, slot: int) -> str:
    _a, _b, c, d = (int(x) for x in ip.split("."))
    value = ((c << 12) | (d << 4) | (slot & 0xF)) ^ _SCRAMBLE
    return "".join(ALPHABET[(value >> shift) & 31] for shift in (15, 10, 5, 0))


def normalise_code(code: str) -> str:
    """Upper case, no spaces (codes never contain 0, O, 1 or I)."""
    return "".join(code.split()).upper()


def read_code(code: str):
    """(c, d, slot) from a code; ValueError when it is not one."""
    code = normalise_code(code)
    if len(code) != 4 or any(ch not in ALPHABET for ch in code):
        raise ValueError("a lobby code is four letters or digits")
    value = 0
    for ch in code:
        value = (value << 5) | ALPHABET.index(ch)
    value ^= _SCRAMBLE
    return (value >> 12) & 0xFF, (value >> 4) & 0xFF, value & 0xF


def port_for(slot: int) -> int:
    return BASE_PORT + slot


def candidates(code: str, local_ips: Iterable[str]) -> List[str]:
    """The addresses the host of `code` can have, seen from this phone."""
    c, d, _slot = read_code(code)
    found: List[str] = []
    for ip in local_ips:
        a, b = ip.split(".")[:2]
        guess = f"{a}.{b}.{c}.{d}"
        if guess not in found:
            found.append(guess)
    return found


# ---------------------------------------------------------------------------
# this phone's addresses
# ---------------------------------------------------------------------------

def _private_rank(ip: str) -> int:
    a, b = (int(x) for x in ip.split(".")[:2])
    if a == 192 and b == 168:
        return 0
    if a == 10:
        return 1
    if a == 172 and 16 <= b <= 31:
        return 2
    if a == 127:
        return 9
    return 5


def local_ipv4s() -> List[str]:
    """This device's IPv4 addresses, the likely local-network one first."""
    found: List[str] = []

    def add(ip: Optional[str]) -> None:
        if ip and ip.count(".") == 3 and not ip.startswith("0.") and ip not in found:
            found.append(ip)

    # Android: every interface, also a hotspot's, through the Java API.
    try:
        from jnius import autoclass   # type: ignore
        interfaces = autoclass("java.net.NetworkInterface").getNetworkInterfaces()
        Inet4 = autoclass("java.net.Inet4Address")
        while interfaces.hasMoreElements():
            nic = interfaces.nextElement()
            if not nic.isUp():
                continue
            addresses = nic.getInetAddresses()
            while addresses.hasMoreElements():
                addr = addresses.nextElement()
                if isinstance(addr, Inet4) or "." in str(addr.getHostAddress()):
                    add(str(addr.getHostAddress()))
    except Exception:
        pass
    # The address a packet to a local network would leave from (sends nothing).
    for target in ("192.168.255.255", "10.255.255.255", "172.31.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 9))
                add(s.getsockname()[0])
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    found.sort(key=_private_rank)
    return found or ["127.0.0.1"]


# ---------------------------------------------------------------------------
# the engine's types as JSON
# ---------------------------------------------------------------------------

_TYPES = {cls.__name__: cls for cls in vars(models).values()
          if isinstance(cls, type) and dataclasses.is_dataclass(cls)}


def encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {"_t": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = encode(getattr(obj, f.name))
        return out
    if isinstance(obj, (list, tuple)):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    return obj


def decode(obj: Any) -> Any:
    """The inverse of encode(); lists come back as tuples (as in the models)."""
    if isinstance(obj, list):
        return tuple(decode(x) for x in obj)
    if isinstance(obj, dict):
        kind = obj.get("_t")
        if kind is not None:
            cls = _TYPES.get(kind)
            if cls is None:
                raise ValueError(f"unknown type {kind!r}")
            return cls(**{k: decode(v) for k, v in obj.items() if k != "_t"})
        return {k: decode(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# connections
# ---------------------------------------------------------------------------

class Connection:
    """One TCP connection carrying JSON messages, one per line. Messages are
    handed to `on_message(conn, msg)` from a reader thread; `on_close(conn)`
    is called once when it ends."""

    def __init__(self, sock: socket.socket, on_message: Callable[["Connection", dict], None],
                 on_close: Callable[["Connection"], None]):
        self.sock = sock
        self.on_message = on_message
        self.on_close = on_close
        self.member_id: Optional[int] = None
        self.closed = False
        self._send_lock = threading.Lock()
        self._closed_lock = threading.Lock()
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        sock.settimeout(None)
        self._thread = threading.Thread(target=self._read, name="perudo-net", daemon=True)

    def start(self) -> "Connection":
        self._thread.start()
        return self

    def send(self, msg: Dict[str, Any]) -> bool:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._send_lock:
                self.sock.sendall(data)
            return True
        except OSError:
            self.close()
            return False

    def close(self) -> None:
        with self._closed_lock:
            if self.closed:
                return
            self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            self.on_close(self)
        except Exception:
            pass

    def _read(self) -> None:
        buffer = b""
        try:
            while not self.closed:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > MAX_LINE and b"\n" not in buffer:
                    break
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except ValueError:
                        continue
                    if isinstance(msg, dict):
                        self.on_message(self, msg)
        except OSError:
            pass
        finally:
            self.close()


def listen(slots: Iterable[int] = range(SLOTS)):
    """A listening socket on the first free slot: (socket, slot)."""
    last: Optional[OSError] = None
    for slot in slots:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port_for(slot)))
            s.listen(16)
            return s, slot
        except OSError as exc:
            last = exc
            s.close()
    raise OSError(f"no free port for a lobby ({last})")


def dial(code: str, local_ips: Optional[List[str]] = None, timeout: float = 3.0) -> socket.socket:
    """Connect to the host of `code`, trying every address it can have at
    once; the first to answer wins."""
    _c, _d, slot = read_code(code)
    port = port_for(slot)
    targets = candidates(code, local_ips or local_ipv4s())
    result: Dict[str, Any] = {}
    done = threading.Event()
    lock = threading.Lock()

    def attempt(ip: str) -> None:
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
        except OSError as exc:
            with lock:
                result.setdefault("errors", []).append(f"{ip}: {exc}")
                if len(result["errors"]) == len(targets):
                    done.set()
            return
        with lock:
            if "sock" in result:
                s.close()
                return
            result["sock"] = s
        done.set()

    for ip in targets:
        threading.Thread(target=attempt, args=(ip,), daemon=True).start()
    done.wait(timeout + 1)
    if "sock" not in result:
        raise ConnectionError("no lobby with that code on this network")
    return result["sock"]
