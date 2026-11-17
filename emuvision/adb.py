"""Thin wrapper around the adb command-line tool.

Design notes
------------
* ``adb connect`` costs ~27 ms (measured on the author's machine) and only has to happen once per device, so the
  handshake is cached per serial.
* If a command later reports the device as gone (emulator restarted), we
  reconnect and retry that command exactly once.
* Nothing here raises on adb failure: callers get a ``CompletedProcess`` with a
  non-zero ``returncode`` and decide for themselves.
"""
from __future__ import annotations

import shutil
import subprocess
import threading

import numpy as np

_STALE_MARKERS = ("not found", "offline", "no devices", "device still connecting", "closed")


def _is_stale(stderr: bytes | None) -> bool:
    """adb's way of saying a cached connection is dead."""
    s = (stderr or b"").decode("utf-8", "ignore").lower()
    return any(m in s for m in _STALE_MARKERS)


class Device:
    """One emulator instance reachable over TCP adb (``127.0.0.1:<port>``)."""

    _connected: set[str] = set()
    _lock = threading.Lock()

    def __init__(self, port: int, adb_path: str | None = None, host: str = "127.0.0.1"):
        self.port = int(port)
        self.serial = f"{host}:{self.port}"
        self.adb_path = adb_path or shutil.which("adb") or "adb"

    # ---- low level -------------------------------------------------------
    def _connect(self) -> None:
        subprocess.run([self.adb_path, "connect", self.serial], capture_output=True, timeout=15)

    def run(self, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
        """Run ``adb -s <serial> <args...>`` and return the completed process."""
        try:
            with Device._lock:
                fresh = self.serial not in Device._connected
                if fresh:
                    self._connect()
                    Device._connected.add(self.serial)
            r = subprocess.run([self.adb_path, "-s", self.serial, *args],
                               capture_output=True, timeout=timeout)
            if not fresh and r.returncode != 0 and _is_stale(r.stderr):
                with Device._lock:
                    self._connect()
                r = subprocess.run([self.adb_path, "-s", self.serial, *args],
                                   capture_output=True, timeout=timeout)
            return r
        except Exception as e:  # timeout, missing binary, ...
            with Device._lock:
                Device._connected.discard(self.serial)
            return subprocess.CompletedProcess(args, 1, b"", str(e).encode())

    def shell(self, cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
        return self.run("shell", cmd, timeout=timeout)

    # ---- input -------------------------------------------------------------
    def tap(self, x: int, y: int) -> subprocess.CompletedProcess:
        return self.shell(f"input tap {int(x)} {int(y)}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> subprocess.CompletedProcess:
        return self.shell(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}")

    # ---- screen ------------------------------------------------------------
    def screencap_png(self) -> bytes | None:
        """PNG bytes of the current frame via ``exec-out`` (no /sdcard round trip)."""
        r = self.run("exec-out", "screencap", "-p", timeout=30)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        return None

    def screencap_bgr(self) -> np.ndarray | None:
        from . import vision
        return vision.decode(self.screencap_png())

    def __repr__(self) -> str:
        return f"Device({self.serial!r})"
