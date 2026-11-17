"""Discover MuMu Player 12 installs and running instances.

``MuMuManager.exe info -v all`` returns JSON describing every instance. Spawning
it costs ~166 ms (measured on the author's machine), and a UI polling every second or two would otherwise keep
launching processes for nothing, so results are cached for a few seconds.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

_INSTALL_NAMES = ("MuMuPlayer", "MuMuPlayerGlobal-12.0", "MuMu Player 12", "MuMuPlayer-12.0", "MuMu")


def _is_root(path: str | None) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "nx_main", "MuMuManager.exe"))


def find_root(hint: str | None = None) -> str | None:
    """Locate the MuMu install folder (the one holding ``nx_main\\MuMuManager.exe``).

    Checks ``hint`` first, then the ``MUMU_ROOT`` environment variable, then the
    usual install locations on drives C-H.
    """
    for cand in (hint, os.environ.get("MUMU_ROOT")):
        if _is_root(cand):
            return cand
    for drive in "CDEFGH":
        for pf in ("Program Files", "Program Files (x86)", ""):
            for base in (os.path.join(f"{drive}:\\", pf, "Netease"), os.path.join(f"{drive}:\\", pf)):
                for name in _INSTALL_NAMES:
                    root = os.path.join(base, name)
                    if _is_root(root):
                        return root
    return None


def find_adb(root: str) -> str | None:
    """The adb.exe bundled with MuMu (layout differs between versions)."""
    for rel in (r"nx_device\12.0\shell\adb.exe", r"nx_main\adb.exe", r"shell\adb.exe"):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    for dirpath, _, files in os.walk(root):
        if "adb.exe" in files:
            return os.path.join(dirpath, "adb.exe")
    return None


def parse_instances(info: dict) -> list[tuple[int, int]]:
    """``MuMuManager info -v all`` JSON -> sorted [(index, adb_port)] of booted,
    non-main instances. Split out so it can be unit tested without MuMu."""
    out = []
    for idx, v in info.items():
        if (isinstance(v, dict) and v.get("is_android_started")
                and not v.get("is_main") and v.get("adb_port")):
            out.append((int(idx), int(v["adb_port"])))
    out.sort()
    return out


class Manager:
    def __init__(self, root: str, ttl: float = 5.0):
        self.root = root
        self.exe = os.path.join(root, "nx_main", "MuMuManager.exe")
        self.ttl = ttl
        self._cache: tuple[float, list[tuple[int, int]]] = (0.0, [])
        self._lock = threading.Lock()

    def instances(self, fresh: bool = False) -> list[tuple[int, int]]:
        now = time.monotonic()
        with self._lock:
            at, data = self._cache
            if not fresh and now - at < self.ttl:
                return list(data)
        try:
            r = subprocess.run([self.exe, "info", "-v", "all"], capture_output=True, timeout=20)
            data = parse_instances(json.loads(r.stdout.decode("utf-8", "ignore")))
        except Exception:
            data = []
        with self._lock:
            self._cache = (time.monotonic(), data)
        return list(data)

    def index_of_port(self, port: int) -> int | None:
        for idx, p in self.instances():
            if p == int(port):
                return idx
        for idx, p in self.instances(fresh=True):
            if p == int(port):
                return idx
        return None
