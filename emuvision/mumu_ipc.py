"""Fast frame grab through MuMu's renderer IPC DLL instead of ``adb screencap``.

Why
---
``adb exec-out screencap -p`` encodes a PNG inside the guest, ships it over the
adb socket, and we decode it again: three expensive steps per frame. The IPC
call copies the raw RGBA framebuffer straight out of the renderer, so a grab
costs roughly a memcpy (~8 ms per 960x540 frame vs. a few hundred ms through
adb, measured on the author's machine with MuMu Player 12). Run
``examples/benchmark_capture.py`` to measure it on yours.

Best effort by design
---------------------
Every failure path returns ``None`` so the caller falls back to adb. Results
are identical either way, only faster. A failed instance is not retried for
``RETRY_AFTER`` seconds so a dead emulator does not cost a DLL call per frame.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time

import cv2
import numpy as np

_IPC_SUBPATH = os.path.join("nx_main", "sdk", "external_renderer_ipc.dll")
RETRY_AFTER = 30.0
CONNECT_TRIES = 3


def dll_path(player_root: str) -> str | None:
    p = os.path.join(player_root, _IPC_SUBPATH)
    return p if os.path.exists(p) else None


class _Connection:
    """One IPC handle to one running instance."""

    def __init__(self, player_root: str, vm_index: int):
        self.handle = 0
        self.ok = False
        self._buf = None
        self._buf_size = 0
        path = dll_path(player_root)
        if not path:
            return
        try:
            dll = ctypes.CDLL(path)
            # restype MUST be set before the first call: the handle is 64-bit and
            # ctypes would otherwise truncate it to int32, silently corrupting the
            # handle for higher instance indexes.
            dll.nemu_connect.restype = ctypes.c_uint64
            dll.nemu_connect.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
            dll.nemu_disconnect.restype = ctypes.c_int
            dll.nemu_disconnect.argtypes = [ctypes.c_uint64]
            dll.nemu_capture_display.restype = ctypes.c_int
            dll.nemu_capture_display.argtypes = [
                ctypes.c_uint64, ctypes.c_uint, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_ubyte),
            ]
            self._dll = dll
            for attempt in range(CONNECT_TRIES):
                self.handle = dll.nemu_connect(player_root, int(vm_index))
                if self.handle:
                    break
                time.sleep(0.4 * (attempt + 1))
            self.ok = bool(self.handle)
        except OSError:
            self.ok = False

    def snap(self) -> np.ndarray | None:
        if not self.ok:
            return None
        try:
            w, h = ctypes.c_int(0), ctypes.c_int(0)
            # buf_size=0 -> the call only reports the display size
            self._dll.nemu_capture_display(self.handle, 0, 0, ctypes.byref(w), ctypes.byref(h), None)
            if w.value <= 0 or h.value <= 0:
                return None
            size = w.value * h.value * 4
            if self._buf is None or self._buf_size != size:   # reuse one buffer per instance
                self._buf = (ctypes.c_ubyte * size)()
                self._buf_size = size
            if self._dll.nemu_capture_display(self.handle, 0, size,
                                              ctypes.byref(w), ctypes.byref(h), self._buf) != 0:
                return None
            img = np.frombuffer(self._buf, dtype=np.uint8).reshape((h.value, w.value, 4))
            # renderer hands back RGBA with the OpenGL origin at the bottom-left
            return cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), 0)
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self.ok and self.handle:
                self._dll.nemu_disconnect(self.handle)
        except Exception:
            pass
        self.handle, self.ok, self._buf, self._buf_size = 0, False, None, 0


class IpcCapture:
    """Pool of IPC connections, one per instance index, safe across threads."""

    def __init__(self, player_root: str):
        self.player_root = player_root
        self._pool: dict[int, _Connection] = {}
        self._failed_at: dict[int, float] = {}
        self._pool_lock = threading.Lock()
        self._vm_locks: dict[int, threading.Lock] = {}

    @property
    def available(self) -> bool:
        return dll_path(self.player_root) is not None

    def _vm_lock(self, vm: int) -> threading.Lock:
        with self._pool_lock:
            return self._vm_locks.setdefault(vm, threading.Lock())

    def _get(self, vm: int) -> _Connection | None:
        with self._pool_lock:
            conn = self._pool.get(vm)
            if conn is not None and conn.ok:
                return conn
            last = self._failed_at.get(vm)
            if last is not None and time.monotonic() - last < RETRY_AFTER:
                return None
        conn = _Connection(self.player_root, vm)
        with self._pool_lock:
            if conn.ok:
                self._pool[vm] = conn
                self._failed_at.pop(vm, None)
                return conn
            self._failed_at[vm] = time.monotonic()
        conn.close()
        return None

    def snap(self, vm_index: int) -> np.ndarray | None:
        """BGR frame of instance ``vm_index`` or None (caller falls back to adb)."""
        with self._vm_lock(vm_index):
            conn = self._get(vm_index)
            if conn is None:
                return None
            img = conn.snap()
            if img is None:  # instance restarted: forget the handle, back off
                with self._pool_lock:
                    self._pool.pop(vm_index, None)
                    self._failed_at[vm_index] = time.monotonic()
                conn.close()
            return img

    def close(self) -> None:
        with self._pool_lock:
            conns = list(self._pool.values())
            self._pool.clear()
            self._failed_at.clear()
        for c in conns:
            c.close()
