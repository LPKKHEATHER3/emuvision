"""One object that answers "what is on screen right now?" for a device.

Tries the MuMu IPC framebuffer first and falls back to ``adb screencap``, so
calling code never has to care which path produced the frame.
"""
from __future__ import annotations

import numpy as np

from .adb import Device
from .mumu_ipc import IpcCapture


class Screen:
    def __init__(self, device: Device, ipc: IpcCapture | None = None, vm_index: int | None = None):
        self.device = device
        self.ipc = ipc
        self.vm_index = vm_index
        self.last_source = None  # "ipc" | "adb" | None, handy for benchmarks/logging

    def grab(self) -> np.ndarray | None:
        if self.ipc is not None and self.vm_index is not None:
            img = self.ipc.snap(self.vm_index)
            if img is not None:
                self.last_source = "ipc"
                return img
        img = self.device.screencap_bgr()
        self.last_source = "adb" if img is not None else None
        return img
