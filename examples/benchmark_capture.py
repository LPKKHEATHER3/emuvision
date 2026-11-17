"""Compare MuMu IPC capture against adb screencap on a live instance.

Checks both things that matter:
  1. speed    - ms per frame for each path
  2. sameness - the IPC frame must match the adb frame. A wrong vertical flip or
     swapped colour channels would still "work" but silently break every
     template match, so the raw frame must beat its own flipped version.

    python examples/benchmark_capture.py [--frames 10]
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emuvision import mumu, vision                      # noqa: E402
from emuvision.adb import Device                        # noqa: E402
from emuvision.mumu_ipc import IpcCapture               # noqa: E402


def timed(fn, n):
    t = time.perf_counter()
    img = None
    for _ in range(n):
        img = fn()
    return img, (time.perf_counter() - t) * 1000 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    root = mumu.find_root()
    if not root:
        sys.exit("MuMu Player not found (set MUMU_ROOT)")
    insts = mumu.Manager(root).instances(fresh=True)
    if not insts:
        sys.exit("no running MuMu instance")
    vm, port = insts[0]
    dev, ipc = Device(port, mumu.find_adb(root)), IpcCapture(root)
    print(f"instance {vm} on port {port}, IPC available: {ipc.available}\n")

    adb_img, adb_ms = timed(dev.screencap_bgr, args.frames)
    ipc_img, ipc_ms = timed(lambda: ipc.snap(vm), args.frames)
    ipc.close()

    print(f"adb screencap : {adb_ms:8.1f} ms/frame")
    if ipc_img is None:
        print("IPC capture   :   unavailable (falls back to adb)")
        return
    print(f"IPC capture   : {ipc_ms:8.1f} ms/frame  ({adb_ms / ipc_ms:.0f}x faster)")

    raw = vision.frame_diff(adb_img, ipc_img)
    flipped = vision.frame_diff(adb_img, cv2.flip(ipc_img, 0))
    print(f"\nmean diff vs adb: {raw:.2f}  (flipped: {flipped:.2f})")
    print("orientation OK" if raw is not None and raw < flipped else "WARNING: orientation mismatch")


if __name__ == "__main__":
    main()
