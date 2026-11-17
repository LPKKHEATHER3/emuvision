"""Wait for a button to appear on an emulator screen, then tap it safely.

    python examples/find_and_tap.py --port 16416 --template ok_button.png
    python examples/find_and_tap.py --port 16416 --template ok_button.png --dry-run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emuvision import mumu, vision                      # noqa: E402
from emuvision.adb import Device                        # noqa: E402
from emuvision.guard import TapGuard                    # noqa: E402
from emuvision.mumu_ipc import IpcCapture               # noqa: E402
from emuvision.screen import Screen                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True, help="adb port of the emulator instance")
    ap.add_argument("--template", required=True, help="PNG crop of the thing to tap")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true", help="find it, but do not tap")
    args = ap.parse_args()

    root = mumu.find_root()
    device = Device(args.port, adb_path=mumu.find_adb(root) if root else None)
    ipc, vm = None, None
    if root:
        ipc, vm = IpcCapture(root), mumu.Manager(root).index_of_port(args.port)
    screen = Screen(device, ipc, vm)
    try:
        run(args, device, screen)
    finally:
        if ipc:
            ipc.close()


def run(args, device, screen):
    frame = screen.grab()
    if frame is None:
        sys.exit(f"could not grab a frame from {device}")
    h, w = frame.shape[:2]
    print(f"{device}: {w}x{h} via {screen.last_source}")

    template = vision.load(args.template)
    guard = TapGuard(send=device.shell, grab=screen.grab, frame_w=w, frame_h=h)

    found = {}

    def detector(f):
        score, cx, cy = vision.best_match(f, template, gate_for=args.threshold)
        if score >= args.threshold:
            found.update(score=score, x=cx, y=cy)
            return True
        return False

    if not guard.wait_for(detector, timeout=args.timeout):
        sys.exit("template not found on screen")
    print(f"found at ({found['x']},{found['y']}) score={found['score']:.3f}")

    if args.dry_run:
        return

    def still_there():
        # re-check right before sending: the UI may have moved on since we looked.
        # A failed grab counts as "not there" (never tap blind).
        f = screen.grab()
        return f is not None and vision.present(f, template, args.threshold)

    if guard.tap(found["x"] / w, found["y"] / h, label="template", verify=still_there) is None:
        sys.exit("screen changed before the tap; nothing was sent")
    print("tapped")


if __name__ == "__main__":
    main()
