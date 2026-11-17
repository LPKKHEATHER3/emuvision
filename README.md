# emuvision

![tests](../../actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue)

Screen-driven automation toolkit for Android emulators: grab frames fast, find
things on screen with OpenCV, read numbers with OCR, and send taps through a
safety layer that refuses to touch the wrong place.

Extracted and cleaned up from a larger automation project I built that drives
several emulator instances in parallel. This repo keeps the reusable engine and
none of the app-specific flows.

## What's inside

| Module | What it does | Interesting bits |
|---|---|---|
| `adb.py` | Wrapper around the adb CLI | Caches the `connect` handshake (~27 ms on the author's machine) per device; auto-reconnects once if the emulator restarted |
| `mumu.py` | Finds MuMu Player installs and running instances | TTL cache so a polling UI doesn't spawn a process every second |
| `mumu_ipc.py` | Grabs the framebuffer through MuMu's renderer DLL via `ctypes` | ~8 ms/frame on the author's machine instead of a PNG encode → socket → decode round trip; 64-bit handle pitfall handled; falls back to adb on any failure |
| `screen.py` | One `grab()` that tries IPC, then adb | Callers never care which path produced the frame |
| `vision.py` | Template matching, multi-match de-duplication, frame diff, hue | Grayscale pre-filter that skipped ~89% of colour matches in the author's runs, **without changing any verdict** (see below) |

All timings and rates in this README and in the module docstrings were measured on the author's
machine (Windows, MuMu Player 12, 960×540 instances). They are not results stored in this repository.
To measure capture speed on your own setup, run `python examples/benchmark_capture.py`.
| `ocr.py` | Tesseract on a screen region | "Refuse rather than guess": detects clipped numbers and returns `None`, rescues the isolated "1" Tesseract drops |
| `guard.py` | `TapGuard`, the only object that sends input | Proportional coordinates, forbidden zones, verify-then-tap, last-moment re-check |

## Design decisions

### 1. All input goes through one choke point
Automation that taps the wrong spot can confirm a purchase or delete data. So
instead of trusting every caller, `TapGuard`:

- works in **proportional coordinates** (0..1) and size-checks every frame it
  polls, aborting with `ResolutionError` if the emulator is not the size the
  coordinates were calibrated on;
- refuses any tap or swipe that lands in a **forbidden zone**, raising *before*
  anything is sent. A caller that has verified which screen it is on can exempt
  specific zones by index, and only those;
- **verifies, then taps**: it polls for a landmark proving it is on the right page
  instead of `sleep()`-and-hope;
- **re-checks at the last moment**: UIs lag, and a popup that has already closed
  exposes a different button underneath. `tap(verify=...)` skips the tap if the
  screen changed while waiting.

The sender and frame source are injected, so all of this is unit tested without
a device.

### 2. A speed-up that provably doesn't change results
On the author's machine, colour template matching was ~5x slower than grayscale, but grayscale
scores drift (worst drift measured on the author's machine: 0.069 over 454 real frames × 15
templates), so
grayscale alone can flip verdicts. `vision.py` uses grayscale only to prove
absence: if the gray score is more than 0.20 (about 3x the worst drift) below
the threshold, no colour match can reach it, so the colour pass is skipped.
Anything closer is still decided by colour. The tests check that both paths
give the same verdict.

### 3. Measured, not guessed
The numbers in this section were measured on the author's machine (MuMu Player 12).

- `cv2.setNumThreads(1)`: matchTemplate at 960×540 took 56.9 ms on 1 thread vs
  55.7 ms on 8, so extra threads only stole CPU from the emulators.
- OCR crops were re-measured after real misreads (a nearby icon once produced
  a believable "477" for an actual 171). That bug is why the clipping guard exists.
- `examples/benchmark_capture.py` reproduces the capture comparison: it measures IPC
  against adb on your machine and checks the frames are identical (catching flipped or
  channel-swapped output). The template-matching numbers above came from the author's own
  frame set, which is not part of this repository.

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v     # 20 tests, no device needed
```

The same suite runs on every push via GitHub Actions (Ubuntu + Windows, Python 3.10–3.12).

With a MuMu Player 12 instance running:

```bash
python examples/benchmark_capture.py
python examples/find_and_tap.py --port 16416 --template ok_button.png --dry-run
```

```python
from emuvision import mumu, vision
from emuvision.adb import Device
from emuvision.guard import TapGuard
from emuvision.mumu_ipc import IpcCapture
from emuvision.screen import Screen

root = mumu.find_root()
vm, port = mumu.Manager(root).instances()[0]
dev = Device(port, mumu.find_adb(root))
screen = Screen(dev, IpcCapture(root), vm)

guard = TapGuard(send=dev.shell, grab=screen.grab,
                 forbidden=[(0.88, 0.54, 1.00, 0.68)])   # e.g. a "Delete" button

settings_icon = vision.load("settings_icon.png")
guard.verify_then_tap(0.88, 0.93, lambda f: vision.present(f, settings_icon), label="settings")
```

## Requirements
- Python 3.10+
- Windows with MuMu Player 12 for the IPC capture (adb capture works with any emulator or device)
- Tesseract OCR, only if you use `emuvision.ocr` readers

## Tech
Python · OpenCV · NumPy · Tesseract · ctypes · adb · unittest

Built with Claude Code as a pair programmer; the design decisions and testing strategy are my own.
