"""TapGuard: the only object allowed to send input to a device.

Automation that taps the wrong place can do real damage (confirm a purchase,
delete something, ...). Rather than trusting every caller to be careful, all
input goes through one choke point that enforces a few rules:

1. **Proportional coordinates.** Callers speak in 0..1 fractions of the frame;
   pixels are computed at runtime, and a resolution check aborts the run if the
   emulator is not the size the coordinates were calibrated on.
2. **Forbidden zones.** Rectangles that must never receive a tap (e.g. a
   "Delete" or "Buy" button). A tap that resolves inside one raises
   ``ForbiddenTapError`` *before* anything is sent. Zones apply everywhere; a
   caller that has verified which screen it is on may explicitly exempt
   specific zone indexes, and only those.
3. **Verify, then tap.** Never tap blindly after ``sleep()``: poll the screen
   until a landmark proves we are on the expected page, then tap.
4. **Last-moment re-check.** UIs lag. A frame that justified a tap may be
   stale by the time the tap is sent (a popup that already closed exposes a
   different button underneath). ``tap(verify=...)`` re-checks right before
   sending and skips the tap if the screen changed.

The sender and frame source are injected, so the whole layer is unit-testable
without a device.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

Box = tuple[float, float, float, float]


class ForbiddenTapError(Exception):
    """A tap resolved inside a forbidden zone. Hard stop."""


class LandmarkNotFound(Exception):
    """The expected page did not appear in time."""


class ResolutionError(Exception):
    """The frame size does not match what the coordinates were calibrated on."""


@dataclass
class TapGuard:
    send: Callable[[str], object]                 # e.g. Device.shell
    grab: Callable[[], object]                    # e.g. Screen.grab
    frame_w: int = 960
    frame_h: int = 540
    forbidden: Sequence[Box] = field(default_factory=list)
    settle_delay: float = 0.3                     # pause before each input so the UI can catch up
    min_swipe_ms: int = 500                       # short swipes fling lists past their target
    sleep: Callable[[float], None] = time.sleep

    # ---- geometry --------------------------------------------------------
    def assert_resolution(self, w: int, h: int) -> None:
        if (w, h) != (self.frame_w, self.frame_h):
            raise ResolutionError(f"frame {w}x{h} != calibrated {self.frame_w}x{self.frame_h}")

    def to_px(self, fx: float, fy: float) -> tuple[int, int]:
        x = min(max(fx * self.frame_w, 0), self.frame_w - 1)
        y = min(max(fy * self.frame_h, 0), self.frame_h - 1)
        return int(round(x)), int(round(y))

    def zone_hits(self, px: int, py: int) -> list[int]:
        """Indexes of forbidden zones containing pixel (px, py)."""
        return [i for i, (x1, y1, x2, y2) in enumerate(self.forbidden)
                if x1 * self.frame_w <= px <= x2 * self.frame_w
                and y1 * self.frame_h <= py <= y2 * self.frame_h]

    def is_forbidden(self, px: int, py: int, allow_zones: Sequence[int] = ()) -> bool:
        return any(i not in allow_zones for i in self.zone_hits(px, py))

    # ---- input -------------------------------------------------------------
    def tap(self, fx: float, fy: float, label: str = "", allow_zones: Sequence[int] = (),
            verify: Callable[[], bool] | None = None):
        """Tap a proportional point. Returns (px, py), or None if ``verify`` vetoed it."""
        px, py = self.to_px(fx, fy)
        if self.is_forbidden(px, py, allow_zones):
            raise ForbiddenTapError(f"tap {label!r} -> ({px},{py}) is inside a forbidden zone")
        self.sleep(self.settle_delay)
        if verify is not None and not verify():
            return None
        self.send(f"input tap {px} {py}")
        return px, py

    def swipe(self, fx1: float, fy1: float, fx2: float, fy2: float, duration_ms: int | None = None):
        d = max(self.min_swipe_ms, duration_ms or self.min_swipe_ms)
        (x1, y1), (x2, y2) = self.to_px(fx1, fy1), self.to_px(fx2, fy2)
        for px, py in ((x1, y1), (x2, y2)):
            if self.is_forbidden(px, py):
                raise ForbiddenTapError(f"swipe endpoint ({px},{py}) is inside a forbidden zone")
        self.sleep(self.settle_delay)
        self.send(f"input swipe {x1} {y1} {x2} {y2} {d}")

    # ---- verify-then-tap -------------------------------------------------------
    def check_frame(self, frame) -> None:
        """Raise ResolutionError if an image frame is not the calibrated size.
        Non-image frames (no ``shape``, e.g. test doubles) are passed through."""
        shape = getattr(frame, "shape", None)
        if shape is not None:
            self.assert_resolution(shape[1], shape[0])

    def wait_for(self, detector: Callable[[object], bool], timeout: float = 30,
                 interval: float = 0.25, clock: Callable[[], float] = time.monotonic) -> bool:
        """Poll frames until ``detector(frame)`` is truthy. Reads only, no input.
        Every grabbed frame is size-checked, so a resized emulator aborts the run
        before any coordinate computed for the old size can be used."""
        end = clock() + timeout
        while True:
            frame = self.grab()
            if frame is not None:
                self.check_frame(frame)
            if frame is not None and detector(frame):
                return True
            if clock() >= end:
                return False
            self.sleep(interval)

    def verify_then_tap(self, fx: float, fy: float, detector: Callable[[object], bool],
                        timeout: float = 30, label: str = ""):
        if not self.wait_for(detector, timeout=timeout):
            raise LandmarkNotFound(f"landmark for {label!r} not seen within {timeout}s")
        return self.tap(fx, fy, label=label)
