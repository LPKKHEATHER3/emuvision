"""OpenCV helpers. Pure image analysis: nothing in here touches a device.

Grayscale pre-filter
--------------------
On the author's machine (MuMu Player 12), a 3-channel ``TM_CCOEFF_NORMED`` match
cost ~5x its 1-channel equivalent, but grayscale scores drift slightly from colour
ones, so grayscale alone is not safe to *decide* with. Measured on the author's
machine over 454 real frames x 15 templates, the worst drift was 0.069 (one true
hit fell from 0.908 to 0.839).

So grayscale is only ever used to say "definitely not here": when the gray best
score is more than ``GATE_MARGIN`` below the caller's threshold, no colour match
could reach the threshold and the expensive pass is skipped. Anything closer is
still decided by the colour match, so verdicts are unchanged by construction.
In the author's runs this skipped ~89% of colour matches.

All numbers here were measured on the author's machine; the frame set is not part
of this repository. ``examples/benchmark_capture.py`` reproduces the capture timings.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# matchTemplate on ~960x540 frames does not parallelise (measured on the author's
# machine: 56.9 ms on one thread vs 55.7 ms on eight), yet OpenCV would still
# spawn a worker per core in every automation thread. One thread keeps the speed
# and frees the cores for the emulators themselves.
cv2.setNumThreads(1)

GATE_MARGIN = 0.20   # ~3x the worst measured colour/gray drift
USE_GRAY_GATE = True


# ---- I/O ----------------------------------------------------------------
def decode(png_bytes: bytes | None) -> np.ndarray | None:
    if not png_bytes:
        return None
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)


def load(path: str) -> np.ndarray | None:
    """Unicode-safe read (``cv2.imread`` fails on non-ASCII paths on Windows)."""
    with open(path, "rb") as f:
        return decode(f.read())


def save(path: str, img: np.ndarray) -> bool:
    """Unicode-safe write."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return bool(ok)


def crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return img[y1:y2, x1:x2].copy()


def crop_prop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop with a proportional (0..1) box so it works at any resolution."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return crop(img, int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))


# ---- grayscale gate -----------------------------------------------------
_tpl_gray: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_frame_gray: tuple[np.ndarray | None, np.ndarray | None] = (None, None)


# Thread safety: several automation lanes call these at once. Each cache entry is
# a (source, gray) tuple that is read ONCE into a local and only ever replaced
# whole, never mutated, so a thread can't see one lane's source paired with
# another lane's gray image. Worst case under a race is a cache miss.
def _gray_tpl(t: np.ndarray) -> np.ndarray:
    hit = _tpl_gray.get(id(t))
    if hit is not None and hit[0] is t:      # keep a ref so id() can't be reused
        return hit[1]
    g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    if len(_tpl_gray) > 64:
        _tpl_gray.clear()
    _tpl_gray[id(t)] = (t, g)
    return g


def _gray_frame(f: np.ndarray) -> np.ndarray:
    global _frame_gray
    src, gray = _frame_gray                  # single read: both halves from the same entry
    if src is f:                             # several templates usually share one frame
        return gray
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    _frame_gray = (f, g)
    return g


def _absent(frame: np.ndarray, template: np.ndarray, threshold: float | None) -> bool:
    """True when grayscale proves no colour match can reach ``threshold``."""
    if not USE_GRAY_GATE or threshold is None or frame.ndim != 3 or template.ndim != 3:
        return False
    res = cv2.matchTemplate(_gray_frame(frame), _gray_tpl(template), cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(res)[1] < threshold - GATE_MARGIN


# ---- matching -----------------------------------------------------------
def best_match(frame: np.ndarray, template: np.ndarray, gate_for: float | None = None):
    """(score, center_x, center_y) of the single best location.

    Passing ``gate_for`` (the threshold you will compare against) enables the
    grayscale pre-filter; the score is then only guaranteed to be *below* that
    threshold when the gate fires, not exact.
    """
    th, tw = template.shape[:2]
    if gate_for is not None and _absent(frame, template, gate_for):
        return 0.0, 0, 0
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, (x, y) = cv2.minMaxLoc(res)
    return float(maxv), x + tw // 2, y + th // 2


def find_all(frame: np.ndarray, template: np.ndarray, threshold: float = 0.85):
    """All matches >= threshold as [(score, cx, cy)], best first, de-duplicated
    so one on-screen object is reported once."""
    if _absent(frame, template, threshold):
        return []
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape[:2]
    ys, xs = np.where(res >= threshold)
    hits = sorted(((float(res[y, x]), int(x + tw // 2), int(y + th // 2))
                   for y, x in zip(ys, xs)), reverse=True)
    kept: list[tuple[float, int, int]] = []
    min_d2 = (min(tw, th) // 2) ** 2
    for s, cx, cy in hits:
        if all((cx - kx) ** 2 + (cy - ky) ** 2 > min_d2 for _, kx, ky in kept):
            kept.append((s, cx, cy))
    return kept


def present(frame: np.ndarray, template: np.ndarray, threshold: float = 0.85) -> bool:
    return best_match(frame, template, gate_for=threshold)[0] >= threshold


# ---- misc ---------------------------------------------------------------
def frame_diff(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Mean absolute pixel difference (0.0 = identical); ~1 ms on a full frame
    on the author's machine.
    Useful for "has the screen settled yet?" checks."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(cv2.absdiff(a, b).mean())


def mean_hue(region: np.ndarray) -> float | None:
    """Mean OpenCV hue (0-179) ignoring gray/white/dark pixels."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s > 60) & (v > 60)
    if not mask.any():
        return None
    return float(h[mask].mean())
