"""Region OCR on top of Tesseract.

Principle: **refuse rather than guess.** Every reader returns ``None`` when it is
not confident. A UI number that silently reads short (e.g. "77" when the screen
says "177") is far more dangerous to downstream logic than no number at all.

Lessons that shaped the preprocessing (all found on real screenshots):
* Upscale 4x before thresholding; small UI fonts are below Tesseract's comfort size.
* A thin isolated "1" is dropped by Tesseract; a small dilation keeps it.
* Glyphs touching the image border get misread, so pad with a white border.
* If digits touch the *left* edge of the crop, leading digits may be cut off:
  report "clipped" instead of a believable but wrong number.
"""
from __future__ import annotations

import os
import re
import shutil

import cv2
import numpy as np

from . import vision

try:
    import pytesseract
except ImportError:  # OCR is optional; vision/guard work without it
    pytesseract = None

_DIGITS = re.compile(r"[0-9]+")


def configure(tesseract_cmd: str | None = None) -> bool:
    """Point pytesseract at a Tesseract binary. Returns True if one was found."""
    if pytesseract is None:
        return False
    cmd = tesseract_cmd or shutil.which("tesseract")
    if not cmd:
        default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        cmd = default if os.path.isfile(default) else None
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd is not None


# ---- preprocessing (each returns black text on white) ---------------------
def prep_color_text(region: np.ndarray, hsv_lo=(15, 60, 120), hsv_hi=(45, 255, 255)) -> np.ndarray:
    """Coloured text on a busy background (default range: yellow/orange)."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.resize(mask, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return _pad(cv2.bitwise_not(mask))


def prep_bright_on_dark(region: np.ndarray, threshold: int = 150) -> np.ndarray:
    """White/light text on a dark pill or bar."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return _pad(cv2.bitwise_not(th))


def _pad(img: np.ndarray, px: int = 20) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)


# ---- guards -----------------------------------------------------------------
def touches_left_edge(region: np.ndarray, threshold: int = 150, cols: int = 2, edge_rows: int = 4) -> bool:
    """True if text-coloured pixels reach the crop's left edge (number may be clipped).

    The top/bottom rows are skipped because rounded UI pills often have a bright
    highlight along their full width that would otherwise trigger on every read.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    band = gray[edge_rows:-edge_rows, :cols] if gray.shape[0] > 2 * edge_rows else gray[:, :cols]
    return bool((band > threshold).any())


def lone_thin_glyph(region: np.ndarray, threshold: int = 150) -> bool:
    """True if the region holds exactly one thin, tall glyph (i.e. a "1").

    Tesseract returns nothing for an isolated "1"; this rescues that case without
    ever firing on wider digits or an empty region.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(th)
    glyphs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if y <= 2 or w > 26 or h < 8 or area < 12:   # edge artifacts, long pill lines
            continue
        glyphs.append((w, h))
    return len(glyphs) == 1 and glyphs[0][0] <= 7 and glyphs[0][1] >= 9


# ---- readers ------------------------------------------------------------------
def read_text(frame: np.ndarray, box, whitelist: str = "", prep=prep_bright_on_dark) -> str:
    """OCR a proportional box as a single line. Returns the raw stripped string."""
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed")
    region = vision.crop_prop(frame, box)
    cfg = "--psm 7" + (f" -c tessedit_char_whitelist={whitelist}" if whitelist else "")
    return pytesseract.image_to_string(prep(region), config=cfg).strip()


def read_int(frame: np.ndarray, box, prep=prep_bright_on_dark, rescue_one: bool = True):
    """Read an integer counter. Returns (value | None, raw_or_reason).

    ``None`` always means "could not read confidently", never zero.
    """
    region = vision.crop_prop(frame, box)
    if touches_left_edge(region):
        return None, "clipped"
    raw = read_text(frame, box, whitelist="0123456789", prep=prep)
    found = _DIGITS.findall(raw.replace(",", ""))
    if found:
        return int(max(found, key=len)), raw
    if rescue_one and lone_thin_glyph(region):
        return 1, "1(fallback)"
    return None, raw
