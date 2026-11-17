"""emuvision — screen-driven automation toolkit for Android emulators.

Modules
-------
adb       Thin, connection-caching wrapper around the adb CLI.
mumu      Discover running MuMu Player instances and their adb ports.
mumu_ipc  Zero-copy frame grab through MuMu's renderer IPC (falls back to adb).
vision    OpenCV helpers: template matching with a grayscale pre-filter, diffs, hue.
ocr       Region OCR on top of Tesseract, with "refuse rather than guess" guards.
guard     The only place that sends input: proportional coords, forbidden zones,
          verify-then-tap.
"""

__version__ = "0.1.0"
