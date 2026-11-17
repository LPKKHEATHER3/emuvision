import os
import tempfile
import unittest

import cv2
import numpy as np

from emuvision import mumu, ocr, vision


def scene():
    """Synthetic 960x540 'screen' with two identical buttons and a distinct icon."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 60, (540, 960, 3), dtype=np.uint8)
    button = np.zeros((40, 80, 3), np.uint8)
    cv2.rectangle(button, (0, 0), (79, 39), (40, 180, 40), -1)
    cv2.putText(button, "OK", (22, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    img[100:140, 100:180] = button
    img[400:440, 700:780] = button
    return img, button


class MatchingTests(unittest.TestCase):
    def test_best_match_finds_center(self):
        img, btn = scene()
        score, cx, cy = vision.best_match(img, btn)
        self.assertGreater(score, 0.99)
        self.assertIn((cx, cy), [(140, 120), (740, 420)])

    def test_find_all_dedupes_to_two_hits(self):
        img, btn = scene()
        hits = vision.find_all(img, btn, threshold=0.9)
        self.assertEqual(sorted((x, y) for _, x, y in hits), [(140, 120), (740, 420)])

    def test_gray_gate_does_not_change_verdicts(self):
        img, btn = scene()
        absent = np.full_like(btn, 200)
        cv2.circle(absent, (40, 20), 15, (0, 0, 255), -1)
        for tpl in (btn, absent):
            vision.USE_GRAY_GATE = False
            plain = vision.present(img, tpl, 0.85)
            vision.USE_GRAY_GATE = True
            gated = vision.present(img, tpl, 0.85)
            self.assertEqual(plain, gated)

    def test_frame_diff(self):
        img, _ = scene()
        self.assertEqual(vision.frame_diff(img, img.copy()), 0.0)
        self.assertIsNone(vision.frame_diff(img, img[:10]))

    def test_unicode_path_roundtrip(self):
        img, _ = scene()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ภาพทดสอบ.png")
            self.assertTrue(vision.save(p, img))
            self.assertTrue(np.array_equal(vision.load(p), img))


class OcrGuardTests(unittest.TestCase):
    """Guards that run without Tesseract installed."""

    def test_left_edge_clip_detected(self):
        region = np.zeros((27, 60, 3), np.uint8)
        region[9:19, 0:3] = 255          # glyph cut by the crop's left edge
        self.assertTrue(ocr.touches_left_edge(region))
        region[:] = 0
        region[0:2, :] = 255             # full-width highlight on the top rows: ignored
        region[9:19, 20:26] = 255
        self.assertFalse(ocr.touches_left_edge(region))

    def test_lone_one_rescue(self):
        region = np.zeros((30, 72, 3), np.uint8)
        region[8:22, 40:44] = 255        # a single thin, tall stroke
        self.assertTrue(ocr.lone_thin_glyph(region))
        region[8:22, 20:34] = 255        # add a wide glyph: no longer a lone "1"
        self.assertFalse(ocr.lone_thin_glyph(region))


class MumuParseTests(unittest.TestCase):
    def test_parse_instances_filters_and_sorts(self):
        info = {
            "0": {"is_main": True, "is_android_started": True, "adb_port": 16384},
            "2": {"is_android_started": True, "adb_port": 16448},
            "1": {"is_android_started": True, "adb_port": 16416},
            "3": {"is_android_started": False, "adb_port": 16480},
        }
        self.assertEqual(mumu.parse_instances(info), [(1, 16416), (2, 16448)])


if __name__ == "__main__":
    unittest.main()
