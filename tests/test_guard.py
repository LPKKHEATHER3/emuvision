import unittest

from emuvision.guard import ForbiddenTapError, LandmarkNotFound, ResolutionError, TapGuard

DELETE_BTN = (0.88, 0.54, 1.00, 0.68)
BUY_ROW = (0.19, 0.77, 0.61, 0.90)


def make_guard(frames=None):
    sent = []
    frames = list(frames or [])

    def grab():
        return frames.pop(0) if frames else None

    g = TapGuard(send=sent.append, grab=grab, forbidden=[DELETE_BTN, BUY_ROW],
                 settle_delay=0, sleep=lambda s: None)
    return g, sent


class GeometryTests(unittest.TestCase):
    def test_resolution_check(self):
        g, _ = make_guard()
        g.assert_resolution(960, 540)
        with self.assertRaises(ResolutionError):
            g.assert_resolution(1280, 720)

    def test_wait_for_aborts_on_wrong_frame_size(self):
        import numpy as np
        g, sent = make_guard(frames=[np.zeros((720, 1280, 3), np.uint8)])
        with self.assertRaises(ResolutionError):
            g.verify_then_tap(0.5, 0.5, lambda f: True, timeout=5)
        self.assertEqual(sent, [])

    def test_wait_for_accepts_calibrated_frame_size(self):
        import numpy as np
        g, sent = make_guard(frames=[np.zeros((540, 960, 3), np.uint8)])
        g.verify_then_tap(0.5, 0.5, lambda f: True, timeout=5)
        self.assertEqual(len(sent), 1)

    def test_proportional_to_pixels_and_clamp(self):
        g, _ = make_guard()
        self.assertEqual(g.to_px(0.5, 0.5), (480, 270))
        self.assertEqual(g.to_px(0.0, 0.0), (0, 0))
        self.assertEqual(g.to_px(1.0, 1.0), (959, 539))
        self.assertEqual(g.to_px(-1, 2), (0, 539))


class ForbiddenZoneTests(unittest.TestCase):
    def test_tap_inside_zone_raises_and_sends_nothing(self):
        g, sent = make_guard()
        with self.assertRaises(ForbiddenTapError):
            g.tap(0.94, 0.60, label="delete")
        self.assertEqual(sent, [])

    def test_safe_tap_is_sent(self):
        g, sent = make_guard()
        self.assertEqual(g.tap(0.5, 0.5), (480, 270))
        self.assertEqual(sent, ["input tap 480 270"])

    def test_allow_zones_only_exempts_named_zone(self):
        g, sent = make_guard()
        g.tap(0.30, 0.80, allow_zones=(1,))          # inside BUY_ROW, explicitly exempted
        self.assertEqual(len(sent), 1)
        with self.assertRaises(ForbiddenTapError):    # DELETE_BTN is still protected
            g.tap(0.94, 0.60, allow_zones=(1,))

    def test_swipe_endpoint_in_zone_raises(self):
        g, sent = make_guard()
        with self.assertRaises(ForbiddenTapError):
            g.swipe(0.5, 0.5, 0.94, 0.60)
        self.assertEqual(sent, [])

    def test_swipe_enforces_minimum_duration(self):
        g, sent = make_guard()
        g.swipe(0.5, 0.5, 0.5, 0.2, duration_ms=100)
        self.assertTrue(sent[0].endswith(" 500"))


class VerifyThenTapTests(unittest.TestCase):
    def test_verify_veto_skips_tap(self):
        g, sent = make_guard()
        self.assertIsNone(g.tap(0.5, 0.5, verify=lambda: False))
        self.assertEqual(sent, [])

    def test_aborts_when_landmark_never_appears(self):
        g, sent = make_guard(frames=["home"] * 5)
        with self.assertRaises(LandmarkNotFound):
            g.verify_then_tap(0.5, 0.5, lambda f: f == "settings", timeout=0)
        self.assertEqual(sent, [])

    def test_taps_once_landmark_appears(self):
        g, sent = make_guard(frames=[None, "loading", "settings"])
        g.verify_then_tap(0.5, 0.5, lambda f: f == "settings", timeout=5)
        self.assertEqual(sent, ["input tap 480 270"])


if __name__ == "__main__":
    unittest.main()
