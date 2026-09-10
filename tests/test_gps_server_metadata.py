import time
import unittest
from unittest.mock import patch

import utils.gps_server as server


class FakeRTK:
    def __init__(self, available=True):
        self.available = available
        self.last_gps_time = time.time() - 0.2
        self.latest_fix_quality = 4

    def get_gps(self):
        return [113.1, 23.2], "poor"

    def has_rtk_fix(self):
        return self.available


class GPSServerMetadataTests(unittest.TestCase):
    def test_rtk_source_has_fix_quality_and_forces_good(self):
        with patch.object(server, "gps_instance", FakeRTK(True)):
            result = server.get_gps()
        self.assertEqual(result["gps_source"], "rtk")
        self.assertEqual(result["gps_state"], "good")
        self.assertEqual(result["fix_quality"], 4)
        self.assertLess(result["age_s"], 1)

    def test_phone_fallback_preserves_accuracy(self):
        with (
            patch.object(server, "gps_instance", FakeRTK(False)),
            patch.object(server, "latest_phone_gps", [113.3, 23.4]),
            patch.object(server, "phone_gps_timestamp", time.time() - 0.5),
            patch.object(server, "latest_phone_accuracy", 6.5),
        ):
            result = server.get_gps()
        self.assertEqual(result["gps_source"], "phone")
        self.assertEqual(result["accuracy_m"], 6.5)
        self.assertIsNone(result["fix_quality"])

    def test_no_fresh_source_returns_error(self):
        with (
            patch.object(server, "gps_instance", None),
            patch.object(server, "latest_phone_gps", [113.3, 23.4]),
            patch.object(server, "phone_gps_timestamp", time.time() - server.PHONE_TIMEOUT - 1),
        ):
            result = server.get_gps()
        self.assertIn("error", result)
        self.assertNotIn("gps_data", result)


if __name__ == "__main__":
    unittest.main()
