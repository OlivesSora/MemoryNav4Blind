import unittest
from unittest.mock import Mock, patch
import requests

from utils.gps import GPS


class GPSAdapterTests(unittest.TestCase):
    @patch("utils.gps.requests.get")
    def test_rich_sample(self, request_get):
        response = Mock(status_code=200)
        response.json.return_value = {
            "gps_data": [113.1, 23.2],
            "gps_state": "good",
            "gps_source": "rtk",
            "coordinate_system": "GCJ-02",
            "server_time_ns": 123,
            "age_s": 0.2,
            "accuracy_m": 0.03,
            "fix_quality": 4,
        }
        request_get.return_value = response
        result = GPS().get_sample()
        self.assertEqual(result["source"], "rtk")
        self.assertEqual(result["fix_quality"], 4)
        self.assertEqual((result["longitude"], result["latitude"]), (113.1, 23.2))
        request_get.assert_called_once_with("http://localhost:9000/gps", timeout=1.0)

    @patch("utils.gps.requests.get")
    def test_legacy_get_gps_remains_compatible(self, request_get):
        response = Mock(status_code=200)
        response.json.return_value = {"gps_data": [113.1, 23.2], "gps_state": "poor", "gps_source": "phone"}
        request_get.return_value = response
        self.assertEqual(GPS().get_gps(), ([113.1, 23.2], "poor"))

    @patch("utils.gps.requests.get", side_effect=requests.RequestException("offline"))
    def test_failure_returns_none(self, _request_get):
        self.assertIsNone(GPS().get_sample())


if __name__ == "__main__":
    unittest.main()
