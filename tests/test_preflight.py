import unittest

from memory_nav.recording.preflight import check_sensors, require_ready


class FakeIMU:
    def __init__(self, ready=True):
        self.ready = ready

    def get_diagnostics(self):
        return {"heading_ready": self.ready, "last_ahrs_age_s": 0.1 if self.ready else None}


class FakeGPS:
    def __init__(self, sample):
        self.sample = sample

    def get_sample(self):
        return self.sample


class PreflightTests(unittest.TestCase):
    def test_ready_sensors(self):
        checks = check_sensors(FakeGPS({"source": "rtk", "state": "good", "age_s": 0.2}), FakeIMU())
        require_ready(checks)
        self.assertTrue(all(check.ready for check in checks))

    def test_stale_gps_and_imu_fail(self):
        checks = check_sensors(FakeGPS({"source": "phone", "state": "poor", "age_s": 11}), FakeIMU(False))
        with self.assertRaisesRegex(RuntimeError, "imu.*gps"):
            require_ready(checks)


if __name__ == "__main__":
    unittest.main()
