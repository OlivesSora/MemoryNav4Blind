import math
import struct
import threading
import unittest
from collections import deque

from utils.imu import IMU


class IMUBufferTests(unittest.TestCase):
    def make_parser(self):
        imu = IMU.__new__(IMU)
        imu.heading_offset_deg = 0.0
        imu._lock = threading.Lock()
        imu._ready_event = threading.Event()
        imu._heading_samples = deque(maxlen=5)
        imu._sample_buffer = deque(maxlen=2)
        imu._sample_buffer_dropped = 0
        imu._heading = imu._angular_velocity = imu._angle = imu._quaternion = None
        imu._ahrs_time_step = imu._last_ahrs_monotonic = None
        imu._gyro = imu._accel = imu._mag = imu._imu_time_step = None
        return imu

    def test_parse_buffers_each_ahrs_snapshot(self):
        imu = self.make_parser()
        raw = struct.pack("<12fq", *([0.0] * 3 + [0.0, 0.0, 9.8] + [1.0, 2.0, 3.0] + [0.0] * 3 + [1_000_000]))
        imu._parse_imu(raw)
        ahrs_values = [0.0, 0.0, 0.0, 0.0, 0.0, math.radians(90), 1.0, 0.0, 0.0, 0.0]
        imu._parse_ahrs(struct.pack("<10fq", *ahrs_values, 2_000_000))
        records = imu.drain_samples()
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["heading_deg"], 90.0, places=4)
        self.assertEqual(records[0]["accel_m_s2"], [0.0, 0.0, 9.800000190734863])

    def test_buffer_overflow_is_diagnosed(self):
        imu = self.make_parser()
        payload = struct.pack("<10fq", *([0.0] * 10), 1)
        imu._parse_ahrs(payload)
        imu._parse_ahrs(payload)
        imu._parse_ahrs(payload)
        self.assertEqual(imu._sample_buffer_dropped, 1)
        self.assertEqual(len(imu.drain_samples(max_count=1)), 1)
        self.assertEqual(len(imu.drain_samples()), 1)
        with self.assertRaises(ValueError):
            imu.drain_samples(0)

    def test_clear_samples_defines_session_boundary(self):
        imu = self.make_parser()
        payload = struct.pack("<10fq", *([0.0] * 10), 1)
        imu._parse_ahrs(payload)
        self.assertEqual(imu.clear_samples(), 1)
        self.assertEqual(imu.drain_samples(), [])


if __name__ == "__main__":
    unittest.main()
