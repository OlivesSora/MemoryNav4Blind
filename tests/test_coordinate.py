import unittest

from memory_nav.trajectory.coordinate import LocalFrame, heading_from_delta, wrap_to_180


class CoordinateTests(unittest.TestCase):
    def test_local_round_trip(self):
        frame = LocalFrame(113.370033, 22.918129)
        coordinate = (113.370633, 22.918529)
        east, north = frame.to_local(*coordinate)
        restored = frame.to_geodetic(east, north)
        self.assertAlmostEqual(restored[0], coordinate[0], places=10)
        self.assertAlmostEqual(restored[1], coordinate[1], places=10)
        self.assertGreater(east, 0)
        self.assertGreater(north, 0)

    def test_heading_convention(self):
        self.assertEqual(heading_from_delta(0, 1), 0)
        self.assertEqual(heading_from_delta(1, 0), 90)
        self.assertEqual(heading_from_delta(0, -1), 180)
        self.assertEqual(heading_from_delta(-1, 0), 270)

    def test_wrap(self):
        self.assertEqual(wrap_to_180(359), -1)
        self.assertEqual(wrap_to_180(-359), 1)
        self.assertEqual(wrap_to_180(180), 180)
        self.assertEqual(wrap_to_180(-180), -180)


if __name__ == "__main__":
    unittest.main()
