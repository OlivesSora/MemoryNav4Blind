import unittest

from memory_nav.models import GPSSample, IMUSample, VIOSample
from memory_nav.trajectory.processing import build_reference, clean_gps_samples, estimate_heading_offset, estimate_vio_transform, resample_polyline, smooth_positions
from memory_nav.trajectory.build_reference import evaluate_quality


def sample(time_s, lon, lat=23.0, source="rtk", state="good"):
    return GPSSample(int(time_s * 1e9), lon, lat, source, state)


class ProcessingTests(unittest.TestCase):
    def test_clean_rejects_duplicate_out_of_order_and_speed_jump(self):
        values = [sample(1, 113.0), sample(2, 113.0), sample(0.5, 113.000001), sample(3, 114.0), sample(4, 113.00001)]
        cleaned, stats = clean_gps_samples(values, max_speed_m_s=4.0)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["rejected_duplicate"], 1)
        self.assertEqual(stats["rejected_out_of_order"], 1)
        self.assertEqual(stats["rejected_speed"], 1)

    def test_resample_includes_endpoint(self):
        points, progress = resample_polyline([(0, 0), (2.2, 0)], 1.0)
        self.assertEqual(list(progress), [0.0, 1.0, 2.0, 2.2])
        self.assertAlmostEqual(points[-1, 0], 2.2)

    def test_smoothing_preserves_endpoints_and_protected(self):
        result = smooth_positions([(0, 0), (1, 10), (2, 0), (3, 0), (4, 0)], [1] * 5, 3, [1])
        self.assertEqual(tuple(result[0]), (0, 0))
        self.assertEqual(tuple(result[1]), (1, 10))
        self.assertEqual(tuple(result[-1]), (4, 0))

    def test_build_reference_straight_east(self):
        values = [sample(i + 1, 113.0 + i * 0.00001) for i in range(5)]
        points, report, _ = build_reference(values, spacing_m=0.5, smoothing_window=3)
        self.assertGreater(report.route_length_m, 3.0)
        self.assertTrue(all(abs(point.heading_deg - 90) < 1 for point in points))
        self.assertAlmostEqual(points[-1].s_m, report.route_length_m)

    def test_imu_heading_offset_is_aligned_and_fused(self):
        gps = [sample(i + 1, 113.0 + i * 0.00001) for i in range(5)]
        imu = [IMUSample(int((i + 1.5) * 1e9), i + 1.5, 80.0, heading_ready=True) for i in range(4)]
        offset, count = estimate_heading_offset(gps, imu)
        self.assertGreater(count, 0)
        self.assertAlmostEqual(offset, 10.0, places=1)
        points, report, _ = build_reference(gps, imu_samples=imu, spacing_m=0.5, smoothing_window=3)
        self.assertAlmostEqual(report.heading_offset_deg, 10.0, places=1)
        self.assertTrue(all(abs(point.heading_deg - 90) < 1 for point in points))

    def test_heading_offset_wraps_across_north(self):
        # Track is north (0 degrees), while IMU reports 350 degrees.
        gps = [GPSSample((i + 1) * 1_000_000_000, 113.0, 23.0 + i * 0.00001, "rtk", "good") for i in range(4)]
        imu = [IMUSample(int((i + 1.5) * 1e9), i + 1.5, 350.0, heading_ready=True) for i in range(3)]
        offset, _ = estimate_heading_offset(gps, imu)
        self.assertAlmostEqual(offset, 10.0, places=1)

    def test_quality_gate_reports_each_failure(self):
        gps = [sample(i + 1, 113.0 + i * 0.00001) for i in range(2)]
        _points, report, _frame = build_reference(gps, spacing_m=0.5, smoothing_window=3)
        ready, reasons = evaluate_quality(report, {
            "minimum_gps_samples": 5,
            "minimum_route_length_m": 10,
            "maximum_gps_gap_s": 0.5,
            "maximum_rejection_ratio": 0.1,
        })
        self.assertFalse(ready)
        self.assertEqual(len(reasons), 3)

    def test_quality_gate_accepts_good_route(self):
        gps = [sample(i + 1, 113.0 + i * 0.00001) for i in range(8)]
        _points, report, _frame = build_reference(gps, spacing_m=0.5, smoothing_window=3)
        ready, reasons = evaluate_quality(report, {
            "minimum_gps_samples": 5,
            "minimum_route_length_m": 3,
            "maximum_gps_gap_s": 2,
            "maximum_rejection_ratio": 0.5,
        })
        self.assertTrue(ready)
        self.assertEqual(reasons, [])

    def test_build_reference_preserves_unmarked_sharp_turn(self):
        # Roughly 2m east followed by 2m north. The corner sample must remain
        # at its raw location even without a camera/manual anchor event.
        gps = [
            GPSSample(1_000_000_000, 113.0, 23.0, "rtk", "good"),
            GPSSample(2_000_000_000, 113.00002, 23.0, "rtk", "good"),
            GPSSample(3_000_000_000, 113.00002, 23.00002, "rtk", "good"),
        ]
        points, _report, frame = build_reference(gps, spacing_m=0.25, smoothing_window=3, turn_protection_deg=30)
        corner_east, corner_north = frame.to_local(gps[1].longitude, gps[1].latitude)
        nearest = min(points, key=lambda point: (point.east_m - corner_east) ** 2 + (point.north_m - corner_north) ** 2)
        self.assertLess(((nearest.east_m - corner_east) ** 2 + (nearest.north_m - corner_north) ** 2) ** 0.5, 0.15)

    def test_vio_rigid_alignment_and_dense_reference(self):
        # VIO x-axis is rotated 90 degrees from EN and translated; it should
        # still become the high-rate reference geometry without scale fitting.
        gps = [sample(index + 1, 113.0, 23.0 + index * 0.00001) for index in range(5)]
        vio = [VIOSample((index + 1) * 1_000_000_000, 0, float(index), 0.0, 0.0) for index in range(9)]
        transform, rms = estimate_vio_transform(vio[:3], [(0, 0), (0, 1), (0, 2)])
        east, north = transform.apply(4, 0)
        self.assertLess(rms, 1e-9)
        self.assertAlmostEqual(east, 0, places=6)
        self.assertAlmostEqual(north, 4, places=6)
        points, report, _ = build_reference(gps, vio_samples=vio, spacing_m=0.25, smoothing_window=3)
        self.assertTrue(report.vio_used)
        self.assertGreater(len(points), len(gps))
        self.assertIsNotNone(report.vio_transform)


if __name__ == "__main__":
    unittest.main()
