import json
import tempfile
import unittest
from pathlib import Path

from memory_nav.tools.hardware_acceptance import evaluate_sequence, merge_report, run_smoke


class FakeGPS:
    def get_sample(self):
        return {"state": "good", "source": "rtk"}


class FakeIMU:
    def get_heading(self):
        return 108.35

    def stop(self):
        pass


class FakeCamera:
    count = 0

    def capture_frame_with_metadata(self, timeout):
        self.count += 1
        return object(), {"frame_count": self.count}

    def release(self):
        pass


class HardwareAcceptanceTests(unittest.TestCase):
    def test_ordered_sequence_ignores_intermediate_values(self):
        result = evaluate_sequence(
            ["normal", "normal", "deviated", "recovering", "normal"],
            ["normal", "deviated", "recovering", "normal"],
        )
        self.assertTrue(result["passed"])
        self.assertFalse(evaluate_sequence(["normal", "deviated"], ["deviated", "normal"])["passed"])

    def test_report_merges_scenarios_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            merge_report(path, "one", {"passed": True})
            report = merge_report(path, "two", {"passed": False})
            self.assertEqual(set(report["scenarios"]), {"one", "two"})
            self.assertFalse(report["all_recorded_scenarios_passed"])
            self.assertEqual(json.loads(path.read_text()), report)

    def test_smoke_with_fake_devices(self):
        result = run_smoke(0.025, 0.001, FakeGPS, FakeIMU, FakeCamera)
        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(result["observations"]["camera_distinct_frames"], 2)


if __name__ == "__main__":
    unittest.main()
