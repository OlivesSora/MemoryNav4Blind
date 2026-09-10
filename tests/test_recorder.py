import tempfile
import unittest

from memory_nav.recording.recorder import SensorRecorder
from memory_nav.recording.session_writer import SessionWriter, read_jsonl


class FakeGPS:
    def __init__(self):
        self.value = ([113.0, 23.0], "good")

    def get_gps(self):
        return self.value


class FakeIMU:
    def __init__(self):
        self.timestamp = 1.0

    def get_ahrs(self):
        return [0, 0, 0], [0, 0, 0], [1, 0, 0, 0], self.timestamp

    def get_imu(self):
        return [0, 0, 0], [0, 0, 9.8], [1, 2, 3], self.timestamp

    def get_heading(self, require_ready=True):
        return 90.0


class FakeBufferedIMU:
    def __init__(self):
        self.records = [{"monotonic_ns": 10, "device_time_s": 1.0, "heading_deg": 90.0, "heading_ready": True}]

    def drain_samples(self):
        records, self.records = self.records, []
        return records


class RecorderTests(unittest.TestCase):
    def test_distinct_samples_and_gps_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            with SessionWriter(directory, "record") as writer:
                imu = FakeIMU()
                recorder = SensorRecorder(writer, FakeGPS(), imu, gps_interval_s=1.0)
                self.assertEqual(recorder.step(1_000_000_000), (True, True))
                self.assertEqual(recorder.step(1_500_000_000), (False, False))
                imu.timestamp = 2.0
                self.assertEqual(recorder.step(2_000_000_000), (True, True))
            self.assertEqual(len(read_jsonl(writer.raw_dir / "imu.jsonl")), 2)
            self.assertEqual(len(read_jsonl(writer.raw_dir / "gps.jsonl")), 2)

    def test_missing_gps_becomes_event(self):
        with tempfile.TemporaryDirectory() as directory:
            gps = FakeGPS()
            gps.value = (None, "poor")
            with SessionWriter(directory, "missing") as writer:
                recorder = SensorRecorder(writer, gps, FakeIMU())
                self.assertFalse(recorder.capture_gps(1_000_000_000))
            events = read_jsonl(writer.raw_dir / "events.jsonl")
            self.assertEqual(events[0]["type"], "gps_unavailable")

    def test_recorder_prefers_lossless_imu_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            with SessionWriter(directory, "buffered") as writer:
                recorder = SensorRecorder(writer, FakeGPS(), FakeBufferedIMU())
                self.assertTrue(recorder.capture_imu(999))
                self.assertFalse(recorder.capture_imu(1000))
            records = read_jsonl(writer.raw_dir / "imu.jsonl")
            self.assertEqual(records[0]["monotonic_ns"], 10)


if __name__ == "__main__":
    unittest.main()
