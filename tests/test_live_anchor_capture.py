import tempfile
import unittest

import numpy as np

from memory_nav.models import GPSSample
from memory_nav.recording.anchor_collector import LiveAnchorCapture
from memory_nav.recording.session_writer import SessionWriter, read_jsonl


class Camera:
    def capture_frame_with_metadata(self):
        checker = (np.indices((40, 40)).sum(axis=0) % 2 * 255).astype(np.uint8)
        return np.stack([checker] * 3, axis=2), {"frame_count": 1, "monotonic_ns": 99}


class LiveAnchorCaptureTests(unittest.TestCase):
    def test_start_interval_turn_and_end_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            with SessionWriter(directory, "anchors") as writer:
                capture = LiveAnchorCapture(writer, Camera(), max_spacing_m=1.0, turn_angle_deg=30, blur_threshold=10)
                first = GPSSample(1, 113.0, 23.0, "rtk", "good")
                second = GPSSample(2, 113.00002, 23.0, "rtk", "good")
                third = GPSSample(3, 113.000021, 23.0, "rtk", "good")
                self.assertTrue(capture.consider(first, 90))
                self.assertTrue(capture.consider(second, 90))
                self.assertTrue(capture.consider(third, 140))
                fourth = GPSSample(4, 113.000022, 23.0, "rtk", "good")
                self.assertFalse(capture.consider(fourth, 140))
                self.assertTrue(capture.finalize(140))
            events = [event for event in read_jsonl(writer.raw_dir / "events.jsonl") if event["type"] == "anchor_image"]
            self.assertEqual([event["kind"] for event in events], ["start", "interval", "turn", "end"])
            self.assertTrue(all(event["clear"] for event in events))
            self.assertTrue(all((writer.route_dir / event["image_path"]).is_file() for event in events))


if __name__ == "__main__":
    unittest.main()
