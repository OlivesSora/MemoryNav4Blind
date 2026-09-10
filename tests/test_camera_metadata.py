import threading
import unittest

import numpy as np

from utils.glasses_camera import Camera


class CameraMetadataTests(unittest.TestCase):
    def test_legacy_capture_can_be_wrapped_with_timestamp(self):
        camera = Camera.__new__(Camera)
        camera._capture_condition = threading.Condition()
        camera.frame_lock = threading.Lock()
        camera.frame_monotonic_ns = 123456
        camera._requested_frame_monotonic_ns = 999
        camera.capture_frame = lambda width=None, height=None, timeout=5.0: (np.zeros((10, 20, 3), dtype=np.uint8), 7)
        image, metadata = camera.capture_frame_with_metadata()
        self.assertEqual(image.shape, (10, 20, 3))
        self.assertEqual(metadata, {"frame_count": 7, "monotonic_ns": 123456, "width": 20, "height": 10})
        _, requested = camera.capture_frame_with_metadata(20, 10)
        self.assertEqual(requested["monotonic_ns"], 999)


if __name__ == "__main__":
    unittest.main()
