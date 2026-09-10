import tempfile
import unittest
from pathlib import Path

import numpy as np

from memory_nav.calibration.collect_calibration import CalibrationCollector
from memory_nav.calibration.import_calibration import convert_kalibr
from memory_nav.calibration.run_kalibr import build_bag_command, build_command
from memory_nav.calibration.validate_calibration import CalibrationError, validate_calibration
from memory_nav.recording.session_writer import read_jsonl


class FakeCamera:
    def capture_frame(self, width=None, height=None):
        return np.zeros((height or 24, width or 32, 3), dtype=np.uint8), 7


class FakeIMU:
    def __init__(self):
        self.time = 1.0

    def get_imu(self):
        return [0.1, 0.2, 0.3], [0, 0, 9.8], [1, 2, 3], self.time


class CalibrationTests(unittest.TestCase):
    def test_collect_frame_and_distinct_imu(self):
        with tempfile.TemporaryDirectory() as directory:
            imu = FakeIMU()
            collector = CalibrationCollector(directory, FakeCamera(), imu)
            self.assertTrue(collector.capture_imu(10))
            self.assertFalse(collector.capture_imu(11))
            imu.time = 2.0
            self.assertTrue(collector.capture_imu(12))
            self.assertTrue(collector.capture_frame(13, 64, 48))
            self.assertEqual(len(read_jsonl(Path(directory) / "imu.jsonl")), 2)
            frame = read_jsonl(Path(directory) / "frames.jsonl")[0]
            self.assertEqual((frame["width"], frame["height"]), (64, 48))
            self.assertTrue((Path(directory) / frame["filename"]).is_file())
            csv_lines = (Path(directory) / "dataset" / "imu0.csv").read_text().splitlines()
            self.assertEqual(csv_lines[0], "timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z")
            self.assertEqual(len(csv_lines), 3)

    def test_validate_transform(self):
        valid = {"schema_version": 1, "camera": {}, "imu": {}, "T_camera_imu": np.eye(4).tolist(), "time_offset_s": 0.01, "quality": {"passed": True}}
        validate_calibration(valid)
        invalid = dict(valid)
        invalid["T_camera_imu"] = [[1, 0], [0, 1]]
        with self.assertRaises(CalibrationError):
            validate_calibration(invalid)

    def test_convert_kalibr_output(self):
        source = {"cam0": {
            "camera_model": "pinhole",
            "intrinsics": [500, 501, 320, 240],
            "distortion_model": "radtan",
            "distortion_coeffs": [0.1, 0.01, 0, 0],
            "resolution": [640, 480],
            "T_cam_imu": np.eye(4).tolist(),
            "timeshift_cam_imu": -0.003,
            "rostopic": "/camera/image",
        }}
        result = convert_kalibr(source, reprojection_error_px=0.4, maximum_error_px=0.8)
        self.assertTrue(result["quality"]["passed"])
        self.assertEqual(result["camera"]["resolution"], [640, 480])
        self.assertEqual(result["time_offset_s"], -0.003)

    def test_build_kalibr_command_validates_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / name for name in ("data.bag", "camera.yaml", "imu.yaml", "target.yaml")]
            for path in inputs:
                path.touch()
            command = build_command(*inputs, root / "output")
            self.assertEqual(command[0], "kalibr_calibrate_imu_camera")
            self.assertIn(str(inputs[0].resolve()), command)
            inputs[0].unlink()
            with self.assertRaises(FileNotFoundError):
                build_command(*inputs, root / "output")

    def test_build_bag_creator_command(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            (dataset / "cam0").mkdir(parents=True)
            (dataset / "imu0.csv").touch()
            output = Path(directory) / "route.bag"
            command = build_bag_command(dataset, output)
            self.assertEqual(command[0], "kalibr_bagcreater")
            self.assertEqual(command[-1], str(output.resolve()))


if __name__ == "__main__":
    unittest.main()
