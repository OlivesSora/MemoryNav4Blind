import json
import tempfile
import unittest
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.recording.anchor_collector import AnchorCandidate, save_anchors
from memory_nav.replay.replay_nav import ReplayRunner, validate_ready_route
from memory_nav.trajectory.coordinate import LocalFrame
from memory_nav.replay.anchor_matcher import VisualMatchResult
import cv2
import numpy as np


class FakeGPS:
    def __init__(self, samples):
        self.samples = iter(samples)

    def get_gps(self):
        return next(self.samples)


class FakeRichGPS:
    def __init__(self, samples):
        self.samples = iter(samples)

    def get_sample(self):
        return next(self.samples)


class FakeIMU:
    def __init__(self, headings):
        self.headings = iter(headings)
        self.stopped = False

    def get_heading(self):
        return next(self.headings)

    def stop(self):
        self.stopped = True


class ReplayRunnerTests(unittest.TestCase):
    def make_route(self, root: Path, anchor_image=False):
        frame = LocalFrame(113.0, 23.0)
        points = []
        for index, east in enumerate((0.0, 10.0, 20.0)):
            longitude, latitude = frame.to_geodetic(east, 0)
            points.append({
                "index": index, "s_m": east, "east_m": east, "north_m": 0.0,
                "longitude": longitude, "latitude": latitude, "heading_deg": 90.0,
                "curvature": 0.0, "quality": "good", "anchor_id": None,
            })
        (root / "reference_trajectory.json").write_text(json.dumps({
            "origin": {"longitude": 113.0, "latitude": 23.0}, "points": points,
        }), encoding="utf-8")
        image_path = None
        if anchor_image:
            (root / "anchors").mkdir()
            image_path = "anchors/turn.jpg"
            cv2.imwrite(str(root / image_path), np.zeros((20, 20, 3), dtype=np.uint8))
        save_anchors(root / "anchors.json", [AnchorCandidate("turn", 1, 10, "turn", "前方左转", True, image_path)])
        return frame

    def test_online_runner_connects_match_anchor_and_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root)
            positions = [frame.to_geodetic(east, 0) for east in (3, 9)]
            gps = FakeGPS([(list(position), "good") for position in positions])
            imu = FakeIMU([90, 90])
            spoken = []
            config = load_config()
            config["voice"]["cooldown_s"] = 0
            runner = ReplayRunner(root, config, gps=gps, imu=imu, speaker=spoken.append)
            first = runner.step()
            second = runner.step()
            runner.close()
            self.assertEqual(first["next_anchor_id"], "turn")
            self.assertEqual([prompt["key"] for prompt in first["prompts"]], ["anchor:turn:advance"])
            self.assertEqual([prompt["key"] for prompt in second["prompts"]], ["anchor:turn:arrival"])
            self.assertCountEqual(spoken, ["前方左转", "前方左转"])
            self.assertTrue(imu.stopped)

    def test_direction_command_is_sent_to_tts_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root)
            position = frame.to_geodetic(3, 0)
            spoken = []
            runner = ReplayRunner(
                root, load_config(), gps=FakeGPS([(list(position), "good")]),
                imu=FakeIMU([90]), speaker=spoken.append, voice_directions=True,
            )
            result = runner.step()
            runner.close()
            self.assertTrue(any(item["key"].startswith("direction:") for item in result["prompts"]))
            self.assertIn(result["command"], spoken)

    def test_online_runner_reports_lost_without_advancing_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_route(root)
            runner = ReplayRunner(root, load_config(), gps=FakeGPS([(None, None)]), imu=FakeIMU([None]))
            result = runner.step()
            runner.close()
            self.assertEqual(result["navigation_state"], "lost")
            self.assertTrue(result["pause_progress"])
            self.assertEqual(result["prompts"][0]["key"], "state:lost")

    def test_phone_source_is_not_inferred_from_good_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root)
            longitude, latitude = frame.to_geodetic(3, 4)
            sample = {"longitude": longitude, "latitude": latitude, "state": "good", "source": "phone"}
            gps = FakeRichGPS([sample, sample, sample])
            runner = ReplayRunner(root, load_config(), gps=gps, imu=FakeIMU([90, 90, 90]))
            result = runner.step()
            result = runner.step()
            result = runner.step()
            runner.close()
            self.assertEqual(result["gps_source"], "phone")
            # 4m would become deviated after three RTK samples but remains
            # only degraded inside the phone threshold (6m).
            self.assertEqual(result["navigation_state"], "degraded")

    def test_visual_anchor_success_and_failure_are_non_blocking(self):
        class Camera:
            def __init__(self, image):
                self.image = image
                self.released = False

            def capture_frame(self):
                return self.image, 1

            def release(self):
                self.released = True

        class Matcher:
            def match(self, reference, current):
                return VisualMatchResult(True, 30, 25, 25 / 30)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root, anchor_image=True)
            position = frame.to_geodetic(9, 0)
            camera = Camera(np.zeros((20, 20, 3), dtype=np.uint8))
            runner = ReplayRunner(root, load_config(), gps=FakeGPS([(list(position), "good")]), imu=FakeIMU([90]), camera=camera, visual_matcher=Matcher())
            result = runner.step()
            runner.close()
            self.assertTrue(result["visual_anchor"]["matched"])
            self.assertEqual(result["navigation_state"], "normal")
            self.assertTrue(camera.released)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root, anchor_image=True)
            (root / "anchors" / "turn.jpg").unlink()
            position = frame.to_geodetic(9, 0)
            runner = ReplayRunner(root, load_config(), gps=FakeGPS([(list(position), "good")]), imu=FakeIMU([90]), camera=Camera(np.zeros((20, 20, 3), dtype=np.uint8)), visual_matcher=Matcher())
            result = runner.step()
            runner.close()
            self.assertFalse(result["visual_anchor"]["matched"])
            self.assertEqual(result["navigation_state"], "normal")

    def test_low_quality_route_is_rejected_before_hardware_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_route(root)
            (root / "manifest.json").write_text(json.dumps({"state": "ERROR"}), encoding="utf-8")
            (root / "quality_report.json").write_text(json.dumps({"ready": False, "failure_reasons": ["too short"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "too short"):
                validate_ready_route(root)
            (root / "manifest.json").write_text(json.dumps({"state": "READY"}), encoding="utf-8")
            (root / "quality_report.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
            validate_ready_route(root)


if __name__ == "__main__":
    unittest.main()
