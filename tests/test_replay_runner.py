import json
import tempfile
import unittest
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.recording.anchor_collector import AnchorCandidate, save_anchors
from memory_nav.replay.replay_nav import HardwareGatewayCamera, ReplayRunner, validate_ready_route
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
    def test_gateway_camera_rejects_repeated_and_old_frames(self):
        import time

        class Agent:
            def __init__(self):
                self.frame_count = 1
                self.received_at_ns = time.time_ns()

            async def get_state_image(self, **kwargs):
                self.last_kwargs = kwargs
                return (
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    self.frame_count,
                    {"received_at_unix_ns": self.received_at_ns},
                )

        agent = Agent()
        camera = HardwareGatewayCamera(agent, max_source_age_s=0.8)
        self.assertIsNotNone(camera.capture_frame()[0])
        self.assertIsNone(camera.capture_frame()[0])
        agent.frame_count = 2
        agent.received_at_ns = time.time_ns() - 2_000_000_000
        self.assertIsNone(camera.capture_frame()[0])
        self.assertTrue(agent.last_kwargs["return_metadata"])

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

    def test_online_segmentation_without_fresh_mask_speaks_stop(self):
        class Camera:
            def capture_frame(self):
                return np.zeros((8, 8, 3), dtype=np.uint8), 1

        class Guard:
            def apply(self, *args):
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root)
            position = frame.to_geodetic(3, 0)
            spoken = []
            runner = ReplayRunner(
                root, load_config(), gps=FakeGPS([(list(position), "good")]),
                imu=FakeIMU([90]), speaker=spoken.append, camera=Camera(),
                voice_directions=True, segmentation_guard=Guard(),
                segmentation_observe_always=True,
            )
            result = runner.step()
            runner.close()
            self.assertEqual(result["command_clock"], None)
            self.assertEqual(result["segmentation"]["status"], "mask_unavailable")
            self.assertEqual([p["key"] for p in result["prompts"]], ["seg:mask_unavailable"])
            self.assertEqual(spoken, ["前方画面未更新，请停止前进"])

    def test_missing_pose_speaks_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_route(root)
            spoken = []
            runner = ReplayRunner(
                root, load_config(), gps=FakeGPS([(None, None)]),
                imu=FakeIMU([None]), speaker=spoken.append,
                voice_directions=True,
            )
            result = runner.step()
            runner.close()
            self.assertEqual([p["key"] for p in result["prompts"]], ["sensor:pose_unavailable"])
            self.assertEqual(spoken, ["定位或朝向不可用，请停止前进"])

    def test_following_writes_outdoor_nav_compatible_gps_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.make_route(root)
            position = frame.to_geodetic(3, 0)
            log_path = root / "follow_output" / "follow-test" / "output.jsonl"
            runner = ReplayRunner(
                root,
                load_config(),
                gps=FakeGPS([(list(position), "good")]),
                imu=FakeIMU([90]),
                follow_log_path=log_path,
            )
            runner.step()
            runner.close()
            record = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["current_pos"], list(position))
            self.assertEqual(record["raw_imu_bearing"], 90.0)
            self.assertIn("log_time", record)
            self.assertIn("route_progress", record)
            self.assertIn("cross_track_error", record)

    def test_online_runner_stays_in_following_state_without_sensor_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_route(root)
            runner = ReplayRunner(root, load_config(), gps=FakeGPS([(None, None)]), imu=FakeIMU([None]))
            result = runner.step()
            runner.close()
            self.assertEqual(result["navigation_state"], "following")
            self.assertFalse(result["pause_progress"])
            self.assertEqual(result["prompts"], [])

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
            # With match-distance quality gates disabled, the 4 m phone
            # offset remains inside the widened deviation threshold (6 m).
            self.assertEqual(result["navigation_state"], "following")

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
            self.assertEqual(result["navigation_state"], "following")
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
            self.assertEqual(result["navigation_state"], "following")

    def test_observe_segmentation_without_pose(self):
        class Guard:
            def __init__(self):
                self.calls = 0
                self.closed = False

            def observe(self, frame_id, frame=None):
                self.calls += 1
                return {"status": "pose_unavailable", "mask_ready": True}

            def close(self):
                self.closed = True

        class Camera:
            def capture_frame(self):
                return np.zeros((20, 20, 3), dtype=np.uint8), 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_route(root)
            guard = Guard()
            runner = ReplayRunner(
                root,
                load_config(),
                gps=FakeGPS([(None, None)]),
                imu=FakeIMU([None]),
                camera=Camera(),
                segmentation_guard=guard,
                segmentation_observe_always=True,
            )
            result = runner.step()
            runner.close()
            self.assertEqual(result["navigation_state"], "following")
            self.assertEqual(result["segmentation"], {"status": "pose_unavailable", "mask_ready": True})
            self.assertEqual(guard.calls, 1)

    def test_observe_segmentation_disabled_by_default_without_pose(self):
        class Guard:
            def __init__(self):
                self.calls = 0

            def observe(self, frame_id, frame=None):
                self.calls += 1
                return {"status": "pose_unavailable", "mask_ready": True}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_route(root)
            guard = Guard()
            runner = ReplayRunner(
                root,
                load_config(),
                gps=FakeGPS([(None, None)]),
                imu=FakeIMU([None]),
                segmentation_guard=guard,
            )
            result = runner.step()
            runner.close()
            self.assertNotIn("segmentation", result)
            self.assertEqual(guard.calls, 0)

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
