import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from memory_nav.segmentation.avoidance import SegmentationAvoidance, SegmentationConfig
from memory_nav.segmentation.catseg_worker import (
    _serve_connection,
    parse_args as worker_parse_args,
)
from memory_nav.segmentation.catseg_trt import preprocess_bgr, probabilities_to_mask
from memory_nav.segmentation.guard import SegmentationGuard
from memory_nav.segmentation.online_provider import (
    CatSegWorkerProvider,
    MaskObservation,
    NullMaskProvider,
)
from memory_nav.segmentation.online_visualizer import SegmentationFrameSaver
from memory_nav.segmentation.protocol import recv_message, send_message
from memory_nav.segmentation.providers import CachedMaskProvider
from memory_nav.segmentation.visualize import draw_mask_overlay


def left_half_mask(height=100, width=100):
    mask = np.zeros((height, width), dtype=bool)
    mask[:, : width // 2] = True
    return mask


class StaticProvider:
    def __init__(self, mask):
        self.mask = mask
        self.closed = False

    def get_mask(self, frame_id, frame=None):
        return self.mask

    def close(self):
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def test_socketpair_round_trip(self):
        sender, receiver = socket.socketpair()
        try:
            frame = np.zeros((4, 5, 3), dtype=np.uint8)
            send_message(sender, {"type": "frame", "image": frame})
            message = recv_message(receiver)
            self.assertEqual(message["type"], "frame")
            self.assertEqual(message["image"].shape, (4, 5, 3))
        finally:
            sender.close()
            receiver.close()

    def test_recv_message_returns_none_on_eof(self):
        sender, receiver = socket.socketpair()
        sender.close()
        try:
            self.assertIsNone(recv_message(receiver))
        finally:
            receiver.close()

    def test_worker_echoes_source_frame_metadata(self):
        sender, receiver = socket.socketpair()
        worker = threading.Thread(
            target=_serve_connection,
            args=(receiver, lambda image: np.ones(image.shape[:2], dtype=bool), 1.0),
        )
        worker.start()
        try:
            send_message(
                sender,
                {
                    "type": "frame",
                    "frame_id": "frame_17",
                    "captured_at_s": 123.5,
                    "image": np.zeros((4, 5, 3), dtype=np.uint8),
                },
            )
            reply = recv_message(sender)
            self.assertEqual(reply["frame_id"], "frame_17")
            self.assertEqual(reply["captured_at_s"], 123.5)
            self.assertEqual(reply["mask"].shape, (4, 5))
            send_message(sender, {"type": "shutdown"})
        finally:
            sender.close()
            worker.join(timeout=1.0)
            receiver.close()


class WorkerArgTests(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = worker_parse_args(["--socket", "/tmp/x.sock"])
        self.assertEqual(args.socket, "/tmp/x.sock")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.backend, "pytorch")
        self.assertEqual(args.input_scale, 1.0)
        self.assertIsNone(args.min_size_test)

    def test_parse_args_tensorrt(self):
        args = worker_parse_args([
            "--socket", "/tmp/x.sock", "--backend", "tensorrt", "--trt-warmup", "2"
        ])
        self.assertEqual(args.backend, "tensorrt")
        self.assertEqual(args.trt_warmup, 2)


class TensorRTHelperTests(unittest.TestCase):
    def test_preprocess_bgr_has_fixed_rgb_nchw_shape(self):
        bgr = np.zeros((20, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10
        bgr[:, :, 1] = 20
        bgr[:, :, 2] = 30
        tensor = preprocess_bgr(bgr)
        self.assertEqual(tensor.shape, (1, 3, 384, 384))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tuple(tensor[0, :, 0, 0]), (30.0, 20.0, 10.0))

    def test_probabilities_to_mask_resizes_walkable_argmax(self):
        scores = np.zeros((1, 3, 2, 2), dtype=np.float32)
        scores[:, 2, :, :1] = 1.0
        scores[:, 1, :, 1:] = 1.0
        mask = probabilities_to_mask(scores, [2], (4, 4))
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(mask[:, :2].all())
        self.assertFalse(mask[:, 2:].any())


class ProviderTests(unittest.TestCase):
    def make_provider(self):
        provider = CatSegWorkerProvider(socket_path="/tmp/does-not-matter.sock")
        provider._started = True  # Avoid spawning the real worker.
        return provider

    @staticmethod
    def set_observation(provider, *, age_s=0.0, frame_id="frame_1"):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        now = time.monotonic()
        provider._latest_observation = MaskObservation(
            frame_id=frame_id,
            captured_at_s=now - age_s,
            completed_at_s=now,
            frame=frame,
            mask=left_half_mask(25, 25).astype(np.uint8) * 255,
            inference_s=0.05,
        )

    def test_to_bgr_converts_rgb(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        bgr = CatSegWorkerProvider._to_bgr(rgb)
        self.assertEqual(tuple(bgr[0, 0]), (0, 0, 255))

    def test_get_mask_returns_latest_and_resizes(self):
        provider = self.make_provider()
        self.set_observation(provider)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = provider.get_mask("frame_1", frame)
        self.assertEqual(mask.shape, (50, 50))
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, -1])

    def test_get_mask_returns_none_without_result(self):
        provider = self.make_provider()
        self.assertIsNone(provider.get_mask("frame_1"))

    def test_get_mask_rejects_observation_older_than_limit(self):
        provider = self.make_provider()
        self.set_observation(provider, age_s=0.41)
        self.assertIsNone(provider.get_mask("frame_2"))
        self.assertIsNotNone(provider.latest_observation())

    def test_provider_uses_host_upload_time_for_new_frame(self):
        class LiveThread:
            def is_alive(self):
                return True

        provider = self.make_provider()
        provider._started = True
        provider._thread = LiveThread()
        uploaded_at_s = time.monotonic() - 0.3
        provider.get_observation(
            "frame_source", np.zeros((8, 8, 3), dtype=np.uint8),
            captured_at_s=uploaded_at_s,
        )
        self.assertAlmostEqual(
            provider._pending_frame.captured_at_s, uploaded_at_s, delta=0.01
        )

    def test_observation_keeps_source_frame_and_id(self):
        provider = self.make_provider()
        self.set_observation(provider, frame_id="frame_source")
        observation = provider.get_observation("frame_current")
        self.assertIsNotNone(observation)
        self.assertEqual(observation.frame_id, "frame_source")
        self.assertEqual(observation.frame.shape, (50, 50, 3))
        self.assertEqual(observation.mask.shape, (50, 50))

    def test_null_provider_never_returns_mask(self):
        provider = NullMaskProvider()
        self.assertIsNone(provider.get_mask("frame_1", np.zeros((4, 4, 3), np.uint8)))
        provider.close()


class GuardVisualizerTests(unittest.TestCase):
    def test_guard_invokes_visualizer_and_closes_provider(self):
        calls = []
        provider = StaticProvider(left_half_mask())
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            provider,
            cooldown_s=0.0,
            visualizer=lambda *args: calls.append(args),
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        decision = guard.apply(3, "3点钟方向前进", "frame_1", frame, now_s=0.0)
        self.assertIsNotNone(decision)
        self.assertEqual(len(calls), 1)
        guard.close()
        self.assertTrue(provider.closed)

    def test_guard_skips_visualizer_without_mask(self):
        calls = []
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            NullMaskProvider(),
            visualizer=lambda *args: calls.append(args),
        )
        guard.apply(3, "3点钟方向前进", "frame_1", np.zeros((10, 10, 3), np.uint8), now_s=0.0)
        self.assertEqual(calls, [])

    def test_guard_visualizes_the_mask_source_frame(self):
        calls = []
        source_frame = np.full((100, 100, 3), 77, dtype=np.uint8)

        class ObservationProvider:
            def get_observation(self, frame_id, frame=None):
                now = time.monotonic()
                return MaskObservation(
                    frame_id="source",
                    captured_at_s=now,
                    completed_at_s=now,
                    frame=source_frame,
                    mask=left_half_mask(),
                    inference_s=0.01,
                )

        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            ObservationProvider(),
            visualizer=lambda *args: calls.append(args),
        )
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        guard.apply(3, "3点钟方向前进", "current", current_frame, now_s=0.0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.array_equal(calls[0][0], source_frame))

class GuardObserveTests(unittest.TestCase):
    def test_observe_returns_starting_without_mask(self):
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()), NullMaskProvider()
        )
        status = guard.observe("frame_1", np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(status, {"status": "starting", "mask_ready": False})

    def test_observe_visualizes_mask_without_command(self):
        calls = []

        class Visualizer:
            def save_mask(self, frame, mapper, mask):
                calls.append((frame.shape, mask.shape))

        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            StaticProvider(left_half_mask()),
            visualizer=Visualizer(),
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        status = guard.observe("frame_1", frame)
        self.assertEqual(status, {"status": "pose_unavailable", "mask_ready": True})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], (100, 100))

    def test_draw_mask_overlay_handles_both_masks(self):
        mapper = SegmentationAvoidance(SegmentationConfig()).mapper
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for mask in (np.ones((100, 100), dtype=bool), np.zeros((100, 100), dtype=bool)):
            annotated = draw_mask_overlay(frame, mapper, mask)
            self.assertEqual(annotated.shape, frame.shape)


class BuildGuardTests(unittest.TestCase):
    def test_online_mode_selects_worker_provider_and_visualizer(self):
        from memory_nav.config import load_config
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args([
            "--route-id", "route",
            "--seg-online",
            "--seg-device", "cpu",
            "--seg-vis-dir", "/tmp/opencode/seg_online_test_vis",
        ])
        guard = build_guard(args, load_config())
        self.assertIsNotNone(guard)
        self.assertIsInstance(guard.provider, CatSegWorkerProvider)
        self.assertEqual(guard.provider.backend, "pytorch")
        self.assertIsNotNone(guard.visualizer)
        guard.close()

    def test_online_mode_uses_configured_tensorrt_backend(self):
        from memory_nav.config import load_config
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args(["--route-id", "route", "--seg-online"])
        config = load_config()
        guard = build_guard(args, config)
        self.assertIsNotNone(guard)
        self.assertEqual(guard.provider.backend, "tensorrt")
        self.assertEqual(guard.provider._min_interval_s, 0.25)
        self.assertEqual(
            guard.provider.max_mask_age_s,
            config["segmentation"]["online"]["max_mask_age_s"],
        )
        guard.close()

    def test_offline_mode_selects_cached_provider(self):
        from memory_nav.config import load_config
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args([
            "--route-id", "route",
            "--walkable-mask-dir", "/tmp/opencode/does_not_exist",
        ])
        guard = build_guard(args, load_config())
        self.assertIsNotNone(guard)
        self.assertIsInstance(guard.provider, CachedMaskProvider)

    def test_online_radius_override_from_cli(self):
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args(["--route-id", "route", "--seg-online", "--seg-radius-ratio", "0.35"])
        guard = build_guard(args, {"segmentation": {}})
        self.assertAlmostEqual(guard.avoidance.config.radius_ratio, 0.35)
        guard.close()

    def test_online_radius_override_from_config(self):
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args(["--route-id", "route", "--seg-online"])
        guard = build_guard(args, {"segmentation": {"online": {"radius_ratio": 0.4}}})
        self.assertAlmostEqual(guard.avoidance.config.radius_ratio, 0.4)
        guard.close()

    def test_online_radius_defaults_to_base(self):
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args(["--route-id", "route", "--seg-online"])
        guard = build_guard(args, {"segmentation": {"radius_ratio": 0.3}})
        self.assertAlmostEqual(guard.avoidance.config.radius_ratio, 0.3)
        guard.close()

    def test_no_segmentation_source_returns_none(self):
        from memory_nav.config import load_config
        from memory_nav.replay.replay_nav_seg import build_guard, parse_args

        args = parse_args(["--route-id", "route"])
        guard = build_guard(args, load_config())
        self.assertIsNone(guard)


class VisualizerTests(unittest.TestCase):
    def test_saver_writes_and_throttles(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = SegmentationFrameSaver(directory, interval_s=0.0)
            result = SegmentationAvoidance(SegmentationConfig()).evaluate(10, left_half_mask())
            mapper = SegmentationAvoidance(SegmentationConfig()).mapper
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            saver(frame, result, mapper, left_half_mask())
            self.assertEqual(saver.saved_count, 1)
            files = list(Path(directory).glob("seg_*.jpg"))
            self.assertEqual(len(files), 1)

            saver.interval_s = 1000.0
            saver(frame, result, mapper, left_half_mask())
            self.assertEqual(saver.saved_count, 1)
            saver.close()

    def test_saver_converts_rgb_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = SegmentationFrameSaver(directory, interval_s=0.0)
            result = SegmentationAvoidance(SegmentationConfig()).evaluate(10, left_half_mask())
            mapper = SegmentationAvoidance(SegmentationConfig()).mapper
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, 0] = 200
            saver(frame, result, mapper, left_half_mask())
            saved = cv2.imread(str(next(Path(directory).glob("seg_*.jpg"))))
            self.assertIsNotNone(saved)
            saver.close()

    def test_saver_save_mask_writes_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = SegmentationFrameSaver(directory, interval_s=0.0)
            mapper = SegmentationAvoidance(SegmentationConfig()).mapper
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            saver.save_mask(frame, mapper, left_half_mask())
            self.assertEqual(saver.saved_count, 1)
            saved = cv2.imread(str(next(Path(directory).glob("seg_*.jpg"))))
            self.assertIsNotNone(saved)
            saver.close()

if __name__ == "__main__":
    unittest.main()
