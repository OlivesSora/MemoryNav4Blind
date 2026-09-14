import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from memory_nav.segmentation.avoidance import SegmentationAvoidance, SegmentationConfig
from memory_nav.segmentation.geometry import (
    SemicircleMapper,
    angle_to_clock,
    clock_angular_distance,
    clock_to_angle_deg,
    normalize_clock,
)
from memory_nav.segmentation.guard import SegmentationGuard
from memory_nav.segmentation.mask_loader import load_mask, save_mask
from memory_nav.segmentation.providers import CachedMaskProvider, SequenceMaskProvider
from memory_nav.segmentation.validate_offline import (
    build_pairs,
    command_clock_from_record,
    main as validate_main,
)
from memory_nav.segmentation.visualize import FALLBACK_COLOR, draw_segmentation


def left_half_mask(height=100, width=100):
    mask = np.zeros((height, width), dtype=bool)
    mask[:, : width // 2] = True
    return mask


class ClockGeometryTests(unittest.TestCase):
    def test_clock_to_angle_and_normalization(self):
        self.assertEqual(clock_to_angle_deg(12), 0.0)
        self.assertEqual(clock_to_angle_deg(3), 90.0)
        self.assertEqual(clock_to_angle_deg(9), -90.0)
        self.assertEqual(normalize_clock(0), 12)
        self.assertEqual(normalize_clock("12"), 12)
        self.assertEqual(angle_to_clock(90.0), 3)
        self.assertEqual(angle_to_clock(-90.0), 9)

    def test_clock_angular_distance_wraps(self):
        self.assertEqual(clock_angular_distance(12, 1), 30.0)
        self.assertEqual(clock_angular_distance(11, 1), 60.0)
        self.assertEqual(clock_angular_distance(3, 9), 180.0)

    def test_hour_clearance_left_half_walkable(self):
        mapper = SemicircleMapper()
        radius = mapper.radius(left_half_mask())
        clearances = mapper.hour_clearance(left_half_mask())
        self.assertEqual(clearances[9].obstacle_pixels, 0)
        self.assertEqual(clearances[10].obstacle_pixels, 0)
        self.assertEqual(clearances[9].clear_distance, radius)
        self.assertGreater(clearances[3].obstacle_pixels, 0)
        self.assertLess(clearances[3].clear_distance, 0.5 * radius)
        self.assertGreater(clearances[12].obstacle_pixels, 0)
        self.assertLess(clearances[12].clear_distance, 0.5 * radius)

    def test_full_mask_all_clear(self):
        mapper = SemicircleMapper()
        mask = np.ones((100, 100), dtype=bool)
        radius = mapper.radius(mask)
        clearances = mapper.hour_clearance(mask)
        for hour, clearance in clearances.items():
            self.assertEqual(clearance.obstacle_pixels, 0, msg=f"hour {hour}")
            self.assertEqual(clearance.clear_distance, radius, msg=f"hour {hour}")


class AvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.avoidance = SegmentationAvoidance(SegmentationConfig())

    def test_consistent_direction_passes(self):
        result = self.avoidance.evaluate(10, left_half_mask())
        self.assertTrue(result.consistent)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.seg_clock, 10)

    def test_blocked_direction_falls_back_to_nearest_walkable(self):
        result = self.avoidance.evaluate(3, left_half_mask())
        self.assertFalse(result.consistent)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.seg_clock, 11)
        self.assertIn(11, result.walkable_hours)

    def test_backward_command_is_out_of_scope(self):
        result = self.avoidance.evaluate(6, left_half_mask())
        self.assertTrue(result.out_of_scope)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.warning, "out_of_scope")

    def test_empty_mask_reports_no_walkable(self):
        result = self.avoidance.evaluate(12, np.zeros((100, 100), dtype=bool))
        self.assertFalse(result.consistent)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.warning, "no_walkable")


class GuardTests(unittest.TestCase):
    class StaticProvider:
        def __init__(self, mask):
            self.mask = mask

        def get_mask(self, frame_id, frame=None):
            return self.mask

    def test_guard_overrides_blocked_direction_and_prompts(self):
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            self.StaticProvider(left_half_mask()),
            cooldown_s=0.0,
        )
        decision = guard.apply(3, "3点钟方向前进", "frame_1", left_half_mask(), now_s=0.0)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.override_used)
        self.assertEqual(decision.command_clock, 11)
        self.assertIn("11点钟", decision.command_text)
        self.assertEqual(len(decision.prompts), 1)
        self.assertEqual(decision.prompts[0].key, "seg:command_blocked:11")

    def test_no_walkable_emits_stop_and_rotate_prompts(self):
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            self.StaticProvider(np.zeros((100, 100), dtype=bool)),
            cooldown_s=5.0,
        )
        decision = guard.apply(12, "12点钟方向直行", "frame_1", np.zeros((100, 100, 3), np.uint8), now_s=0.0)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.override_used)
        keys = [prompt.key for prompt in decision.prompts]
        self.assertEqual(keys, ["seg:no_walkable", "seg:no_walkable_rotate"])
        self.assertEqual(decision.command_text, "前方未检测到可通行区域，请停止前进")
        self.assertIn("请旋转一下", decision.prompts[1].text)
        # Stable keys: rotating (different clock) does not re-trigger within cooldown.
        repeat = guard.apply(3, "3点钟方向前进", "frame_2", np.zeros((100, 100, 3), np.uint8), now_s=0.1)
        self.assertEqual(repeat.prompts, ())

    def test_custom_no_walkable_texts_are_used(self):
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            self.StaticProvider(np.zeros((100, 100), dtype=bool)),
            cooldown_s=0.0,
            no_walkable_text="STOP",
            rotate_text="TURN",
        )
        decision = guard.apply(12, "12点钟方向直行", "frame_1", np.zeros((100, 100, 3), np.uint8), now_s=0.0)
        self.assertEqual([prompt.text for prompt in decision.prompts], ["STOP", "TURN"])

    def test_guard_returns_none_without_mask(self):
        guard = SegmentationGuard(
            SegmentationAvoidance(SegmentationConfig()),
            CachedMaskProvider("/nonexistent"),
        )
        self.assertIsNone(guard.apply(3, "3点钟方向前进", "frame_1", now_s=0.0))

    def test_sequence_provider_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [root / "frame_1.jpg", root / "frame_2.jpg"]
            for frame in frames:
                cv2.imwrite(str(frame), np.zeros((10, 10, 3), dtype=np.uint8))
                save_mask(root / f"{frame.stem}_walkable.png", left_half_mask(10, 10))
            provider = SequenceMaskProvider(root, frames, loop=False)
            self.assertIsNotNone(provider.get_mask("ignored"))
            self.assertIsNotNone(provider.get_mask("ignored"))
            self.assertIsNone(provider.get_mask("ignored"))


class MaskLoaderTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_1_walkable.png"
            save_mask(path, left_half_mask(20, 20))
            loaded = load_mask(path)
            self.assertEqual(loaded.dtype, np.bool_)
            self.assertTrue(loaded[0, 0])
            self.assertFalse(loaded[0, -1])


class VisualizeTests(unittest.TestCase):
    def test_draw_returns_same_shape(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = SegmentationAvoidance(SegmentationConfig()).evaluate(10, left_half_mask())
        annotated = draw_segmentation(frame, result, SemicircleMapper(), left_half_mask())
        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated.sum()), 0)

    def test_orange_arrow_only_when_direction_is_modified(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mapper = SemicircleMapper()
        mask = left_half_mask()
        avoidance = SegmentationAvoidance(SegmentationConfig())

        def orange_pixels(image):
            diff = np.abs(image.astype(int) - np.array(FALLBACK_COLOR)).sum(axis=2)
            return int((diff < 30).sum())

        consistent = avoidance.evaluate(10, mask)
        self.assertFalse(consistent.fallback_used)
        self.assertEqual(orange_pixels(draw_segmentation(frame, consistent, mapper, mask)), 0)

        fallback = avoidance.evaluate(3, mask)
        self.assertTrue(fallback.fallback_used)
        self.assertGreater(orange_pixels(draw_segmentation(frame, fallback, mapper, mask)), 0)


class ValidateOfflineTests(unittest.TestCase):
    def test_command_clock_derivation(self):
        self.assertEqual(command_clock_from_record({"command_clock": "3"}), 3)
        self.assertEqual(command_clock_from_record({"angle_diff": -140.5}), 7)
        self.assertEqual(command_clock_from_record({"guide": "2点钟方向前进"}), 2)
        self.assertEqual(command_clock_from_record({"guide": "继续直行"}), 12)
        self.assertIsNone(command_clock_from_record({"guide": "未知"}))

    def test_build_pairs_sequential_maps_all_frames(self):
        records = [{"index": index} for index in range(5)]
        frames = [Path(f"frame_{index}.jpg") for index in range(3)]
        pairs = build_pairs(records, frames, None, "sequential")
        self.assertEqual([frame for _record, frame in pairs], frames)

    def test_validate_main_writes_metrics_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            masks = root / "masks"
            output = root / "out"
            frames.mkdir()
            masks.mkdir()
            record = {
                "log_time": "2026-08-21T18:12:26.700+08:00",
                "command_clock": 3,
                "guide": "3点钟方向前进",
            }
            log = root / "output.jsonl"
            log.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            frame = frames / "frame_000001.jpg"
            cv2.imwrite(str(frame), np.zeros((100, 100, 3), dtype=np.uint8))
            save_mask(masks / "frame_000001_walkable.png", left_half_mask())
            code = validate_main([
                "--log", str(log),
                "--frames", str(frames),
                "--masks", str(masks),
                "--output", str(output),
                "--assoc", "sequential",
            ])
            self.assertEqual(code, 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["evaluated"], 1)
            self.assertEqual(summary["fallback"], 1)
            metrics = (output / "seg_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(metrics), 1)
            self.assertEqual(json.loads(metrics[0])["seg_clock"], 11)


if __name__ == "__main__":
    unittest.main()
