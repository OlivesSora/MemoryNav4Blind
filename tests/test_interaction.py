import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from memory_nav.interaction.voice_prompt import Prompt, VoicePlaybackWorker, VoicePromptScheduler
from memory_nav.models import ReferencePoint
from memory_nav.recording.anchor_collector import generate_anchor_candidates, is_image_clear, load_anchors, save_anchors
from memory_nav.replay.anchor_matcher import XFeatAnchorMatcher, verify_correspondences


class InteractionTests(unittest.TestCase):
    def test_prompt_cooldown_and_anchor_once(self):
        scheduler = VoicePromptScheduler(10)
        prompt = Prompt("deviation", "已偏离", 100)
        self.assertIsNotNone(scheduler.request(prompt, 0))
        self.assertIsNone(scheduler.request(prompt, 5))
        self.assertIsNotNone(scheduler.request(prompt, 11))
        self.assertIsNotNone(scheduler.anchor_prompt("a1", "advance", "前方左转", 0))
        self.assertIsNone(scheduler.anchor_prompt("a1", "advance", "前方左转", 20))

    def test_anchor_start_interval_end(self):
        points = [ReferencePoint(i, float(i * 10), float(i * 10), 0, 113, 23, 90) for i in range(6)]
        anchors = generate_anchor_candidates(points, max_spacing_m=20)
        self.assertEqual([a.kind for a in anchors], ["start", "interval", "interval", "end"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.json"
            save_anchors(path, anchors)
            self.assertEqual(load_anchors(path), anchors)

    def test_blur_detection(self):
        flat = np.zeros((50, 50), dtype=np.uint8)
        checker = (np.indices((50, 50)).sum(axis=0) % 2 * 255).astype(np.uint8)
        self.assertFalse(is_image_clear(flat, 10))
        self.assertTrue(is_image_clear(checker, 10))

    def test_visual_geometry_verification(self):
        rng = np.random.default_rng(2)
        reference = rng.uniform(0, 100, (30, 2)).astype(np.float32)
        current = reference + np.array([12, -7], dtype=np.float32)
        result = verify_correspondences(reference, current, minimum_matches=20)
        self.assertTrue(result.matched)
        self.assertGreaterEqual(result.inlier_ratio, 0.9)

    def test_visual_geometry_rejects_too_few(self):
        reference = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        result = verify_correspondences(reference, reference, minimum_matches=4)
        self.assertFalse(result.matched)

    def test_voice_worker_contains_playback_failure(self):
        played = []

        def speaker(text):
            played.append(text)
            if text == "bad":
                raise RuntimeError("audio unavailable")

        worker = VoicePlaybackWorker(speaker)
        self.assertTrue(worker.submit(Prompt("ok", "ok", 1)))
        self.assertTrue(worker.submit(Prompt("bad", "bad", 10)))
        worker.join()
        worker.close()
        self.assertCountEqual(played, ["ok", "bad"])
        self.assertEqual(len(worker.errors), 1)

    def test_xfeat_loads_local_weights_without_hub_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "accelerated_features" / "weights").mkdir(parents=True)
            weights = root / "accelerated_features" / "weights" / "xfeat.pt"
            weights.touch()
            model = Mock()
            model.dev = "cpu"
            with patch("torch.hub.load", return_value=model) as hub_load, patch("torch.load", return_value={"weight": 1}) as torch_load:
                matcher = XFeatAnchorMatcher(root)
                self.assertIs(matcher._load(), model)
            self.assertFalse(hub_load.call_args.kwargs["pretrained"])
            torch_load.assert_called_once_with(weights, map_location="cpu", weights_only=True)
            model.net.load_state_dict.assert_called_once_with({"weight": 1})


if __name__ == "__main__":
    unittest.main()
