import unittest
import json
import tempfile
from pathlib import Path

from memory_nav.recording.anchor_collector import AnchorCandidate
from memory_nav.tools.confirm_anchor import refresh_route_state, update_anchor


class ConfirmAnchorTests(unittest.TestCase):
    def setUp(self):
        self.anchors = [AnchorCandidate("a1", 1, 2, "turn", image_path="anchors/a1.jpg")]

    def test_confirm_requires_and_stores_prompt(self):
        confirmed = update_anchor(self.anchors, "a1", "  前方左转  ")
        self.assertTrue(confirmed[0].confirmed)
        self.assertEqual(confirmed[0].prompt_text, "前方左转")
        with self.assertRaises(ValueError):
            update_anchor(self.anchors, "a1", " ")

    def test_reject_and_unknown(self):
        self.assertEqual(update_anchor(self.anchors, "a1", reject=True), [])
        with self.assertRaises(ValueError):
            update_anchor(self.anchors, "missing", "提示")

    def test_review_gate_advances_manifest_only_when_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory)
            (route / "manifest.json").write_text(json.dumps({"state": "REVIEW_REQUIRED"}), encoding="utf-8")
            (route / "quality_report.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
            self.assertEqual(refresh_route_state(route, self.anchors), "REVIEW_REQUIRED")
            confirmed = update_anchor(self.anchors, "a1", "前方左转")
            self.assertEqual(refresh_route_state(route, confirmed), "READY")
            manifest = json.loads((route / "manifest.json").read_text())
            self.assertEqual(manifest["pending_anchor_count"], 0)


if __name__ == "__main__":
    unittest.main()
