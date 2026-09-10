import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from memory_nav.models import GPSSample
from memory_nav.recording.session_writer import SessionWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CLITests(unittest.TestCase):
    def test_build_and_offline_replay_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            routes = root / "routes"
            with SessionWriter(routes, "integration") as writer:
                for index in range(8):
                    writer.append("gps", GPSSample((index + 1) * 1_000_000_000, 113.0 + index * 0.00001, 23.0, "rtk", "good"))
                writer.append("events", {
                    "type": "anchor_image", "kind": "turn", "monotonic_ns": 4_000_000_000,
                    "longitude": 113.00003, "latitude": 23.0, "clear": True,
                    "image_path": "anchors/candidate.jpg",
                })
                (writer.anchors_dir / "candidate.jpg").write_bytes(b"test-image-placeholder")
            config = {
                "routes_dir": str(routes),
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            build_script = PROJECT_ROOT / "memory_nav" / "scripts" / "build_reference.sh"
            subprocess.run([str(build_script), "--route-id", "integration", "--config", str(config_path)], cwd=root, check=True, capture_output=True, text=True)
            route_dir = routes / "integration"
            self.assertTrue((route_dir / "reference_trajectory.json").is_file())
            manifest = json.loads((route_dir / "manifest.json").read_text())
            self.assertEqual(manifest["state"], "READY")
            self.assertEqual(manifest["pending_anchor_count"], 0)
            self.assertGreaterEqual(manifest["anchor_count"], 2)
            anchors = json.loads((route_dir / "anchors.json").read_text())["anchors"]
            self.assertEqual((anchors[0]["kind"], anchors[-1]["kind"]), ("start", "end"))
            image_anchors = [anchor for anchor in anchors if anchor.get("image_path")]
            self.assertEqual(len(image_anchors), 1)
            self.assertEqual(image_anchors[0]["image_path"], "anchors/candidate.jpg")

            replay_input = root / "replay.jsonl"
            replay_input.write_text("\n".join(json.dumps({"longitude": 113.0 + index * 0.00001, "latitude": 23.0, "heading_deg": 90, "source": "rtk"}) for index in range(8)) + "\n", encoding="utf-8")
            replay_output = root / "result.jsonl"
            replay_svg = root / "result.svg"
            replay_script = PROJECT_ROOT / "memory_nav" / "scripts" / "replay_offline.sh"
            subprocess.run([str(replay_script), "--route-id", "integration", "--input", str(replay_input), "--output", str(replay_output), "--svg", str(replay_svg), "--config", str(config_path)], cwd="/tmp", check=True, capture_output=True, text=True)
            results = [json.loads(line) for line in replay_output.read_text().splitlines()]
            self.assertEqual(len(results), 8)
            self.assertTrue(all(result["match_quality"] == "good" for result in results))
            progress = [result["matched_s_m"] for result in results]
            self.assertEqual(progress, sorted(progress))
            self.assertIn("<svg", replay_svg.read_text(encoding="utf-8"))

    def test_script_missing_required_argument_is_nonzero(self):
        script = PROJECT_ROOT / "memory_nav" / "scripts" / "build_reference.sh"
        result = subprocess.run([str(script)], cwd="/tmp", capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_all_start_scripts_resolve_project_from_other_cwd(self):
        scripts_dir = PROJECT_ROOT / "memory_nav" / "scripts"
        for name in ("start_calibration.sh", "start_recording.sh", "build_reference.sh", "start_replay.sh", "replay_offline.sh", "hardware_acceptance.sh"):
            with self.subTest(script=name):
                result = subprocess.run([str(scripts_dir / name), "--help"], cwd="/tmp", capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
