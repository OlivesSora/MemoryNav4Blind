import json
import tempfile
import unittest
from pathlib import Path

from memory_nav.models import GPSSample
from memory_nav.recording.session_writer import AsyncSessionWriter, SessionError, SessionWriter, read_jsonl, validate_route_id


class SessionTests(unittest.TestCase):
    def test_session_writes_manifest_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            with SessionWriter(directory, "route-01") as writer:
                writer.append("gps", GPSSample(1, 113.0, 23.0, "rtk", "good"))
            route = Path(directory) / "route-01"
            manifest = json.loads((route / "manifest.json").read_text())
            self.assertEqual(manifest["state"], "FINALIZING")
            self.assertEqual(manifest["sample_counts"]["gps"], 1)
            self.assertEqual(len(read_jsonl(route / "raw" / "gps.jsonl")), 1)

    def test_duplicate_route_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SessionWriter(directory, "same")
            writer.close()
            with self.assertRaises(SessionError):
                SessionWriter(directory, "same")

    def test_route_id_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            validate_route_id("../outside")

    def test_async_writer_drains_before_close(self):
        with tempfile.TemporaryDirectory() as directory:
            with AsyncSessionWriter(directory, "async") as writer:
                for index in range(100):
                    writer.append("events", {"index": index})
            records = read_jsonl(writer.raw_dir / "events.jsonl")
            self.assertEqual([record["index"] for record in records], list(range(100)))


if __name__ == "__main__":
    unittest.main()
