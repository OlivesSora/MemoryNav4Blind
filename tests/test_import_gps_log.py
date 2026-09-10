import json
import tempfile
import unittest
from pathlib import Path

from memory_nav.recording.session_writer import read_jsonl
from memory_nav.tools.import_gps_log import convert_records, import_route, parse_epoch_ns, read_trimmed_lines


def record(log_time, lon, lat, heading):
    return json.dumps({
        "log_time": log_time,
        "current_pos": [lon, lat],
        "raw_imu_bearing": heading,
    })


class ImportGpsLogTests(unittest.TestCase):
    def _write_log(self, root: Path) -> Path:
        path = root / "output.jsonl"
        path.write_text("\n".join([
            record("2026-08-16T18:01:12.396+08:00", 113.407646, 23.046712, 267.44),
            record("2026-08-16T18:01:15.000+08:00", 113.407650, 23.046715, 267.50),
            record("2026-08-16T18:01:18.000+08:00", 113.407700, 23.046720, 267.60),
            record("2026-08-16T18:01:21.000+08:00", 113.407800, 23.046730, 267.70),
        ]) + "\n", encoding="utf-8")
        return path

    def test_parse_epoch_ns_handles_offsets(self):
        self.assertEqual(parse_epoch_ns("1970-01-01T00:00:01+00:00"), 1_000_000_000)
        self.assertEqual(parse_epoch_ns("1970-01-01T00:00:00+00:00"), 0)

    def test_convert_records_builds_gps_and_imu_on_shared_base(self):
        lines = read_trimmed_lines(self._write_log(Path(tempfile.mkdtemp())), skip=1)
        gps, imu, skipped = convert_records(lines)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(gps), 3)
        self.assertEqual(len(imu), 3)
        self.assertEqual(gps[0].monotonic_ns, 0)
        self.assertEqual(gps[0].longitude, 113.407650)
        self.assertEqual(gps[0].source, "phone")
        self.assertEqual(gps[0].state, "good")
        self.assertTrue(imu[0].heading_ready)
        self.assertAlmostEqual(imu[0].heading_deg, 267.50)

    def test_import_route_writes_raw_streams_and_trimmed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = self._write_log(root)
            route_dir = import_route(input_path, "imported", {"routes_dir": str(root / "routes")}, skip=1)
            gps = read_jsonl(route_dir / "raw" / "gps.jsonl")
            imu = read_jsonl(route_dir / "raw" / "imu.jsonl")
            self.assertEqual(len(gps), 3)
            self.assertEqual(len(imu), 3)
            trimmed = (route_dir / "source" / "output_trimmed.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(trimmed), 3)
            manifest = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sample_counts"]["gps"], 3)
            self.assertEqual(manifest["trimmed_lines"], 1)


if __name__ == "__main__":
    unittest.main()
