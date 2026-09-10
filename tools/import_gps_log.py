"""Import an L4 outdoor_nav ``output.jsonl`` log as a MemoryNav route.

The L4 log stores one navigation record per line with ``log_time`` (wall clock),
``current_pos`` (``[longitude, latitude]`` in GCJ-02) and ``raw_imu_bearing``.
This tool trims the leading lines, writes a ``source/output_trimmed.jsonl`` copy
for provenance, and derives ``raw/gps.jsonl`` + ``raw/imu.jsonl`` so the existing
``memory_nav.trajectory.build_reference`` pipeline can build a reference route.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.models import GPSSample, IMUSample
from memory_nav.recording.session_writer import SessionWriter, validate_route_id


LOCAL_TZ = timezone(timedelta(hours=8))


def parse_epoch_ns(value: str) -> int:
    """Parse an ISO-8601 timestamp into epoch nanoseconds."""
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=LOCAL_TZ)
    return int(timestamp.timestamp() * 1e9)


def read_trimmed_lines(path: str | Path, skip: int) -> list[str]:
    if skip < 0:
        raise ValueError("skip must be non-negative")
    with Path(path).open("r", encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()]
    return lines[skip:]


def convert_records(lines: list[str]) -> tuple[list[GPSSample], list[IMUSample], int]:
    """Turn trimmed log lines into GPS/IMU samples sharing one time base."""
    records = [json.loads(line) for line in lines]
    if not records:
        raise ValueError("no records remain after trimming")
    base_ns = parse_epoch_ns(str(records[0]["log_time"]))
    gps_samples: list[GPSSample] = []
    imu_samples: list[IMUSample] = []
    skipped = 0
    for record in records:
        position = record.get("current_pos")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            skipped += 1
            continue
        log_time = record.get("log_time")
        if not log_time:
            skipped += 1
            continue
        monotonic_ns = parse_epoch_ns(str(log_time)) - base_ns
        if monotonic_ns < 0:
            skipped += 1
            continue
        longitude, latitude = float(position[0]), float(position[1])
        gps_samples.append(GPSSample(
            monotonic_ns=monotonic_ns,
            longitude=longitude,
            latitude=latitude,
            source="phone",
            state="good",
            coordinate_system="GCJ-02",
            server_time_ns=base_ns + monotonic_ns,
        ))
        heading = record.get("raw_imu_bearing")
        if heading is not None:
            imu_samples.append(IMUSample(
                monotonic_ns=monotonic_ns,
                device_time_s=None,
                heading_deg=float(heading),
                heading_ready=True,
            ))
    if len(gps_samples) < 2:
        raise ValueError("fewer than two usable GPS records after trimming")
    return gps_samples, imu_samples, skipped


def import_route(input_path: str | Path, route_id: str, config: dict, skip: int) -> Path:
    route_id = validate_route_id(route_id)
    lines = read_trimmed_lines(input_path, skip)
    gps_samples, imu_samples, skipped = convert_records(lines)
    metadata = {
        "recorder": "import-gps-log-v1",
        "source_log": str(Path(input_path).resolve()),
        "trimmed_lines": skip,
    }
    writer = SessionWriter(config["routes_dir"], route_id, metadata)
    try:
        source_dir = writer.route_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "output_trimmed.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for sample in gps_samples:
            writer.append("gps", sample)
        for sample in imu_samples:
            writer.append("imu", sample)
    finally:
        writer.close()
    print(json.dumps({
        "route_id": route_id,
        "route_dir": str(writer.route_dir),
        "gps_samples": len(gps_samples),
        "imu_samples": len(imu_samples),
        "skipped_records": skipped,
    }, ensure_ascii=False))
    return writer.route_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="L4 output.jsonl to import")
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--skip", type=int, default=0, help="leading lines to drop before import")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    import_route(args.input, args.route_id, config, args.skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
