"""Bounded real-device acceptance runs with machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


REPORT_VERSION = 1


def contains_ordered(values: Iterable[str], expected: Iterable[str]) -> bool:
    """Return whether every expected value occurs in order (duplicates allowed)."""
    wanted = iter(expected)
    current = next(wanted, None)
    if current is None:
        return True
    for value in values:
        if value == current:
            current = next(wanted, None)
            if current is None:
                return True
    return False


def evaluate_sequence(values: list[str], expected: list[str]) -> dict[str, Any]:
    passed = contains_ordered(values, expected)
    return {
        "passed": passed,
        "observed": values,
        "expected_order": expected,
        "failure_reasons": [] if passed else ["未观察到要求的完整状态顺序"],
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def merge_report(path: Path, scenario: str, result: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"version": REPORT_VERSION, "scenarios": {}}
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("version") != REPORT_VERSION:
            raise ValueError("不支持的验收报告版本")
    report.setdefault("scenarios", {})[scenario] = result
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    report["all_recorded_scenarios_passed"] = bool(report["scenarios"]) and all(
        item.get("passed") is True for item in report["scenarios"].values()
    )
    _atomic_json(path, report)
    return report


def run_smoke(
    duration: float,
    interval: float,
    gps_factory: Callable[[], Any],
    imu_factory: Callable[[], Any],
    camera_factory: Callable[[], Any],
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    gps, imu, camera = gps_factory(), None, None
    gps_samples: list[dict[str, Any]] = []
    headings: list[float] = []
    frames: list[int] = []
    failures: list[str] = []
    try:
        imu = imu_factory()
        camera = camera_factory()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            sample = gps.get_sample()
            if sample:
                gps_samples.append(sample)
            heading = imu.get_heading()
            if heading is not None:
                headings.append(float(heading))
            image, metadata = camera.capture_frame_with_metadata(timeout=max(interval, 0.1))
            if image is not None and metadata and metadata.get("frame_count") is not None:
                frames.append(int(metadata["frame_count"]))
            time.sleep(interval)
    finally:
        if camera is not None:
            camera.release()
        if imu is not None:
            imu.stop()
    if not gps_samples:
        failures.append("GPS 无有效样本")
    elif not any(item.get("state") == "good" for item in gps_samples):
        failures.append("GPS 未出现 good 状态")
    if not headings:
        failures.append("IMU 无有效航向")
    if len(set(frames)) < 2:
        failures.append("相机未观察到至少两个不同帧")
    return {
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failure_reasons": failures,
        "observations": {
            "gps_sample_count": len(gps_samples),
            "gps_sources": sorted({str(item.get("source", "unknown")) for item in gps_samples}),
            "gps_states": sorted({str(item.get("state", "unknown")) for item in gps_samples}),
            "imu_heading_count": len(headings),
            "imu_heading_min_deg": min(headings) if headings else None,
            "imu_heading_max_deg": max(headings) if headings else None,
            "camera_distinct_frames": len(set(frames)),
        },
        "environment": {"hostname": platform.node(), "python": platform.python_version()},
    }


def _load_values(path: Path, field: str) -> list[str]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if field not in record:
            raise ValueError(f"第 {line_number} 行缺少字段 {field!r}")
        values.append(str(record[field]))
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MemoryNav 实机验收与 JSON 证据报告")
    parser.add_argument("--report", type=Path, required=True, help="累计验收报告路径")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="同时检查 GPS、IMU 和相机")
    smoke.add_argument("--duration", type=float, default=10.0)
    smoke.add_argument("--interval", type=float, default=0.2)
    sequence = subparsers.add_parser("sequence", help="校验 JSONL 中的状态转换顺序")
    sequence.add_argument("--scenario", required=True)
    sequence.add_argument("--input", type=Path, required=True)
    sequence.add_argument("--field", required=True)
    sequence.add_argument("--expect", nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        if args.duration <= 0 or args.interval <= 0:
            parser.error("duration 和 interval 必须大于 0")
        from utils.gps import GPS
        from utils.glasses_camera import Camera
        from utils.imu import IMU

        result = run_smoke(args.duration, args.interval, GPS, IMU, Camera)
        scenario = "device_smoke"
    else:
        values = _load_values(args.input, args.field)
        result = evaluate_sequence(values, args.expect)
        result.update({"input": str(args.input), "field": args.field})
        scenario = args.scenario
    merge_report(args.report, scenario, result)
    print(json.dumps({"scenario": scenario, **result}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
