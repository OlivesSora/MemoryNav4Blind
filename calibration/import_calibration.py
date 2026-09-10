"""Convert Kalibr camchain-imucam YAML into MemoryNav calibration YAML."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from memory_nav.calibration.validate_calibration import validate_calibration


def convert_kalibr(data: dict, reprojection_error_px: float, maximum_error_px: float = 1.0) -> dict:
    camera = data.get("cam0")
    if not isinstance(camera, dict):
        raise ValueError("Kalibr output does not contain cam0")
    required = ("camera_model", "intrinsics", "distortion_model", "distortion_coeffs", "resolution", "T_cam_imu", "timeshift_cam_imu")
    missing = [key for key in required if key not in camera]
    if missing:
        raise ValueError(f"cam0 is missing: {', '.join(missing)}")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Kalibr camchain-imucam",
        "camera": {
            "model": camera["camera_model"],
            "intrinsics": camera["intrinsics"],
            "distortion_model": camera["distortion_model"],
            "distortion_coefficients": camera["distortion_coeffs"],
            "resolution": camera["resolution"],
        },
        "imu": {"topic": camera.get("rostopic")},
        "T_camera_imu": camera["T_cam_imu"],
        "time_offset_s": camera["timeshift_cam_imu"],
        "quality": {
            "passed": float(reprojection_error_px) <= float(maximum_error_px),
            "reprojection_error_px": float(reprojection_error_px),
            "maximum_error_px": float(maximum_error_px),
        },
    }
    validate_calibration(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kalibr_yaml", type=Path)
    parser.add_argument("output_yaml", type=Path)
    parser.add_argument("--reprojection-error", type=float, required=True)
    parser.add_argument("--maximum-error", type=float, default=1.0)
    args = parser.parse_args(argv)
    source = yaml.safe_load(args.kalibr_yaml.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        parser.error("Kalibr YAML root must be a mapping")
    result = convert_kalibr(source, args.reprojection_error, args.maximum_error)
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return 0 if result["quality"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
