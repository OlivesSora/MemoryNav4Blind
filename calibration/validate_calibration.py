"""Validate the versioned Camera-IMU calibration YAML contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


class CalibrationError(ValueError):
    pass


def validate_calibration(data: dict) -> None:
    required = ("schema_version", "camera", "imu", "T_camera_imu", "time_offset_s", "quality")
    missing = [name for name in required if name not in data]
    if missing:
        raise CalibrationError(f"missing calibration keys: {', '.join(missing)}")
    matrix = np.asarray(data["T_camera_imu"], dtype=float)
    if matrix.shape != (4, 4):
        raise CalibrationError("T_camera_imu must be a 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise CalibrationError("invalid homogeneous transform bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise CalibrationError("T_camera_imu rotation is not orthonormal")
    if not isinstance(data["time_offset_s"], (int, float)):
        raise CalibrationError("time_offset_s must be numeric")
    quality = data["quality"]
    if not isinstance(quality, dict) or "passed" not in quality:
        raise CalibrationError("quality.passed is required")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration")
    args = parser.parse_args(argv)
    data = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        parser.error("calibration root must be a mapping")
    validate_calibration(data)
    print("calibration schema is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
