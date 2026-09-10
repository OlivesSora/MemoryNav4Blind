"""Run Kalibr camera-IMU calibration with explicit, inspectable arguments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_command(bag: Path, camera_yaml: Path, imu_yaml: Path, target_yaml: Path, output_dir: Path, require_bag: bool = True) -> list[str]:
    paths = (bag, camera_yaml, imu_yaml, target_yaml) if require_bag else (camera_yaml, imu_yaml, target_yaml)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return [
        "kalibr_calibrate_imu_camera",
        "--bag", str(bag.resolve()),
        "--cam", str(camera_yaml.resolve()),
        "--imu", str(imu_yaml.resolve()),
        "--target", str(target_yaml.resolve()),
        "--dont-show-report",
        "--bag-freq", "20.0",
    ]


def run_kalibr(command: list[str], output_dir: Path) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError("kalibr_calibrate_imu_camera is not installed or not on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=output_dir, check=True)


def build_bag_command(dataset_dir: Path, output_bag: Path) -> list[str]:
    if not (dataset_dir / "cam0").is_dir() or not (dataset_dir / "imu0.csv").is_file():
        raise FileNotFoundError("dataset must contain cam0/ and imu0.csv")
    return ["kalibr_bagcreater", "--folder", str(dataset_dir.resolve()), "--output-bag", str(output_bag.resolve())]


def create_bag(command: list[str]) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError("kalibr_bagcreater is not installed or not on PATH")
    Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="ROS bag containing camera and IMU topics")
    parser.add_argument("--camera-yaml", type=Path, required=True)
    parser.add_argument("--imu-yaml", type=Path, required=True)
    parser.add_argument("--target-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dataset", type=Path, help="optional collected dataset to convert before calibration")
    args = parser.parse_args(argv)
    if args.dataset:
        bag_command = build_bag_command(args.dataset, args.bag)
        if args.print_only:
            print(" ".join(bag_command))
        else:
            create_bag(bag_command)
    command = build_command(args.bag, args.camera_yaml, args.imu_yaml, args.target_yaml, args.output, require_bag=not (args.dataset and args.print_only))
    if args.print_only:
        print(" ".join(command))
    else:
        run_kalibr(command, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
