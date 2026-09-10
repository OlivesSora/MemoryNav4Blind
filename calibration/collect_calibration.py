"""Collect timestamped AprilGrid images and synchronized IMU snapshots."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import cv2


class CalibrationCollector:
    def __init__(self, output_dir: str | Path, camera, imu):
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"calibration output is not empty: {self.output_dir}")
        self.dataset_dir = self.output_dir / "dataset"
        self.images_dir = self.dataset_dir / "cam0"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.imu = imu
        self.imu_path = self.output_dir / "imu.jsonl"
        self.imu_csv_path = self.dataset_dir / "imu0.csv"
        self.imu_csv_path.write_text("timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n", encoding="utf-8")
        self.frames_path = self.output_dir / "frames.jsonl"
        self._frame_index = 0
        self._last_imu_time = None

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()

    def capture_imu(self, monotonic_ns: int) -> bool:
        gyro, accel, _mag, device_time = self.imu.get_imu()
        if device_time is None or device_time == self._last_imu_time or gyro is None or accel is None:
            return False
        self._append(self.imu_path, {
            "monotonic_ns": monotonic_ns,
            "device_time_s": device_time,
            "gyro_rad_s": list(gyro),
            "accel_m_s2": list(accel),
        })
        with self.imu_csv_path.open("a", encoding="utf-8") as stream:
            values = [monotonic_ns, *gyro, *accel]
            stream.write(",".join(str(value) for value in values) + "\n")
            stream.flush()
        self._last_imu_time = device_time
        return True

    def capture_frame(self, monotonic_ns: int, width: int | None = None, height: int | None = None) -> bool:
        capture_with_metadata = getattr(self.camera, "capture_frame_with_metadata", None)
        if capture_with_metadata is not None:
            image, metadata = capture_with_metadata(width=width, height=height)
            source_frame = None if metadata is None else metadata.get("frame_count")
            monotonic_ns = monotonic_ns if metadata is None or metadata.get("monotonic_ns") is None else metadata["monotonic_ns"]
        else:
            image, source_frame = self.camera.capture_frame(width=width, height=height) if width is not None else self.camera.capture_frame()
        if image is None:
            return False
        filename = f"{monotonic_ns}.png"
        # Existing camera adapters return RGB.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (self.images_dir / filename).exists():
            raise FileExistsError(f"duplicate camera timestamp: {monotonic_ns}")
        if not cv2.imwrite(str(self.images_dir / filename), bgr):
            raise OSError(f"failed to write calibration frame: {filename}")
        self._append(self.frames_path, {
            "index": self._frame_index,
            "monotonic_ns": monotonic_ns,
            "source_frame": source_frame,
            "filename": f"dataset/cam0/{filename}",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        })
        self._frame_index += 1
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--image-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.image_fps <= 0:
        parser.error("duration and image-fps must be positive")
    if (args.width is None) != (args.height is None):
        parser.error("width and height must be provided together")
    from utils.glasses_camera import Camera
    from utils.imu import IMU

    camera, imu = Camera(), IMU()
    collector = CalibrationCollector(args.output, camera, imu)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    next_frame = started
    try:
        while running and time.monotonic() - started < args.duration:
            now = time.monotonic()
            timestamp = time.monotonic_ns()
            collector.capture_imu(timestamp)
            if now >= next_frame:
                collector.capture_frame(timestamp, args.width, args.height)
                next_frame += 1.0 / args.image_fps
            time.sleep(0.002)
    finally:
        imu.stop()
        camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
