"""Collect timestamped AprilGrid images and synchronized IMU snapshots."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import cv2


DEFAULT_LIVE_PREVIEW_PATH = (
    Path(__file__).resolve().parents[2] / "calibration_data" / "tmp.jpg"
)


class CalibrationCollector:
    def __init__(
        self,
        output_dir: str | Path,
        camera,
        imu,
        require_device_timestamps=False,
        live_preview_path: str | Path | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"calibration output is not empty: {self.output_dir}")
        self.dataset_dir = self.output_dir / "dataset"
        self.images_dir = self.dataset_dir / "cam0"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.imu = imu
        self.require_device_timestamps = require_device_timestamps
        self.live_preview_path = (
            None if live_preview_path is None else Path(live_preview_path).resolve()
        )
        if self.live_preview_path is not None:
            self.live_preview_path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_path = self.output_dir / "clock_sync.jsonl"
        self.diagnostics_path = self.output_dir / "diagnostics.jsonl"
        self.imu_path = self.output_dir / "imu.jsonl"
        self.imu_csv_path = self.dataset_dir / "imu0.csv"
        self.imu_csv_path.write_text("timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n", encoding="utf-8")
        self.frames_path = self.output_dir / "frames.jsonl"
        self._frame_index = 0
        self._last_imu_time = None
        self._last_frame_time = None
        self._last_source_frame = None

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()

    def _write_live_preview(self, bgr) -> None:
        """Atomically replace the JPEG read by live calibration monitors."""
        if self.live_preview_path is None:
            return
        temporary = self.live_preview_path.with_name(
            f".{self.live_preview_path.name}.writing.jpg"
        )
        try:
            if not cv2.imwrite(
                str(temporary), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
            ):
                raise OSError(f"failed to write live preview: {temporary}")
            temporary.replace(self.live_preview_path)
        finally:
            if temporary.exists():
                temporary.unlink()

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

    def capture_pending_imu(self) -> int:
        """Write all raw IMU frames when the adapter exposes a frame buffer."""
        drain = getattr(self.imu, "drain_imu_samples", None)
        if drain is None:
            if self.require_device_timestamps:
                raise ValueError("aligned collection requires IMU.drain_imu_samples()")
            return int(self.capture_imu(time.monotonic_ns()))

        records = []
        csv_rows = []
        for sample in drain():
            device_time = sample.get("device_time_s")
            gyro = sample.get("gyro_rad_s")
            accel = sample.get("accel_m_s2")
            monotonic_ns = sample.get("monotonic_ns")
            if self.require_device_timestamps:
                if type(sample.get("device_timestamp_us")) is not int:
                    raise ValueError("missing integer IMU device_timestamp_us")
                if device_time is None:
                    raise ValueError("missing IMU device_time_s")
                if self._last_imu_time is not None and device_time <= self._last_imu_time:
                    raise ValueError("IMU clock repeated or reset; start a new session")
            if (
                device_time is None
                or device_time == self._last_imu_time
                or gyro is None
                or accel is None
                or monotonic_ns is None
            ):
                continue
            records.append({
                **sample,
                "monotonic_ns": monotonic_ns,
                "device_time_s": device_time,
                "gyro_rad_s": list(gyro),
                "accel_m_s2": list(accel),
            })
            csv_rows.append([monotonic_ns, *gyro, *accel])
            self._last_imu_time = device_time

        if not records:
            return 0
        with self.imu_path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        with self.imu_csv_path.open("a", encoding="utf-8") as stream:
            for values in csv_rows:
                stream.write(",".join(str(value) for value in values) + "\n")
        return len(records)

    def capture_frame(self, monotonic_ns: int, width: int | None = None, height: int | None = None) -> bool:
        metadata = {}
        capture_with_metadata = getattr(self.camera, "capture_frame_with_metadata", None)
        if capture_with_metadata is not None:
            image, metadata = capture_with_metadata(width=width, height=height)
            source_frame = None if metadata is None else metadata.get("frame_count")
            monotonic_ns = monotonic_ns if metadata is None or metadata.get("monotonic_ns") is None else metadata["monotonic_ns"]
        else:
            image, source_frame = self.camera.capture_frame(width=width, height=height) if width is not None else self.camera.capture_frame()
        if image is None:
            return False
        metadata = metadata or {}
        if monotonic_ns == self._last_frame_time:
            return False
        if self.require_device_timestamps:
            required = ("camera_timestamp_ns", "camera_clock_id", "source_frame_id", "timestamp_semantics")
            if any(metadata.get(key) is None for key in required):
                raise ValueError("camera sender lacks exposure timestamp metadata; see Basalt/TIME_ALIGNMENT_ZH.md")
            source_id = (metadata["camera_clock_id"], metadata["source_frame_id"])
            if source_id == self._last_source_frame:
                return False
            if self._last_source_frame is not None:
                if source_id[0] != self._last_source_frame[0] or source_id[1] < self._last_source_frame[1]:
                    raise ValueError("camera clock/frame counter reset; start a new session")
            self._last_source_frame = source_id
        filename = f"{monotonic_ns}.png"
        # Existing camera adapters return RGB.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (self.images_dir / filename).exists():
            raise FileExistsError(f"duplicate camera timestamp: {monotonic_ns}")
        if not cv2.imwrite(str(self.images_dir / filename), bgr):
            raise OSError(f"failed to write calibration frame: {filename}")
        self._write_live_preview(bgr)
        self._append(self.frames_path, {
            **metadata,
            "index": self._frame_index,
            "monotonic_ns": monotonic_ns,
            "source_frame": source_frame,
            "filename": f"dataset/cam0/{filename}",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        })
        self._frame_index += 1
        self._last_frame_time = monotonic_ns
        return True

    def capture_sync(self):
        drain = getattr(self.camera, "drain_clock_sync_samples", None)
        if drain is not None:
            for sample in drain():
                self._append(self.sync_path, sample)

    def capture_diagnostics(self):
        record = {"monotonic_ns": time.monotonic_ns()}
        for name, device in (("camera", self.camera), ("imu", self.imu)):
            getter = getattr(device, "get_diagnostics", None)
            if getter is not None:
                record[name] = getter()
        self._append(self.diagnostics_path, record)


def legacy_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--image-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--require-device-timestamps", action="store_true",
                        help="Fail if the sender or IMU lacks device sampling timestamps")
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.image_fps <= 0:
        parser.error("duration and image-fps must be positive")
    if (args.width is None) != (args.height is None):
        parser.error("width and height must be provided together")
    if args.require_device_timestamps and args.width is not None:
        parser.error("configure a fixed continuous stream on the sender; do not request one-shot frames")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty")
    from utils.glasses_camera import Camera
    from utils.imu import IMU

    camera, imu = Camera(), IMU()
    if hasattr(imu, "clear_imu_samples"):
        imu.clear_imu_samples()
    collector = CalibrationCollector(
        args.output,
        camera,
        imu,
        args.require_device_timestamps,
        live_preview_path=DEFAULT_LIVE_PREVIEW_PATH,
    )
    collector.capture_diagnostics()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    next_frame = started
    next_diagnostics = started + 1.0
    try:
        while running and time.monotonic() - started < args.duration:
            now = time.monotonic()
            timestamp = time.monotonic_ns()
            collector.capture_pending_imu()
            collector.capture_sync()
            if now >= next_diagnostics:
                collector.capture_diagnostics()
                next_diagnostics = now + 1.0
            if now >= next_frame:
                collector.capture_frame(timestamp, args.width, args.height)
                next_frame = now + 1.0 / args.image_fps
            time.sleep(0.002)
    finally:
        try:
            imu.stop()
            collector.capture_pending_imu()
            collector.capture_sync()
            collector.capture_diagnostics()
        finally:
            camera.release()
    if args.require_device_timestamps and collector._frame_index == 0:
        raise ValueError("no camera frames received; no usable calibration session was recorded")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--imu-source", choices=["android", "n100"], default="android")
    args, rest = parser.parse_known_args(argv)
    if args.imu_source == "n100":
        return legacy_main(rest)
    from .android_capture import main as android_main
    return android_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
