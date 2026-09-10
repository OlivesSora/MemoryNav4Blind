"""Record timestamped snapshots from the existing GPS and IMU adapters."""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.models import GPSSample, IMUSample
from memory_nav.recording.preflight import check_sensors, require_ready
from memory_nav.recording.session_writer import AsyncSessionWriter, SessionWriter
from memory_nav.recording.anchor_collector import LiveAnchorCapture


LOGGER = logging.getLogger(__name__)


class SensorRecorder:
    """Best-effort recorder over current polling APIs.

    This records every distinct IMU device timestamp visible to the client.
    A future callback/buffer in ``utils.imu`` is still required to guarantee
    that no high-rate serial frames are skipped.
    """

    def __init__(self, writer: SessionWriter, gps, imu, gps_interval_s: float = 1.0, anchor_capture: LiveAnchorCapture | None = None, vio=None):
        if gps_interval_s <= 0:
            raise ValueError("gps_interval_s must be positive")
        self.writer = writer
        self.gps = gps
        self.imu = imu
        self.gps_interval_s = gps_interval_s
        self._last_gps_s = float("-inf")
        self._last_ahrs_device_time = None
        self._last_raw_device_time = None
        self.anchor_capture = anchor_capture
        self.vio = vio

    def capture_imu(self, monotonic_ns: int) -> bool:
        drain = getattr(self.imu, "drain_samples", None)
        if drain is not None:
            buffered = drain()
            for record in buffered:
                self.writer.append("imu", IMUSample.from_dict(record))
            return bool(buffered)
        angular_velocity, _angle, quaternion, ahrs_time = self.imu.get_ahrs()
        _gyro, accel, mag, raw_time = self.imu.get_imu()
        if ahrs_time is None and raw_time is None:
            return False
        if ahrs_time == self._last_ahrs_device_time and raw_time == self._last_raw_device_time:
            return False
        heading = self.imu.get_heading(require_ready=False)
        sample = IMUSample(
            monotonic_ns=monotonic_ns,
            device_time_s=ahrs_time if ahrs_time is not None else raw_time,
            heading_deg=heading,
            angular_velocity_rad_s=None if angular_velocity is None else tuple(angular_velocity),
            quaternion_wxyz=None if quaternion is None else tuple(quaternion),
            accel_m_s2=None if accel is None else tuple(accel),
            mag=None if mag is None else tuple(mag),
            heading_ready=heading is not None,
        )
        self.writer.append("imu", sample)
        self._last_ahrs_device_time = ahrs_time
        self._last_raw_device_time = raw_time
        return True

    def capture_gps(self, monotonic_ns: int) -> bool:
        now_s = monotonic_ns / 1e9
        if now_s - self._last_gps_s < self.gps_interval_s:
            return False
        self._last_gps_s = now_s
        get_sample = getattr(self.gps, "get_sample", None)
        rich_sample = get_sample() if get_sample is not None else None
        if rich_sample is not None:
            # Use this step's monotonic timestamp so GPS and IMU share the
            # recorder clock boundary; preserve server time separately.
            rich_sample["monotonic_ns"] = monotonic_ns
            sample = GPSSample.from_dict(rich_sample)
            self.writer.append("gps", sample)
            if self.anchor_capture:
                self.anchor_capture.consider(sample, self.imu.get_heading(require_ready=False))
            return True
        coordinate, state = self.gps.get_gps()
        if coordinate is None or len(coordinate) != 2:
            self.writer.append("events", {"monotonic_ns": monotonic_ns, "type": "gps_unavailable", "state": state})
            return False
        source = "rtk" if state == "good" else "phone"
        sample = GPSSample(monotonic_ns, float(coordinate[0]), float(coordinate[1]), source, state or "poor")
        self.writer.append("gps", sample)
        if self.anchor_capture:
            self.anchor_capture.consider(sample, self.imu.get_heading(require_ready=False))
        return True

    def step(self, monotonic_ns: int | None = None) -> tuple[bool, bool]:
        timestamp = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        imu_written, gps_written = self.capture_imu(timestamp), self.capture_gps(timestamp)
        if self.vio is not None:
            for sample in self.vio.drain():
                self.writer.append("vio", sample)
        return imu_written, gps_written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--poll-hz", type=float, default=100.0)
    parser.add_argument("--gps-interval", type=float, default=1.0)
    parser.add_argument("--camera", action="store_true", help="capture anchor image candidates during recording")
    parser.add_argument("--vio-topic", help="subscribe to this ROS 2 nav_msgs/Odometry topic")
    parser.add_argument("--ros-image-topic", help="subscribe to an existing ROS 2 sensor_msgs/Image stream for anchor capture")
    parser.add_argument("--imu-port", default="/dev/imu")
    args = parser.parse_args(argv)
    if args.poll_hz <= 0:
        parser.error("--poll-hz must be positive")
    config = load_config(args.config)
    from utils.gps import GPS

    if args.vio_topic:
        # VINS bridge already owns /dev/imu and publishes /imu0; do not open
        # the serial port a second time (double-read starves both readers).
        from memory_nav.vio import ImuSubscriber
        imu = ImuSubscriber("/imu0")
    else:
        from utils.imu import IMU
        imu = IMU(port=args.imu_port)
    gps = GPS()
    camera = None
    vio = None
    if args.camera or args.ros_image_topic:
        if args.ros_image_topic and args.vio_topic:
            from memory_nav.vio import VioSubscriber
            vio = VioSubscriber(args.vio_topic, image_topic=args.ros_image_topic)
            vio.start()
            camera = vio
        elif args.ros_image_topic:
            from memory_nav.vio import VioSubscriber
            vio = VioSubscriber("/odometry", image_topic=args.ros_image_topic)
            vio.start()
            camera = vio
        else:
            from utils.glasses_camera import Camera
            camera = Camera()
    if args.vio_topic and vio is None:
        from memory_nav.vio import VioSubscriber
        vio = VioSubscriber(args.vio_topic)
        vio.start()
    if not imu.wait_for_heading(15.0):
        LOGGER.warning("IMU heading not ready after 15s; continuing (preflight may report imu not ready)")
    checks = check_sensors(gps, imu, camera)
    for check in checks:
        LOGGER.info("preflight %s: ready=%s %s", check.name, check.ready, check.detail)
    # IMU + GPS are mandatory; the camera/vio source (ROS /cam0/image_raw from
    # the VINS bridge) is best-effort and may not be publishing yet when we
    # start. Record anyway; anchors/vio fill in as those topics appear.
    require_ready([check for check in checks if check.name != "camera"])
    for check in checks:
        if check.name == "camera" and not check.ready:
            LOGGER.warning(
                "camera not ready yet (%s); recording continues and captures once /cam0/image_raw publishes",
                check.detail,
            )
    clear_samples = getattr(imu, "clear_samples", None)
    if clear_samples is not None:
        discarded = clear_samples()
        LOGGER.info("discarded %d pre-session IMU samples", discarded)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        metadata = {"recorder": "buffered-v1"}
        if args.vio_topic:
            metadata["vio_topic"] = args.vio_topic
        with AsyncSessionWriter(config["routes_dir"], args.route_id, metadata) as writer:
            anchor_capture = None
            if camera is not None:
                anchor_capture = LiveAnchorCapture(
                    writer, camera,
                    max_spacing_m=float(config["anchors"]["max_spacing_m"]),
                    turn_angle_deg=float(config["anchors"]["turn_angle_deg"]),
                    blur_threshold=float(config["anchors"]["blur_threshold"]),
                )
            recorder = SensorRecorder(writer, gps, imu, args.gps_interval, anchor_capture, vio)
            interval = 1.0 / args.poll_hz
            while running:
                recorder.step()
                time.sleep(interval)
            if anchor_capture is not None:
                anchor_capture.finalize(imu.get_heading(require_ready=False))
    finally:
        imu.stop()
        if vio is not None:
            vio.close()
        if camera is not None:
            camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
