"""ROS 2 adapter and small, dependency-light VIO helpers."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
import numpy as np

from memory_nav.models import VIOSample


class VioSubscriber:
    """Subscribe to VINS odometry in a background ROS executor.

    Importing rclpy is deliberately delayed so offline tools do not require a
    ROS installation.
    """

    def __init__(self, topic: str, queue_size: int = 4096, image_topic: str | None = None):
        self.topic = topic
        self.image_topic = image_topic
        self._samples: queue.Queue[VIOSample] = queue.Queue(queue_size)
        self._latest: VIOSample | None = None
        self._latest_image = None
        self._latest_image_ns: int | None = None
        self._dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-nav-vio", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        rclpy.init(args=None)
        node = rclpy.create_node("memory_nav_vio")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                         reliability=ReliabilityPolicy.BEST_EFFORT)

        def callback(message: Odometry) -> None:
            stamp = message.header.stamp
            pose = message.pose.pose
            sample = VIOSample(
                monotonic_ns=time.monotonic_ns(),
                ros_time_ns=int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
                x_m=float(pose.position.x), y_m=float(pose.position.y), z_m=float(pose.position.z),
                quaternion_xyzw=(float(pose.orientation.x), float(pose.orientation.y),
                                 float(pose.orientation.z), float(pose.orientation.w)),
                frame_id=message.header.frame_id or "world",
            )
            self._latest = sample
            try:
                self._samples.put_nowait(sample)
            except queue.Full:
                self._dropped += 1

        node.create_subscription(Odometry, self.topic, callback, qos)
        if self.image_topic:
            def image_callback(message: Image) -> None:
                try:
                    height, width = int(message.height), int(message.width)
                    channels = 1 if message.encoding.lower() in ("mono8", "8uc1") else 3
                    raw = np.frombuffer(message.data, dtype=np.uint8)
                    row = int(message.step) if message.step else width * channels
                    image = raw.reshape(height, row)[:, :width * channels].reshape(height, width, channels).copy()
                    if channels == 1:
                        import cv2
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif message.encoding.lower() in ("bgr8", "8uc3"):
                        import cv2
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self._latest_image = image
                    self._latest_image_ns = time.monotonic_ns()
                except Exception:
                    return
            node.create_subscription(Image, self.image_topic, image_callback, qos)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            while not self._stop.is_set():
                executor.spin_once(timeout_sec=0.1)
        finally:
            executor.remove_node(node)
            node.destroy_node()
            rclpy.shutdown()

    def drain(self) -> list[VIOSample]:
        result: list[VIOSample] = []
        while True:
            try:
                result.append(self._samples.get_nowait())
            except queue.Empty:
                return result

    def latest(self) -> VIOSample | None:
        return self._latest

    def capture_frame_with_metadata(self, timeout: float = 1.0):
        deadline = time.monotonic() + timeout
        while self._latest_image is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._latest_image is None:
            return None, None
        image = self._latest_image.copy()
        return image, {"monotonic_ns": self._latest_image_ns, "width": int(image.shape[1]), "height": int(image.shape[0])}

    def capture_frame(self, timeout: float = 1.0):
        value, metadata = self.capture_frame_with_metadata(timeout)
        return value, None if metadata is None else metadata.get("monotonic_ns")

    def diagnostics(self) -> dict[str, int | None]:
        latest = self._latest
        return {"dropped": self._dropped, "age_ms": None if latest is None else int((time.monotonic_ns() - latest.monotonic_ns) / 1e6)}

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    release = close


def _quaternion_yaw_deg(qw, qx, qy, qz) -> float:
    """Yaw (degrees, [0, 360)) from a (w,x,y,z) quaternion.

    Fallback heading when the bridge has not embedded the true N100 AHRS
    heading into orientation_covariance[0].  Convention may differ by a fixed
    offset from the magnetometer heading; the offline pipeline auto-calibrates
    a heading offset, so an absolute offset here is tolerated.
    """
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return yaw % 360.0


class ImuSubscriber:
    """Subscribe to bridge /imu0 and expose a utils.imu.IMU-like interface.

    The bridge encodes the true N100 AHRS heading (degrees, [0, 360)) into
    ``orientation_covariance[0]``; a negative value means unknown, in which
    case heading falls back to the quaternion yaw.  Reading /imu0 instead of
    opening /dev/imu lets recorder/replay run alongside the VINS bridge that
    already owns the serial port.
    """

    def __init__(self, topic: str = "/imu0", queue_size: int = 4096):
        self.topic = topic
        self._lock = threading.Lock()
        self._queue: deque = deque(maxlen=queue_size)
        self._heading: float | None = None
        self._last_ns: int | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-nav-imu", daemon=True)
        self._thread.start()

    def _yaw_deg(self, qw, qx, qy, qz) -> float:
        return _quaternion_yaw_deg(qw, qx, qy, qz)

    def _run(self) -> None:
        import rclpy
        from sensor_msgs.msg import Imu
        from rclpy.executors import SingleThreadedExecutor

        context = rclpy.Context()
        rclpy.init(context=context)
        node = rclpy.create_node("memory_nav_imu", context=context)
        context.on_shutdown(self._stop.set)

        def callback(msg: Imu) -> None:
            now = time.monotonic_ns()
            qw = float(msg.orientation.w)
            qx = float(msg.orientation.x)
            qy = float(msg.orientation.y)
            qz = float(msg.orientation.z)
            embedded = float(msg.orientation_covariance[0])
            heading = embedded if embedded >= 0.0 else self._yaw_deg(qw, qx, qy, qz)
            stamp = msg.header.stamp
            record = {
                "monotonic_ns": now,
                "device_time_s": float(stamp.sec) + float(stamp.nanosec) * 1e-9,
                "heading_deg": float(heading) % 360.0,
                "angular_velocity_rad_s": [
                    float(msg.angular_velocity.x),
                    float(msg.angular_velocity.y),
                    float(msg.angular_velocity.z),
                ],
                "quaternion_wxyz": [qw, qx, qy, qz],
                "accel_m_s2": [
                    float(msg.linear_acceleration.x),
                    float(msg.linear_acceleration.y),
                    float(msg.linear_acceleration.z),
                ],
                "mag": None,
                "heading_ready": True,
            }
            with self._lock:
                self._heading = record["heading_deg"]
                self._last_ns = now
                self._queue.append(record)
                self._ready.set()

        node.create_subscription(Imu, self.topic, callback, 10)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        try:
            while not self._stop.is_set():
                executor.spin_once(timeout_sec=0.05)
        finally:
            executor.remove_node(node)
            node.destroy_node()
            context.try_shutdown()

    def wait_for_heading(self, timeout=8.0):
        return self._ready.wait(timeout)

    def get_heading(self, require_ready=True):
        if require_ready and not self._ready.is_set():
            return None
        with self._lock:
            return self._heading

    def get_diagnostics(self):
        with self._lock:
            age = None if self._last_ns is None else (time.monotonic_ns() - self._last_ns) / 1e9
            return {
                "heading_ready": self._ready.is_set(),
                "last_ahrs_age_s": age,
                "sample_buffer_size": len(self._queue),
                "heading_deg": self._heading,
            }

    def drain_samples(self, max_count=None):
        with self._lock:
            count = len(self._queue) if max_count is None else min(max_count, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def clear_samples(self):
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def get_ahrs(self):
        with self._lock:
            latest = self._queue[-1] if self._queue else None
            if latest is None:
                return None, None, None, None
            return (
                latest["angular_velocity_rad_s"],
                None,
                latest["quaternion_wxyz"],
                latest["device_time_s"],
            )

    def get_imu(self):
        with self._lock:
            latest = self._queue[-1] if self._queue else None
            if latest is None:
                return None, None, None, None
            return (
                latest["angular_velocity_rad_s"],
                latest["accel_m_s2"],
                latest["mag"],
                latest["device_time_s"],
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    close = stop
    release = stop


class OnlineVioAligner:
    """Estimate a fixed VINS-world -> local EN transform from live GPS pairs."""

    def __init__(self, minimum_pairs: int = 3, max_pair_age_s: float = 1.5, max_rms_m: float = 5.0):
        self.minimum_pairs = minimum_pairs
        self.max_pair_age_ns = int(max_pair_age_s * 1e9)
        self.max_rms_m = max_rms_m
        self._pairs: deque[tuple[VIOSample, tuple[float, float]]] = deque(maxlen=60)
        self.transform = None
        self.rms_m: float | None = None
        self._last_sample_ns: int | None = None

    def add(self, sample: VIOSample | None, east: float, north: float) -> None:
        if sample is None or abs(time.monotonic_ns() - sample.monotonic_ns) > self.max_pair_age_ns:
            return
        self._last_sample_ns = sample.monotonic_ns
        if self._pairs and sample.monotonic_ns == self._pairs[-1][0].monotonic_ns:
            return
        self._pairs.append((sample, (east, north)))
        if len(self._pairs) >= self.minimum_pairs:
            from memory_nav.trajectory.processing import estimate_vio_transform
            transform, rms = estimate_vio_transform([item[0] for item in self._pairs], [item[1] for item in self._pairs])
            self.rms_m = rms
            self.transform = transform if rms <= self.max_rms_m else None

    @property
    def state(self) -> str:
        if self._last_sample_ns is not None and time.monotonic_ns() - self._last_sample_ns > self.max_pair_age_ns:
            return "lost"
        if self.transform is not None:
            return "aligned"
        return "warming_up" if len(self._pairs) < self.minimum_pairs else "degraded"

    def project(self, sample: VIOSample | None) -> tuple[float, float] | None:
        if sample is None or self.transform is None or time.monotonic_ns() - sample.monotonic_ns > self.max_pair_age_ns:
            return None
        return self.transform.apply(sample.x_m, sample.y_m)
