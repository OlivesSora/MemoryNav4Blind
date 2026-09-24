"""Online CAT-Seg mask provider backed by a persistent worker subprocess.

The provider owns a ``catseg``-environment worker process (detectron2) and
talks to it over a Unix domain socket. Inference runs on a background thread
with a single-slot frame buffer: the newest frame replaces any queued frame.
Every result retains its source frame id, capture timestamp and pixels, and
results older than the configured age limit are rejected.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from memory_nav.config import PROJECT_ROOT
from memory_nav.segmentation.protocol import recv_message, send_message

LOGGER = logging.getLogger(__name__)

DEFAULT_WORKER_PYTHON = os.path.expanduser("~/anaconda3/envs/catseg/bin/python")
DEFAULT_CATSEG_DIR = "/home/wheeltec/projects/blind-nav-server/CAT-Seg"
DEFAULT_WALKABLE_NAMES = ("pavement", "road", "stairs")
DEFAULT_TRT_PYTHON_PATH = "/usr/lib/python3.10/dist-packages"


@dataclass(frozen=True)
class MaskObservation:
    """A mask paired with the exact camera frame that produced it."""

    frame_id: str
    captured_at_s: float
    completed_at_s: float
    frame: np.ndarray
    mask: np.ndarray
    inference_s: float

    def age_s(self, now_s: float | None = None) -> float:
        now = time.monotonic() if now_s is None else now_s
        return max(0.0, now - self.captured_at_s)


@dataclass(frozen=True)
class _PendingFrame:
    frame_id: str
    captured_at_s: float
    frame_bgr: np.ndarray
    source_frame: np.ndarray


class NullMaskProvider:
    """Provider that never yields a mask; keeps navigation running unchanged."""

    def get_mask(self, frame_id, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        pass


class CatSegWorkerProvider:
    def __init__(
        self,
        worker_python: str = DEFAULT_WORKER_PYTHON,
        socket_path: str | Path | None = None,
        catseg_dir: str | Path = DEFAULT_CATSEG_DIR,
        config: str = "configs/vitb_384.yaml",
        weights: str = "model_base.pth",
        device: str = "cuda",
        backend: str = "pytorch",
        trt_clip_engine: str | Path | None = None,
        trt_aggregator_engine: str | Path | None = None,
        trt_python_path: str = DEFAULT_TRT_PYTHON_PATH,
        trt_warmup: int = 1,
        walkable_names: Sequence[str] = DEFAULT_WALKABLE_NAMES,
        input_scale: float = 0.5,
        min_size_test: int | None = None,
        max_hz: float = 4.0,
        max_mask_age_s: float = 0.4,
        connect_timeout_s: float = 60.0,
        min_free_mb: int = 3000,
    ):
        if not 0.0 < input_scale <= 1.0:
            raise ValueError("input_scale must be in (0, 1]")
        if max_hz <= 0:
            raise ValueError("max_hz must be positive")
        if max_mask_age_s <= 0:
            raise ValueError("max_mask_age_s must be positive")
        if backend not in {"pytorch", "tensorrt"}:
            raise ValueError("backend must be 'pytorch' or 'tensorrt'")
        if backend == "tensorrt" and device != "cuda":
            raise ValueError("TensorRT backend requires device='cuda'")
        if trt_warmup < 0:
            raise ValueError("trt_warmup must be non-negative")
        self.worker_python = str(Path(worker_python).expanduser())
        self.catseg_dir = Path(catseg_dir).expanduser()
        self.config = config
        self.weights = weights
        self.device = device
        self.backend = backend
        export_dir = self.catseg_dir / "export" / "catseg_stages"
        self.trt_clip_engine = Path(
            trt_clip_engine or export_dir / "catseg_clip_stage_fp16.engine"
        ).expanduser()
        self.trt_aggregator_engine = Path(
            trt_aggregator_engine
            or export_dir / "catseg_aggregator_stage_fp16.engine"
        ).expanduser()
        self.trt_python_path = trt_python_path
        self.trt_warmup = int(trt_warmup)
        self.walkable_names = tuple(walkable_names)
        self.input_scale = float(input_scale)
        self.min_size_test = min_size_test
        self.socket_path = (
            Path(socket_path).expanduser()
            if socket_path
            else Path(f"/tmp/memory_nav_catseg_{os.getpid()}_{id(self):x}.sock")
        )
        self._min_interval_s = 1.0 / max_hz
        self.max_mask_age_s = float(max_mask_age_s)
        self._connect_timeout_s = connect_timeout_s
        self._min_free_mb = int(min_free_mb)
        self._lock = threading.Lock()
        self._pending_frame: Optional[_PendingFrame] = None
        self._pending_event = threading.Event()
        self._latest_observation: Optional[MaskObservation] = None
        self._stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._connected = False
        self._last_inference_started_s: Optional[float] = None
        self._frames_sent = 0
        self._errors = 0
        self._last_error: Optional[str] = None
        self._worker_log_path = self.socket_path.with_suffix(".log")
        self._worker_log_stream = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._log_memory_preflight()
        self._spawn_worker()
        self._thread = threading.Thread(
            target=self._run, name="memory-nav-catseg", daemon=True
        )
        self._thread.start()

    def _log_memory_preflight(self) -> None:
        """Report free memory before loading CAT-Seg (Jetson shares RAM/GPU).

        The worker loads the model on the unified GPU memory, so a low
        ``MemAvailable`` (or exhausted CMA) makes ``build_predictor`` fail with
        ``NvMapMemAllocInternalTagged: error 12``.  Log it explicitly instead of
        letting the worker crash silently.
        """
        info = {}
        try:
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    key, _, value = line.partition(":")
                    info[key.strip()] = value.strip()
        except OSError as exc:
            LOGGER.warning("could not read /proc/meminfo: %s", exc)
            return
        available_mb = self._meminfo_mb(info.get("MemAvailable"))
        cma_free_mb = self._meminfo_mb(info.get("CmaFree"))
        LOGGER.info(
            "CAT-Seg %s preflight: MemAvailable=%sMB CmaFree=%sMB "
            "(need >= %sMB for CUDA model load)",
            self.backend,
            "?" if available_mb is None else available_mb,
            "?" if cma_free_mb is None else cma_free_mb,
            self._min_free_mb,
        )
        if available_mb is not None and available_mb < self._min_free_mb:
            LOGGER.warning(
                "low free memory (%sMB < %sMB); CAT-Seg may fail to load on CUDA. "
                "Free memory (e.g. stop llama-server) or use --seg-device cpu. "
                "Worker stderr: %s",
                available_mb,
                self._min_free_mb,
                self._worker_log_path,
            )

    @staticmethod
    def _meminfo_mb(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        try:
            return int(value.split()[0]) // 1024
        except (ValueError, IndexError):
            return None

    def _spawn_worker(self) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        )
        command = [
            self.worker_python,
            "-m",
            "memory_nav.segmentation.catseg_worker",
            "--socket",
            str(self.socket_path),
            "--backend",
            self.backend,
            "--catseg-dir",
            str(self.catseg_dir),
            "--config",
            self.config,
            "--weights",
            self.weights,
            "--device",
            self.device,
            "--walkable-names",
            ",".join(self.walkable_names),
            "--input-scale",
            str(self.input_scale),
        ]
        if self.backend == "tensorrt":
            command.extend(
                [
                    "--trt-clip-engine",
                    str(self.trt_clip_engine),
                    "--trt-aggregator-engine",
                    str(self.trt_aggregator_engine),
                    "--trt-python-path",
                    self.trt_python_path,
                    "--trt-warmup",
                    str(self.trt_warmup),
                ]
            )
        if self.min_size_test is not None:
            command.extend(["--min-size-test", str(int(self.min_size_test))])
        LOGGER.info("starting CAT-Seg worker: %s", " ".join(command))
        try:
            self._worker_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._worker_log_stream = open(
                self._worker_log_path, "a", encoding="utf-8", buffering=1
            )
        except OSError as exc:
            LOGGER.warning(
                "could not open CAT-Seg worker log %s: %s", self._worker_log_path, exc
            )
            self._worker_log_stream = None
        # Keep stdout (worker "ready on ..." line) on the console; capture the
        # traceback/stderr to a file so an OOM crash is not silent.
        self._process = subprocess.Popen(command, env=env, stderr=self._worker_log_stream)

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + self._connect_timeout_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"CAT-Seg worker exited with code {self._process.returncode}"
                )
            if self.socket_path.exists():
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    connection.connect(str(self.socket_path))
                    return connection
                except OSError:
                    connection.close()
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for CAT-Seg worker socket")

    def _run(self) -> None:
        connection = None
        try:
            connection = self._connect()
            self._connected = True
            LOGGER.info("connected to CAT-Seg worker at %s", self.socket_path)
            while not self._stop.is_set():
                if not self._pending_event.wait(timeout=0.2):
                    continue
                # Wait first, then take the single-slot buffer. Frames that
                # arrive while throttled replace older ones, so inference
                # always starts from the newest available camera frame.
                self._throttle()
                with self._lock:
                    pending = self._pending_frame
                    self._pending_frame = None
                    self._pending_event.clear()
                if pending is None:
                    continue
                inference_started_s = time.monotonic()
                send_message(
                    connection,
                    {
                        "type": "frame",
                        "frame_id": pending.frame_id,
                        "captured_at_s": pending.captured_at_s,
                        "image": pending.frame_bgr,
                    },
                )
                reply = recv_message(connection)
                if reply is None:
                    raise ConnectionError("CAT-Seg worker closed the connection")
                if reply.get("type") != "mask":
                    raise RuntimeError(f"CAT-Seg worker error: {reply.get('error')}")
                if str(reply.get("frame_id")) != pending.frame_id:
                    raise RuntimeError(
                        "CAT-Seg worker returned a mismatched frame: "
                        f"expected {pending.frame_id}, received {reply.get('frame_id')}"
                    )
                completed_at_s = time.monotonic()
                observation = MaskObservation(
                    frame_id=pending.frame_id,
                    captured_at_s=float(reply.get("captured_at_s", pending.captured_at_s)),
                    completed_at_s=completed_at_s,
                    frame=pending.source_frame,
                    mask=np.asarray(reply["mask"]),
                    inference_s=float(reply.get("elapsed_s", 0.0)),
                )
                with self._lock:
                    self._latest_observation = observation
                self._last_inference_started_s = inference_started_s
                self._frames_sent += 1
        except Exception as exc:  # Degrade to no-mask without killing navigation.
            if not self._stop.is_set():
                self._errors += 1
                self._last_error = str(exc)
                LOGGER.warning(
                    "CAT-Seg worker thread stopped: %s (see %s)",
                    exc,
                    self._worker_log_path,
                )
        finally:
            if connection is not None:
                connection.close()

    def _throttle(self) -> None:
        if self._last_inference_started_s is None:
            return
        wait = self._min_interval_s - (
            time.monotonic() - self._last_inference_started_s
        )
        if wait > 0:
            self._stop.wait(timeout=wait)

    # -- MaskProvider interface -------------------------------------------
    @staticmethod
    def _to_bgr(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    def get_mask(
        self, frame_id, frame: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        observation = self.get_observation(frame_id, frame)
        return None if observation is None else observation.mask.astype(bool)

    def get_observation(
        self, frame_id, frame: Optional[np.ndarray] = None,
        captured_at_s: float | None = None,
    ) -> Optional[MaskObservation]:
        """Queue ``frame`` and return only a recent, correctly paired result."""
        if not self._started:
            self.start()
        if frame is not None and self._thread is not None and self._thread.is_alive():
            now_s = time.monotonic()
            source_time_s = now_s if captured_at_s is None else min(now_s, captured_at_s)
            source_frame = np.asarray(frame).copy()
            with self._lock:
                self._pending_frame = _PendingFrame(
                    frame_id=str(frame_id),
                    captured_at_s=source_time_s,
                    frame_bgr=self._to_bgr(source_frame),
                    source_frame=source_frame,
                )
                self._pending_event.set()
        observation = self.latest_observation()
        if observation is None or observation.age_s() > self.max_mask_age_s:
            return None

        return observation

    def latest_observation(self) -> Optional[MaskObservation]:
        """Return the latest source-paired result, even when too old for navigation."""
        with self._lock:
            observation = self._latest_observation
        if observation is None:
            return None
        raw_mask = np.asarray(observation.mask)
        if raw_mask.dtype == bool and raw_mask.shape[:2] == observation.frame.shape[:2]:
            return observation
        result = raw_mask.astype(bool)
        if result.shape[:2] != observation.frame.shape[:2]:
            result = (
                cv2.resize(
                    result.astype(np.uint8),
                    (observation.frame.shape[1], observation.frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        return MaskObservation(
            frame_id=observation.frame_id,
            captured_at_s=observation.captured_at_s,
            completed_at_s=observation.completed_at_s,
            frame=observation.frame,
            mask=result,
            inference_s=observation.inference_s,
        )

    def diagnostics(self) -> dict:
        with self._lock:
            observation = self._latest_observation
        return {
            "backend": self.backend,
            "max_hz": round(1.0 / self._min_interval_s, 3),
            "max_mask_age_s": self.max_mask_age_s,
            "connected": self._connected,
            "frames_sent": self._frames_sent,
            "errors": self._errors,
            "has_mask": observation is not None,
            "latest_frame_id": None if observation is None else observation.frame_id,
            "mask_age_ms": None if observation is None else round(observation.age_s() * 1000.0, 1),
            "capture_to_result_ms": None if observation is None else round(
                (observation.completed_at_s - observation.captured_at_s) * 1000.0,
                1,
            ),
            "inference_ms": None if observation is None else round(observation.inference_s * 1000.0, 1),
            "worker_alive": self._process is not None and self._process.poll() is None,
            "worker_returncode": None if self._process is None else self._process.poll(),
            "last_error": self._last_error,
            "worker_log_path": str(self._worker_log_path),
        }

    def close(self) -> None:
        self._stop.set()
        self._pending_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._worker_log_stream is not None:
            try:
                self._worker_log_stream.close()
            except OSError:
                pass
            self._worker_log_stream = None
        try:
            self.socket_path.unlink()
        except OSError:
            pass
