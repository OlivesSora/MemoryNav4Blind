"""Online CAT-Seg mask provider backed by a persistent worker subprocess.

The provider owns a ``catseg``-environment worker process (detectron2) and
talks to it over a Unix domain socket.  Inference is slow relative to the
navigation loop, so the provider runs it on a background thread with a
single-slot frame buffer: the newest frame replaces any queued frame and
``get_mask`` always returns the most recent mask without blocking.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
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
        walkable_names: Sequence[str] = DEFAULT_WALKABLE_NAMES,
        input_scale: float = 0.5,
        min_size_test: int | None = None,
        max_hz: float = 1.0,
        connect_timeout_s: float = 60.0,
        min_free_mb: int = 3000,
    ):
        if not 0.0 < input_scale <= 1.0:
            raise ValueError("input_scale must be in (0, 1]")
        if max_hz <= 0:
            raise ValueError("max_hz must be positive")
        self.worker_python = str(Path(worker_python).expanduser())
        self.catseg_dir = Path(catseg_dir).expanduser()
        self.config = config
        self.weights = weights
        self.device = device
        self.walkable_names = tuple(walkable_names)
        self.input_scale = float(input_scale)
        self.min_size_test = min_size_test
        self.socket_path = (
            Path(socket_path).expanduser()
            if socket_path
            else Path(f"/tmp/memory_nav_catseg_{os.getpid()}_{id(self):x}.sock")
        )
        self._min_interval_s = 1.0 / max_hz
        self._connect_timeout_s = connect_timeout_s
        self._min_free_mb = int(min_free_mb)
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_event = threading.Event()
        self._latest_mask: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._connected = False
        self._last_inference_s: Optional[float] = None
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
            "CAT-Seg preflight: MemAvailable=%sMB CmaFree=%sMB (need >= %sMB for CUDA model load)",
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
                with self._lock:
                    frame = self._pending_frame
                    self._pending_frame = None
                    self._pending_event.clear()
                if frame is None:
                    continue
                self._throttle()
                send_message(connection, {"type": "frame", "image": frame})
                reply = recv_message(connection)
                if reply is None:
                    raise ConnectionError("CAT-Seg worker closed the connection")
                if reply.get("type") != "mask":
                    raise RuntimeError(f"CAT-Seg worker error: {reply.get('error')}")
                with self._lock:
                    self._latest_mask = np.asarray(reply["mask"])
                self._last_inference_s = time.monotonic()
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
        if self._last_inference_s is None:
            return
        wait = self._min_interval_s - (time.monotonic() - self._last_inference_s)
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
        if not self._started:
            self.start()
        if frame is not None and self._thread is not None and self._thread.is_alive():
            with self._lock:
                self._pending_frame = self._to_bgr(frame)
                self._pending_event.set()
        with self._lock:
            mask = self._latest_mask
        if mask is None:
            return None
        result = np.asarray(mask).astype(bool)
        if frame is not None and result.shape[:2] != np.asarray(frame).shape[:2]:
            result = (
                cv2.resize(
                    result.astype(np.uint8),
                    (np.asarray(frame).shape[1], np.asarray(frame).shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        return result

    def diagnostics(self) -> dict:
        return {
            "connected": self._connected,
            "frames_sent": self._frames_sent,
            "errors": self._errors,
            "has_mask": self._latest_mask is not None,
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
