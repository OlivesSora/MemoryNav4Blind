"""Crash-tolerant route directory and JSONL session writer."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from memory_nav import SCHEMA_VERSION


ROUTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class SessionError(RuntimeError):
    pass


def validate_route_id(route_id: str) -> str:
    if not ROUTE_ID_RE.fullmatch(route_id):
        raise ValueError("route_id must be 1-64 safe filename characters")
    return route_id


def atomic_write_json(path: Path, data: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class SessionWriter:
    """Append samples immediately while updating metadata atomically."""

    def __init__(self, routes_dir: str | Path, route_id: str, metadata: Mapping[str, Any] | None = None):
        self.route_id = validate_route_id(route_id)
        self.route_dir = Path(routes_dir).resolve() / self.route_id
        if self.route_dir.exists():
            raise SessionError(f"route already exists: {self.route_dir}")
        self.raw_dir = self.route_dir / "raw"
        self.anchors_dir = self.route_dir / "anchors"
        self.raw_dir.mkdir(parents=True)
        self.anchors_dir.mkdir()
        self._lock = threading.Lock()
        self._closed = False
        self._streams = {
            name: (self.raw_dir / f"{name}.jsonl").open("a", encoding="utf-8", buffering=1)
            for name in ("gps", "imu", "vio", "events")
        }
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "route_id": self.route_id,
            "state": "RECORDING",
            "coordinate_system": "GCJ-02",
            "start_utc": datetime.now(timezone.utc).isoformat(),
            "start_monotonic_ns": time.monotonic_ns(),
            "sample_counts": {"gps": 0, "imu": 0, "vio": 0, "events": 0},
        }
        if metadata:
            self.manifest.update(dict(metadata))
        atomic_write_json(self.route_dir / "manifest.json", self.manifest)

    def append(self, stream_name: str, sample: Mapping[str, Any] | Any) -> None:
        if stream_name not in self._streams:
            raise ValueError(f"unknown stream: {stream_name}")
        if hasattr(sample, "to_dict"):
            sample = sample.to_dict()
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping or expose to_dict()")
        payload = json.dumps(dict(sample), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise SessionError("session is closed")
            self._streams[stream_name].write(payload + "\n")
            self._streams[stream_name].flush()
            self.manifest["sample_counts"][stream_name] += 1

    def close(self, state: str = "FINALIZING") -> None:
        with self._lock:
            if self._closed:
                return
            for stream in self._streams.values():
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            self._closed = True
            self.manifest.update({
                "state": state,
                "end_utc": datetime.now(timezone.utc).isoformat(),
                "end_monotonic_ns": time.monotonic_ns(),
            })
            atomic_write_json(self.route_dir / "manifest.json", self.manifest)

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close("ERROR" if exc_type else "FINALIZING")


class AsyncSessionWriter:
    """Bounded producer/consumer wrapper that keeps sensor loops off disk I/O."""

    def __init__(self, routes_dir: str | Path, route_id: str, metadata: Mapping[str, Any] | None = None, queue_size: int = 8192):
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.session = SessionWriter(routes_dir, route_id, metadata)
        self.route_dir = self.session.route_dir
        self.raw_dir = self.session.raw_dir
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue(queue_size)
        self._error: Exception | None = None
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="memory-nav-writer", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                stream_name, sample = item
                self.session.append(stream_name, sample)
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def append(self, stream_name: str, sample: Mapping[str, Any] | Any) -> None:
        if self._closed:
            raise SessionError("session is closed")
        if self._error:
            raise SessionError("background session writer failed") from self._error
        try:
            self._queue.put_nowait((stream_name, sample))
        except queue.Full as exc:
            raise SessionError("session writer queue is full; samples would be lost") from exc

    def close(self, state: str = "FINALIZING") -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(None)
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            self.session.close("ERROR")
            raise SessionError("background session writer did not stop")
        final_state = "ERROR" if self._error else state
        self.session.close(final_state)
        if self._error:
            raise SessionError("background session writer failed") from self._error

    def __enter__(self) -> "AsyncSessionWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close("ERROR" if exc_type else "FINALIZING")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise SessionError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(value)
    return records
