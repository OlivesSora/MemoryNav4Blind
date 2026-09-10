"""Anchor candidate generation and image quality checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from memory_nav.models import ReferencePoint
from memory_nav.trajectory.coordinate import wrap_to_180
from memory_nav.recording.session_writer import atomic_write_json
from memory_nav.models import GPSSample
from memory_nav.trajectory.coordinate import LocalFrame


@dataclass(frozen=True)
class AnchorCandidate:
    anchor_id: str
    point_index: int
    s_m: float
    kind: str
    prompt_text: str = ""
    confirmed: bool = False
    image_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def blur_score(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_image_clear(image: np.ndarray, threshold: float = 100.0) -> bool:
    return blur_score(image) >= threshold


def generate_anchor_candidates(points: Sequence[ReferencePoint], max_spacing_m: float = 20.0, turn_angle_deg: float = 35.0) -> list[AnchorCandidate]:
    if len(points) < 2:
        raise ValueError("at least two reference points are required")
    selected: list[tuple[int, str]] = [(0, "start")]
    last_s = points[0].s_m
    for index in range(1, len(points) - 1):
        turn = abs(wrap_to_180(points[index].heading_deg - points[index - 1].heading_deg))
        if turn >= turn_angle_deg:
            selected.append((index, "turn"))
            last_s = points[index].s_m
        elif points[index].s_m - last_s >= max_spacing_m:
            selected.append((index, "interval"))
            last_s = points[index].s_m
    if selected[-1][0] != len(points) - 1:
        selected.append((len(points) - 1, "end"))
    return [AnchorCandidate(f"anchor_{order:04d}", index, points[index].s_m, kind) for order, (index, kind) in enumerate(selected)]


def save_anchors(path: str | Path, anchors: Sequence[AnchorCandidate]) -> None:
    identifiers = [anchor.anchor_id for anchor in anchors]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("anchor IDs must be unique")
    atomic_write_json(Path(path), {"anchors": [anchor.to_dict() for anchor in anchors]})


def load_anchors(path: str | Path) -> list[AnchorCandidate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("anchors"), list):
        raise ValueError("anchors file must contain an anchors list")
    anchors = [AnchorCandidate(**item) for item in data["anchors"]]
    if len({anchor.anchor_id for anchor in anchors}) != len(anchors):
        raise ValueError("anchor IDs must be unique")
    return anchors


class LiveAnchorCapture:
    """Capture route-time image candidates without deciding confirmation."""

    def __init__(self, writer, camera, max_spacing_m: float = 20.0, turn_angle_deg: float = 35.0, blur_threshold: float = 100.0):
        self.writer = writer
        self.camera = camera
        self.max_spacing_m = max_spacing_m
        self.turn_angle_deg = turn_angle_deg
        self.blur_threshold = blur_threshold
        self._frame: LocalFrame | None = None
        self._last_anchor_position: tuple[float, float] | None = None
        self._last_heading: float | None = None
        self._last_sample: GPSSample | None = None
        self._last_anchor_timestamp: int | None = None
        self._count = 0

    def consider(self, sample: GPSSample, heading_deg: float | None) -> bool:
        if self._frame is None:
            self._frame = LocalFrame(sample.longitude, sample.latitude)
        position = self._frame.to_local(sample.longitude, sample.latitude)
        if self._last_anchor_position is None:
            kind = "start"
        else:
            distance = float(np.linalg.norm(np.subtract(position, self._last_anchor_position)))
            turn = 0.0 if heading_deg is None or self._last_heading is None else abs(wrap_to_180(heading_deg - self._last_heading))
            kind = "turn" if turn >= self.turn_angle_deg else "interval" if distance >= self.max_spacing_m else None
        self._last_sample = sample
        if kind is None:
            return False
        return self._capture(sample, heading_deg, kind, position)

    def finalize(self, heading_deg: float | None) -> bool:
        if self._last_sample is None or self._frame is None:
            return False
        if self._last_anchor_timestamp == self._last_sample.monotonic_ns:
            return False
        position = self._frame.to_local(self._last_sample.longitude, self._last_sample.latitude)
        return self._capture(self._last_sample, heading_deg, "end", position)

    def _capture(self, sample: GPSSample, heading_deg: float | None, kind: str, position: tuple[float, float]) -> bool:
        capture_with_metadata = getattr(self.camera, "capture_frame_with_metadata", None)
        if capture_with_metadata:
            image, metadata = capture_with_metadata()
            frame_timestamp = None if metadata is None else metadata.get("monotonic_ns")
        else:
            image, frame_count = self.camera.capture_frame()
            metadata = {"frame_count": frame_count}
            frame_timestamp = None
        if image is None:
            self.writer.append("events", {"type": "anchor_image_failed", "kind": kind, "monotonic_ns": sample.monotonic_ns})
            return False
        score = blur_score(image)
        filename = f"candidate_{self._count:04d}.jpg"
        image_path = self.writer.route_dir / "anchors" / filename
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        if not cv2.imwrite(str(image_path), bgr):
            raise OSError(f"failed to save anchor image: {image_path}")
        self.writer.append("events", {
            "type": "anchor_image",
            "kind": kind,
            "monotonic_ns": sample.monotonic_ns,
            "frame_monotonic_ns": frame_timestamp,
            "longitude": sample.longitude,
            "latitude": sample.latitude,
            "heading_deg": heading_deg,
            "image_path": f"anchors/{filename}",
            "blur_score": score,
            "clear": score >= self.blur_threshold,
            "frame_metadata": metadata,
        })
        self._count += 1
        self._last_anchor_position = position
        self._last_heading = heading_deg
        self._last_anchor_timestamp = sample.monotonic_ns
        return True
