"""Clock-direction to image-semicircle geometry.

Convention
----------
The tracking loop produces clock directions where ``12`` is straight
ahead, ``3`` is to the right and ``9`` is to the left (clockwise positive).
The image mapping places a fixed-length semicircle with its origin at the
bottom-center of the frame and opens toward the top of the image:

    12 (image up, angle 0)
    11                1
    10                  2
    9  (left)    (right) 3

``clock_to_angle_deg`` returns the angle measured from the image-up
direction, positive clockwise, so ``12 -> 0``, ``3 -> +90`` and
``9 -> -90``.  Pixel offsets at radius ``r`` are ``dx = r*sin(a)`` and
``dy = -r*cos(a)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import cv2
import numpy as np


FORWARD_HOURS: tuple[int, ...] = (9, 10, 11, 12, 1, 2, 3)
HOUR_WIDTH_DEG = 30.0


@dataclass(frozen=True)
class HourClearance:
    """Raw clearance metrics for one clock hour's angular sector."""

    clear_distance: float
    obstacle_pixels: int
    wedge_pixels: int


def normalize_clock(clock: int | str) -> int:
    """Normalise a clock direction to the range 1..12 (12 instead of 0)."""
    try:
        hour = int(clock)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid clock direction: {clock!r}") from exc
    hour %= 12
    return 12 if hour == 0 else hour


def clock_to_angle_deg(clock: int | str) -> float:
    """Angle from image-up in degrees, positive clockwise, in (-180, 180]."""
    angle = (normalize_clock(clock) % 12) * HOUR_WIDTH_DEG
    if angle > 180.0:
        angle -= 360.0
    return angle


def angle_to_clock(angle_deg: float) -> int:
    """Nearest clock direction for an image angle (positive clockwise)."""
    hour = int(round(float(angle_deg) / HOUR_WIDTH_DEG)) % 12
    return 12 if hour == 0 else hour


def clock_angular_distance(first: int | str, second: int | str) -> float:
    """Absolute angular distance between two clock directions in degrees."""
    delta = clock_to_angle_deg(first) - clock_to_angle_deg(second)
    return abs((delta + 180.0) % 360.0 - 180.0)


def _polar_points(cx: float, cy: float, radius: float, angles_deg: np.ndarray) -> np.ndarray:
    radians = np.radians(angles_deg)
    x = cx + radius * np.sin(radians)
    y = cy - radius * np.cos(radians)
    return np.stack([x, y], axis=1)


class SemicircleMapper:
    """Fixed-length forward semicircle anchored at the bottom of the frame."""

    def __init__(
        self,
        origin_x_ratio: float = 0.5,
        origin_y_ratio: float = 1.0,
        radius_ratio: float = 0.25,
        inner_radius_ratio: float = 0.05,
        half_width_deg: float = HOUR_WIDTH_DEG / 2.0,
        polygon_samples: int = 24,
    ):
        if not 0.0 <= origin_x_ratio <= 1.0:
            raise ValueError("origin_x_ratio must be in [0, 1]")
        if not 0.0 <= origin_y_ratio <= 1.0:
            raise ValueError("origin_y_ratio must be in [0, 1]")
        if radius_ratio <= 0.0:
            raise ValueError("radius_ratio must be positive")
        if not 0.0 <= inner_radius_ratio < radius_ratio:
            raise ValueError("inner_radius_ratio must be in [0, radius_ratio)")
        if not 0.0 < half_width_deg <= HOUR_WIDTH_DEG / 2.0:
            raise ValueError("half_width_deg must be in (0, 15]")
        if polygon_samples < 3:
            raise ValueError("polygon_samples must be at least 3")
        self.origin_x_ratio = float(origin_x_ratio)
        self.origin_y_ratio = float(origin_y_ratio)
        self.radius_ratio = float(radius_ratio)
        self.inner_radius_ratio = float(inner_radius_ratio)
        self.half_width_deg = float(half_width_deg)
        self.polygon_samples = int(polygon_samples)

    def _shape(self, mask: np.ndarray) -> tuple[int, int]:
        if mask.ndim != 2:
            raise ValueError("walkable mask must be a 2-D array")
        height, width = mask.shape
        if height < 2 or width < 2:
            raise ValueError("walkable mask is too small")
        return height, width

    def origin(self, mask: np.ndarray) -> tuple[float, float]:
        height, width = self._shape(mask)
        return (self.origin_x_ratio * (width - 1), self.origin_y_ratio * (height - 1))

    def radius(self, mask: np.ndarray) -> float:
        height, _width = self._shape(mask)
        return self.radius_ratio * height

    def inner_radius(self, mask: np.ndarray) -> float:
        height, _width = self._shape(mask)
        return self.inner_radius_ratio * height

    def sector_polygon(self, mask: np.ndarray, clock: int | str) -> np.ndarray:
        """Integer polygon for one clock hour's angular sector."""
        cx, cy = self.origin(mask)
        outer_radius = self.radius(mask)
        inner_radius = self.inner_radius(mask)
        center = clock_to_angle_deg(clock)
        angles = np.linspace(
            center - self.half_width_deg,
            center + self.half_width_deg,
            self.polygon_samples,
        )
        outer = _polar_points(cx, cy, outer_radius, angles)
        inner = _polar_points(cx, cy, inner_radius, angles[::-1])
        return np.vstack([outer, inner]).astype(np.int32)

    def hour_clearance(
        self,
        mask: np.ndarray,
        hours: Iterable[int] = FORWARD_HOURS,
    ) -> dict[int, HourClearance]:
        """Clearance metrics for each clock hour's sector.

        ``clear_distance`` is the Euclidean distance from the semicircle
        origin to the nearest non-walkable pixel inside the sector, capped at
        the sector's outer radius when no obstacle is present.
        """
        walkable = np.asarray(mask).astype(bool)
        height, width = self._shape(walkable)
        cx, cy = self.origin(walkable)
        outer_radius = self.radius(walkable)
        clearances: dict[int, HourClearance] = {}
        for hour in hours:
            normalized = normalize_clock(hour)
            sector = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(sector, [self.sector_polygon(walkable, normalized)], 1)
            wedge = sector.astype(bool)
            wedge_pixels = int(np.count_nonzero(wedge))
            obstacles = np.logical_and(wedge, ~walkable)
            obstacle_pixels = int(np.count_nonzero(obstacles))
            if obstacle_pixels == 0:
                clear_distance = float(outer_radius)
            else:
                ys, xs = np.nonzero(obstacles)
                distances = np.hypot(xs - cx, ys - cy)
                clear_distance = float(distances.min())
            clearances[normalized] = HourClearance(
                clear_distance=clear_distance,
                obstacle_pixels=obstacle_pixels,
                wedge_pixels=wedge_pixels,
            )
        return clearances

    def arc_points(self, mask: np.ndarray, samples: int = 64) -> np.ndarray:
        """Polyline of the forward semicircle arc for drawing."""
        cx, cy = self.origin(mask)
        radius = self.radius(mask)
        angles = np.linspace(-90.0, 90.0, samples)
        return _polar_points(cx, cy, radius, angles).astype(np.int32)

    def to_dict(self) -> dict[str, float]:
        return {
            "origin_x_ratio": self.origin_x_ratio,
            "origin_y_ratio": self.origin_y_ratio,
            "radius_ratio": self.radius_ratio,
            "inner_radius_ratio": self.inner_radius_ratio,
            "half_width_deg": self.half_width_deg,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SemicircleMapper":
        known = {key: mapping[key] for key in (
            "origin_x_ratio",
            "origin_y_ratio",
            "radius_ratio",
            "inner_radius_ratio",
            "half_width_deg",
            "polygon_samples",
        ) if key in mapping}
        return cls(**known)  # type: ignore[arg-type]
