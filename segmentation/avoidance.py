"""Compare a tracking clock direction with a walkable segmentation mask.

The evaluator maps the commanded clock direction and the walkable mask onto
the same bottom-center semicircle.  A forward clock hour is feasible when
the nearest non-walkable pixel in its sector is beyond the semicircle radius
(i.e. the sector is clear, up to a tiny noise tolerance).  When the commanded
sector (and its tolerance band) overlaps a feasible sector the tracking
direction is validated and returned unchanged; otherwise the nearest feasible
forward clock direction is returned as a fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from memory_nav.segmentation.geometry import (
    FORWARD_HOURS,
    HOUR_WIDTH_DEG,
    SemicircleMapper,
    clock_angular_distance,
    normalize_clock,
)


@dataclass(frozen=True)
class SegmentationConfig:
    origin_x_ratio: float = 0.5
    origin_y_ratio: float = 1.0
    radius_ratio: float = 0.25
    inner_radius_ratio: float = 0.05
    half_width_deg: float = HOUR_WIDTH_DEG / 2.0
    max_obstacle_ratio: float = 0.005
    command_tolerance_hours: int = 1
    forward_hours: tuple[int, ...] = FORWARD_HOURS

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_obstacle_ratio <= 1.0:
            raise ValueError("max_obstacle_ratio must be in [0, 1]")
        if self.command_tolerance_hours < 0:
            raise ValueError("command_tolerance_hours cannot be negative")
        if not self.forward_hours:
            raise ValueError("forward_hours cannot be empty")

    def mapper(self) -> SemicircleMapper:
        return SemicircleMapper(
            origin_x_ratio=self.origin_x_ratio,
            origin_y_ratio=self.origin_y_ratio,
            radius_ratio=self.radius_ratio,
            inner_radius_ratio=self.inner_radius_ratio,
            half_width_deg=self.half_width_deg,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SegmentationConfig":
        known = {
            key: mapping[key]
            for key in cls.__dataclass_fields__
            if key in mapping and key != "forward_hours"
        }
        if "forward_hours" in mapping:
            known["forward_hours"] = tuple(int(hour) for hour in mapping["forward_hours"])
        return cls(**known)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SegmentationResult:
    command_clock: int
    seg_clock: int
    consistent: bool
    fallback_used: bool
    out_of_scope: bool
    clear_distances: dict[int, float]
    obstacle_pixels: dict[int, int]
    walkable_hours: tuple[int, ...]
    intersection_hours: tuple[int, ...]
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SegmentationAvoidance:
    def __init__(self, config: SegmentationConfig | None = None):
        self.config = config or SegmentationConfig()
        self.mapper = self.config.mapper()

    def _feasible(self, clearance) -> bool:
        if clearance.wedge_pixels <= 0:
            return False
        return (
            clearance.obstacle_pixels
            <= self.config.max_obstacle_ratio * clearance.wedge_pixels
        )

    def evaluate(
        self,
        command_clock: int | str,
        walkable_mask: np.ndarray,
    ) -> SegmentationResult:
        command = normalize_clock(command_clock)
        forward = tuple(normalize_clock(hour) for hour in self.config.forward_hours)
        clearances = self.mapper.hour_clearance(walkable_mask, forward)
        clear_distances = {
            hour: clearances[hour].clear_distance for hour in forward
        }
        obstacle_pixels = {
            hour: clearances[hour].obstacle_pixels for hour in forward
        }
        walkable_hours = tuple(
            hour for hour in forward if self._feasible(clearances[hour])
        )

        if command not in forward:
            return SegmentationResult(
                command_clock=command,
                seg_clock=command,
                consistent=False,
                fallback_used=False,
                out_of_scope=True,
                clear_distances=clear_distances,
                obstacle_pixels=obstacle_pixels,
                walkable_hours=walkable_hours,
                intersection_hours=(),
                warning="out_of_scope",
            )

        tolerance_deg = self.config.command_tolerance_hours * HOUR_WIDTH_DEG
        band = tuple(
            hour
            for hour in forward
            if clock_angular_distance(hour, command) <= tolerance_deg + 1e-9
        )
        intersection = tuple(hour for hour in band if hour in walkable_hours)
        if intersection:
            return SegmentationResult(
                command_clock=command,
                seg_clock=command,
                consistent=True,
                fallback_used=False,
                out_of_scope=False,
                clear_distances=clear_distances,
                obstacle_pixels=obstacle_pixels,
                walkable_hours=walkable_hours,
                intersection_hours=intersection,
                warning=None,
            )

        if walkable_hours:
            fallback = min(
                walkable_hours,
                key=lambda hour: (clock_angular_distance(hour, command), hour),
            )
            return SegmentationResult(
                command_clock=command,
                seg_clock=fallback,
                consistent=False,
                fallback_used=True,
                out_of_scope=False,
                clear_distances=clear_distances,
                obstacle_pixels=obstacle_pixels,
                walkable_hours=walkable_hours,
                intersection_hours=(),
                warning="command_blocked",
            )

        return SegmentationResult(
            command_clock=command,
            seg_clock=command,
            consistent=False,
            fallback_used=False,
            out_of_scope=False,
            clear_distances=clear_distances,
            obstacle_pixels=obstacle_pixels,
            walkable_hours=(),
            intersection_hours=(),
            warning="no_walkable",
        )
