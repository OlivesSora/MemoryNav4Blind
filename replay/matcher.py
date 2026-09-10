"""Progress-constrained nearest segment matching."""

from __future__ import annotations

import math
from typing import Sequence

from memory_nav.models import MatchResult, ReferencePoint
from memory_nav.trajectory.coordinate import heading_from_delta, wrap_to_180


def project_to_segment(
    east_m: float, north_m: float, start: ReferencePoint, end: ReferencePoint
) -> tuple[float, float, float, float, float]:
    dx = end.east_m - start.east_m
    dy = end.north_m - start.north_m
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        raise ValueError("zero-length route segment")
    t = max(0.0, min(1.0, ((east_m - start.east_m) * dx + (north_m - start.north_m) * dy) / length_sq))
    projected_east = start.east_m + t * dx
    projected_north = start.north_m + t * dy
    offset_east = east_m - projected_east
    offset_north = north_m - projected_north
    distance = math.hypot(offset_east, offset_north)
    cross = dx * (north_m - start.north_m) - dy * (east_m - start.east_m)
    signed_distance = distance if cross > 0 else -distance if cross < 0 else 0.0
    s_m = start.s_m + t * (end.s_m - start.s_m)
    return projected_east, projected_north, signed_distance, distance, s_m


class RouteMatcher:
    def __init__(
        self,
        points: Sequence[ReferencePoint],
        backward_window_m: float = 2.0,
        forward_window_m: float = 20.0,
        initial_search_m: float = 15.0,
        good_distance_m: float = 3.0,
        lost_distance_m: float = 8.0,
        heading_weight: float = 0.03,
        max_heading_error_deg: float = 100.0,
    ):
        if len(points) < 2:
            raise ValueError("route needs at least two points")
        if any(points[index + 1].s_m <= points[index].s_m for index in range(len(points) - 1)):
            raise ValueError("route progress must be strictly increasing")
        self.points = list(points)
        self.backward_window_m = backward_window_m
        self.forward_window_m = forward_window_m
        self.initial_search_m = initial_search_m
        self.good_distance_m = good_distance_m
        self.lost_distance_m = lost_distance_m
        self.heading_weight = heading_weight
        self.max_heading_error_deg = max_heading_error_deg
        self.last_s_m: float | None = None

    def reset(self) -> None:
        self.last_s_m = None

    def is_complete(self, tolerance_m: float = 2.0) -> bool:
        if tolerance_m < 0:
            raise ValueError("tolerance_m cannot be negative")
        return self.last_s_m is not None and self.last_s_m >= self.points[-1].s_m - tolerance_m

    def _candidate_indices(self) -> range:
        if self.last_s_m is None:
            indices = [index for index in range(len(self.points) - 1) if self.points[index].s_m <= self.initial_search_m]
            return range(indices[0], indices[-1] + 1) if indices else range(0)
        minimum = self.last_s_m - self.backward_window_m
        maximum = self.last_s_m + self.forward_window_m
        indices = [i for i in range(len(self.points) - 1) if self.points[i + 1].s_m >= minimum and self.points[i].s_m <= maximum]
        return range(indices[0], indices[-1] + 1) if indices else range(0)

    def match(self, east_m: float, north_m: float, heading_deg: float) -> MatchResult:
        best = None
        for index in self._candidate_indices():
            start, end = self.points[index], self.points[index + 1]
            projected_east, projected_north, cross_track, distance, s_m = project_to_segment(east_m, north_m, start, end)
            if self.last_s_m is None and s_m > self.initial_search_m:
                continue
            segment_heading = heading_from_delta(end.east_m - start.east_m, end.north_m - start.north_m)
            heading_error = wrap_to_180(heading_deg - segment_heading)
            if abs(heading_error) > self.max_heading_error_deg:
                heading_penalty = 1000.0
            else:
                heading_penalty = self.heading_weight * abs(heading_error)
            backwards_penalty = 0.0
            if self.last_s_m is not None and s_m < self.last_s_m:
                backwards_penalty = (self.last_s_m - s_m) * 2.0
            score = distance + heading_penalty + backwards_penalty
            if best is None or score < best[0]:
                best = (score, index, projected_east, projected_north, cross_track, distance, s_m, heading_error)
        if best is None:
            raise RuntimeError("no route segment in progress window")
        _, index, projected_east, projected_north, cross_track, distance, s_m, heading_error = best
        if self.last_s_m is None and (distance > self.initial_search_m or abs(heading_error) > self.max_heading_error_deg):
            quality = "lost"
        elif distance <= self.good_distance_m:
            quality = "good"
        elif distance <= self.lost_distance_m:
            quality = "degraded"
        else:
            quality = "lost"
        if quality != "lost":
            self.last_s_m = s_m if self.last_s_m is None else max(self.last_s_m, s_m)
        return MatchResult(index, s_m, projected_east, projected_north, cross_track, heading_error, distance, quality)
