"""Render the tracking direction and walkable sectors on a camera frame."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from memory_nav.segmentation.avoidance import SegmentationResult
from memory_nav.segmentation.geometry import SemicircleMapper, clock_to_angle_deg


WALKABLE_COLOR = (0, 200, 0)
BLOCKED_COLOR = (0, 0, 220)
COMMAND_COLOR = (255, 128, 0)
FALLBACK_COLOR = (0, 165, 255)
CLEAR_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
ARC_COLOR = (255, 255, 255)


def _endpoint(mapper: SemicircleMapper, mask: np.ndarray, clock: float, radius: float) -> tuple[int, int]:
    cx, cy = mapper.origin(mask)
    angle = math.radians(clock_to_angle_deg(clock))
    return int(round(cx + radius * math.sin(angle))), int(round(cy - radius * math.cos(angle)))


def draw_segmentation(
    frame_bgr: np.ndarray,
    result: SegmentationResult,
    mapper: SemicircleMapper,
    walkable_mask: Optional[np.ndarray] = None,
    show_mask: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` annotated with the segmentation mapping."""
    canvas = np.asarray(frame_bgr).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")
    height, width = canvas.shape[:2]
    reference = walkable_mask if walkable_mask is not None else np.zeros((height, width), dtype=bool)

    if show_mask and walkable_mask is not None:
        mask = np.asarray(walkable_mask).astype(bool)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST) > 127
        tint = np.zeros_like(canvas)
        tint[mask] = WALKABLE_COLOR
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0.0)

    # Sector fills for every forward hour.
    overlay = canvas.copy()
    for hour in result.clear_distances:
        walkable = hour in result.walkable_hours
        color = WALKABLE_COLOR if walkable else BLOCKED_COLOR
        polygon = mapper.sector_polygon(reference, hour)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0.0)

    cx, cy = mapper.origin(reference)
    center = (int(round(cx)), int(round(cy)))
    radius = mapper.radius(reference)

    # Semicircle arc, hour spokes/labels, and clear-distance markers.
    cv2.polylines(canvas, [mapper.arc_points(reference)], False, ARC_COLOR, 2, cv2.LINE_AA)
    for hour, clear_distance in result.clear_distances.items():
        tip = _endpoint(mapper, reference, hour, radius)
        cv2.line(canvas, center, tip, ARC_COLOR, 1, cv2.LINE_AA)
        marker_radius = min(max(float(clear_distance), 0.0), radius)
        marker = _endpoint(mapper, reference, hour, marker_radius)
        cv2.circle(canvas, marker, 3, CLEAR_COLOR, -1, cv2.LINE_AA)
        label = str(hour)
        lx = int(round(cx + 0.88 * radius * math.sin(math.radians(clock_to_angle_deg(hour)))))
        ly = int(round(cy - 0.88 * radius * math.cos(math.radians(clock_to_angle_deg(hour)))))
        cv2.putText(canvas, label, (lx - 6, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (lx - 6, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARC_COLOR, 1, cv2.LINE_AA)

    command_tip = _endpoint(mapper, reference, result.command_clock, radius * 0.92)
    cv2.arrowedLine(canvas, center, command_tip, COMMAND_COLOR, 3, cv2.LINE_AA, tipLength=0.15)
    if result.fallback_used:
        fallback_tip = _endpoint(mapper, reference, result.seg_clock, radius * 0.92)
        cv2.arrowedLine(canvas, center, fallback_tip, FALLBACK_COLOR, 3, cv2.LINE_AA, tipLength=0.15)

    status = "consistent" if result.consistent else (
        "fallback" if result.fallback_used else (result.warning or "no_walkable")
    )
    clear_text = " ".join(
        f"{hour}:{int(round(clear))}" for hour, clear in result.clear_distances.items()
    )
    lines = [
        f"cmd={result.command_clock} seg={result.seg_clock} [{status}] r={int(round(radius))}px",
        "walkable=" + ",".join(str(hour) for hour in result.walkable_hours),
        "clear(px)=" + clear_text,
    ]
    for index, line in enumerate(lines):
        y = 20 + index * 20
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return canvas


def draw_mask_overlay(
    frame_bgr: np.ndarray,
    mapper: SemicircleMapper,
    walkable_mask: np.ndarray,
    label: str = "pose_unavailable [mask only]",
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` tinted with the walkable mask.

    Unlike :func:`draw_segmentation` this needs no tracked command, so it is
    used when the navigation pose is unavailable.
    """
    canvas = np.asarray(frame_bgr).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")
    height, width = canvas.shape[:2]
    mask = np.asarray(walkable_mask).astype(bool)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8) * 255,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 127
    tint = np.zeros_like(canvas)
    tint[mask] = WALKABLE_COLOR
    canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0.0)
    cv2.polylines(canvas, [mapper.arc_points(mask)], False, ARC_COLOR, 2, cv2.LINE_AA)
    cx, cy = mapper.origin(mask)
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), 3, CLEAR_COLOR, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return canvas
