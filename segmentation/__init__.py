"""Map tracking clock directions onto a walkable-segmentation semicircle.

The tracking loop reports a clock direction (12 = straight, 3 = right,
9 = left).  This package projects that direction onto a fixed-length
semicircle anchored at the bottom-center of the camera image and compares
it with a CAT-Seg walkable mask, so the caller can detect when the tracked
direction is blocked and fall back to the nearest walkable direction.
"""

from __future__ import annotations

from memory_nav.segmentation.avoidance import (
    SegmentationAvoidance,
    SegmentationConfig,
    SegmentationResult,
)
from memory_nav.segmentation.geometry import (
    FORWARD_HOURS,
    HourClearance,
    SemicircleMapper,
    angle_to_clock,
    clock_angular_distance,
    clock_to_angle_deg,
    normalize_clock,
)
from memory_nav.segmentation.guard import SegmentationDecision, SegmentationGuard
from memory_nav.segmentation.mask_loader import load_mask, mask_path_for, save_mask
from memory_nav.segmentation.online_provider import (
    CatSegWorkerProvider,
    MaskObservation,
    NullMaskProvider,
)
from memory_nav.segmentation.online_visualizer import SegmentationFrameSaver
from memory_nav.segmentation.providers import (
    CachedMaskProvider,
    MaskProvider,
    SequenceMaskProvider,
)
from memory_nav.segmentation.visualize import draw_segmentation

__all__ = [
    "FORWARD_HOURS",
    "CachedMaskProvider",
    "CatSegWorkerProvider",
    "HourClearance",
    "MaskProvider",
    "MaskObservation",
    "NullMaskProvider",
    "SegmentationAvoidance",
    "SegmentationConfig",
    "SegmentationDecision",
    "SegmentationFrameSaver",
    "SegmentationGuard",
    "SegmentationResult",
    "SemicircleMapper",
    "SequenceMaskProvider",
    "angle_to_clock",
    "clock_angular_distance",
    "clock_to_angle_deg",
    "draw_segmentation",
    "load_mask",
    "mask_path_for",
    "normalize_clock",
    "save_mask",
]
