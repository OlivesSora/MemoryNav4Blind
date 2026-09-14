"""Throttled disk visualization for online segmentation decisions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from memory_nav.segmentation.avoidance import SegmentationResult
from memory_nav.segmentation.geometry import SemicircleMapper
from memory_nav.segmentation.visualize import draw_segmentation


class SegmentationFrameSaver:
    """Write annotated frames to ``vis_dir`` at a limited rate.

    Frames arriving from the navigation loop are RGB; ``draw_segmentation``
    expects BGR, so the conversion happens here.
    """

    def __init__(
        self,
        vis_dir: str | Path,
        interval_s: float = 1.0,
        show: bool = False,
        prefix: str = "seg",
    ):
        if interval_s < 0:
            raise ValueError("interval_s cannot be negative")
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.show = show
        self.prefix = prefix
        self._last_saved = None
        self.saved_count = 0

    def _to_bgr(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 2:
            return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if array.shape[2] == 4:
            return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def __call__(
        self,
        frame: np.ndarray,
        result: SegmentationResult,
        mapper: SemicircleMapper,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        now = time.monotonic()
        if self._last_saved is not None and now - self._last_saved < self.interval_s:
            return
        self._last_saved = now
        annotated = draw_segmentation(self._to_bgr(frame), result, mapper, mask)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((now % 1) * 1000):03d}"
        path = self.vis_dir / f"{self.prefix}_{stamp}.jpg"
        cv2.imwrite(str(path), annotated)
        self.saved_count += 1
        if self.show:
            cv2.imshow("memory_nav_segmentation", annotated)
            cv2.waitKey(1)

    def close(self) -> None:
        if self.show:
            try:
                cv2.destroyWindow("memory_nav_segmentation")
            except cv2.error:
                pass
