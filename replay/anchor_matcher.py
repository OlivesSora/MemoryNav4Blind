"""Lazy XFeat anchor matcher with RANSAC geometry verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualMatchResult:
    matched: bool
    match_count: int
    inlier_count: int
    inlier_ratio: float


def verify_correspondences(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    minimum_matches: int = 20,
    minimum_inlier_ratio: float = 0.35,
    ransac_threshold_px: float = 4.0,
) -> VisualMatchResult:
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    current = np.asarray(current_points, dtype=np.float32).reshape(-1, 2)
    if len(reference) != len(current):
        raise ValueError("correspondence arrays must have equal length")
    match_count = len(reference)
    if match_count < 4:
        return VisualMatchResult(False, match_count, 0, 0.0)
    _, mask = cv2.findHomography(reference, current, cv2.USAC_MAGSAC, ransac_threshold_px)
    inlier_count = 0 if mask is None else int(mask.ravel().sum())
    ratio = inlier_count / match_count
    matched = match_count >= minimum_matches and ratio >= minimum_inlier_ratio
    return VisualMatchResult(matched, match_count, inlier_count, ratio)


class XFeatAnchorMatcher:
    """Load the repository's local XFeat model only when first needed."""

    def __init__(self, project_root: str | Path, top_k: int = 2048, minimum_matches: int = 20, minimum_inlier_ratio: float = 0.35):
        self.project_root = Path(project_root).resolve()
        self.top_k = top_k
        self.minimum_matches = minimum_matches
        self.minimum_inlier_ratio = minimum_inlier_ratio
        self._model = None

    def _load(self):
        if self._model is None:
            import torch

            repository = self.project_root / "accelerated_features"
            if not repository.is_dir():
                raise RuntimeError(f"local XFeat repository is missing: {repository}")
            weights_candidates = (repository / "weights" / "xfeat.pt", self.project_root / "ckpt" / "xfeat.pt")
            weights_path = next((path for path in weights_candidates if path.is_file()), None)
            if weights_path is None:
                raise RuntimeError("local XFeat weights are missing")
            # ``pretrained=True`` in the bundled hubconf downloads from
            # GitHub.  Build locally and load the checked-out weights instead.
            self._model = torch.hub.load(str(repository), "XFeat", source="local", pretrained=False, top_k=self.top_k)
            state = torch.load(weights_path, map_location=self._model.dev, weights_only=True)
            self._model.net.load_state_dict(state)
        return self._model

    def match(self, reference_bgr: np.ndarray, current_bgr: np.ndarray) -> VisualMatchResult:
        if reference_bgr is None or current_bgr is None:
            raise ValueError("both anchor images are required")
        model = self._load()
        reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
        points_reference, points_current = model.match_xfeat(reference_rgb, current_rgb, top_k=self.top_k)
        if hasattr(points_reference, "detach"):
            points_reference = points_reference.detach().cpu().numpy()
            points_current = points_current.detach().cpu().numpy()
        return verify_correspondences(points_reference, points_current, self.minimum_matches, self.minimum_inlier_ratio)
