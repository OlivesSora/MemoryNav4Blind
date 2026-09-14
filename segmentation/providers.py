"""Mask providers decouple the segmentation evaluator from its source.

The default providers read precomputed masks produced by
``memory_nav.segmentation.precompute_masks`` in the ``catseg`` conda
environment.  Providers return ``None`` when no mask is available so the
navigation loop can degrade gracefully instead of blocking.

A future real-time provider (e.g. invoking CAT-Seg in a subprocess) only
needs to implement :class:`MaskProvider` and can use the ``frame`` argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from memory_nav.segmentation.mask_loader import MASK_SUFFIX, load_mask, mask_path_for


@runtime_checkable
class MaskProvider(Protocol):
    def get_mask(
        self, frame_id: str | Path, frame: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Return a boolean walkable mask for the frame or ``None``."""
        ...


class CachedMaskProvider:
    """Look up precomputed walkable masks by frame path or frame stem."""

    def __init__(self, mask_dir: str | Path):
        self.mask_dir = Path(mask_dir)

    def _candidates(self, frame_id: str | Path) -> tuple[Path, ...]:
        path = Path(frame_id)
        if path.suffix:
            return (mask_path_for(self.mask_dir, path), self.mask_dir / f"{path.stem}.png")
        return (
            self.mask_dir / f"{path.name}{MASK_SUFFIX}.png",
            self.mask_dir / f"{path.name}.png",
        )

    def get_mask(
        self, frame_id: str | Path, frame: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        for candidate in self._candidates(frame_id):
            if candidate.is_file():
                return load_mask(candidate)
        return None


class SequenceMaskProvider:
    """Iterate precomputed masks in lock-step with a frame sequence.

    Useful for replaying a recorded run through the online loop: each call
    advances to the next frame/mask pair.
    """

    def __init__(self, mask_dir: str | Path, frames: list[str | Path], loop: bool = True):
        if not frames:
            raise ValueError("frames cannot be empty")
        self.mask_dir = Path(mask_dir)
        self.frames = [Path(frame) for frame in frames]
        self.loop = loop
        self._index = 0

    def get_mask(
        self, frame_id: str | Path, frame: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if self._index >= len(self.frames):
            if not self.loop:
                return None
            self._index = 0
        current = self.frames[self._index]
        self._index += 1
        mask_path = self.mask_dir / f"{current.stem}{MASK_SUFFIX}.png"
        if not mask_path.is_file():
            return None
        return load_mask(mask_path)
