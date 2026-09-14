"""Read and write cached binary walkable masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


MASK_SUFFIX = "_walkable"


def mask_path_for(mask_dir: str | Path, frame: str | Path) -> Path:
    """Return the cached mask path for a frame path or frame stem."""
    stem = Path(frame).stem
    return Path(mask_dir) / f"{stem}{MASK_SUFFIX}.png"


def load_mask(path: str | Path) -> np.ndarray:
    """Load a cached walkable mask as a boolean array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to read walkable mask: {path}")
    return image > 127


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    """Persist a boolean walkable mask as a 0/255 PNG."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255
    if not cv2.imwrite(str(destination), binary):
        raise IOError(f"unable to write walkable mask: {destination}")


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask to ``(height, width)`` using nearest-neighbour sampling."""
    height, width = shape
    source = np.asarray(mask).astype(np.uint8) * 255
    if source.shape[:2] == (height, width):
        return np.asarray(mask).astype(bool)
    resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 127
