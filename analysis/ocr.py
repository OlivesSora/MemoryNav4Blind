"""Optional OCR text extraction from anchor images.

Uses EasyOCR (ch_sim + en) when its models are available.  Because the model
weights are downloaded on first use and may be unavailable on offline machines,
the engine degrades gracefully: if the model cannot be initialised, every image
simply yields an empty string.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OcrEngine:
    def __init__(self, languages: tuple[str, ...] = ("ch_sim", "en")):
        self.languages = list(languages)
        self._reader = None
        self._error: str | None = None

    def _load(self):
        if self._reader is None and self._error is None:
            try:
                import easyocr

                self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                self._error = str(exc)
        return self._reader

    def extract(self, frame_bgr: np.ndarray) -> str:
        """Return the recognised text in *frame_bgr*, or ``""`` if unavailable."""
        reader = self._load()
        if reader is None:
            return ""
        try:
            results = reader.readtext(frame_bgr, detail=0)
            return " ".join(str(text).strip() for text in results if str(text).strip())
        except Exception:
            return ""

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def error(self) -> str | None:
        return self._error