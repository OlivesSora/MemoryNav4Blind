"""Global image embedding with the cached offline DINOv2 ViT-B/14 model.

DINOv2 is used to produce a compact, self-supervised global visual descriptor for
every keypoint frame.  These descriptors serve as the per-anchor ``视觉特征`` and
drive the visual-change detector ``d_vis`` used during adaptive segmentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub"
DINOV2_HUB_REPO = CACHE_DIR / "facebookresearch_dinov2_main"


class DinoV2Embedder:
    """Lazy loader for the offline DINOv2 ViT-B/14 model."""

    def __init__(self, repo_path: str | Path = DINOV2_HUB_REPO, model_name: str = "dinov2_vitb14"):
        self.repo_path = Path(repo_path)
        self.model_name = model_name
        self._model = None
        self._device = "cuda" if _torch_available() and _cuda_available() else "cpu"

    def _load(self):
        if self._model is None:
            import torch

            if not self.repo_path.is_dir():
                raise RuntimeError(f"cached DINOv2 hub repository missing: {self.repo_path}")
            self._model = torch.hub.load(
                str(self.repo_path), self.model_name, source="local", pretrained=True
            ).eval()
            self._model.to(self._device)
        return self._model

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a normalized 768-D CLS-token embedding for a BGR frame."""
        import cv2
        import torch
        from torchvision import transforms

        model = self._load()
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = transforms.ToTensor()(image)
        normalize = transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD)
        tensor = normalize(tensor).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = model.forward_features(tensor)["x_norm_clstoken"]
        vector = features.squeeze(0).float().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - DinoV2Embedder.cosine_similarity(a, b)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()