"""Precompute CAT-Seg walkable masks for a set of frames.

Run this module with the ``catseg`` conda environment (which provides
detectron2), for example::

    ~/anaconda3/envs/catseg/bin/python -m memory_nav.segmentation.precompute_masks \
        --frames /path/to/frames --masks /path/to/walkable_masks

Each frame produces ``<stem>_walkable.png`` (255 = walkable) plus a
``manifest.json`` describing the model and walkable classes used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

DEFAULT_CATSEG_DIR = os.getenv(
    "CATSEG_DIR", "/home/wheeltec/projects/blind-nav-server/CAT-Seg"
)
DEFAULT_WALKABLE_NAMES = ("pavement", "road", "stairs")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def build_predictor(
    catseg_dir: Path,
    config_path: str,
    weights_path: str,
    device: str,
    extra_options: list | None = None,
):
    if str(catseg_dir) not in sys.path:
        sys.path.insert(0, str(catseg_dir))
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config

    from cat_seg import add_cat_seg_config

    config = catseg_dir / config_path
    weights = catseg_dir / weights_path
    if not config.is_file():
        raise FileNotFoundError(f"CAT-Seg config not found: {config}")
    if not weights.is_file():
        raise FileNotFoundError(f"CAT-Seg weights not found: {weights}")

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(config))
    options = ["MODEL.WEIGHTS", str(weights), "MODEL.DEVICE", device]
    if extra_options:
        options.extend(extra_options)
    cfg.merge_from_list(options)
    cfg.freeze()
    return DefaultPredictor(cfg)


def collect_frames(frames: Path) -> list[Path]:
    if frames.is_dir():
        paths = sorted(
            path
            for path in frames.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
    elif frames.is_file():
        paths = [frames]
    else:
        raise FileNotFoundError(f"frames path does not exist: {frames}")
    if not paths:
        raise FileNotFoundError(f"no supported images found in: {frames}")
    return paths


def load_walkable_ids(catseg_dir: Path, names: tuple[str, ...]) -> list[int]:
    labels_path = catseg_dir / "datasets" / "coco.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    missing = [name for name in names if name not in labels]
    if missing:
        raise ValueError(f"walkable classes not in dataset labels: {', '.join(missing)}")
    return [labels.index(name) for name in names]


def extract_mask(image: np.ndarray, predictor, walkable_ids: list[int]) -> np.ndarray:
    with torch.inference_mode():
        prediction = predictor(image)
    class_map = prediction["sem_seg"].argmax(dim=0).detach().cpu().numpy()
    return np.isin(class_map, walkable_ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="Frame image or directory")
    parser.add_argument("--masks", type=Path, required=True, help="Output mask directory")
    parser.add_argument("--catseg-dir", type=Path, default=Path(DEFAULT_CATSEG_DIR))
    parser.add_argument("--config", default="configs/vitb_384.yaml")
    parser.add_argument("--weights", default="model_base.pth")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--walkable-names", default=",".join(DEFAULT_WALKABLE_NAMES))
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing masks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.frames = args.frames.expanduser().resolve()
    args.masks = args.masks.expanduser().resolve()
    args.catseg_dir = args.catseg_dir.expanduser().resolve()
    walkable_names = tuple(
        name.strip() for name in args.walkable_names.split(",") if name.strip()
    )
    frames = collect_frames(args.frames)
    args.masks.mkdir(parents=True, exist_ok=True)
    walkable_ids = load_walkable_ids(args.catseg_dir, walkable_names)
    print(f"walkable_ids: {dict(zip(walkable_names, walkable_ids))}")
    # CAT-Seg configs reference dataset paths relative to the project root.
    os.chdir(args.catseg_dir)
    predictor = build_predictor(args.catseg_dir, args.config, args.weights, args.device)

    started = time.perf_counter()
    written = 0
    for frame_path in frames:
        destination = args.masks / f"{frame_path.stem}_walkable.png"
        if destination.is_file() and not args.overwrite:
            continue
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skipping unreadable frame: {frame_path}")
            continue
        mask = extract_mask(image, predictor, walkable_ids)
        cv2.imwrite(str(destination), mask.astype(np.uint8) * 255)
        written += 1

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "catseg_dir": str(args.catseg_dir),
        "config": args.config,
        "weights": args.weights,
        "device": args.device,
        "walkable_names": list(walkable_names),
        "walkable_ids": walkable_ids,
        "frame_count": len(frames),
        "written": written,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (args.masks / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {written} masks to {args.masks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
