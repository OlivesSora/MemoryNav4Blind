"""Persistent CAT-Seg walkable-mask service for online navigation.

Run this module with the ``catseg`` conda environment (detectron2).  It
loads the CAT-Seg predictor once and serves binary BGR frames over a Unix
domain socket, returning a walkable mask (255 = walkable).  The online
provider owns the process and communicates with it via
:mod:`memory_nav.segmentation.protocol`.

Example::

    ~/anaconda3/envs/catseg/bin/python -m memory_nav.segmentation.catseg_worker \
        --socket /tmp/memory_nav_catseg.sock --device cuda --input-scale 0.5

For Jetson deployments, ``--backend tensorrt`` uses the exported FP16 engines
and avoids importing PyTorch or Detectron2 in the worker.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CATSEG_DIR = os.getenv(
    "CATSEG_DIR", "/home/wheeltec/projects/blind-nav-server/CAT-Seg"
)
DEFAULT_WALKABLE_NAMES = "pavement,road,stairs"
DEFAULT_TRT_EXPORT_DIR = Path(DEFAULT_CATSEG_DIR) / "export" / "catseg_stages"
DEFAULT_TRT_CLIP_ENGINE = DEFAULT_TRT_EXPORT_DIR / "catseg_clip_stage_fp16.engine"
DEFAULT_TRT_AGGREGATOR_ENGINE = (
    DEFAULT_TRT_EXPORT_DIR / "catseg_aggregator_stage_fp16.engine"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="Unix domain socket path")
    parser.add_argument(
        "--backend", default="pytorch", choices=("pytorch", "tensorrt")
    )
    parser.add_argument("--catseg-dir", type=Path, default=Path(DEFAULT_CATSEG_DIR))
    parser.add_argument("--config", default="configs/vitb_384.yaml")
    parser.add_argument("--weights", default="model_base.pth")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--walkable-names", default=DEFAULT_WALKABLE_NAMES)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Downscale frames before inference (0 < scale <= 1)",
    )
    parser.add_argument(
        "--min-size-test",
        type=int,
        default=None,
        help="Override INPUT.MIN_SIZE_TEST to reduce GPU memory",
    )
    parser.add_argument(
        "--trt-clip-engine", type=Path, default=DEFAULT_TRT_CLIP_ENGINE
    )
    parser.add_argument(
        "--trt-aggregator-engine",
        type=Path,
        default=DEFAULT_TRT_AGGREGATOR_ENGINE,
    )
    parser.add_argument(
        "--trt-python-path", default="/usr/lib/python3.10/dist-packages"
    )
    parser.add_argument("--trt-warmup", type=int, default=1)
    return parser.parse_args(argv)


def _serve_connection(
    connection, infer_mask, default_scale: float, allow_input_scale: bool = True
) -> None:
    from memory_nav.segmentation.protocol import recv_message, send_message

    while True:
        try:
            message = recv_message(connection)
        except (ConnectionError, OSError, ValueError):
            return
        if message is None or message.get("type") == "shutdown":
            return
        if message.get("type") != "frame":
            send_message(connection, {"type": "error", "error": f"unknown request: {message.get('type')}"})
            continue
        image = message.get("image")
        if image is None or not isinstance(image, np.ndarray):
            send_message(connection, {"type": "error", "error": "missing frame"})
            continue
        requested_scale = float(message.get("scale", default_scale))
        frame_id = str(message.get("frame_id", ""))
        captured_at_s = message.get("captured_at_s")
        scale = requested_scale if allow_input_scale else 1.0
        started = time.perf_counter()
        try:
            processed = image
            if 0.0 < scale < 1.0:
                processed = cv2.resize(
                    image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            mask = infer_mask(processed)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            send_message(
                connection,
                {
                    "type": "mask",
                    "frame_id": frame_id,
                    "captured_at_s": captured_at_s,
                    "mask": mask.astype(np.uint8) * 255,
                    "elapsed_s": time.perf_counter() - started,
                    "scale": scale,
                },
            )
        except Exception as exc:  # Keep the service alive across per-frame errors.
            send_message(connection, {"type": "error", "error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.catseg_dir = args.catseg_dir.expanduser().resolve()
    socket_path = Path(args.socket).expanduser()
    if socket_path.exists():
        socket_path.unlink()

    walkable_names = tuple(
        name.strip() for name in args.walkable_names.split(",") if name.strip()
    )
    close_predictor = None
    if args.backend == "tensorrt":
        if args.device != "cuda":
            raise ValueError("TensorRT backend requires --device cuda")
        from memory_nav.segmentation.catseg_trt import CatSegTensorRTPredictor

        predictor = CatSegTensorRTPredictor(
            clip_engine=args.trt_clip_engine,
            aggregator_engine=args.trt_aggregator_engine,
            labels_path=args.catseg_dir / "datasets" / "coco.json",
            walkable_names=walkable_names,
            tensorrt_python_path=args.trt_python_path,
            warmup=args.trt_warmup,
        )
        walkable_ids = predictor.walkable_ids
        infer_mask = predictor.predict
        close_predictor = predictor.close
        allow_input_scale = False
    else:
        from memory_nav.segmentation.precompute_masks import (
            build_predictor,
            extract_mask,
            load_walkable_ids,
        )

        walkable_ids = load_walkable_ids(args.catseg_dir, walkable_names)
        os.chdir(args.catseg_dir)
        extra_options = None
        if args.min_size_test is not None:
            extra_options = ["INPUT.MIN_SIZE_TEST", int(args.min_size_test)]
        predictor = build_predictor(
            args.catseg_dir, args.config, args.weights, args.device, extra_options
        )

        def infer_mask(image):
            return extract_mask(image, predictor, walkable_ids)

        allow_input_scale = True

    print(
        f"[catseg-worker] backend={args.backend} "
        f"walkable_ids={dict(zip(walkable_names, walkable_ids))}",
        flush=True,
    )
    print(f"[catseg-worker] ready on {socket_path}", flush=True)

    stop = {"flag": False}

    def _handle_signal(*_args) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)
        server.settimeout(0.5)
        while not stop["flag"]:
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                _serve_connection(
                    connection,
                    infer_mask,
                    args.input_scale,
                    allow_input_scale=allow_input_scale,
                )
    finally:
        server.close()
        if close_predictor is not None:
            close_predictor()
        try:
            socket_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
