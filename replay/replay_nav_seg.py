"""Online follow loop with walkable-segmentation avoidance.

This is the segmentation-aware counterpart of
``memory_nav.replay.replay_nav``.  It maps the tracked clock direction onto
the bottom-center semicircle of the current frame, compares it with a
walkable mask and, when the tracked direction is blocked, returns the
nearest walkable clock direction.

Two mask sources are supported:

* ``--seg-online``: a persistent CAT-Seg worker (``catseg`` conda env) is
  spawned automatically and queried over a Unix socket.  The newest mask is
  reused by the navigation loop, so inference never blocks it.
* ``--walkable-mask-dir``: precomputed mask PNGs, optionally replayed in
  lock-step with ``--simulate-frames``.

Visualization writes annotated frames to ``--seg-vis-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.config import PROJECT_ROOT
from memory_nav.recording.session_writer import validate_route_id
from memory_nav.replay.anchor_matcher import XFeatAnchorMatcher
from memory_nav.replay.replay_nav import (
    HardwareGatewayCamera,
    HardwareGatewayGPS,
    HardwareGatewayIMU,
    ReplayRunner,
    create_hardware_agent_client,
    validate_ready_route,
)
from memory_nav.segmentation.avoidance import SegmentationAvoidance, SegmentationConfig
from memory_nav.segmentation.guard import SegmentationGuard
from memory_nav.segmentation.online_provider import (
    DEFAULT_CATSEG_DIR,
    DEFAULT_WALKABLE_NAMES,
    DEFAULT_WORKER_PYTHON,
    CatSegWorkerProvider,
)
from memory_nav.segmentation.online_visualizer import SegmentationFrameSaver
from memory_nav.segmentation.providers import CachedMaskProvider, SequenceMaskProvider
from memory_nav.vio import VioSubscriber

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_NO_WALKABLE_TEXT = "前方未检测到可通行区域，请停止前进"
DEFAULT_ROTATE_TEXT = "无可行区域，请旋转一下"


def _sequence_frames(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    frames = sorted(
        item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise FileNotFoundError(f"no frames found in: {path}")
    return frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--visual-anchors", action="store_true")
    parser.add_argument("--vio-topic")
    parser.add_argument("--no-vio", action="store_true", help="Ignore VINS and use the hardware-gateway camera")
    parser.add_argument("--ros-image-topic")
    parser.add_argument("--follow-log", type=Path)
    parser.add_argument("--walkable-mask-dir", type=Path, help="Precomputed walkable mask directory")
    parser.add_argument("--simulate-frames", type=Path, help="Frame sequence for mask lock-step replay")
    parser.add_argument("--seg-cooldown", type=float, default=5.0)
    parser.add_argument("--seg-online", action="store_true", help="Run live CAT-Seg in a worker subprocess")
    parser.add_argument("--seg-radius-ratio", type=float, help="Online judgment semicircle radius ratio (overrides config)")
    parser.add_argument("--seg-worker-python", help="catseg conda python used to launch the worker")
    parser.add_argument("--seg-socket", help="Unix socket path for the worker")
    parser.add_argument("--seg-catseg-dir", help="CAT-Seg project directory")
    parser.add_argument("--seg-config", help="CAT-Seg config (default configs/vitb_384.yaml)")
    parser.add_argument("--seg-weights", help="CAT-Seg weights (default model_base.pth)")
    parser.add_argument("--seg-device", choices=("cuda", "cpu"), help="Worker inference device")
    parser.add_argument("--seg-walkable-names", help="Comma-separated walkable class names")
    parser.add_argument("--seg-input-scale", type=float, help="Downscale before inference (0, 1]")
    parser.add_argument("--seg-min-size-test", type=int, help="Override INPUT.MIN_SIZE_TEST")
    parser.add_argument("--seg-max-hz", type=float, help="Maximum worker inference rate")
    parser.add_argument("--seg-vis-dir", type=Path, help="Directory for annotated frames")
    parser.add_argument("--seg-vis-interval", type=float, default=1.0, help="Seconds between saved frames")
    parser.add_argument("--seg-show", action="store_true", help="Also show a live window if a display exists")
    return parser.parse_args(argv)


def _online_provider(args: argparse.Namespace, mapping: dict) -> CatSegWorkerProvider:
    online = dict(mapping.get("online", {}))
    walkable = args.seg_walkable_names or online.get("walkable_names") or ",".join(DEFAULT_WALKABLE_NAMES)
    if isinstance(walkable, (list, tuple)):
        walkable = ",".join(str(name) for name in walkable)
    return CatSegWorkerProvider(
        worker_python=args.seg_worker_python or online.get("worker_python") or DEFAULT_WORKER_PYTHON,
        socket_path=args.seg_socket or online.get("socket_path"),
        catseg_dir=args.seg_catseg_dir or mapping.get("catseg_project_dir") or DEFAULT_CATSEG_DIR,
        config=args.seg_config or mapping.get("catseg_config") or "configs/vitb_384.yaml",
        weights=args.seg_weights or mapping.get("catseg_weights") or "model_base.pth",
        device=args.seg_device or online.get("device") or mapping.get("catseg_device") or "cuda",
        walkable_names=[name.strip() for name in str(walkable).split(",") if name.strip()],
        input_scale=args.seg_input_scale if args.seg_input_scale is not None else float(online.get("input_scale", 0.5)),
        min_size_test=args.seg_min_size_test if args.seg_min_size_test is not None else online.get("min_size_test"),
        max_hz=args.seg_max_hz if args.seg_max_hz is not None else float(online.get("max_hz", 1.0)),
        min_free_mb=int(online.get("min_free_mb", 3000)),
    )


def build_guard(args: argparse.Namespace, config: dict) -> SegmentationGuard | None:
    mapping = dict(config.get("segmentation", {}))
    messages = dict(mapping.get("messages", {}))
    guard_text = {
        "no_walkable_text": messages.get("no_walkable", DEFAULT_NO_WALKABLE_TEXT),
        "rotate_text": messages.get("no_walkable_rotate", DEFAULT_ROTATE_TEXT),
    }
    if args.seg_online:
        online = dict(mapping.get("online", {}))
        if args.seg_radius_ratio is not None:
            mapping["radius_ratio"] = args.seg_radius_ratio
        elif online.get("radius_ratio") is not None:
            mapping["radius_ratio"] = float(online["radius_ratio"])
        avoidance = SegmentationAvoidance(SegmentationConfig.from_mapping(mapping))
        provider = _online_provider(args, mapping)
        vis_dir = args.seg_vis_dir or online.get("vis_dir")
        visualizer = None
        if vis_dir:
            vis_dir = Path(vis_dir)
            if not vis_dir.is_absolute():
                vis_dir = PROJECT_ROOT / vis_dir
            interval = args.seg_vis_interval
            if interval == 1.0 and "vis_interval_s" in online:
                interval = float(online["vis_interval_s"])
            visualizer = SegmentationFrameSaver(vis_dir, interval_s=interval, show=args.seg_show)
        return SegmentationGuard(
            avoidance,
            provider,
            cooldown_s=args.seg_cooldown,
            visualizer=visualizer,
            **guard_text,
        )
    if args.walkable_mask_dir is not None:
        mapping["mask_dir"] = str(args.walkable_mask_dir)
    mask_dir = mapping.get("mask_dir")
    if not mask_dir:
        return None
    avoidance = SegmentationAvoidance(SegmentationConfig.from_mapping(mapping))
    if args.simulate_frames is not None:
        provider = SequenceMaskProvider(mask_dir, _sequence_frames(args.simulate_frames))
    else:
        provider = CachedMaskProvider(mask_dir)
    return SegmentationGuard(avoidance, provider, cooldown_s=args.seg_cooldown, **guard_text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    config = load_config(args.config)
    route_dir = Path(config["routes_dir"]) / validate_route_id(args.route_id)
    try:
        validate_ready_route(route_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc))
    guard = build_guard(args, config)
    if guard is None:
        LOGGER.warning("no segmentation source configured; running without avoidance")
    logging.basicConfig(level=logging.INFO)
    follow_log_path = args.follow_log or (
        route_dir / "follow_output" / f"follow-{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "output.jsonl"
    )
    hardware_agent = create_hardware_agent_client()
    gps = HardwareGatewayGPS(hardware_agent)
    imu = HardwareGatewayIMU(hardware_agent)
    speaker = None
    if args.voice:
        speaker = lambda text: asyncio.run(
            hardware_agent.play_text(text, need_feedback=False)
        )
    camera = visual_matcher = vio = None
    needs_image = args.visual_anchors or args.seg_online
    if args.vio_topic and not args.no_vio:
        image_topic = args.ros_image_topic if needs_image else None
        vio = VioSubscriber(args.vio_topic, image_topic=image_topic)
        vio.start()
        if needs_image and image_topic:
            camera = vio
    if needs_image and camera is None:
        camera = HardwareGatewayCamera(hardware_agent)
    if args.visual_anchors:
        visual_matcher = XFeatAnchorMatcher(PROJECT_ROOT)
    runner = ReplayRunner(
        route_dir,
        config,
        gps=gps,
        imu=imu,
        speaker=speaker,
        camera=camera,
        visual_matcher=visual_matcher,
        vio=vio,
        voice_directions=args.voice,
        follow_log_path=follow_log_path,
        segmentation_guard=guard,
    )
    LOGGER.info("recording follow GPS to %s", follow_log_path)
    signal.signal(signal.SIGINT, lambda *_: setattr(runner, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(runner, "running", False))
    diagnostics_provider = getattr(guard, "provider", None) if guard is not None else None
    next_diagnostics_s = time.monotonic() + 5.0
    try:
        while runner.running:
            print(json.dumps(runner.step(), ensure_ascii=False), flush=True)
            now = time.monotonic()
            if diagnostics_provider is not None and now >= next_diagnostics_s:
                get_diagnostics = getattr(diagnostics_provider, "diagnostics", None)
                if get_diagnostics is not None:
                    LOGGER.info("segmentation provider: %s", get_diagnostics())
                next_diagnostics_s = now + 5.0
            time.sleep(args.interval)
    finally:
        runner.close()
        if guard is not None:
            guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

