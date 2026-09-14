"""Offline validation of the clock-direction / walkable-segmentation mapping.

Given an ``output.jsonl`` follow log, a directory of frames and a directory
of precomputed walkable masks, this tool maps each tracked clock direction
onto the bottom-center semicircle, compares it with the walkable region and
writes per-frame metrics plus annotated images.

Example::

    python -m memory_nav.segmentation.validate_offline \
        --log /path/to/output.jsonl \
        --frames /path/to/frames \
        --masks /path/to/walkable_masks \
        --output /path/to/seg_validation
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import cv2

from memory_nav.config import load_config
from memory_nav.segmentation.avoidance import SegmentationAvoidance, SegmentationConfig
from memory_nav.segmentation.mask_loader import load_mask, resize_mask
from memory_nav.segmentation.visualize import draw_segmentation


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
CLOCK_PATTERN = re.compile(r"([0-9]{1,2})\s*点钟")


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in log: {path}")
    return records


def collect_frames(frames: Path) -> list[Path]:
    if frames.is_file():
        return [frames]
    paths = sorted(
        path for path in frames.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no frames found in: {frames}")
    return paths


def command_clock_from_record(record: dict) -> int | None:
    value = record.get("command_clock")
    if value is not None:
        try:
            hour = int(value) % 12
            return 12 if hour == 0 else hour
        except (TypeError, ValueError):
            pass
    angle_diff = record.get("angle_diff")
    if angle_diff is not None:
        try:
            hour = int(round(float(angle_diff) / 30.0)) % 12
            return 12 if hour == 0 else hour
        except (TypeError, ValueError):
            pass
    text = record.get("guide") or record.get("route_instruction") or ""
    match = CLOCK_PATTERN.search(str(text))
    if match:
        hour = int(match.group(1)) % 12
        return 12 if hour == 0 else hour
    if "直行" in str(text) or "向前" in str(text):
        return 12
    return None


def resolve_frame_by_number(frames: list[Path], number: int) -> Path | None:
    for path in frames:
        if path.stem == f"frame_{number}" or path.stem == f"frame_{number:06d}":
            return path
    return None


def resolve_frame_path(record: dict, frames: list[Path], frames_dir: Path | None) -> Path | None:
    image_path = record.get("image_path")
    if image_path:
        candidate = Path(image_path)
        if candidate.is_file():
            return candidate
        if frames_dir is not None:
            local = frames_dir / candidate.name
            if local.is_file():
                return local
    timestamp = record.get("camera_time_stamp")
    if timestamp is not None:
        try:
            found = resolve_frame_by_number(frames, int(timestamp))
        except (TypeError, ValueError):
            found = None
        if found is not None:
            return found
    return None


def sequential_pairs(records: list[dict], frames: list[Path]) -> list[tuple[dict, Path]]:
    if len(frames) == 1:
        return [(records[0], frames[0])]
    pairs: list[tuple[dict, Path]] = []
    last = len(records) - 1
    for index, frame in enumerate(frames):
        record_index = round(index * last / (len(frames) - 1))
        pairs.append((records[record_index], frame))
    return pairs


def build_pairs(
    records: list[dict], frames: list[Path], frames_dir: Path | None, association: str
) -> list[tuple[dict, Path]]:
    if association == "sequential":
        return sequential_pairs(records, frames)
    pairs: list[tuple[dict, Path]] = []
    for record in records:
        frame = resolve_frame_path(record, frames, frames_dir)
        if frame is not None:
            pairs.append((record, frame))
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="outdoor_nav output.jsonl")
    parser.add_argument("--frames", type=Path, required=True, help="Frame image or directory")
    parser.add_argument("--masks", type=Path, required=True, help="Precomputed walkable mask directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", help="MemoryNav config with a segmentation block")
    parser.add_argument(
        "--assoc",
        choices=("sequential", "path", "frame"),
        default="sequential",
        help="How to pair log records with frames",
    )
    parser.add_argument("--no-visualize", action="store_true", help="Skip annotated images")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_records(args.log)
    frames = collect_frames(args.frames)
    frames_dir = args.frames if args.frames.is_dir() else None
    args.output.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output / "vis"
    if not args.no_visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    seg_mapping = config.get("segmentation", {})
    if args.masks is not None:
        seg_mapping = {**seg_mapping, "mask_dir": str(args.masks)}
    seg_config = SegmentationConfig.from_mapping(seg_mapping)
    avoidance = SegmentationAvoidance(seg_config)

    pairs = build_pairs(records, frames, frames_dir, args.assoc)
    metrics_path = args.output / "seg_metrics.jsonl"
    summary = {
        "log": str(args.log),
        "frames": len(frames),
        "records": len(records),
        "association": args.assoc,
        "evaluated": 0,
        "consistent": 0,
        "fallback": 0,
        "out_of_scope": 0,
        "no_walkable": 0,
        "missing_mask": 0,
        "unresolved_command": 0,
        "consistency_rate_in_scope": None,
        "fallback_rate_in_scope": None,
    }

    with metrics_path.open("w", encoding="utf-8") as stream:
        for record, frame_path in pairs:
            mask_path = args.masks / f"{frame_path.stem}_walkable.png"
            if not mask_path.is_file():
                summary["missing_mask"] += 1
                continue
            command_clock = command_clock_from_record(record)
            if command_clock is None:
                summary["unresolved_command"] += 1
                continue
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            mask = load_mask(mask_path)
            if mask.shape[:2] != frame.shape[:2]:
                mask = resize_mask(mask, frame.shape[:2])
            result = avoidance.evaluate(command_clock, mask)
            summary["evaluated"] += 1
            if result.out_of_scope:
                summary["out_of_scope"] += 1
            elif result.consistent:
                summary["consistent"] += 1
            elif result.fallback_used:
                summary["fallback"] += 1
            else:
                summary["no_walkable"] += 1

            entry = {
                "log_time": record.get("log_time"),
                "frame": str(frame_path),
                "command_clock": command_clock,
                "guide": record.get("guide"),
                **asdict(result),
            }
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()

            if not args.no_visualize:
                annotated = draw_segmentation(frame, result, avoidance.mapper, mask)
                cv2.imwrite(str(vis_dir / f"{frame_path.stem}_seg.jpg"), annotated)

    in_scope = summary["evaluated"] - summary["out_of_scope"]
    if in_scope > 0:
        summary["consistency_rate_in_scope"] = summary["consistent"] / in_scope
        summary["fallback_rate_in_scope"] = summary["fallback"] / in_scope
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
