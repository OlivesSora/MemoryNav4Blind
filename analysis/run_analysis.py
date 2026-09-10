"""CLI orchestrator for the offline memory-vs-test anchor analysis.

Reads the recorded GPS ``output.jsonl`` and frame directories for a memory
(reference) trajectory and a test trajectory, performs adaptive anchor
segmentation, generates 1 Hz clock-direction commands, computes test-vs-memory
follow metrics, and writes JSONL outputs plus visualizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_nav.analysis.core import (
    AnalysisConfig,
    Session,
    enrich_test_anchors_with_reference,
    load_records,
    match_memory_to_reference,
    parse_frame_dir_start,
    read_frame,
    run_session,
)
from memory_nav.analysis.embedding import DinoV2Embedder
from memory_nav.analysis.text_extract import VisionTextExtractor


class _NoEmbedding:
    """Explicitly disable DINO while retaining image and route visualizations."""

    def embed(self, _frame):
        return None


def enrich_anchors_with_text(session: Session, extractor: VisionTextExtractor) -> None:
    """Fill each anchor's ``image_text`` from the vision-language model."""
    for anchor in session.anchors:
        frame = read_frame(session, anchor.frame_index)
        if frame is not None:
            anchor.image_text = extractor.extract(frame)


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-gps", required=True, help="memory trajectory output.jsonl")
    parser.add_argument("--test-gps", required=True, help="test trajectory output.jsonl")
    parser.add_argument("--memory-frames", required=True, help="memory frames directory")
    parser.add_argument("--test-frames", required=True, help="test frames directory")
    parser.add_argument("--fps", type=float, default=None, help="frame rate for frame-GPS alignment; default auto-inferred from frame count and GPS duration")
    parser.add_argument("--tau-theta", type=float, default=30.0, help="heading-change anchor threshold (deg)")
    parser.add_argument("--tau-vis", type=float, default=0.15, help="visual-change anchor threshold (DINOv2 cosine distance)")
    parser.add_argument("--tau-s", type=float, default=20.0, help="spacing anchor threshold (m)")
    parser.add_argument("--visual-window", type=float, default=2.0, help="visual-change window (s)")
    parser.add_argument("--lookahead", type=float, default=8.0, help="distance (m) ahead on the reference to guide toward (next point)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--map", help="optional navigation_map.html basemap for the interactive page")
    parser.add_argument("--skip-vis", action="store_true", help="skip matplotlib/HTML visualizations")
    parser.add_argument("--skip-dino", action="store_true", help="skip DINOv2 inference; keep route/frame visualizations")
    parser.add_argument("--vision-model", default="Qwen/Qwen3-VL-8B-Instruct", help="vision-language model for text extraction")
    parser.add_argument("--api-key-file", default=None, help="JSON array file of ModelScope API keys (default: L4 config copy/api_keys.json)")
    args = parser.parse_args(argv)

    cfg = AnalysisConfig(
        fps=args.fps,
        tau_theta_deg=args.tau_theta,
        tau_vis=args.tau_vis,
        tau_s_m=args.tau_s,
        visual_window_s=args.visual_window,
        lookahead_m=args.lookahead,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    memory = Session(
        name="memory",
        records=load_records(args.memory_gps),
        frames_dir=Path(args.memory_frames),
        frame_start_epoch=parse_frame_dir_start(Path(args.memory_frames).name),
    )
    test = Session(
        name="test",
        records=load_records(args.test_gps),
        frames_dir=Path(args.test_frames),
        frame_start_epoch=parse_frame_dir_start(Path(args.test_frames).name),
    )

    embedder = _NoEmbedding() if args.skip_dino else DinoV2Embedder()
    run_session(memory, cfg, embedder, out)
    run_session(test, cfg, embedder, out)

    embed_cache_memory = _make_cache(memory, embedder)
    embed_cache_test = _make_cache(test, embedder)
    metrics = match_memory_to_reference(memory, test, memory.anchors, embed_cache_test, lookahead_m=cfg.lookahead_m)

    enrich_test_anchors_with_reference(test.anchors, metrics)
    extractor = VisionTextExtractor(
        api_key_file=args.api_key_file,
        model=args.vision_model,
    )
    enrich_anchors_with_text(memory, extractor)
    enrich_anchors_with_text(test, extractor)

    _write_jsonl(out / "memory_anchors.jsonl", memory.anchors)
    _write_jsonl(out / "test_anchors.jsonl", test.anchors)
    _write_jsonl(out / "memory_commands.jsonl", memory.commands)
    _write_jsonl(out / "test_commands.jsonl", test.commands)
    _write_jsonl(out / "match_metrics.jsonl", metrics)

    summary = {
        "config": cfg.__dict__,
        "text_extraction": {
            "available": extractor.available,
            "error": extractor.error,
            "model": args.vision_model,
        },
        "memory": _session_summary(memory),
        "test": _session_summary(test),
        "follow": {
            "mean_cross_track_error_m": _mean([abs(m.cross_track_error_m) for m in metrics]),
            "mean_heading_error_deg": _mean([abs(m.heading_error_deg) for m in metrics]),
            "mean_visual_sim": _mean([m.visual_sim for m in metrics if m.visual_sim is not None]),
            "good_quality_ratio": _ratio([m.match_quality == "good" for m in metrics]),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_vis:
        from memory_nav.analysis import vis_matplotlib

        vis_matplotlib.render(memory, test, metrics, out)

        from memory_nav.analysis import vis_html

        vis_html.render(memory, test, metrics, out, args.map)
        vis_html.render_session_page(memory, out)
        vis_html.render_session_page(test, out, reference=memory, metrics=[m.to_dict() for m in metrics])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"outputs written to {out}")
    return 0


def _make_cache(session: Session, embedder: DinoV2Embedder):
    from memory_nav.analysis.core import EmbeddingCache
    return EmbeddingCache(session, embedder)


def _session_summary(session: Session) -> dict:
    return {
        "records": len(session.records),
        "keypoints": len(session.keypoints),
        "anchors": len(session.anchors),
        "route_length_m": session.keypoints[-1].s_m if session.keypoints else 0.0,
        "anchor_reasons": _reason_counts(session.anchors),
        "alignment": {
            "source": session.alignment_source,
            "calibration_points": len(session.frame_calibration) if session.frame_calibration else 0,
            "fps": round(session.fps, 3) if session.fps else None,
            "frame_idx_min": session.frame_idx_min,
            "frame_idx_max": session.frame_idx_max,
            "n_clamped_last": session.n_clamped_last,
            "n_duplicate_frames": session.n_duplicate_frames,
        },
    }


def _reason_counts(anchors) -> dict[str, int]:
    counts: dict[str, int] = {}
    for anchor in anchors:
        for part in anchor.reason.split("+"):
            counts[part] = counts.get(part, 0) + 1
    return counts


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and v == v]
    if not values:
        return None
    return sum(values) / len(values)


def _ratio(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


if __name__ == "__main__":
    raise SystemExit(main())
