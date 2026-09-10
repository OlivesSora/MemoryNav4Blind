"""CLI for turning a recorded GPS JSONL stream into a reference route."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.models import GPSSample, IMUSample, VIOSample
from memory_nav.recording.session_writer import atomic_write_json, read_jsonl, validate_route_id
from memory_nav.recording.anchor_collector import AnchorCandidate, generate_anchor_candidates, save_anchors
from memory_nav.trajectory.processing import build_reference


def evaluate_quality(report, settings: dict) -> tuple[bool, list[str]]:
    reasons = []
    if report.accepted_samples < int(settings["minimum_gps_samples"]):
        reasons.append(f"accepted GPS samples {report.accepted_samples} < {int(settings['minimum_gps_samples'])}")
    if report.route_length_m < float(settings["minimum_route_length_m"]):
        reasons.append(f"route length {report.route_length_m:.2f}m < {float(settings['minimum_route_length_m']):.2f}m")
    if report.max_gap_s > float(settings["maximum_gps_gap_s"]):
        reasons.append(f"maximum GPS gap {report.max_gap_s:.2f}s > {float(settings['maximum_gps_gap_s']):.2f}s")
    rejected = report.rejected_out_of_order + report.rejected_duplicate + report.rejected_speed
    rejection_ratio = rejected / report.input_samples if report.input_samples else 1.0
    if rejection_ratio > float(settings["maximum_rejection_ratio"]):
        reasons.append(f"GPS rejection ratio {rejection_ratio:.3f} > {float(settings['maximum_rejection_ratio']):.3f}")
    return not reasons, reasons


def build_route(route_dir: Path, config: dict) -> None:
    gps_path = route_dir / "raw" / "gps.jsonl"
    samples = [GPSSample.from_dict(record) for record in read_jsonl(gps_path)]
    imu_path = route_dir / "raw" / "imu.jsonl"
    imu_samples = [IMUSample.from_dict(record) for record in read_jsonl(imu_path)] if imu_path.is_file() else []
    vio_path = route_dir / "raw" / "vio.jsonl"
    vio_samples = [VIOSample.from_dict(record) for record in read_jsonl(vio_path)] if vio_path.is_file() else []
    events_path = route_dir / "raw" / "events.jsonl"
    events = read_jsonl(events_path) if events_path.is_file() else []
    protected_times = [
        int(event["monotonic_ns"])
        for event in events
        if event.get("type") in ("anchor", "anchor_image") and event.get("monotonic_ns") is not None
    ]
    settings = config["trajectory"]
    points, report, frame = build_reference(
        samples,
        imu_samples=imu_samples,
        protected_monotonic_ns=protected_times,
        turn_protection_deg=float(config["anchors"]["turn_angle_deg"]),
        spacing_m=float(settings["resample_spacing_m"]),
        smoothing_window=int(settings["smoothing_window"]),
        max_speed_m_s=float(settings["max_speed_m_s"]),
        duplicate_distance_m=float(settings["duplicate_distance_m"]),
        vio_samples=vio_samples,
        vio_max_alignment_rms_m=float(config["vio"]["maximum_alignment_rms_m"]),
    )
    atomic_write_json(route_dir / "reference_trajectory.json", {
        "coordinate_system": "GCJ-02",
        "origin": {"longitude": frame.origin_longitude, "latitude": frame.origin_latitude},
        "points": [point.to_dict() for point in points],
    })
    quality = asdict(report)
    quality["rejection_ratio"] = (
        report.rejected_out_of_order + report.rejected_duplicate + report.rejected_speed
    ) / report.input_samples if report.input_samples else 1.0
    quality["ready"], quality["failure_reasons"] = evaluate_quality(report, config["quality"])
    atomic_write_json(route_dir / "quality_report.json", quality)
    anchors_path = route_dir / "anchors.json"
    if not anchors_path.exists():
        anchor_settings = config["anchors"]
        anchors = generate_anchor_candidates(
            points,
            max_spacing_m=float(anchor_settings["max_spacing_m"]),
            turn_angle_deg=float(anchor_settings["turn_angle_deg"]),
        )
        for event_index, event in enumerate(events):
            if event.get("type") != "anchor_image" or not event.get("clear") or not event.get("image_path"):
                continue
            try:
                east, north = frame.to_local(float(event["longitude"]), float(event["latitude"]))
            except (KeyError, TypeError, ValueError):
                continue
            nearest = min(points, key=lambda point: (point.east_m - east) ** 2 + (point.north_m - north) ** 2)
            anchors.append(AnchorCandidate(
                anchor_id=f"image_{event_index:04d}",
                point_index=nearest.index,
                s_m=nearest.s_m,
                kind=str(event.get("kind", "manual")),
                prompt_text="",
                confirmed=False,
                image_path=str(event["image_path"]),
            ))
        anchors.sort(key=lambda anchor: (anchor.s_m, anchor.image_path is None, anchor.anchor_id))
        save_anchors(anchors_path, anchors)
    manifest_path = route_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchor_records = json.loads(anchors_path.read_text(encoding="utf-8"))["anchors"]
    pending_anchor_count = sum(not anchor.get("confirmed", False) for anchor in anchor_records)
    # One-shot route generation is usable immediately; anchors remain editable
    # afterwards and visual verification stays opt-in during replay.
    if config["anchors"].get("auto_confirm", False) and pending_anchor_count:
        for anchor in anchor_records:
            anchor["confirmed"] = True
        atomic_write_json(anchors_path, {"anchors": anchor_records})
        pending_anchor_count = 0
    manifest["state"] = "ERROR" if not quality["ready"] else "REVIEW_REQUIRED" if pending_anchor_count else "READY"
    manifest["local_origin"] = {"longitude": frame.origin_longitude, "latitude": frame.origin_latitude}
    manifest["anchor_count"] = len(anchor_records)
    manifest["pending_anchor_count"] = pending_anchor_count
    atomic_write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    route_dir = Path(config["routes_dir"]) / validate_route_id(args.route_id)
    if not route_dir.is_dir():
        parser.error(f"route does not exist: {route_dir}")
    build_route(route_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
