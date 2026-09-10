"""Replay timestamped local positions against a built reference route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.models import ReferencePoint
from memory_nav.recording.session_writer import read_jsonl, validate_route_id
from memory_nav.replay.deviation import DeviationMonitor
from memory_nav.replay.matcher import RouteMatcher
from memory_nav.trajectory.coordinate import LocalFrame


def write_svg(path: str | Path, points: list[ReferencePoint], results: list[dict], width: int = 900, height: int = 600) -> None:
    east = [point.east_m for point in points]
    north = [point.north_m for point in points]
    margin = 30.0
    span_east = max(max(east) - min(east), 1.0)
    span_north = max(max(north) - min(north), 1.0)
    scale = min((width - 2 * margin) / span_east, (height - 2 * margin) / span_north)

    def pixel(east_m: float, north_m: float) -> tuple[float, float]:
        return margin + (east_m - min(east)) * scale, height - margin - (north_m - min(north)) * scale

    route_pixels = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pixel(point.east_m, point.north_m) for point in points))
    colors = {"good": "#19a974", "degraded": "#ffb000", "lost": "#d62728"}
    circles = []
    for result in results:
        x, y = pixel(float(result["east_m"]), float(result["north_m"]))
        color = colors.get(result["match_quality"], "#555")
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<polyline points="{route_pixels}" fill="none" stroke="#2463eb" stroke-width="3"/>'
        + "".join(circles)
        + "</svg>\n"
    )
    Path(path).write_text(svg, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--input", required=True, help="JSONL with longitude, latitude and heading_deg")
    parser.add_argument("--output", help="output JSONL; defaults to stdout")
    parser.add_argument("--svg", help="optional SVG visualization path")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    route_dir = Path(config["routes_dir"]) / validate_route_id(args.route_id)
    route = json.loads((route_dir / "reference_trajectory.json").read_text(encoding="utf-8"))
    points = [ReferencePoint(**point) for point in route["points"]]
    frame = LocalFrame(route["origin"]["longitude"], route["origin"]["latitude"])
    matcher = RouteMatcher(points, **config["matching"])
    monitor = DeviationMonitor(**config["deviation"])
    output_lines = []
    results = []
    for sample in read_jsonl(args.input):
        east, north = frame.to_local(float(sample["longitude"]), float(sample["latitude"]))
        match = matcher.match(east, north, float(sample["heading_deg"]))
        state = monitor.update(match.cross_track_error_m, match.match_quality, sample.get("source", "rtk"))
        if match.match_quality == "good" and matcher.is_complete(float(config["anchors"]["arrival_distance_m"])):
            state = monitor.complete()
        result = {**match.to_dict(), "navigation_state": state.state.value, "pause_progress": state.pause_progress}
        results.append(result)
        output_lines.append(json.dumps(result, ensure_ascii=False))
    payload = "\n".join(output_lines) + ("\n" if output_lines else "")
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.svg:
        write_svg(args.svg, points, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
