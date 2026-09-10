"""Render a recorded reference trajectory (and its anchors) to a PNG.

Reads ``<routes_dir>/<route_id>/reference_trajectory.json`` and, when present,
``anchors.json``, then writes ``<route_dir>/reference_trajectory.png``.

Usage:
    python -m memory_nav.analysis.plot_reference --route-id ID [--config FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memory_nav.config import load_config  # noqa: E402
from memory_nav.recording.session_writer import validate_route_id  # noqa: E402

_KIND_MARKER = {
    "start": ("o", "green"),
    "turn": ("^", "red"),
    "interval": ("s", "orange"),
    "end": ("*", "black"),
    "image": ("D", "purple"),
}


def plot_route(route_dir: Path, out_path: Path) -> None:
    ref = json.loads((route_dir / "reference_trajectory.json").read_text(encoding="utf-8"))
    points = ref["points"]
    lons = [float(p["longitude"]) for p in points]
    lats = [float(p["latitude"]) for p in points]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(lons, lats, "-", color="#1f77b4", linewidth=2, label="reference")

    anchors_path = route_dir / "anchors.json"
    if anchors_path.is_file():
        anchors = json.loads(anchors_path.read_text(encoding="utf-8")).get("anchors", [])
        by_index = {int(p["index"]): p for p in points}
        for anchor in anchors:
            point = by_index.get(anchor.get("point_index"))
            if point is None:
                continue
            kind = anchor.get("kind", "image")
            marker, color = _KIND_MARKER.get(kind, _KIND_MARKER["image"])
            ax.plot(float(point["longitude"]), float(point["latitude"]), marker,
                    color=color, markersize=9, label=kind if kind not in ax.get_legend_handles_labels()[1] else None)
            ax.annotate(anchor.get("anchor_id", ""), (float(point["longitude"]), float(point["latitude"])),
                        textcoords="offset points", xytext=(6, 6), fontsize=7, color=color)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Reference trajectory: {route_dir.name}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--out", help="override output PNG path")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    route_dir = Path(config["routes_dir"]) / validate_route_id(args.route_id)
    required = route_dir / "reference_trajectory.json"
    if not required.is_file():
        parser.error(f"route has no reference_trajectory.json: {required}")
    out = Path(args.out) if args.out else route_dir / "reference_trajectory.png"
    plot_route(route_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
