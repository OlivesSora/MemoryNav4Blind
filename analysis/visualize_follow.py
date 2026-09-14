"""Visualize each follow trajectory against the reference (source) trajectory.

Reads the source/reference GPS log (``<route>/source/output_trimmed.jsonl``) and
every ``<route>/follow_output/*/output.jsonl``, matches each follow trajectory
back onto the reference route in a shared local metric frame, and renders
per-follow static figures + an offline HTML page plus a top-level overview.

The follow logs contain no frame/image data, so this tool is GPS-only: it reuses
``memory_nav.analysis.core`` for record loading, 1 Hz keypoint resampling and the
reference matcher, and reports cross-track / heading error as the tracking
metrics.

Usage:
    python3 -m memory_nav.analysis.visualize_follow \
        --reference memory_nav/routes/suishi-2/source/output_trimmed.jsonl \
        --follow-root memory_nav/routes/suishi-2/follow_output \
        --out memory_nav/analysis/test0910
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from memory_nav.analysis.core import (
    FollowMetric,
    Session,
    build_keypoints,
    load_records,
    match_memory_to_reference,
)
from memory_nav.trajectory.coordinate import LocalFrame

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_REFERENCE = "#1976d2"
COLOR_FOLLOW = "#ef6c00"
COLOR_GOOD = "#15803d"
COLOR_DEGRADED = "#d97706"
COLOR_LOST = "#dc2626"

GOOD_DISTANCE_M = 3.0
LOST_DISTANCE_M = 8.0


def classify_quality(cross_track: float | None) -> str:
    if cross_track is None or cross_track != cross_track:
        return "lost"
    magnitude = abs(cross_track)
    if magnitude <= GOOD_DISTANCE_M:
        return "good"
    if magnitude <= LOST_DISTANCE_M:
        return "degraded"
    return "lost"


def quality_color(quality: str) -> str:
    return {"good": COLOR_GOOD, "degraded": COLOR_DEGRADED, "lost": COLOR_LOST}.get(
        quality, COLOR_LOST
    )


def follow_metrics(metrics: list[FollowMetric]) -> list[FollowMetric]:
    for metric in metrics:
        metric.match_quality = classify_quality(metric.cross_track_error_m)
    return metrics


def _numeric(metrics: list[FollowMetric], key: str) -> list[float]:
    values: list[float] = []
    for metric in metrics:
        value = getattr(metric, key)
        if value is None or value != value:
            continue
        values.append(float(value))
    return values


def summarize(
    name: str,
    reference_keypoints,
    follow_keypoints,
    metrics: list[FollowMetric],
) -> dict[str, Any]:
    cross = [abs(v) for v in _numeric(metrics, "cross_track_error_m")]
    heading = [abs(v) for v in _numeric(metrics, "heading_error_deg")]
    quality = [m.match_quality for m in metrics]

    def _pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
        return ordered[index]

    def _ratio(label: str) -> float | None:
        if not quality:
            return None
        return sum(1 for q in quality if q == label) / len(quality)

    return {
        "follow_name": name,
        "reference_points": len(reference_keypoints),
        "reference_length_m": round(reference_keypoints[-1].s_m, 1)
        if reference_keypoints
        else 0.0,
        "follow_points": len(follow_keypoints),
        "follow_length_m": round(follow_keypoints[-1].s_m, 1) if follow_keypoints else 0.0,
        "cross_track_error_m": {
            "mean": round(statistics.fmean(cross), 2) if cross else None,
            "median": round(statistics.median(cross), 2) if cross else None,
            "p90": round(_pct(cross, 90), 2) if cross else None,
            "max": round(max(cross), 2) if cross else None,
        },
        "heading_error_deg": {
            "mean_abs": round(statistics.fmean(heading), 2) if heading else None,
        },
        "quality_ratio": {
            "good": round(_ratio("good"), 3) if quality else None,
            "degraded": round(_ratio("degraded"), 3) if quality else None,
            "lost": round(_ratio("lost"), 3) if quality else None,
        },
    }


def _write_jsonl(path: Path, metrics: list[FollowMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric.to_dict(), ensure_ascii=False) + "\n")


def _elapsed_s(metrics: list[FollowMetric]) -> list[float]:
    if not metrics:
        return []
    start = metrics[0].epoch_s
    return [m.epoch_s - start for m in metrics]


def render_route_overlay(
    reference_keypoints,
    follow_keypoints,
    metrics: list[FollowMetric],
    name: str,
    out: Path,
) -> None:
    ref_east = [p.east for p in reference_keypoints]
    ref_north = [p.north for p in reference_keypoints]
    follow_east = [p.east for p in follow_keypoints]
    follow_north = [p.north for p in follow_keypoints]

    cross = [abs(m.cross_track_error_m) for m in metrics]
    finite_cross = [c for c in cross if c == c]
    cap = max(finite_cross) if finite_cross else 0.0
    norm = Normalize(vmin=0.0, vmax=max(cap, 1.0))

    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(15, 7))

    ax_map.plot(ref_east, ref_north, "-", color=COLOR_REFERENCE, linewidth=2.5, label="参考轨迹 source")
    segments = [
        [(follow_east[i], follow_north[i]), (follow_east[i + 1], follow_north[i + 1])]
        for i in range(len(follow_east) - 1)
    ]
    values = [cross[i] if i < len(cross) else cross[-1] for i in range(len(segments))]
    line_collection = LineCollection(
        segments, cmap="RdYlGn_r", norm=norm, linewidth=3, label="跟随轨迹 follow"
    )
    line_collection.set_array(values)
    ax_map.add_collection(line_collection)
    if ref_east:
        ax_map.plot(ref_east[0], ref_north[0], "o", color=COLOR_REFERENCE, markersize=9)
        ax_map.plot(ref_east[-1], ref_north[-1], "s", color=COLOR_REFERENCE, markersize=9)
    if follow_east:
        ax_map.plot(follow_east[0], follow_north[0], "o", color=COLOR_FOLLOW, markersize=9)
        ax_map.plot(follow_east[-1], follow_north[-1], "s", color=COLOR_FOLLOW, markersize=9)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"{name}\n参考轨迹 vs 跟随轨迹（颜色=横向误差）")
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.grid(alpha=0.3)
    fig.colorbar(line_collection, ax=ax_map, label="横向误差 (m)")

    elapsed = _elapsed_s(metrics)
    colors = [quality_color(m.match_quality) for m in metrics]
    ax_err.scatter(elapsed, cross, c=colors, s=16, alpha=0.7)
    ax_err.set_title("横向误差随时间变化")
    ax_err.set_xlabel("时间 (s，相对跟随起点)")
    ax_err.set_ylabel("横向误差 (m)")
    ax_err.axhline(GOOD_DISTANCE_M, color=COLOR_GOOD, linestyle="--", alpha=0.6, label="good ≤ 3 m")
    ax_err.axhline(LOST_DISTANCE_M, color=COLOR_LOST, linestyle="--", alpha=0.6, label="lost ≥ 8 m")
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_follow_metrics(
    metrics: list[FollowMetric],
    name: str,
    out: Path,
) -> None:
    elapsed = _elapsed_s(metrics)
    cross = [abs(m.cross_track_error_m) for m in metrics]
    heading = [abs(m.heading_error_deg) for m in metrics]
    progress = [m.ref_s_m for m in metrics]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    colors = [quality_color(m.match_quality) for m in metrics]
    axes[0].scatter(elapsed, cross, c=colors, s=16, alpha=0.7)
    axes[0].axhline(GOOD_DISTANCE_M, color=COLOR_GOOD, linestyle="--", alpha=0.6)
    axes[0].axhline(LOST_DISTANCE_M, color=COLOR_LOST, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("横向误差 (m)")
    axes[0].set_title(f"{name} · 跟随指标（相对时间）")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(elapsed, heading, c=colors, s=16, alpha=0.7)
    axes[1].set_ylabel("朝向误差 (°)")
    axes[1].set_xlabel("时间 (s，相对跟随起点)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _svg_overlay(
    reference_keypoints,
    follow_keypoints,
    metrics: list[FollowMetric],
    width: int = 620,
    height: int = 620,
) -> str:
    pts = list(reference_keypoints) + list(follow_keypoints)
    lngs = [p.east for p in pts]
    lats = [p.north for p in pts]
    if not lngs:
        return "<svg></svg>"
    min_x, max_x = min(lngs), max(lngs)
    min_y, max_y = min(lats), max(lats)
    pad = max((max_x - min_x), (max_y - min_y), 1.0) * 0.04
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    def px(x: float, y: float) -> tuple[float, float]:
        return (
            (x - min_x) / (max_x - min_x) * width,
            (max_y - y) / (max_y - min_y) * height,
        )

    parts: list[str] = []
    parts.append(
        _svg_polyline(reference_keypoints, px, COLOR_REFERENCE, 2.5, dash="6 4")
    )
    for i in range(len(follow_keypoints) - 1):
        quality = metrics[i].match_quality if i < len(metrics) else "lost"
        color = quality_color(quality)
        a = px(follow_keypoints[i].east, follow_keypoints[i].north)
        b = px(follow_keypoints[i + 1].east, follow_keypoints[i + 1].north)
        parts.append(
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
            f'stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
    for label, kp, color in (
        ("ref_start", reference_keypoints[0] if reference_keypoints else None, COLOR_REFERENCE),
        ("follow_start", follow_keypoints[0] if follow_keypoints else None, COLOR_FOLLOW),
    ):
        if kp is None:
            continue
        x, y = px(kp.east, kp.north)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="1.5"><title>{label}</title></circle>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
    )


def _svg_polyline(keypoints, px, color: str, width: float, dash: str | None = None) -> str:
    if not keypoints:
        return ""
    d = "M " + " L ".join(
        f"{px(p.east, p.north)[0]:.1f},{px(p.east, p.north)[1]:.1f}" for p in keypoints
    )
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round"{dash_attr}/>'
    )


def render_html(
    name: str,
    summary: dict[str, Any],
    reference_keypoints,
    follow_keypoints,
    metrics: list[FollowMetric],
    out: Path,
) -> None:
    svg = _svg_overlay(reference_keypoints, follow_keypoints, metrics)

    metric_rows: list[str] = []
    for m in metrics:
        cross = m.cross_track_error_m
        cross_html = f"{abs(cross):.1f}" if cross == cross else "—"
        heading = m.heading_error_deg
        heading_html = f"{abs(heading):.1f}" if heading == heading else "—"
        quality = m.match_quality
        metric_rows.append(
            f"<tr><td>{m.index}</td><td>{m.time_iso[11:19]}</td>"
            f"<td class='{quality}'>{quality}</td>"
            f"<td>{cross_html}</td><td>{heading_html}</td>"
            f"<td>{m.ref_s_m:.1f}</td></tr>"
        )
    metric_table = (
        "<table><thead><tr><th>#</th><th>时间</th><th>匹配</th>"
        "<th>横向误差(m)</th><th>朝向误差(°)</th><th>参考进度(m)</th></tr></thead>"
        "<tbody>" + "".join(metric_rows) + "</tbody></table>"
    )

    cross = summary["cross_track_error_m"]
    quality = summary["quality_ratio"]
    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} · 跟随轨迹对比</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; color: #1f2937; background: #f3f4f6; }}
  header {{ background: #111827; color: #fff; padding: 12px 18px; }}
  header h1 {{ margin: 0; font-size: 18px; }}
  header .summary {{ color: #9ca3af; font-size: 13px; margin-top: 2px; }}
  main {{ display: flex; gap: 16px; padding: 16px; align-items: flex-start; flex-wrap: wrap; }}
  #mapwrap {{ flex: 1 1 540px; min-width: 420px; background: #fff; border-radius: 8px;
             padding: 10px; box-shadow: 0 1px 8px rgba(0,0,0,.12); }}
  #mapwrap svg {{ width: 100%; height: auto; background: #eef2f7; border-radius: 6px; }}
  #panel {{ flex: 1 1 340px; min-width: 300px; background: #fff; border-radius: 8px;
            padding: 14px; box-shadow: 0 1px 8px rgba(0,0,0,.12); }}
  .legend {{ margin: 6px 0 10px; font-size: 12px; color: #4b5563; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 4px 6px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  thead th {{ position: sticky; top: 0; background: #f9fafb; }}
  .good {{ color: #15803d; font-weight: 600; }}
  .degraded {{ color: #d97706; font-weight: 600; }}
  .lost {{ color: #dc2626; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>跟随轨迹对比 · {name}</h1>
  <div class="summary">参考轨迹 {summary['reference_points']} 点（{summary['reference_length_m']} m）；
  跟随 {summary['follow_points']} 点（{summary['follow_length_m']} m）。点击表格行可查看横向/朝向误差。</div>
</header>
<main>
  <div id="mapwrap">
    <div class="legend">
      <span><i class="swatch" style="background:{COLOR_REFERENCE}"></i> 参考轨迹（虚线）</span>
      <span><i class="swatch" style="background:{COLOR_FOLLOW}"></i> 跟随轨迹（颜色=误差）</span>
      <span><i class="swatch" style="background:{COLOR_GOOD}"></i> good ≤ 3m</span>
      <span><i class="swatch" style="background:{COLOR_DEGRADED}"></i> degraded ≤ 8m</span>
      <span><i class="swatch" style="background:{COLOR_LOST}"></i> lost &gt; 8m</span>
    </div>
    {svg}
  </div>
  <div id="panel">
    <table>
      <tr><th>平均横向误差</th><td>{cross['mean'] if cross['mean'] is not None else '—'} m</td></tr>
      <tr><th>中位横向误差</th><td>{cross['median'] if cross['median'] is not None else '—'} m</td></tr>
      <tr><th>90 分位横向误差</th><td>{cross['p90'] if cross['p90'] is not None else '—'} m</td></tr>
      <tr><th>最大横向误差</th><td>{cross['max'] if cross['max'] is not None else '—'} m</td></tr>
      <tr><th>平均朝向误差</th><td>{summary['heading_error_deg']['mean_abs'] if summary['heading_error_deg']['mean_abs'] is not None else '—'}°</td></tr>
      <tr><th>good / degraded / lost</th><td>{quality['good']} / {quality['degraded']} / {quality['lost']}</td></tr>
    </table>
  </div>
</main>
<section style="padding:0 16px 24px;">
  <h3 style="margin:8px 0">逐点跟随指标</h3>
  <div style="max-height:420px; overflow:auto; background:#fff; border-radius:8px; padding:8px; box-shadow:0 1px 8px rgba(0,0,0,.12);">
    {metric_table}
  </div>
</section>
</body>
</html>
"""
    out.write_text(page, encoding="utf-8")


def render_overview_png(reference_keypoints, follow_groups, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(
        [p.east for p in reference_keypoints],
        [p.north for p in reference_keypoints],
        "-", color=COLOR_REFERENCE, linewidth=3, label="参考轨迹 source",
    )
    cmap = plt.get_cmap("tab10")
    for i, (name, follow_keypoints) in enumerate(follow_groups):
        color = cmap(i % 10)
        ax.plot(
            [p.east for p in follow_keypoints],
            [p.north for p in follow_keypoints],
            "-", color=color, linewidth=1.6, label=name,
        )
    ax.set_aspect("equal")
    ax.set_title("全部跟随轨迹 vs 参考轨迹")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_index_html(summaries: list[dict[str, Any]], out: Path) -> None:
    rows: list[str] = []
    for s in summaries:
        cross = s["cross_track_error_m"]
        quality = s["quality_ratio"]
        rows.append(
            f"<tr><td><a href='{s['follow_name']}/follow_comparison.html'>{s['follow_name']}</a></td>"
            f"<td>{s['follow_points']}</td>"
            f"<td>{cross['mean'] if cross['mean'] is not None else '—'}</td>"
            f"<td>{cross['p90'] if cross['p90'] is not None else '—'}</td>"
            f"<td>{s['heading_error_deg']['mean_abs'] if s['heading_error_deg']['mean_abs'] is not None else '—'}</td>"
            f"<td>{quality['good']}</td><td>{quality['degraded']}</td><td>{quality['lost']}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>跟随轨迹</th><th>点数</th><th>平均横向误差(m)</th>"
        "<th>90分位横向误差(m)</th><th>平均朝向误差(°)</th>"
        "<th>good</th><th>degraded</th><th>lost</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>跟随轨迹跟踪情况总览</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; color: #1f2937; background: #f3f4f6; }}
  header {{ background: #111827; color: #fff; padding: 12px 18px; }}
  header h1 {{ margin: 0; font-size: 18px; }}
  header .summary {{ color: #9ca3af; font-size: 13px; margin-top: 2px; }}
  main {{ padding: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff;
          border-radius: 8px; box-shadow: 0 1px 8px rgba(0,0,0,.12); overflow: hidden; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  thead th {{ background: #f9fafb; position: sticky; top: 0; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 1px 8px rgba(0,0,0,.12); }}
</style>
</head>
<body>
<header>
  <h1>跟随轨迹跟踪情况总览 · suishi-2</h1>
  <div class="summary">每条跟随轨迹相对参考轨迹（source/output_trimmed.jsonl）的横向误差与朝向误差。点击名称进入详情页。</div>
</header>
<main>
  {table}
  <h3 style="margin:20px 0 8px">全部轨迹叠加</h3>
  <img src="overview.png" alt="overview">
</main>
</body>
</html>
"""
    out.write_text(page, encoding="utf-8")


def build_session(name: str, path: Path, frame: LocalFrame) -> Session:
    session = Session(
        name=name,
        records=load_records(path),
        frames_dir=None,
        fps=1.0,
        frame=frame,
    )
    session.keypoints = build_keypoints(session)
    return session


def render_follow(
    reference: Session,
    output_file: Path,
    out_root: Path,
) -> tuple[dict[str, Any], Session]:
    name = output_file.parent.name
    follow = build_session(name, output_file, reference.frame)
    metrics = follow_metrics(
        match_memory_to_reference(reference, follow, [], None)
    )
    summary = summarize(name, reference.keypoints, follow.keypoints, metrics)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    render_route_overlay(
        reference.keypoints, follow.keypoints, metrics, name, out_dir / "route_overlay.png"
    )
    render_follow_metrics(metrics, name, out_dir / "follow_metrics.png")
    render_html(
        name, summary, reference.keypoints, follow.keypoints, metrics, out_dir / "follow_comparison.html"
    )
    _write_jsonl(out_dir / "match_metrics.jsonl", metrics)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, follow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--follow-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    reference_file = args.reference
    follow_root = args.follow_root
    if not reference_file.is_file():
        parser.error(f"reference JSONL not found: {reference_file}")
    if not follow_root.is_dir():
        parser.error(f"follow root not found: {follow_root}")

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    reference_records = load_records(reference_file)
    origin = reference_records[0]
    frame = LocalFrame(origin.longitude, origin.latitude)
    reference = build_session("memory", reference_file, frame)

    output_files = sorted(follow_root.glob("*/output.jsonl"))
    if not output_files:
        parser.error(f"no output.jsonl under {follow_root}")

    summaries: list[dict[str, Any]] = []
    follow_groups: list[tuple[str, Any]] = []
    for output_file in output_files:
        summary, follow = render_follow(reference, output_file, out_root)
        summaries.append(summary)
        follow_groups.append((summary["follow_name"], follow.keypoints))
        print(f"rendered {summary['follow_name']}: "
              f"mean_cross={summary['cross_track_error_m']['mean']}m, "
              f"good={summary['quality_ratio']['good']}")

    render_overview_png(reference.keypoints, follow_groups, out_root / "overview.png")
    render_index_html(summaries, out_root / "index.html")
    (out_root / "overview_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"overview written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
