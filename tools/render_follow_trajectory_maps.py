"""Render reference-vs-follow trajectory maps for a MemoryNav route.

The JSONL logs store ``current_pos`` as ``[longitude, latitude]``.  This
tool reads the reference log and every ``follow_output/*/output.jsonl`` file,
then writes a Leaflet HTML page into each follow-output directory.

Example:
    python3 -m memory_nav.tools.render_follow_trajectory_maps \\
        --route-dir memory_nav/routes/suishi-2
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAP_NAME = "trajectory_comparison.html"
GAODE_TILES = (
    "https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1"
    "&style=8&x={x}&y={y}&z={z}"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting the physical line for malformed input."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def current_points(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep valid current-position points and the fields useful in a popup."""
    points: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        position = row.get("current_pos")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        try:
            longitude, latitude = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            continue
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            continue
        points.append(
            {
                "index": index,
                "lng": longitude,
                "lat": latitude,
                "time": row.get("log_time"),
                "guide": row.get("guide"),
                "route_instruction": row.get("route_instruction"),
                "gps_action": row.get("gps_action"),
                "cur_bearing": row.get("cur_bearing"),
                "distance": row.get("distance"),
                "cross_track_error": row.get("cross_track_error"),
            }
        )
    return points


def haversine_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    radius_m = 6_371_000.0
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    d_lat = lat2 - lat1
    d_lng = math.radians(b["lng"] - a["lng"])
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(h))


def path_length_m(points: list[dict[str, Any]]) -> float:
    return sum(haversine_m(a, b) for a, b in zip(points, points[1:]))


def mean_numeric(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def json_for_html(value: Any) -> str:
    """Embed JSON safely even if an instruction contains a closing script tag."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_html(
    *,
    route_name: str,
    output_name: str,
    reference: list[dict[str, Any]],
    follow: list[dict[str, Any]],
    follow_row_count: int,
) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    summary = {
        "reference_points": len(reference),
        "reference_length_m": round(path_length_m(reference), 1),
        "follow_rows": follow_row_count,
        "follow_points": len(follow),
        "follow_length_m": round(path_length_m(follow), 1),
        "mean_cross_track_error_m": (
            round(mean_numeric(follow, "cross_track_error"), 2)
            if mean_numeric(follow, "cross_track_error") is not None
            else None
        ),
    }
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{route_name} · {output_name} · 轨迹对比</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body, #map {{ height: 100%; margin: 0; }}
    body {{ color: #172033; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
    .panel {{ position: fixed; z-index: 1000; top: 16px; left: 16px; width: min(360px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: auto; padding: 14px 16px; border-radius: 10px; background: rgba(255,255,255,.94); box-shadow: 0 2px 16px rgba(0,0,0,.20); }}
    h1 {{ margin: 0 0 6px; font-size: 17px; }}
    .subtle {{ color: #586174; font-size: 12px; line-height: 1.45; word-break: break-all; }}
    .legend {{ display: flex; gap: 13px; margin: 12px 0 8px; font-size: 13px; flex-wrap: wrap; }}
    .swatch {{ display: inline-block; width: 22px; height: 4px; margin: 0 5px 3px 0; vertical-align: middle; border-radius: 2px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    td {{ padding: 5px 0; border-top: 1px solid #e7eaf0; }} td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .notice {{ margin-top: 10px; padding: 8px; border-radius: 6px; color: #7a3d00; background: #fff3cd; font-size: 13px; line-height: 1.45; }}
    .leaflet-popup-content {{ min-width: 220px; line-height: 1.45; }}
    .popup-title {{ font-weight: 650; margin-bottom: 5px; }}
    .popup-row {{ margin: 2px 0; }}
    .leaflet-control-layers {{ font-size: 13px; }}
    @media (max-width: 600px) {{ .panel {{ top: 8px; left: 8px; width: calc(100vw - 16px); max-height: 38vh; }} }}
  </style>
</head>
<body>
  <div id="map"></div>
  <aside class="panel">
    <h1>记忆轨迹与跟随轨迹对比</h1>
    <div class="subtle">路线：{route_name}<br>输出：{output_name}<br>生成时间：{generated_at}</div>
    <div class="legend"><span><i class="swatch" style="background:#1976d2"></i>参考轨迹</span><span><i class="swatch" style="background:#ef6c00"></i>记忆输出</span></div>
    <table id="summary"></table>
    <div id="notice"></div>
    <div class="subtle" style="margin-top:9px">点击起终点或轨迹采样点可查看时间、指令和偏差。朝向箭头依据 <code>cur_bearing</code> 绘制（北为 0°、顺时针）；坐标使用 <code>current_pos</code>。</div>
  </aside>
  <script>
    const reference = {json_for_html(reference)};
    const follow = {json_for_html(follow)};
    const summary = {json_for_html(summary)};
    const map = L.map('map', {{ zoomControl: true, preferCanvas: true, maxZoom: 22 }});
    const gaode = L.tileLayer('{GAODE_TILES}', {{ maxZoom: 22, maxNativeZoom: 18, attribution: '高德地图' }}).addTo(map);
    const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 22, maxNativeZoom: 19, attribution: '© OpenStreetMap contributors' }});
    const refLayer = L.layerGroup().addTo(map);
    const followLayer = L.layerGroup().addTo(map);
    const markerLayer = L.layerGroup().addTo(map);
    const bearingLayer = L.layerGroup().addTo(map);
    L.control.layers({{ '高德地图': gaode, 'OpenStreetMap': osm }}, {{ '参考轨迹': refLayer, '记忆输出轨迹': followLayer, '起终点与采样点': markerLayer, 'cur_bearing 朝向': bearingLayer }}, {{ collapsed: false }}).addTo(map);

    function latLngs(points) {{ return points.map(p => [p.lat, p.lng]); }}
    function escapeHtml(value) {{ return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    function formatMeters(value) {{ return value == null ? '—' : `${{Number(value).toFixed(1)}} m`; }}
    function hasBearing(point) {{ return point.cur_bearing !== null && point.cur_bearing !== '' && Number.isFinite(Number(point.cur_bearing)); }}
    function popup(point, label) {{
      return `<div class="popup-title">${{escapeHtml(label)}}（第 ${{point.index + 1}} 条）</div>` +
        `<div class="popup-row">时间：${{escapeHtml(point.time || '—')}}</div>` +
        `<div class="popup-row">坐标：${{point.lat.toFixed(7)}}, ${{point.lng.toFixed(7)}}</div>` +
        (point.guide ? `<div class="popup-row">引导：${{escapeHtml(point.guide)}}</div>` : '') +
        (point.route_instruction ? `<div class="popup-row">路线指令：${{escapeHtml(point.route_instruction)}}</div>` : '') +
        (point.gps_action ? `<div class="popup-row">GPS 动作：${{escapeHtml(point.gps_action)}}</div>` : '') +
        (hasBearing(point) ? `<div class="popup-row">当前朝向：${{Number(point.cur_bearing).toFixed(1)}}°</div>` : '') +
        (point.cross_track_error != null ? `<div class="popup-row">横向误差：${{formatMeters(point.cross_track_error)}}</div>` : '');
    }}
    function addEndpoint(point, label, color, layer) {{
      L.circleMarker([point.lat, point.lng], {{ radius: 8, color, weight: 2, fillColor: '#fff', fillOpacity: 1 }}).bindPopup(popup(point, label)).addTo(layer);
    }}
    function addSamples(points, label, color) {{
      if (!points.length) return;
      const stride = Math.max(1, Math.ceil(points.length / 120));
      points.forEach((point, i) => {{
        if (i !== 0 && i !== points.length - 1 && i % stride !== 0) return;
        L.circleMarker([point.lat, point.lng], {{ radius: 3, color, weight: 1, fillColor: color, fillOpacity: .7 }}).bindPopup(popup(point, label)).addTo(markerLayer);
        const bearing = Number(point.cur_bearing);
        if (hasBearing(point)) {{
          const arrow = L.divIcon({{
            className: '', iconSize: [18, 18], iconAnchor: [9, 9],
            html: `<div style="width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-bottom:13px solid ${{color}};filter:drop-shadow(0 0 1px #fff);transform:rotate(${{bearing}}deg);transform-origin:50% 65%;"></div>`
          }});
          L.marker([point.lat, point.lng], {{ icon: arrow, keyboard: false, interactive: false }}).addTo(bearingLayer);
        }}
      }});
    }}
    if (reference.length) {{
      L.polyline(latLngs(reference), {{ color: '#1976d2', weight: 5, opacity: .82, lineJoin: 'round' }}).addTo(refLayer).bindTooltip('参考轨迹');
      addEndpoint(reference[0], '参考起点', '#1976d2', markerLayer);
      if (reference.length > 1) addEndpoint(reference.at(-1), '参考终点', '#1976d2', markerLayer);
      addSamples(reference, '参考轨迹采样点', '#1976d2');
    }}
    if (follow.length) {{
      L.polyline(latLngs(follow), {{ color: '#ef6c00', weight: 5, opacity: .88, lineJoin: 'round' }}).addTo(followLayer).bindTooltip('记忆输出轨迹');
      addEndpoint(follow[0], '跟随起点', '#ef6c00', markerLayer);
      if (follow.length > 1) addEndpoint(follow.at(-1), '跟随终点', '#ef6c00', markerLayer);
      addSamples(follow, '记忆输出采样点', '#ef6c00');
    }}
    const allPoints = reference.concat(follow);
    if (allPoints.length) map.fitBounds(L.latLngBounds(latLngs(allPoints)), {{ padding: [36, 36] }});
    else map.setView([23.047, 113.408], 16);

    const statRows = [
      ['参考轨迹点', summary.reference_points], ['参考轨迹长度', formatMeters(summary.reference_length_m)],
      ['输出日志行', summary.follow_rows], ['有效跟随点', summary.follow_points],
      ['跟随轨迹长度', formatMeters(summary.follow_length_m)], ['平均横向误差', formatMeters(summary.mean_cross_track_error_m)]
    ];
    document.querySelector('#summary').innerHTML = statRows.map(([name, value]) => `<tr><td>${{name}}</td><td>${{value}}</td></tr>`).join('');
    if (!follow.length) document.querySelector('#notice').innerHTML = '该输出日志没有有效的 <code>current_pos</code>，因此页面只显示参考轨迹。';
  </script>
</body>
</html>
"""


def render_one(reference: list[dict[str, Any]], output_file: Path, route_name: str) -> tuple[int, int]:
    rows = read_jsonl(output_file)
    follow = current_points(rows)
    destination = output_file.parent / DEFAULT_MAP_NAME
    destination.write_text(
        render_html(
            route_name=route_name,
            output_name=output_file.parent.name,
            reference=reference,
            follow=follow,
            follow_row_count=len(rows),
        ),
        encoding="utf-8",
    )
    return len(rows), len(follow)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, required=True, help="route directory containing source/ and follow_output/")
    parser.add_argument("--reference", type=Path, help="reference JSONL; defaults to <route-dir>/source/output_trimmed.jsonl")
    parser.add_argument("--follow-root", type=Path, help="follow output root; defaults to <route-dir>/follow_output")
    args = parser.parse_args(argv)

    route_dir = args.route_dir.resolve()
    reference_file = args.reference or route_dir / "source" / "output_trimmed.jsonl"
    follow_root = args.follow_root or route_dir / "follow_output"
    if not reference_file.is_file():
        parser.error(f"reference JSONL not found: {reference_file}")
    if not follow_root.is_dir():
        parser.error(f"follow output directory not found: {follow_root}")

    reference = current_points(read_jsonl(reference_file))
    if not reference:
        parser.error(f"reference JSONL contains no valid current_pos: {reference_file}")
    output_files = sorted(follow_root.glob("*/output.jsonl"))
    if not output_files:
        parser.error(f"no output.jsonl files found under: {follow_root}")

    for output_file in output_files:
        rows, points = render_one(reference, output_file, route_dir.name)
        print(f"{output_file.parent / DEFAULT_MAP_NAME} ({points}/{rows} current-position points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
