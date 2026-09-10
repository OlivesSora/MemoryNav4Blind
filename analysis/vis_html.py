"""Interactive HTML visualization mirroring the style of ``gps_ana.py``.

Produces a fully self-contained offline HTML page (no CDN / tile dependency):

* an SVG map of the memory (reference) and test routes with clickable anchors,
* a detail panel per anchor showing its original image, extracted semantic text
  (``guide``/``route_instruction``/``gps_action``), command, and DINOv2 visual
  feature summary,
* a follow-metrics table proving the test trajectory's follow capability.

Anchor images are embedded as base64 data URIs so the single HTML file is
portable and viewable offline.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np

COLOR_MEMORY = "#7c3aed"
COLOR_TEST = "#0d9488"
COLOR_ANCHOR = "#dc2626"

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; color: #1f2937; background: #f3f4f6; }
  header { background: #111827; color: #fff; padding: 12px 18px; }
  header h1 { margin: 0; font-size: 18px; }
  header .summary { color: #9ca3af; font-size: 13px; margin-top: 2px; }
  main { display: flex; gap: 16px; padding: 16px; align-items: flex-start; flex-wrap: wrap; }
  #mapwrap { flex: 1 1 540px; min-width: 420px; background: #fff; border-radius: 8px;
             padding: 10px; box-shadow: 0 1px 8px rgba(0,0,0,.12); }
  #mapwrap svg { width: 100%; height: auto; background: #eef2f7; border-radius: 6px; }
  #panel { flex: 1 1 380px; min-width: 340px; background: #fff; border-radius: 8px;
           padding: 14px; box-shadow: 0 1px 8px rgba(0,0,0,.12); }
  .legend { margin: 6px 0 10px; font-size: 12px; color: #4b5563; }
  .legend span { white-space: nowrap; margin-right: 8px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
  #panel .guide { margin: 8px 0; padding: 9px; border-left: 4px solid #7c3aed;
                  border-radius: 4px; background: #f5f3ff; font-weight: 600;
                  word-break: break-word; }
  #panel img { display: block; width: 100%; max-height: 240px; margin: 8px 0;
               border-radius: 5px; object-fit: contain; background: #111827; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 4px 0; text-align: left; vertical-align: top;
           border-bottom: 1px solid #e5e7eb; word-break: break-word; }
  th { width: 34%; color: #6b7280; font-weight: 500; }
  .metric-good { color: #15803d; font-weight: 600; }
  .metric-bad { color: #dc2626; font-weight: 600; }
  #metrics table { font-size: 11px; }
  #metrics th, #metrics td { padding: 2px 6px; }
  #metrics thead th { position: sticky; top: 0; background: #f9fafb; }
  .clock-grid { display: grid; grid-template-columns: repeat(4, auto); gap: 4px 12px;
                font-size: 12px; margin-top: 6px; }
</style>
"""


def _embed_image(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _feature_summary(embedding) -> dict:
    if embedding is None:
        return {}
    vector = np.asarray(embedding, dtype=np.float32)
    return {
        "dim": int(vector.shape[0]),
        "norm": round(float(np.linalg.norm(vector)), 4),
        "l2": round(float(np.linalg.norm(vector - vector.mean())), 4),
        "hash": _stable_hash(vector),
    }


def _stable_hash(vector: np.ndarray) -> str:
    import hashlib
    return hashlib.sha1(vector.astype(np.float32).tobytes()).hexdigest()[:12]


def render_session_page(session, out: Path, reference=None, metrics=None) -> Path:
    """Render a single self-contained HTML page for one trajectory.

    ``reference`` (a memory Session) overlays the historical reference route,
    and ``metrics`` (FollowMetric list aligned by index to ``session.keypoints``)
    adds per-point reference-follow annotations (e.g. "9点钟方向前进约X米").
    """
    out.mkdir(parents=True, exist_ok=True)
    anchor_dir = out / "anchors"
    anchors = [_serialize_anchor(a, anchor_dir) for a in session.anchors]
    route = [[round(p.longitude, 7), round(p.latitude, 7)] for p in session.keypoints]
    is_test = session.name == "test"
    title = "记忆轨迹 (memory) 锚点分析" if not is_test else "测试轨迹 (test) 跟随记忆轨迹分析"

    rows = anchors
    reference_route = None
    if reference is not None:
        reference_route = [[round(p.longitude, 7), round(p.latitude, 7)] for p in reference.keypoints]

    # Merge each test anchor's matched-reference info for the map overlay.
    if metrics is not None:
        metric_by_index = {m["index"]: m for m in metrics}
        for a in anchors:
            m = metric_by_index.get(a["index"])
            if m is not None:
                a["ref_longitude"] = m["ref_longitude"]
                a["ref_latitude"] = m["ref_latitude"]
                a["target_heading_deg"] = m["target_heading_deg"]

    svg = _render_svg({session.name: route}, [(session.name, anchors)], reference_route=reference_route)
    panel_default = _anchor_panel_html(rows[0] if rows else None)

    # Geo bounds matching _render_svg, for the JS overlay coordinate transform.
    all_pts = list(route) + (reference_route or [])
    pad = 0.00012
    min_lng = min(p[0] for p in all_pts) - pad
    max_lng = max(p[0] for p in all_pts) + pad
    min_lat = min(p[1] for p in all_pts) - pad
    max_lat = max(p[1] for p in all_pts) + pad

    command_rows = []
    for command in session.commands:
        command_rows.append(
            f"<tr><td>{command.index}</td><td>{command.time_iso[11:19]}</td>"
            f"<td>{command.clock_dir}点</td><td>{command.command}</td>"
            f"<td>{command.distance_m:.0f}</td></tr>"
        )
    command_table = ("<table><thead><tr><th>#</th><th>时间</th><th>时钟</th><th>指令</th><th>距离(m)</th></tr></thead>"
                     "<tbody>" + "".join(command_rows) + "</tbody></table>")

    follow_section = ""
    if metrics is not None:
        follow_rows = []
        for m in sorted(metrics, key=lambda x: x["index"]):
            if m["index"] >= len(session.keypoints):
                continue
            quality = "good" if m["match_quality"] == "good" else "degraded" if m["match_quality"] == "degraded" else "lost"
            cross = m["cross_track_error_m"]
            cross_html = f"{cross:.1f}" if cross == cross else "—"
            heading = m["heading_error_deg"]
            heading_html = f"{abs(heading):.1f}" if heading == heading else "—"
            vis = m["visual_sim"]
            vis_html = f"{vis:.2f}" if vis is not None else "—"
            follow_rows.append(
                f"<tr><td>{m['index']}</td><td>{m['time_iso'][11:19]}</td>"
                f"<td><b>{m['clock_dir']}点钟</b></td><td>{m['command']}</td>"
                f"<td>{m['nearest_anchor_id'] or '—'}</td>"
                f"<td class='{quality}'>{quality}</td><td>{cross_html}</td>"
                f"<td>{heading_html}</td><td>{vis_html}</td></tr>"
            )
        follow_table = ("<table><thead><tr><th>#</th><th>时间</th><th>时钟</th><th>跟随动作（参考记忆轨迹）</th>"
                        "<th>参考锚点</th><th>匹配</th><th>横向误差(m)</th><th>朝向误差(°)</th><th>视觉相似</th></tr></thead>"
                        "<tbody>" + "".join(follow_rows) + "</tbody></table>")
        follow_section = f"""
<section id="follow" style="padding:0 16px 24px;">
  <h3 style="margin:8px 0">参考轨迹跟随标注（依据历史记忆轨迹，逐点给出盲人可执行动作）</h3>
  <div style="max-height:460px; overflow:auto; background:#fff; border-radius:8px; padding:8px; box-shadow:0 1px 8px rgba(0,0,0,.12);">
  {follow_table}
  </div>
</section>"""

    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{CSS}
</head>
<body>
<header>
  <h1>{title} · {session.name}</h1>
  <div class="summary">共 {len(session.keypoints)} 个 1Hz 关键点，{len(anchors)} 个自适应锚点。点击地图锚点查看原图、语义文本与视觉特征。</div>
</header>
<main>
  <div id="mapwrap">
    <div class="legend">
      <span><i class="swatch" style="background:{COLOR_MEMORY if is_test else COLOR_TEST}"></i> {('记忆参考轨迹 memory' if is_test else session.name + ' 轨迹')}</span>
      <span><i class="swatch" style="background:{COLOR_TEST if is_test else COLOR_TEST}"></i> 测试轨迹 test</span>
      <span><i class="swatch" style="background:{COLOR_ANCHOR}"></i> 锚点 anchor</span>
    </div>
    {svg}
  </div>
  <div id="panel">{panel_default}</div>
</main>
<section id="metrics" style="padding:0 16px 24px;">
  <h3 style="margin:8px 0">1Hz 时钟指令（盲人可执行动作）</h3>
  <div style="max-height:420px; overflow:auto; background:#fff; border-radius:8px; padding:8px; box-shadow:0 1px 8px rgba(0,0,0,.12);">
  {command_table}
  </div>
</section>
{follow_section}
<script>
const anchors = {json.dumps(rows, ensure_ascii=False)};
const MINL = {min_lng}, MAXL = {max_lng}, MINT = {min_lat}, MAXT = {max_lat}, MW = 620, MH = 620;
function px(lng, lat) {{
  return {{ x: (lng - MINL) / (MAXL - MINL) * MW, y: (MAXT - lat) / (MAXT - MINT) * MH }};
}}
function setMapOverlay(a) {{
  const dot = document.getElementById('nearest-dot');
  const g = document.getElementById('direction-arrow');
  const line = document.getElementById('arrow-line');
  const head = document.getElementById('arrow-head');
  if (!a || a.ref_longitude === undefined || a.ref_longitude !== a.ref_longitude || a.target_heading_deg === undefined) {{
    dot.setAttribute('opacity', 0); g.setAttribute('opacity', 0);
    return;
  }}
  const r = px(a.ref_longitude, a.ref_latitude);
  dot.setAttribute('cx', r.x); dot.setAttribute('cy', r.y); dot.setAttribute('opacity', 1);
  const p = px(a.longitude, a.latitude);
  const b = a.target_heading_deg * Math.PI / 180;
  const L = 46, ah = 12, half = 6;
  const sx = Math.sin(b), sy = -Math.cos(b);
  const ex = p.x + L * sx, ey = p.y + L * sy;
  const bx = ex - ah * sx, by = ey - ah * sy;
  const pvx = -sy, pvy = sx;
  line.setAttribute('x1', p.x); line.setAttribute('y1', p.y);
  line.setAttribute('x2', bx); line.setAttribute('y2', by);
  head.setAttribute('points', ex + ',' + ey + ' ' + (bx + pvx * half) + ',' + (by + pvy * half) + ' ' + (bx - pvx * half) + ',' + (by - pvy * half));
  g.setAttribute('opacity', 1);
}}
function selectAnchor(idx) {{
  const a = anchors[idx];
  const el = document.getElementById('panel');
  if (!a) {{ el.innerHTML = '<div>无锚点数据</div>'; return; }}
  el.innerHTML = a.html;
  setMapOverlay(a);
}}
</script>
</body>
</html>"""
    target = out / f"{session.name}_trajectory.html"
    target.write_text(page, encoding="utf-8")
    return target


def render(memory, test, metrics, out: Path, map_path: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    anchor_dir = out / "anchors"
    data = {
        "memory_anchors": [_serialize_anchor(a, anchor_dir) for a in memory.anchors],
        "test_anchors": [_serialize_anchor(a, anchor_dir) for a in test.anchors],
        "routes": {
            "memory": [[round(p.longitude, 7), round(p.latitude, 7)] for p in memory.keypoints],
            "test": [[round(p.longitude, 7), round(p.latitude, 7)] for p in test.keypoints],
        },
        "metrics": [m.to_dict() for m in metrics],
        "summary": {
            "memory_anchors": len(memory.anchors),
            "test_anchors": len(test.anchors),
            "test_points": len(test.keypoints),
        },
    }
    page = _build_page(data)
    target = out / "analysis.html"
    target.write_text(page, encoding="utf-8")
    if map_path:
        _inject_into_basemap(map_path, data, out)
    return target


def _serialize_anchor(anchor, anchor_dir: Path) -> dict:
    record = anchor.to_dict()
    record["feature"] = _feature_summary(record.get("embedding"))
    record.pop("embedding", None)
    record["image_data"] = _embed_image(anchor_dir / f"{anchor.anchor_id}.jpg")
    record["html"] = _anchor_panel_html(record)
    record.pop("image_data", None)
    return record


def _build_page(data: dict) -> str:
    memory_anchors = data["memory_anchors"]
    test_anchors = data["test_anchors"]
    metrics = data["metrics"]
    route = data["routes"]

    anchor_rows = memory_anchors + test_anchors

    svg = _render_svg(route, [("memory", memory_anchors), ("test", test_anchors)])
    panel_default = _anchor_panel_html(anchor_rows[0] if anchor_rows else None)
    metric_rows = _metrics_table(metrics)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>离线记忆/测试轨迹锚点分析</title>
{CSS}
</head>
<body>
<header>
  <h1>MemoryNav 离线锚点分析 · 记忆轨迹 vs 测试轨迹</h1>
  <div class="summary">{data['summary']['test_points']} 个测试关键点；记忆锚点 {data['summary']['memory_anchors']} 个；测试锚点 {data['summary']['test_anchors']} 个。点击地图上的锚点查看原始图像、语义文本与视觉特征。</div>
</header>
<main>
  <div id="mapwrap">
    <div class="legend">
      <span><i class="swatch" style="background:{COLOR_MEMORY}"></i> 记忆轨迹 memory</span>
      <span><i class="swatch" style="background:{COLOR_TEST}"></i> 测试轨迹 test</span>
      <span><i class="swatch" style="background:{COLOR_ANCHOR}"></i> 锚点 anchor</span>
      <span>时钟指令：<b>12点=正前方</b>，右转方向=1,2,3点钟（3=右90°），左转=11,10,9点钟（9=左90°），6点=回头。</span>
    </div>
    {svg}
  </div>
  <div id="panel">{panel_default}</div>
</main>
<section id="metrics" style="padding:0 16px 24px;">
  <h3 style="margin:8px 0">测试轨迹跟随指标（test follow capability）</h3>
  <div style="max-height:420px; overflow:auto; background:#fff; border-radius:8px; padding:8px; box-shadow:0 1px 8px rgba(0,0,0,.12);">
  {metric_rows}
  </div>
</section>
<script>
const anchors = {json.dumps(anchor_rows, ensure_ascii=False)};
function selectAnchor(idx) {{
  const a = anchors[idx];
  const el = document.getElementById('panel');
  if (!a) {{ el.innerHTML = '<div>无锚点数据</div>'; return; }}
  el.innerHTML = a.html;
}}
</script>
</body>
</html>"""


def _render_svg(route, anchor_groups, reference_route=None) -> str:
    pts = [p for series in route.values() for p in series]
    if reference_route:
        pts = pts + reference_route
    lngs = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    if not lngs:
        return "<svg></svg>"
    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)
    pad = 0.00012
    min_lng -= pad; max_lng += pad; min_lat -= pad; max_lat += pad
    width, height = 620, 620
    def px(lng, lat):
        x = (lng - min_lng) / (max_lng - min_lng) * width
        y = (max_lat - lat) / (max_lat - min_lat) * height
        return x, y
    def polyline(series, color, weight, dash=None):
        d = "M " + " L ".join(f"{px(lng, lat)[0]:.1f},{px(lng, lat)[1]:.1f}" for lng, lat in series)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{weight}" stroke-linejoin="round"{dash_attr}/>'
    paths = []
    if reference_route:
        paths.append(polyline(reference_route, COLOR_MEMORY, 2, dash="6 4"))
    for name, series in route.items():
        color = COLOR_MEMORY if name == "memory" else COLOR_TEST
        paths.append(polyline(series, color, 3))
    circles = []
    idx = 0
    for name, group in anchor_groups:
        for a in group:
            x, y = px(a["longitude"], a["latitude"])
            circles.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{COLOR_ANCHOR}" '
                f'stroke="#fff" stroke-width="1.5" cursor="pointer" '
                f'onclick="selectAnchor({idx})"><title>{a["anchor_id"]} · {a["reason"]}</title></circle>'
            )
            idx += 1
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(paths) + "".join(circles)
            + '<circle id="nearest-dot" cx="0" cy="0" r="7" fill="#2563eb" stroke="#fff" stroke-width="1.5" opacity="0"/>'
            + '<g id="direction-arrow" opacity="0">'
            + '<line id="arrow-line" x1="0" y1="0" x2="0" y2="0" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>'
            + '<polygon id="arrow-head" points="0,0 0,0 0,0" fill="#f59e0b"/>'
            + '</g></svg>')


def _anchor_panel_html(a) -> str:
    if a is None:
        return "<div>点击地图锚点查看详情</div>"
    feats = a.get("feature", {}) or {}
    feat_str = ", ".join(f"{k}={v}" for k, v in feats.items()) or "无"
    suggested = a.get("suggested_direction") or ""
    image_text = (a.get("image_text") or "").strip()
    suggested_row = f"<tr><th>建议方向(参考轨迹)</th><td>{suggested}</td></tr>" if suggested else ""
    text_row = f"<tr><th>图像文本(VLM)</th><td>{image_text}</td></tr>" if image_text else ""
    return f"""<div class="guide">{a['command']}</div>
{'' if not a.get('image_data') else f"<img src='{a['image_data']}' alt='{a['anchor_id']} 原图'>"}
<table>
  <tr><th>锚点</th><td>{a['anchor_id']} · 触发: {a['reason']}</td></tr>
  <tr><th>轨迹</th><td>{a['session']}</td></tr>
  <tr><th>时间</th><td>{a['time_iso']}</td></tr>
  <tr><th>坐标</th><td>{a['longitude']:.7f}, {a['latitude']:.7f}</td></tr>
  <tr><th>进度 s</th><td>{a['s_m']:.1f} m</td></tr>
  <tr><th>朝向</th><td>{a['heading_deg']:.1f}°</td></tr>
  <tr><th>guide 指令</th><td>{a['guide']}</td></tr>
  <tr><th>路线指令</th><td>{a['route_instruction']}</td></tr>
  <tr><th>GPS 动作</th><td>{a['gps_action']}</td></tr>
  {suggested_row}
  {text_row}
  <tr><th>视觉特征(DINOv2)</th><td>{feat_str}</td></tr>
</table>"""


def _metrics_table(metrics) -> str:
    if not metrics:
        return "<div>无跟随指标</div>"
    rows = []
    for m in metrics:
        quality = "good" if m["match_quality"] == "good" else "degraded" if m["match_quality"] == "degraded" else "lost"
        vis = m["visual_sim"]
        vis_html = f"{vis:.2f}" if vis is not None else "—"
        vis_class = "metric-good" if (vis is not None and vis > 0.7) else "metric-bad" if (vis is not None and vis < 0.5) else ""
        cross = m["cross_track_error_m"]
        cross_html = f"{cross:.1f}" if cross == cross else "—"
        heading = m["heading_error_deg"]
        heading_html = f"{abs(heading):.1f}" if heading == heading else "—"
        rows.append(
            f"<tr><td>{m['index']}</td><td>{m['time_iso'][11:19]}</td>"
            f"<td>{m['clock_dir']}点</td><td>{m['command']}</td>"
            f"<td>{m['nearest_anchor_id'] or '—'}</td>"
            f"<td class='{quality}'>{quality}</td>"
            f"<td>{cross_html}</td><td>{heading_html}</td>"
            f"<td class='{vis_class}'>{vis_html}</td></tr>"
        )
    return ("<table><thead><tr><th>#</th><th>时间</th><th>时钟</th><th>跟随指令</th>"
            "<th>最近锚点</th><th>匹配</th><th>横向误差(m)</th><th>朝向误差(°)</th><th>视觉相似</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


def _inject_into_basemap(map_path: str, data: dict, out: Path) -> Path:
    import re
    path = Path(map_path)
    if not path.is_file():
        return out / "analysis.html"
    html = path.read_text(encoding="utf-8")
    match = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map\s*\(", html)
    if not match:
        return out / "analysis.html"
    map_name = match.group(1)
    script = f"""
<script>
(() => {{
  const map = window.{map_name};
  const anchors = {json.dumps(data['memory_anchors'] + data['test_anchors'], ensure_ascii=False)};
  const route = {json.dumps(data['routes'], ensure_ascii=False)};
  if (!map) return;
  const memLayer = L.polyline(route.memory.map(p=>[p[1],p[0]]), {{color:'#7c3aed', weight:3, opacity:.8}}).addTo(map);
  const testLayer = L.polyline(route.test.map(p=>[p[1],p[0]]), {{color:'#0d9488', weight:3, opacity:.8}}).addTo(map);
  anchors.forEach(a => {{
    L.circleMarker([a.latitude, a.longitude], {{radius:7, color:'#dc2626', fillColor:'#dc2626', fillOpacity:.85}})
      .bindPopup(`<b>${{a.anchor_id}}</b> (${{a.reason}})<br>guide: ${{a.guide}}<br>指令: ${{a.command}}`)
      .addTo(map);
  }});
}})();
</script>"""
    page = html.replace("</body>", script + "</body>")
    target = out / "analysis_map.html"
    target.write_text(page, encoding="utf-8")
    return target