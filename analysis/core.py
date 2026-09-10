"""Adaptive anchor segmentation, clock-direction commands, and memory/test metrics.

Ingests the recorded GPS ``output.jsonl`` streams and the extracted frame
directories, aligns frames to GPS by ``fps`` + a start-time offset, resamples to
1 Hz keypoints, then segments each trajectory into anchors when any of the
following holds:

* turning: ``|Δθ_t| > τ_θ`` (heading change),  ``τ_θ = 30°``
* visual change: ``d_vis(t, t+w) > τ_vis`` (DINOv2 cosine distance)
* spacing: ``s_t - s_{last anchor} > τ_s`` (long straight)

Every 1 Hz keypoint is also turned into a blind-executable clock-direction
command (``9点钟方向前进约X米``).  The test trajectory is matched back onto the
memory (reference) trajectory to prove follow capability via cross-track error,
heading error, and visual similarity against the nearest reference anchor.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from memory_nav.analysis.embedding import DinoV2Embedder
from memory_nav.replay.matcher import RouteMatcher
from memory_nav.trajectory.coordinate import LocalFrame, heading_from_delta, wrap_to_180

LOCAL_TZ = timezone(timedelta(hours=8))
FRAME_RE = re.compile(r"^(?P<prefix>\d{8}_\d{6})")

CLOCK_LABELS = {
    0: "12", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
    6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "11",
}


@dataclass
class GpsRecord:
    line_number: int
    log_time: str
    epoch_s: float
    longitude: float
    latitude: float
    guide: str = ""
    gps_action: str = ""
    raw_gps_action: str = ""
    route_instruction: str = ""
    cur_bearing: float | None = None
    target_bearing: float | None = None
    angle_diff: float | None = None
    distance: float | None = None
    route_progress: float | None = None
    cross_track_error: float | None = None
    image_path: str = ""


@dataclass
class Keypoint:
    index: int
    epoch_s: float
    time_iso: str
    longitude: float
    latitude: float
    east: float
    north: float
    s_m: float
    heading_deg: float
    frame_index: int
    guide: str = ""
    route_instruction: str = ""
    gps_action: str = ""


@dataclass
class Anchor:
    anchor_id: str
    session: str
    index: int
    epoch_s: float
    time_iso: str
    longitude: float
    latitude: float
    east: float
    north: float
    s_m: float
    heading_deg: float
    frame_index: int
    reason: str
    guide: str = ""
    route_instruction: str = ""
    gps_action: str = ""
    command: str = ""
    clock_dir: str = ""
    embedding: np.ndarray | None = None
    image_file: str = ""
    suggested_direction: str = ""
    suggested_clock_dir: str = ""
    suggested_distance_m: float = 0.0
    image_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.embedding is not None:
            data["embedding"] = self.embedding.astype(np.float32).tolist()
        else:
            data["embedding"] = None
        return data


@dataclass
class Command:
    session: str
    index: int
    epoch_s: float
    time_iso: str
    longitude: float
    latitude: float
    clock_dir: str
    action: str
    command: str
    distance_m: float
    heading_deg: float
    target_heading_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FollowMetric:
    index: int
    epoch_s: float
    time_iso: str
    longitude: float
    latitude: float
    ref_s_m: float
    cross_track_error_m: float
    heading_error_deg: float
    match_quality: str
    ref_longitude: float
    ref_latitude: float
    target_heading_deg: float
    lookahead_longitude: float
    lookahead_latitude: float
    visual_sim: float | None
    nearest_anchor_id: str | None
    distance_to_anchor_m: float | None
    forward_distance_m: float | None
    command: str
    clock_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisConfig:
    fps: float | None = None
    tau_theta_deg: float = 30.0
    tau_vis: float = 0.15
    tau_s_m: float = 20.0
    visual_window_s: float = 2.0
    lookahead_m: float = 8.0


@dataclass
class Session:
    name: str
    records: list[GpsRecord] = field(default_factory=list)
    keypoints: list[Keypoint] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    frames_dir: Path | None = None
    frame_start_epoch: float | None = None
    fps: float | None = None
    frame: LocalFrame | None = None
    frame_idx_min: int | None = None
    frame_idx_max: int | None = None
    n_clamped_last: int = 0
    n_duplicate_frames: int = 0
    frame_calibration: list | None = None
    alignment_source: str = "fps"


def parse_frame_dir_start(dir_name: str) -> float:
    match = FRAME_RE.match(Path(dir_name).name)
    if not match:
        raise ValueError(f"cannot parse frame-directory start time: {dir_name}")
    dt = datetime.strptime(match.group("prefix"), "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
    return dt.timestamp()


def parse_log_time(value: str) -> float:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.timestamp()


def load_records(jsonl_path: str | Path) -> list[GpsRecord]:
    path = Path(jsonl_path)
    records: list[GpsRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = _parse_record_line(line, line_number)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError(f"no valid GPS records in {path}")
    records.sort(key=lambda r: r.epoch_s)
    return records


def _parse_record_line(line: str, line_number: int) -> GpsRecord | None:
    import json

    try:
        source = json.loads(line)
    except json.JSONDecodeError:
        return None
    pos = source.get("current_pos")
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    try:
        longitude, latitude = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        return None
    log_time = source.get("log_time")
    if not log_time:
        return None
    epoch_s = parse_log_time(log_time)

    def _f(key: str) -> float | None:
        try:
            return float(source[key])
        except (TypeError, ValueError, KeyError):
            return None

    return GpsRecord(
        line_number=line_number,
        log_time=log_time,
        epoch_s=epoch_s,
        longitude=longitude,
        latitude=latitude,
        guide=str(source.get("guide") or ""),
        gps_action=str(source.get("gps_action") or ""),
        raw_gps_action=str(source.get("raw_gps_action") or ""),
        route_instruction=str(source.get("route_instruction") or ""),
        cur_bearing=_f("cur_bearing"),
        target_bearing=_f("target_bearing"),
        angle_diff=_f("angle_diff"),
        distance=_f("distance"),
        route_progress=_f("route_progress"),
        cross_track_error=_f("cross_track_error"),
        image_path=str(source.get("image_path") or ""),
    )


def align_frame_index(epoch_s: float, frame_start_epoch: float, fps: float, n_frames: int) -> int:
    if n_frames <= 0:
        raise ValueError("no frames available")
    seconds = (epoch_s - frame_start_epoch) * fps
    return max(0, min(n_frames - 1, int(round(seconds))))


def build_frame_calibration(records: list[GpsRecord]) -> list[tuple[float, int]]:
    """Ground-truth frame<-time pairs from ``image_path`` in the GPS log.

    Each log record logs the frame it captured at that moment.  That frame is
    pixel-identical to the frame-folder image at ``logged_index - 1``.
    """
    pairs: list[tuple[float, int]] = []
    for record in records:
        path = record.image_path
        if not path:
            continue
        match = re.search(r"frame_(\d+)", path)
        if not match:
            continue
        frame = int(match.group(1)) - 1
        if frame < 0:
            continue
        pairs.append((record.epoch_s, frame))
    pairs.sort(key=lambda p: p[0])
    deduped: list[tuple[float, int]] = []
    seen: set[int] = set()
    for epoch, frame in pairs:
        if frame not in seen:
            seen.add(frame)
            deduped.append((epoch, frame))
    return deduped


def frame_for_time(epoch_s: float, calibration: list[tuple[float, int]], n_frames: int, edge_fps: float | None = None) -> int:
    """Map a time to a frame index by interpolation (edge_fps for extrapolation)."""
    if not calibration:
        raise ValueError("empty frame calibration")
    times = [p[0] for p in calibration]
    frames = [p[1] for p in calibration]
    if epoch_s <= times[0]:
        if edge_fps is not None:
            value = frames[0] + (epoch_s - times[0]) * edge_fps
        else:
            value = frames[0]
    elif epoch_s >= times[-1]:
        if edge_fps is not None:
            value = frames[-1] + (epoch_s - times[-1]) * edge_fps
        else:
            value = frames[-1]
    else:
        lo = 0
        while lo + 1 < len(times) and times[lo + 1] < epoch_s:
            lo += 1
        hi = lo + 1
        span = times[hi] - times[lo]
        if span <= 0:
            return frames[lo]
        local_fps = (frames[hi] - frames[lo]) / span
        value = frames[lo] + (epoch_s - times[lo]) * local_fps
    return max(0, min(n_frames - 1, int(round(value))))


def calibrate_global_fps(calibration: list[tuple[float, int]]) -> float | None:
    """Robust global frame rate from the ground-truth (time, frame) pairs."""
    import numpy as np

    if len(calibration) < 3:
        return None
    x = np.array([p[0] for p in calibration], dtype=np.float64)
    y = np.array([p[1] for p in calibration], dtype=np.float64)
    xc = x - x.mean()
    fps, intercept = np.polyfit(xc, y, 1)
    if not (np.isfinite(fps) and fps > 0):
        return None
    return float(fps)


def infer_fps(session: Session) -> float:
    """Infer the frame rate so the walked/GPS window maps into the frame range.

    ``fps = (n_frames - 1) / (gps_end - frame_dir_start)`` makes the last GPS
    keypoint land on the final frame, so no keypoint clamps to the folder edge.
    """
    if not session.records:
        raise ValueError(f"session {session.name}: cannot infer fps without GPS records")
    if session.frame_start_epoch is None:
        raise ValueError(f"session {session.name}: frame_start_epoch required to infer fps")
    n_frames = len(list(session.frames_dir.glob("frame_*.jpg"))) if session.frames_dir else 0
    if n_frames < 2:
        raise ValueError(f"session {session.name}: not enough frames to infer fps")
    gps_end = session.records[-1].epoch_s
    duration_s = gps_end - session.frame_start_epoch
    if duration_s <= 0:
        raise ValueError(f"session {session.name}: invalid frame-start/GPS duration")
    return (n_frames - 1) / duration_s


def resample_1hz(records: list[GpsRecord]) -> list[GpsRecord]:
    by_second: dict[int, GpsRecord] = {}
    for record in records:
        second = int(record.epoch_s)
        if second not in by_second or record.epoch_s < by_second[second].epoch_s:
            by_second[second] = record
    return [by_second[k] for k in sorted(by_second)]


def build_keypoints(session: Session) -> list[Keypoint]:
    records = resample_1hz(session.records)
    if not records:
        raise ValueError(f"session {session.name}: no GPS records after 1 Hz resampling")
    if session.frame is None:
        session.frame = LocalFrame(records[0].longitude, records[0].latitude)
    local = [session.frame.to_local(r.longitude, r.latitude) for r in records]
    s_accum: list[float] = [0.0]
    for index in range(1, len(local)):
        s_accum.append(s_accum[-1] + math.hypot(local[index][0] - local[index - 1][0], local[index][1] - local[index - 1][1]))
    n_frames = len(list(session.frames_dir.glob("frame_*.jpg"))) if session.frames_dir else 0
    calibration = build_frame_calibration(session.records)
    if calibration and len(calibration) >= 2 and n_frames:
        session.frame_calibration = calibration
        session.alignment_source = "calibration"
        session.fps = calibrate_global_fps(calibration)
        edge_fps = session.fps
    else:
        session.frame_calibration = None
        session.alignment_source = "fps"
        if session.fps is None:
            session.fps = infer_fps(session)
        edge_fps = float(session.fps)
    fps = float(session.fps) if session.fps else None
    frame_indices: list[int] = []
    keypoints: list[Keypoint] = []
    for index, (record, (east, north), s_m) in enumerate(zip(records, local, s_accum)):
        heading = record.cur_bearing
        if heading is None or not math.isfinite(heading):
            if index > 0:
                dx = east - local[index - 1][0]
                dy = north - local[index - 1][1]
                heading = heading_from_delta(dx, dy) if (dx != 0 or dy != 0) else keypoints[-1].heading_deg
            else:
                heading = 0.0
        if n_frames:
            if session.frame_calibration:
                frame_index = frame_for_time(record.epoch_s, session.frame_calibration, n_frames, edge_fps)
            else:
                frame_index = align_frame_index(record.epoch_s, session.frame_start_epoch, fps, n_frames)
        else:
            frame_index = -1
        frame_indices.append(frame_index)
        keypoints.append(Keypoint(
            index=index,
            epoch_s=record.epoch_s,
            time_iso=datetime.fromtimestamp(record.epoch_s, tz=LOCAL_TZ).isoformat(),
            longitude=record.longitude,
            latitude=record.latitude,
            east=east,
            north=north,
            s_m=s_m,
            heading_deg=heading % 360.0,
            frame_index=frame_index,
            guide=record.guide,
            route_instruction=record.route_instruction,
            gps_action=record.gps_action,
        ))
    if frame_indices and n_frames > 0:
        valid = [i for i in frame_indices if 0 <= i < n_frames]
        if valid:
            session.frame_idx_min = min(valid)
            session.frame_idx_max = max(valid)
            session.n_clamped_last = sum(1 for i in frame_indices if i == n_frames - 1)
            session.n_duplicate_frames = len(valid) - len(set(valid))
    return keypoints


def read_frame(session: Session, frame_index: int) -> np.ndarray | None:
    if frame_index < 0 or session.frames_dir is None:
        return None
    path = session.frames_dir / f"frame_{frame_index:06d}.jpg"
    if not path.is_file():
        return None
    return cv2.imread(str(path))


class EmbeddingCache:
    def __init__(self, session: Session, embedder: DinoV2Embedder):
        self.session = session
        self.embedder = embedder
        self._cache: dict[int, np.ndarray] = {}

    def get(self, frame_index: int) -> np.ndarray | None:
        if frame_index in self._cache:
            return self._cache[frame_index]
        frame = read_frame(self.session, frame_index)
        if frame is None:
            return None
        vector = self.embedder.embed(frame)
        self._cache[frame_index] = vector
        return vector


def clock_dir_from_delta(delta_deg: float) -> str:
    rel = wrap_to_180(delta_deg)
    hour = round(rel / 30.0) % 12
    return CLOCK_LABELS[hour]


def format_suggested_direction(clock_dir: str, distance_m: float) -> str:
    """Reference-follow suggested direction: ``向X点钟方向前进XX米``."""
    distance = max(0.0, float(distance_m))
    if clock_dir == "12":
        return f"向前直行{distance:.0f}米" if distance > 0.5 else "原地停住"
    return f"向{clock_dir}点钟方向前进{distance:.0f}米"


def enrich_test_anchors_with_reference(anchors: list[Anchor], metrics: list[FollowMetric]) -> None:
    """Fill each test anchor's reference-based suggested direction from its follow metric."""
    metric_by_index = {metric.index: metric for metric in metrics}
    for anchor in anchors:
        metric = metric_by_index.get(anchor.index)
        if metric is None:
            continue
        distance = metric.forward_distance_m if metric.forward_distance_m is not None else (metric.distance_to_anchor_m or 0.0)
        anchor.suggested_clock_dir = metric.clock_dir
        anchor.suggested_distance_m = distance
        anchor.suggested_direction = format_suggested_direction(metric.clock_dir, distance)


def make_command(heading_deg: float, target_heading_deg: float, distance_m: float) -> tuple[str, str]:
    rel = wrap_to_180(target_heading_deg - heading_deg)
    hour = round(rel / 30.0) % 12
    clock = CLOCK_LABELS[hour]
    distance = max(0.0, float(distance_m))
    if abs(rel) <= 22.5:
        return clock, f"{clock}点钟方向直行约{distance:.0f}米" if distance > 0.5 else f"{clock}点钟方向原地停住"
    if hour == 6:
        return clock, f"转向至{clock}点钟方向（回头），前进约{distance:.0f}米"
    if abs(rel) > 120:
        return clock, f"原地转至{clock}点钟方向，然后前进约{distance:.0f}米"
    return clock, f"{clock}点钟方向前进约{distance:.0f}米"


def segment_anchors(
    session: Session,
    embed_cache: EmbeddingCache,
    cfg: AnalysisConfig,
    command_lookup: dict[int, tuple[str, str]],
) -> list[Anchor]:
    points = session.keypoints
    if len(points) < 2:
        raise ValueError(f"session {session.name}: not enough keypoints for segmentation")
    emb: list[np.ndarray | None] = [embed_cache.get(p.frame_index) for p in points]

    def d_vis_from(t: int, from_index: int) -> float | None:
        if emb[t] is None or from_index < 0 or emb[from_index] is None:
            return None
        return DinoV2Embedder.cosine_distance(emb[t], emb[from_index])

    last_anchor_index = 0
    anchors: list[Anchor] = []
    for t, point in enumerate(points):
        if t == 0:
            reason = "start"
        else:
            reasons: list[str] = []
            turn = abs(wrap_to_180(point.heading_deg - points[t - 1].heading_deg))
            if turn > cfg.tau_theta_deg:
                reasons.append("turn")
            distance_vis = d_vis_from(t, last_anchor_index)
            moved_since_anchor = point.s_m - points[last_anchor_index].s_m
            if distance_vis is not None and moved_since_anchor >= 3.0 and distance_vis > cfg.tau_vis:
                reasons.append("visual")
            if point.s_m - points[last_anchor_index].s_m > cfg.tau_s_m:
                reasons.append("spacing")
            reason = "+".join(reasons) if reasons else ""
        if t == len(points) - 1:
            reason = reason or "end"
        if not reason:
            continue
        clock, command = command_lookup.get(t, ("12", "直行"))
        anchors.append(Anchor(
            anchor_id=f"{session.name}_a{t:03d}",
            session=session.name,
            index=t,
            epoch_s=point.epoch_s,
            time_iso=point.time_iso,
            longitude=point.longitude,
            latitude=point.latitude,
            east=point.east,
            north=point.north,
            s_m=point.s_m,
            heading_deg=point.heading_deg,
            frame_index=point.frame_index,
            reason=reason,
            guide=point.guide,
            route_instruction=point.route_instruction,
            gps_action=point.gps_action,
            command=command,
            clock_dir=clock,
            embedding=emb[t],
            image_file=f"frame_{point.frame_index:06d}.jpg" if point.frame_index >= 0 else "",
        ))
        last_anchor_index = t
    return anchors


def next_anchor_distance_s(points: list[Keypoint], anchors: list[Anchor], t: int) -> float:
    s = points[t].s_m
    for anchor in anchors:
        if anchor.s_m > s + 0.5:
            return anchor.s_m - s
    return points[-1].s_m - s


def build_commands(session: Session, anchors: list[Anchor]) -> list[Command]:
    points = session.keypoints
    commands: list[Command] = []
    for t, point in enumerate(points):
        target = points[t + 1].heading_deg if t + 1 < len(points) else point.heading_deg
        distance = next_anchor_distance_s(points, anchors, t)
        clock, text = make_command(point.heading_deg, target, distance)
        commands.append(Command(
            session=session.name,
            index=t,
            epoch_s=point.epoch_s,
            time_iso=point.time_iso,
            longitude=point.longitude,
            latitude=point.latitude,
            clock_dir=clock,
            action="turn" if clock not in ("12", "6") and abs(wrap_to_180(target - point.heading_deg)) > 22.5 else "straight",
            command=text,
            distance_m=distance,
            heading_deg=point.heading_deg,
            target_heading_deg=target,
        ))
    return commands


def build_command_lookup(commands: list[Command]) -> dict[int, tuple[str, str]]:
    return {command.index: (command.clock_dir, command.command) for command in commands}


def adaptive_forward_window(
    reference_points,
    close_dist_m: float = 6.0,
    margin_m: float = 40.0,
    floor_m: float = 30.0,
    cap_m: float = 300.0,
) -> float:
    """Forward matching window wide enough to bridge closed loops in the route.

    A reference route that loops (revisits the same spatial area far apart in
    arc-length) can trap progress matching.  This computes the largest arc-length
    gap between spatially-close reference points and widens the forward window so
    the matcher can skip the loop's redundant arc instead of stalling.
    """
    max_gap = 0.0
    points = reference_points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j].east_m - points[i].east_m
            dy = points[j].north_m - points[i].north_m
            if dx * dx + dy * dy <= close_dist_m * close_dist_m:
                gap = points[j].s_m - points[i].s_m
                if gap > max_gap:
                    max_gap = gap
    window = max_gap + margin_m
    return min(cap_m, max(floor_m, window))


def match_memory_to_reference(
    memory: Session,
    test: Session,
    memory_anchors: list[Anchor],
    embed_cache: EmbeddingCache,
    lookahead_m: float = 8.0,
) -> list[FollowMetric]:
    from memory_nav.models import ReferencePoint

    reference_points: list[ReferencePoint] = []
    for point in memory.keypoints:
        if not reference_points or point.s_m > reference_points[-1].s_m:
            reference_points.append(ReferencePoint(
                index=point.index,
                s_m=point.s_m,
                east_m=point.east,
                north_m=point.north,
                longitude=point.longitude,
                latitude=point.latitude,
                heading_deg=point.heading_deg,
            ))
    if len(reference_points) < 2:
        raise ValueError("memory trajectory has too few strictly-increasing reference points")
    matcher = RouteMatcher(reference_points, forward_window_m=adaptive_forward_window(reference_points))

    def interpolate_ref_heading(s: float) -> float | None:
        points = memory.keypoints
        if s is None or not points:
            return None
        if s <= points[0].s_m:
            return points[0].heading_deg
        if s >= points[-1].s_m:
            return points[-1].heading_deg
        for index in range(len(points) - 1):
            if points[index].s_m <= s <= points[index + 1].s_m:
                low, high = points[index], points[index + 1]
                span = high.s_m - low.s_m
                if span <= 0:
                    return low.heading_deg
                weight = (s - low.s_m) / span
                delta = wrap_to_180(high.heading_deg - low.heading_deg)
                return (low.heading_deg + delta * weight) % 360.0
        return points[-1].heading_deg

    def nearest_anchor(point) -> tuple[Anchor | None, float | None]:
        if not memory_anchors:
            return None, None
        best = min(memory_anchors, key=lambda a: (a.east - point.east) ** 2 + (a.north - point.north) ** 2)
        distance = math.hypot(best.east - point.east, best.north - point.north)
        return best, distance

    def interpolate_ref_xy(s: float) -> tuple[float, float]:
        if s <= reference_points[0].s_m:
            return reference_points[0].east_m, reference_points[0].north_m
        if s >= reference_points[-1].s_m:
            return reference_points[-1].east_m, reference_points[-1].north_m
        for index in range(len(reference_points) - 1):
            low, high = reference_points[index], reference_points[index + 1]
            if low.s_m <= s <= high.s_m:
                span = high.s_m - low.s_m
                if span <= 0:
                    return low.east_m, low.north_m
                weight = (s - low.s_m) / span
                return low.east_m + weight * (high.east_m - low.east_m), low.north_m + weight * (high.north_m - low.north_m)
        return reference_points[-1].east_m, reference_points[-1].north_m

    max_ref_s = -math.inf

    def bearing_to(tx: float, ty: float, fallback: float) -> float:
        dx = tx - point.east
        dy = ty - point.north
        if dx == 0 and dy == 0:
            return fallback
        return heading_from_delta(dx, dy)

    metrics: list[FollowMetric] = []
    for t, point in enumerate(test.keypoints):
        heading = point.heading_deg
        try:
            result = matcher.match(point.east, point.north, heading)
        except RuntimeError:
            result = None
        ref_s = None
        cross_track = None
        heading_error = None
        quality = "lost"
        if result is not None:
            ref_s = result.matched_s_m
            cross_track = result.cross_track_error_m
            heading_error = result.heading_error_deg
            quality = result.match_quality

        # Monotonic forward progress: never let the matched position drift
        # backwards in time, so the guidance never asks the user to backtrack.
        if ref_s is not None:
            max_ref_s = max(max_ref_s, ref_s)
        eff_ref_s = max_ref_s if math.isfinite(max_ref_s) else None

        # Guide toward the NEXT point ahead on the reference (a lookahead point),
        # strictly forward so there is no backtracking.
        forward_dist = None
        if eff_ref_s is not None:
            target_s = min(eff_ref_s + lookahead_m, reference_points[-1].s_m)
            tx, ty = interpolate_ref_xy(target_s)
            target_heading = bearing_to(tx, ty, heading)
            forward_dist = max(0.0, target_s - eff_ref_s)
            if memory.frame is not None:
                lookahead_lng, lookahead_lat = memory.frame.to_geodetic(tx, ty)
            else:
                lookahead_lng = lookahead_lat = float("nan")
        else:
            target_heading = heading
            lookahead_lng = lookahead_lat = float("nan")

        anchor, distance_to_anchor = nearest_anchor(point)
        visual_sim = None
        if anchor is not None and anchor.embedding is not None and distance_to_anchor is not None and distance_to_anchor <= 12.0:
            test_emb = embed_cache.get(point.frame_index)
            if test_emb is not None:
                visual_sim = DinoV2Embedder.cosine_similarity(anchor.embedding, test_emb)
        if result is not None and memory.frame is not None:
            ref_lng, ref_lat = memory.frame.to_geodetic(result.east_m, result.north_m)
        else:
            ref_lng = ref_lat = float("nan")
        clock, text = make_command(heading, target_heading, forward_dist if forward_dist is not None else 0.0)
        metrics.append(FollowMetric(
            index=t,
            epoch_s=point.epoch_s,
            time_iso=point.time_iso,
            longitude=point.longitude,
            latitude=point.latitude,
            ref_s_m=eff_ref_s if eff_ref_s is not None else float("nan"),
            cross_track_error_m=cross_track if cross_track is not None else float("nan"),
            heading_error_deg=heading_error if heading_error is not None else float("nan"),
            match_quality=quality,
            ref_longitude=ref_lng,
            ref_latitude=ref_lat,
            target_heading_deg=target_heading,
            lookahead_longitude=lookahead_lng,
            lookahead_latitude=lookahead_lat,
            visual_sim=visual_sim,
            nearest_anchor_id=anchor.anchor_id if anchor else None,
            distance_to_anchor_m=distance_to_anchor,
            forward_distance_m=forward_dist,
            command=text,
            clock_dir=clock,
        ))
    return metrics


def run_session(
    session: Session,
    cfg: AnalysisConfig,
    embedder: DinoV2Embedder,
    cache_dir: Path,
) -> Session:
    session.fps = cfg.fps
    session.keypoints = build_keypoints(session)
    embed_cache = EmbeddingCache(session, embedder)
    commands = build_commands(session, [])
    anchors = segment_anchors(session, embed_cache, cfg, build_command_lookup(commands))
    session.commands = build_commands(session, anchors)
    session.anchors = anchors
    _save_anchor_embeddings(session, cache_dir)
    return session


def _save_anchor_embeddings(session: Session, cache_dir: Path) -> None:
    anchor_dir = cache_dir / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    for anchor in session.anchors:
        if anchor.embedding is not None:
            path = anchor_dir / f"{anchor.anchor_id}.npy"
            np.save(path, anchor.embedding)
        frame = read_frame(session, anchor.frame_index)
        if frame is not None:
            image_path = anchor_dir / f"{anchor.anchor_id}.jpg"
            cv2.imwrite(str(image_path), frame)
            anchor.image_file = str(image_path.name)
