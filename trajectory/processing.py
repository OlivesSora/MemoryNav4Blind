"""GPS cleaning, smoothing and reference-trajectory generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from memory_nav.models import GPSSample, IMUSample, ReferencePoint, VIOSample
from memory_nav.trajectory.coordinate import LocalFrame, heading_from_delta, wrap_to_180


@dataclass(frozen=True)
class ProcessingReport:
    input_samples: int
    accepted_samples: int
    rejected_out_of_order: int
    rejected_duplicate: int
    rejected_speed: int
    route_length_m: float
    max_gap_s: float
    heading_alignment_samples: int
    heading_offset_deg: float | None
    vio_used: bool = False
    vio_samples: int = 0
    vio_alignment_pairs: int = 0
    vio_alignment_rms_m: float | None = None
    vio_transform: dict[str, float] | None = None


@dataclass(frozen=True)
class VioTransform:
    """Rigid mapping from VINS horizontal coordinates to local EN metres."""

    cos_yaw: float
    sin_yaw: float
    east_offset_m: float
    north_offset_m: float

    def apply(self, x_m: float, y_m: float) -> tuple[float, float]:
        return (self.cos_yaw * x_m - self.sin_yaw * y_m + self.east_offset_m,
                self.sin_yaw * x_m + self.cos_yaw * y_m + self.north_offset_m)

    def to_dict(self) -> dict[str, float]:
        return {"cos_yaw": self.cos_yaw, "sin_yaw": self.sin_yaw,
                "east_offset_m": self.east_offset_m, "north_offset_m": self.north_offset_m}


def estimate_vio_transform(
    samples: Sequence[VIOSample], targets: Sequence[tuple[float, float]],
) -> tuple[VioTransform, float]:
    """Least-squares 2D rigid alignment, deliberately without scale fitting."""
    if len(samples) != len(targets) or len(samples) < 2:
        raise ValueError("at least two matched VIO/GPS pairs are required")
    source = np.asarray([(item.x_m, item.y_m) for item in samples], dtype=float)
    destination = np.asarray(targets, dtype=float)
    source_center = source.mean(axis=0)
    destination_center = destination.mean(axis=0)
    covariance = (source - source_center).T @ (destination - destination_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    offset = destination_center - rotation @ source_center
    fitted = source @ rotation.T + offset
    rms = float(np.sqrt(np.mean(np.sum((fitted - destination) ** 2, axis=1))))
    return VioTransform(float(rotation[0, 0]), float(rotation[1, 0]), float(offset[0]), float(offset[1])), rms


def match_vio_to_gps(
    gps_samples: Sequence[GPSSample], vio_samples: Sequence[VIOSample], frame: LocalFrame,
    max_time_delta_s: float = 1.5,
) -> tuple[list[VIOSample], list[tuple[float, float]]]:
    """Pair each trusted GPS observation with its nearest VIO observation."""
    ordered = sorted(vio_samples, key=lambda item: item.monotonic_ns)
    if not ordered:
        return [], []
    times = np.asarray([item.monotonic_ns for item in ordered], dtype=np.int64)
    max_delta_ns = int(max_time_delta_s * 1e9)
    matched_vio: list[VIOSample] = []
    targets: list[tuple[float, float]] = []
    used: set[int] = set()
    for gps in gps_samples:
        if gps.state != "good":
            continue
        insertion = int(np.searchsorted(times, gps.monotonic_ns))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(ordered)]
        if not candidates:
            continue
        index = min(candidates, key=lambda item: abs(int(times[item]) - gps.monotonic_ns))
        if index in used or abs(int(times[index]) - gps.monotonic_ns) > max_delta_ns:
            continue
        used.add(index)
        matched_vio.append(ordered[index])
        targets.append(frame.to_local(gps.longitude, gps.latitude))
    return matched_vio, targets


def clean_gps_samples(
    samples: Iterable[GPSSample],
    max_speed_m_s: float = 4.0,
    duplicate_distance_m: float = 0.05,
) -> tuple[list[GPSSample], dict[str, int | float]]:
    """Keep input order and reject samples that cannot form a physical track."""
    source = list(samples)
    if not source:
        return [], {"input_samples": 0, "rejected_out_of_order": 0, "rejected_duplicate": 0, "rejected_speed": 0, "max_gap_s": 0.0}
    frame = LocalFrame(source[0].longitude, source[0].latitude)
    accepted: list[GPSSample] = []
    positions: list[tuple[float, float]] = []
    counters: dict[str, int | float] = {
        "input_samples": len(source),
        "rejected_out_of_order": 0,
        "rejected_duplicate": 0,
        "rejected_speed": 0,
        "max_gap_s": 0.0,
    }
    for sample in source:
        position = frame.to_local(sample.longitude, sample.latitude)
        if accepted:
            dt = (sample.monotonic_ns - accepted[-1].monotonic_ns) / 1e9
            if dt <= 0:
                counters["rejected_out_of_order"] += 1
                continue
            counters["max_gap_s"] = max(float(counters["max_gap_s"]), dt)
            distance = math.dist(position, positions[-1])
            if distance < duplicate_distance_m:
                counters["rejected_duplicate"] += 1
                continue
            if distance / dt > max_speed_m_s:
                counters["rejected_speed"] += 1
                continue
        accepted.append(sample)
        positions.append(position)
    return accepted, counters


def _quality_weight(sample: GPSSample) -> float:
    if sample.source == "rtk" and sample.state == "good":
        return 4.0
    if sample.state == "good":
        return 2.0
    return 1.0


def smooth_positions(
    positions: Sequence[tuple[float, float]],
    weights: Sequence[float],
    window: int = 5,
    protected_indices: Iterable[int] = (),
) -> np.ndarray:
    if len(positions) != len(weights):
        raise ValueError("positions and weights must have equal length")
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd number")
    points = np.asarray(positions, dtype=float)
    if len(points) <= 2 or window == 1:
        return points.copy()
    protected = {0, len(points) - 1, *protected_indices}
    radius = window // 2
    result = points.copy()
    sample_weights = np.asarray(weights, dtype=float)
    for index in range(1, len(points) - 1):
        if index in protected:
            continue
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        local_weights = sample_weights[start:end]
        result[index] = np.average(points[start:end], axis=0, weights=local_weights)
    return result


def resample_polyline(points: Sequence[tuple[float, float]] | np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 2 or len(points_array) < 2:
        raise ValueError("at least two 2D points are required")
    segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-9))
    points_array = points_array[keep]
    if len(points_array) < 2:
        raise ValueError("trajectory has zero length")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points_array, axis=0), axis=1))))
    total = float(cumulative[-1])
    targets = np.arange(0.0, total, spacing_m)
    if len(targets) == 0 or not math.isclose(float(targets[-1]), total):
        targets = np.append(targets, total)
    east = np.interp(targets, cumulative, points_array[:, 0])
    north = np.interp(targets, cumulative, points_array[:, 1])
    return np.column_stack((east, north)), targets


def _headings_and_curvature(points: np.ndarray, cumulative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.diff(points, axis=0)
    segment_headings = np.array([heading_from_delta(delta[0], delta[1]) for delta in deltas])
    headings = np.empty(len(points))
    headings[:-1] = segment_headings
    headings[-1] = segment_headings[-1]
    curvature = np.zeros(len(points))
    for index in range(1, len(points) - 1):
        ds = cumulative[index + 1] - cumulative[index - 1]
        if ds > 1e-9:
            curvature[index] = math.radians(wrap_to_180(headings[index] - headings[index - 1])) / (ds / 2.0)
    return headings, curvature


def _circular_mean_deg(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    if not values:
        raise ValueError("at least one angle is required")
    radians = np.radians(np.asarray(values, dtype=float))
    sample_weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    x = float(np.sum(np.cos(radians) * sample_weights))
    y = float(np.sum(np.sin(radians) * sample_weights))
    if math.hypot(x, y) < 1e-9:
        raise ValueError("angles have no defined circular mean")
    return math.degrees(math.atan2(y, x)) % 360.0


def estimate_heading_offset(gps_samples: Sequence[GPSSample], imu_samples: Sequence[IMUSample], minimum_speed_m_s: float = 0.3) -> tuple[float | None, int]:
    """Estimate ``course - IMU heading`` from moving GPS intervals."""
    usable_imu = sorted((sample for sample in imu_samples if sample.heading_ready and sample.heading_deg is not None), key=lambda sample: sample.monotonic_ns)
    if not usable_imu or len(gps_samples) < 2:
        return None, 0
    frame = LocalFrame(gps_samples[0].longitude, gps_samples[0].latitude)
    positions = [frame.to_local(sample.longitude, sample.latitude) for sample in gps_samples]
    imu_times = np.array([sample.monotonic_ns for sample in usable_imu], dtype=np.int64)
    offsets: list[float] = []
    weights: list[float] = []
    for index in range(len(gps_samples) - 1):
        dt = (gps_samples[index + 1].monotonic_ns - gps_samples[index].monotonic_ns) / 1e9
        if dt <= 0:
            continue
        east_delta = positions[index + 1][0] - positions[index][0]
        north_delta = positions[index + 1][1] - positions[index][1]
        distance = math.hypot(east_delta, north_delta)
        speed = distance / dt
        if speed < minimum_speed_m_s:
            continue
        midpoint = (gps_samples[index].monotonic_ns + gps_samples[index + 1].monotonic_ns) // 2
        insertion = int(np.searchsorted(imu_times, midpoint))
        candidates = [candidate for candidate in (insertion - 1, insertion) if 0 <= candidate < len(usable_imu)]
        nearest = min(candidates, key=lambda candidate: abs(int(imu_times[candidate]) - midpoint))
        # Do not align against an IMU value outside the GPS interval.
        if abs(int(imu_times[nearest]) - midpoint) > max(int(dt * 0.75e9), 500_000_000):
            continue
        course = heading_from_delta(east_delta, north_delta)
        offsets.append(wrap_to_180(course - float(usable_imu[nearest].heading_deg)))
        weights.append(distance)
    if not offsets:
        return None, 0
    return wrap_to_180(_circular_mean_deg(offsets, weights)), len(offsets)


def _fuse_reference_headings(
    geometric_headings: np.ndarray,
    cumulative: np.ndarray,
    gps_samples: Sequence[GPSSample],
    imu_samples: Sequence[IMUSample],
    offset_deg: float | None,
) -> np.ndarray:
    if offset_deg is None:
        return geometric_headings
    usable = sorted((sample for sample in imu_samples if sample.heading_ready and sample.heading_deg is not None), key=lambda sample: sample.monotonic_ns)
    if not usable:
        return geometric_headings
    gps_frame = LocalFrame(gps_samples[0].longitude, gps_samples[0].latitude)
    gps_positions = np.asarray([gps_frame.to_local(sample.longitude, sample.latitude) for sample in gps_samples])
    gps_s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(gps_positions, axis=0), axis=1))))
    if gps_s[-1] <= 0:
        return geometric_headings
    reference_times = np.interp(cumulative / cumulative[-1] * gps_s[-1], gps_s, [sample.monotonic_ns for sample in gps_samples])
    imu_times = np.asarray([sample.monotonic_ns for sample in usable], dtype=float)
    imu_unwrapped = np.unwrap(np.radians([sample.heading_deg for sample in usable]))
    interpolated = np.degrees(np.interp(reference_times, imu_times, imu_unwrapped)) + offset_deg
    fused = np.empty_like(geometric_headings)
    for index, (geometry, imu_heading) in enumerate(zip(geometric_headings, interpolated)):
        fused[index] = _circular_mean_deg([float(geometry), float(imu_heading)], [1.0, 2.0])
    return fused


def build_reference(
    samples: Iterable[GPSSample],
    imu_samples: Iterable[IMUSample] = (),
    protected_monotonic_ns: Iterable[int] = (),
    turn_protection_deg: float = 35.0,
    spacing_m: float = 0.5,
    smoothing_window: int = 5,
    max_speed_m_s: float = 4.0,
    duplicate_distance_m: float = 0.05,
    vio_samples: Iterable[VIOSample] = (),
    vio_max_alignment_rms_m: float = 5.0,
) -> tuple[list[ReferencePoint], ProcessingReport, LocalFrame]:
    source = list(samples)
    imu_source = list(imu_samples)
    cleaned, stats = clean_gps_samples(source, max_speed_m_s, duplicate_distance_m)
    if len(cleaned) < 2:
        raise ValueError("at least two valid GPS samples are required")
    frame = LocalFrame(cleaned[0].longitude, cleaned[0].latitude)
    positions = [frame.to_local(sample.longitude, sample.latitude) for sample in cleaned]
    vio_source = list(vio_samples)
    matched_vio, vio_targets = match_vio_to_gps(cleaned, vio_source, frame)
    vio_transform = None
    vio_rms = None
    if len(matched_vio) >= 3:
        candidate, vio_rms = estimate_vio_transform(matched_vio, vio_targets)
        if vio_rms <= vio_max_alignment_rms_m:
            vio_transform = candidate
    protected_times = list(protected_monotonic_ns)
    protected_indices = {
        min(range(len(cleaned)), key=lambda index: abs(cleaned[index].monotonic_ns - timestamp))
        for timestamp in protected_times
    }
    if turn_protection_deg > 0 and len(positions) >= 3:
        segment_headings = [
            heading_from_delta(positions[index + 1][0] - positions[index][0], positions[index + 1][1] - positions[index][1])
            for index in range(len(positions) - 1)
        ]
        protected_indices.update(
            index for index in range(1, len(positions) - 1)
            if abs(wrap_to_180(segment_headings[index] - segment_headings[index - 1])) >= turn_protection_deg
        )
    # VIO supplies the dense local geometry only after it has proved coherent
    # with GPS. GPS remains the global reference and fallback source.
    if vio_transform is not None:
        ordered_vio = sorted(vio_source, key=lambda item: item.monotonic_ns)
        positions = [vio_transform.apply(item.x_m, item.y_m) for item in ordered_vio]
        if len(positions) < 2:
            positions = [frame.to_local(sample.longitude, sample.latitude) for sample in cleaned]
            vio_transform = None
        else:
            protected_indices = {0, len(positions) - 1}
    smoothed = smooth_positions(positions, [1.0] * len(positions) if vio_transform else [_quality_weight(sample) for sample in cleaned], smoothing_window, protected_indices)
    resampled, cumulative = resample_polyline(smoothed, spacing_m)
    geometric_headings, _ = _headings_and_curvature(resampled, cumulative)
    heading_offset, alignment_count = estimate_heading_offset(cleaned, imu_source)
    headings = _fuse_reference_headings(geometric_headings, cumulative, cleaned, imu_source, heading_offset)
    # Curvature must follow the final fused heading rather than the GPS-only heading.
    curvature = np.zeros(len(headings))
    for index in range(1, len(headings) - 1):
        ds = cumulative[index + 1] - cumulative[index - 1]
        if ds > 1e-9:
            curvature[index] = math.radians(wrap_to_180(headings[index] - headings[index - 1])) / (ds / 2.0)
    references: list[ReferencePoint] = []
    for index, ((east, north), s_m, heading, curve) in enumerate(zip(resampled, cumulative, headings, curvature)):
        longitude, latitude = frame.to_geodetic(float(east), float(north))
        references.append(ReferencePoint(
            index=index,
            s_m=float(s_m),
            east_m=float(east),
            north_m=float(north),
            longitude=longitude,
            latitude=latitude,
            heading_deg=float(heading),
            curvature=float(curve),
            quality="good",
        ))
    report = ProcessingReport(
        input_samples=len(source),
        accepted_samples=len(cleaned),
        rejected_out_of_order=int(stats["rejected_out_of_order"]),
        rejected_duplicate=int(stats["rejected_duplicate"]),
        rejected_speed=int(stats["rejected_speed"]),
        route_length_m=float(cumulative[-1]),
        max_gap_s=float(stats["max_gap_s"]),
        heading_alignment_samples=alignment_count,
        heading_offset_deg=heading_offset,
        vio_used=vio_transform is not None,
        vio_samples=len(vio_source),
        vio_alignment_pairs=len(matched_vio),
        vio_alignment_rms_m=vio_rms,
        vio_transform=None if vio_transform is None else vio_transform.to_dict(),
    )
    return references, report, frame
