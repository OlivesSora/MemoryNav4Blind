"""Versioned serializable data models shared by recording and replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GPSSample:
    monotonic_ns: int
    longitude: float
    latitude: float
    source: str = "unknown"
    state: str = "poor"
    coordinate_system: str = "GCJ-02"
    server_time_ns: int | None = None
    age_s: float | None = None
    accuracy_m: float | None = None
    fix_quality: int | None = None

    def __post_init__(self) -> None:
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns cannot be negative")
        if not -180.0 <= self.longitude <= 180.0 or not -90.0 <= self.latitude <= 90.0:
            raise ValueError("invalid longitude/latitude")
        if self.coordinate_system != "GCJ-02":
            raise ValueError("only GCJ-02 is currently supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPSSample":
        return cls(**{field: data.get(field) for field in cls.__dataclass_fields__ if field in data})


@dataclass(frozen=True)
class IMUSample:
    monotonic_ns: int
    device_time_s: float | None
    heading_deg: float | None
    angular_velocity_rad_s: tuple[float, float, float] | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    accel_m_s2: tuple[float, float, float] | None = None
    mag: tuple[float, float, float] | None = None
    heading_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IMUSample":
        converted = dict(data)
        for key in ("angular_velocity_rad_s", "quaternion_wxyz", "accel_m_s2", "mag"):
            if converted.get(key) is not None:
                converted[key] = tuple(converted[key])
        return cls(**{field: converted.get(field) for field in cls.__dataclass_fields__ if field in converted})


@dataclass(frozen=True)
class VIOSample:
    """A VINS odometry sample, timestamped on the MemoryNav host clock."""

    monotonic_ns: int
    ros_time_ns: int
    x_m: float
    y_m: float
    z_m: float
    quaternion_xyzw: tuple[float, float, float, float] | None = None
    frame_id: str = "world"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VIOSample":
        converted = dict(data)
        if converted.get("quaternion_xyzw") is not None:
            converted["quaternion_xyzw"] = tuple(converted["quaternion_xyzw"])
        return cls(**{field: converted.get(field) for field in cls.__dataclass_fields__ if field in converted})


@dataclass(frozen=True)
class ReferencePoint:
    index: int
    s_m: float
    east_m: float
    north_m: float
    longitude: float
    latitude: float
    heading_deg: float
    curvature: float = 0.0
    quality: str = "good"
    anchor_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchResult:
    segment_index: int
    matched_s_m: float
    east_m: float
    north_m: float
    cross_track_error_m: float
    heading_error_deg: float
    distance_m: float
    match_quality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
