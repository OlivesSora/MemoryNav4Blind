"""Small-area local tangent-plane conversion for GCJ-02 coordinates."""

from __future__ import annotations

import math


EARTH_RADIUS_M = 6_378_137.0


class LocalFrame:
    def __init__(self, origin_longitude: float, origin_latitude: float):
        if not -180 <= origin_longitude <= 180 or not -90 <= origin_latitude <= 90:
            raise ValueError("invalid local-frame origin")
        self.origin_longitude = float(origin_longitude)
        self.origin_latitude = float(origin_latitude)
        self._origin_lat_rad = math.radians(self.origin_latitude)
        self._cos_origin = math.cos(self._origin_lat_rad)
        if abs(self._cos_origin) < 1e-8:
            raise ValueError("local frame is undefined at the poles")

    def to_local(self, longitude: float, latitude: float) -> tuple[float, float]:
        east = EARTH_RADIUS_M * math.radians(longitude - self.origin_longitude) * self._cos_origin
        north = EARTH_RADIUS_M * math.radians(latitude - self.origin_latitude)
        return east, north

    def to_geodetic(self, east_m: float, north_m: float) -> tuple[float, float]:
        longitude = self.origin_longitude + math.degrees(east_m / (EARTH_RADIUS_M * self._cos_origin))
        latitude = self.origin_latitude + math.degrees(north_m / EARTH_RADIUS_M)
        return longitude, latitude


def wrap_to_180(angle_deg: float) -> float:
    value = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 and angle_deg > 0 else value


def heading_from_delta(east_delta: float, north_delta: float) -> float:
    """Heading in degrees: north=0, east=90, clockwise positive."""
    if east_delta == 0 and north_delta == 0:
        raise ValueError("zero displacement has no heading")
    return math.degrees(math.atan2(east_delta, north_delta)) % 360.0
