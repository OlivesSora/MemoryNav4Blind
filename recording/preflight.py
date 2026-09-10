"""Non-mutating sensor readiness checks before a route is created."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorCheck:
    name: str
    ready: bool
    detail: str


def check_sensors(gps, imu, camera=None) -> list[SensorCheck]:
    checks: list[SensorCheck] = []
    diagnostics = imu.get_diagnostics()
    heading_ready = bool(diagnostics.get("heading_ready"))
    age = diagnostics.get("last_ahrs_age_s")
    # IMU readiness is gated only on a settled heading; the AHRS freshness
    # (age_s) is reported for diagnostics but no longer fails the check.
    imu_ready = heading_ready
    checks.append(SensorCheck("imu", imu_ready, f"heading_ready={heading_ready}, age_s={age}"))

    rich_getter = getattr(gps, "get_sample", None)
    if rich_getter is not None:
        sample = rich_getter()
        gps_ready = sample is not None and sample.get("age_s", 0.0) is not None and float(sample.get("age_s", 0.0)) <= 10.0
        detail = "unavailable" if sample is None else f"source={sample.get('source')}, state={sample.get('state')}, age_s={sample.get('age_s')}"
    else:
        coordinate, state = gps.get_gps()
        gps_ready = coordinate is not None
        detail = f"state={state}"
    checks.append(SensorCheck("gps", gps_ready, detail))

    if camera is not None:
        image, metadata = camera.capture_frame_with_metadata(timeout=2.0) if hasattr(camera, "capture_frame_with_metadata") else camera.capture_frame()
        checks.append(SensorCheck("camera", image is not None, "unavailable" if image is None else f"frame={metadata}"))
    return checks


def require_ready(checks: list[SensorCheck]) -> None:
    failed = [check for check in checks if not check.ready]
    if failed:
        summary = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise RuntimeError(f"sensor preflight failed: {summary}")
