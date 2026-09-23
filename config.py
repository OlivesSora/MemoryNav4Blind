"""Configuration loading with explicit validation and no cwd dependency."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "memory_nav.yaml"


class ConfigError(ValueError):
    """Raised when a MemoryNav configuration is invalid."""


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML config: {path}") from exc
    if not isinstance(value, dict):
        raise ConfigError("config root must be a mapping")
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults and optionally merge a user configuration."""
    config = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        user_path = Path(path).expanduser().resolve()
        if user_path != DEFAULT_CONFIG_PATH:
            config = _merge(deepcopy(config), _read_yaml(user_path))
    _validate(config)
    routes_dir = Path(config["routes_dir"])
    if not routes_dir.is_absolute():
        routes_dir = PROJECT_ROOT / routes_dir
    config["routes_dir"] = str(routes_dir.resolve())
    return config


def _validate(config: Mapping[str, Any]) -> None:
    required = ("schema_version", "coordinate_system", "routes_dir", "trajectory", "quality", "matching", "deviation", "anchors", "voice", "vio")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"missing config keys: {', '.join(missing)}")
    if config["coordinate_system"] != "GCJ-02":
        raise ConfigError("coordinate_system must currently be GCJ-02")
    positive = {
        "trajectory.resample_spacing_m": config["trajectory"]["resample_spacing_m"],
        "trajectory.max_speed_m_s": config["trajectory"]["max_speed_m_s"],
        "quality.minimum_gps_samples": config["quality"]["minimum_gps_samples"],
        "quality.minimum_route_length_m": config["quality"]["minimum_route_length_m"],
        "quality.maximum_gps_gap_s": config["quality"]["maximum_gps_gap_s"],
        "deviation.warning_m": config["deviation"]["warning_m"],
        "deviation.severe_m": config["deviation"]["severe_m"],
    }
    invalid = [name for name, value in positive.items() if not isinstance(value, (int, float)) or value <= 0]
    if invalid:
        raise ConfigError(f"values must be positive: {', '.join(invalid)}")
    if config["deviation"]["severe_m"] <= config["deviation"]["warning_m"]:
        raise ConfigError("deviation.severe_m must exceed warning_m")
    ratio = config["quality"]["maximum_rejection_ratio"]
    if not isinstance(ratio, (int, float)) or not 0 <= ratio < 1:
        raise ConfigError("quality.maximum_rejection_ratio must be in [0, 1)")
    _validate_segmentation(config)


def _validate_segmentation(config: Mapping[str, Any]) -> None:
    segmentation = config.get("segmentation")
    if segmentation is None:
        return
    if not isinstance(segmentation, Mapping):
        raise ConfigError("segmentation must be a mapping")
    for key in ("origin_x_ratio", "origin_y_ratio"):
        value = segmentation.get(key, 0.5)
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ConfigError(f"segmentation.{key} must be in [0, 1]")
    radius = segmentation.get("radius_ratio", 0.25)
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ConfigError("segmentation.radius_ratio must be positive")
    inner = segmentation.get("inner_radius_ratio", 0.05)
    if not isinstance(inner, (int, float)) or not 0.0 <= inner < radius:
        raise ConfigError("segmentation.inner_radius_ratio must be in [0, radius_ratio)")
    obstacle_ratio = segmentation.get("max_obstacle_ratio", 0.005)
    if not isinstance(obstacle_ratio, (int, float)) or not 0.0 <= obstacle_ratio <= 1.0:
        raise ConfigError("segmentation.max_obstacle_ratio must be in [0, 1]")
    tolerance = segmentation.get("command_tolerance_hours", 1)
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ConfigError("segmentation.command_tolerance_hours must be a non-negative integer")
    forward = segmentation.get("forward_hours", [9, 10, 11, 12, 1, 2, 3])
    if not forward:
        raise ConfigError("segmentation.forward_hours cannot be empty")
    messages = segmentation.get("messages")
    if messages is not None:
        if not isinstance(messages, Mapping):
            raise ConfigError("segmentation.messages must be a mapping")
        for key in ("no_walkable", "no_walkable_rotate"):
            value = messages.get(key)
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"segmentation.messages.{key} must be a string")
    online = segmentation.get("online")
    if online is None:
        return
    if not isinstance(online, Mapping):
        raise ConfigError("segmentation.online must be a mapping")
    backend = online.get("backend", "pytorch")
    if backend not in {"pytorch", "tensorrt"}:
        raise ConfigError("segmentation.online.backend must be pytorch or tensorrt")
    if backend == "tensorrt" and online.get("device", "cuda") != "cuda":
        raise ConfigError("segmentation.online TensorRT backend requires device cuda")
    trt_warmup = online.get("trt_warmup", 1)
    if not isinstance(trt_warmup, int) or trt_warmup < 0:
        raise ConfigError("segmentation.online.trt_warmup must be a non-negative integer")
    scale = online.get("input_scale", 0.5)
    if not isinstance(scale, (int, float)) or not 0.0 < scale <= 1.0:
        raise ConfigError("segmentation.online.input_scale must be in (0, 1]")
    max_hz = online.get("max_hz", 1.0)
    if not isinstance(max_hz, (int, float)) or max_hz <= 0:
        raise ConfigError("segmentation.online.max_hz must be positive")
    vis_interval = online.get("vis_interval_s", 1.0)
    if not isinstance(vis_interval, (int, float)) or vis_interval < 0:
        raise ConfigError("segmentation.online.vis_interval_s cannot be negative")
    min_size = online.get("min_size_test")
    if min_size is not None and (not isinstance(min_size, int) or min_size <= 0):
        raise ConfigError("segmentation.online.min_size_test must be a positive integer")
    online_radius = online.get("radius_ratio")
    if online_radius is not None:
        if not isinstance(online_radius, (int, float)) or online_radius <= 0:
            raise ConfigError("segmentation.online.radius_ratio must be positive")
        if online_radius <= inner:
            raise ConfigError("segmentation.online.radius_ratio must exceed inner_radius_ratio")
