"""Hysteretic deviation state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NavigationState(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    DEVIATED = "deviated"
    SEVERE = "severe"
    RECOVERING = "recovering"
    LOST = "lost"
    COMPLETED = "completed"


@dataclass(frozen=True)
class StateUpdate:
    state: NavigationState
    changed: bool
    pause_progress: bool


class DeviationMonitor:
    def __init__(self, warning_m: float = 3.0, severe_m: float = 5.0, phone_scale: float = 2.0, enter_samples: int = 3, recover_samples: int = 3):
        if warning_m <= 0 or severe_m <= warning_m:
            raise ValueError("invalid deviation thresholds")
        self.warning_m = warning_m
        self.severe_m = severe_m
        self.phone_scale = phone_scale
        self.enter_samples = enter_samples
        self.recover_samples = recover_samples
        self.state = NavigationState.NORMAL
        self._outside_count = 0
        self._inside_count = 0

    def update(self, cross_track_m: float | None, match_quality: str, source: str = "rtk") -> StateUpdate:
        previous = self.state
        if cross_track_m is None or match_quality == "lost":
            self.state = NavigationState.LOST
            self._outside_count = self._inside_count = 0
        else:
            scale = self.phone_scale if source != "rtk" else 1.0
            warning = self.warning_m * scale
            severe = self.severe_m * scale
            distance = abs(cross_track_m)
            if distance >= warning:
                self._outside_count += 1
                self._inside_count = 0
                if self._outside_count >= self.enter_samples:
                    self.state = NavigationState.SEVERE if distance >= severe else NavigationState.DEVIATED
                elif self.state in (NavigationState.NORMAL, NavigationState.RECOVERING, NavigationState.LOST):
                    self.state = NavigationState.DEGRADED
            else:
                self._inside_count += 1
                self._outside_count = 0
                if self.state in (NavigationState.DEVIATED, NavigationState.SEVERE, NavigationState.LOST):
                    self.state = NavigationState.RECOVERING
                if self._inside_count >= self.recover_samples:
                    self.state = NavigationState.NORMAL
                if self.state == NavigationState.NORMAL and match_quality == "degraded":
                    self.state = NavigationState.DEGRADED
        return StateUpdate(self.state, self.state != previous, self.state in (NavigationState.DEVIATED, NavigationState.SEVERE, NavigationState.LOST))

    def complete(self) -> StateUpdate:
        previous = self.state
        self.state = NavigationState.COMPLETED
        return StateUpdate(self.state, self.state != previous, True)
