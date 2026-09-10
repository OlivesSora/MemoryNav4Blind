"""Turn match/state updates into deduplicated anchor and safety prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from memory_nav.interaction.voice_prompt import Prompt, VoicePromptScheduler
from memory_nav.recording.anchor_collector import AnchorCandidate


@dataclass(frozen=True)
class GuidanceUpdate:
    prompts: tuple[Prompt, ...]
    next_anchor_id: str | None
    distance_to_next_anchor_m: float | None
    visual_anchor: AnchorCandidate | None = None


class GuidanceController:
    def __init__(self, anchors: Sequence[AnchorCandidate], advance_distance_m: float = 8.0, arrival_distance_m: float = 2.0, cooldown_s: float = 10.0):
        if advance_distance_m <= arrival_distance_m or arrival_distance_m < 0:
            raise ValueError("advance distance must exceed non-negative arrival distance")
        self.anchors = sorted((anchor for anchor in anchors if anchor.confirmed), key=lambda anchor: anchor.s_m)
        self.advance_distance_m = advance_distance_m
        self.arrival_distance_m = arrival_distance_m
        self.scheduler = VoicePromptScheduler(cooldown_s)
        self._next_anchor = 0
        self._previous_state = "normal"

    @staticmethod
    def _anchor_text(anchor: AnchorCandidate, phase: str) -> str:
        if anchor.prompt_text:
            return anchor.prompt_text
        if anchor.kind == "end":
            return "即将到达终点" if phase == "advance" else "已到达终点"
        return "即将到达导航锚点" if phase == "advance" else "已到达导航锚点"

    def update(self, matched_s_m: float | None, navigation_state: str, cross_track_error_m: float | None, pause_progress: bool, now_s: float) -> GuidanceUpdate:
        prompts: list[Prompt] = []
        if navigation_state != self._previous_state:
            if navigation_state in ("deviated", "severe"):
                side = "左侧" if (cross_track_error_m or 0) > 0 else "右侧"
                distance = 0.0 if cross_track_error_m is None else abs(cross_track_error_m)
                text = f"已偏离路线{side}{distance:.1f}米，请返回路线"
                prompt = self.scheduler.request(Prompt(f"state:{navigation_state}", text, 100), now_s)
            elif navigation_state == "lost":
                prompt = self.scheduler.request(Prompt("state:lost", "定位信号丢失，请暂停并等待定位恢复", 100), now_s)
            elif navigation_state in ("recovering", "normal") and self._previous_state in ("deviated", "severe", "lost", "recovering"):
                prompt = self.scheduler.request(Prompt("state:recovered", "已恢复到参考路线", 80), now_s)
            elif navigation_state == "completed":
                prompt = self.scheduler.request(Prompt("state:completed", "路线重演已完成", 80), now_s)
            else:
                prompt = None
            if prompt:
                prompts.append(prompt)
        self._previous_state = navigation_state

        visual_anchor = None
        if matched_s_m is not None and not pause_progress:
            while self._next_anchor < len(self.anchors) and matched_s_m > self.anchors[self._next_anchor].s_m + self.arrival_distance_m:
                self._next_anchor += 1
            if self._next_anchor < len(self.anchors):
                anchor = self.anchors[self._next_anchor]
                distance = anchor.s_m - matched_s_m
                if distance <= self.advance_distance_m:
                    prompt = self.scheduler.anchor_prompt(anchor.anchor_id, "advance", self._anchor_text(anchor, "advance"), now_s)
                    if prompt:
                        prompts.append(prompt)
                if abs(distance) <= self.arrival_distance_m:
                    prompt = self.scheduler.anchor_prompt(anchor.anchor_id, "arrival", self._anchor_text(anchor, "arrival"), now_s)
                    if prompt:
                        prompts.append(prompt)
                    if anchor.image_path:
                        visual_anchor = anchor
                    self._next_anchor += 1

        if self._next_anchor < len(self.anchors) and matched_s_m is not None:
            next_anchor = self.anchors[self._next_anchor]
            return GuidanceUpdate(tuple(prompts), next_anchor.anchor_id, next_anchor.s_m - matched_s_m, visual_anchor)
        return GuidanceUpdate(tuple(prompts), None, None, visual_anchor)
