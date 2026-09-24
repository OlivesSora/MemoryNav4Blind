"""Apply segmentation avoidance to a tracked clock direction.

The guard queries a :class:`~memory_nav.segmentation.providers.MaskProvider`
for the current frame, evaluates the walkable region and, when the tracked
direction is blocked, returns a corrected direction plus a voice prompt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from memory_nav.interaction.voice_prompt import Prompt, VoicePromptScheduler
from memory_nav.segmentation.avoidance import SegmentationAvoidance, SegmentationResult
from memory_nav.segmentation.providers import MaskProvider

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationDecision:
    result: SegmentationResult
    command_clock: int
    command_text: str
    override_used: bool
    prompts: tuple[Prompt, ...] = ()


class SegmentationGuard:
    def __init__(
        self,
        avoidance: SegmentationAvoidance,
        provider: MaskProvider,
        cooldown_s: float = 5.0,
        visualizer: Optional[Callable[[np.ndarray, SegmentationResult, object, Optional[np.ndarray]], None]] = None,
        no_walkable_text: str = "前方未检测到可通行区域，请停止前进",
        rotate_text: str = "无可行区域，请旋转一下",
    ):
        self.avoidance = avoidance
        self.provider = provider
        self.scheduler = VoicePromptScheduler(cooldown_s)
        self.visualizer = visualizer
        self.no_walkable_text = no_walkable_text
        self.rotate_text = rotate_text

    def _get_mask_and_frame(
        self, frame_id: str, frame: Optional[np.ndarray],
        captured_at_s: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return a mask with the exact source frame used to infer it."""
        get_observation = getattr(self.provider, "get_observation", None)
        if get_observation is not None:
            observation = (
                get_observation(frame_id, frame, captured_at_s=captured_at_s)
                if captured_at_s is not None else get_observation(frame_id, frame)
            )
            if observation is None:
                return None, None
            return (
                np.asarray(observation.mask).astype(bool),
                observation.frame,
            )
        return self.provider.get_mask(frame_id, frame), frame

    def _text_for(self, result: SegmentationResult, command_text: str) -> str:
        if result.consistent or result.out_of_scope:
            return command_text
        if result.fallback_used:
            return f"前方不可通行，建议向{result.seg_clock}点钟方向前进"
        return self.no_walkable_text

    def apply(
        self,
        command_clock: int | str,
        command_text: str,
        frame_id: str,
        frame: Optional[np.ndarray] = None,
        now_s: Optional[float] = None,
        captured_at_s: Optional[float] = None,
    ) -> Optional[SegmentationDecision]:
        mask, source_frame = self._get_mask_and_frame(frame_id, frame, captured_at_s)
        if mask is None:
            return None
        result = self.avoidance.evaluate(command_clock, mask)
        if self.visualizer is not None and source_frame is not None:
            try:
                self.visualizer(source_frame, result, self.avoidance.mapper, mask)
            except Exception as exc:  # Visualization must never break navigation.
                LOGGER.warning("segmentation visualization failed: %s", exc)
        text = self._text_for(result, command_text)
        override_used = result.fallback_used or result.warning == "no_walkable"
        now = time.monotonic() if now_s is None else now_s
        prompts: list[Prompt] = []
        if result.fallback_used:
            prompt = self.scheduler.request(
                Prompt(f"seg:command_blocked:{result.seg_clock}", text, 90), now
            )
            if prompt is not None:
                prompts.append(prompt)
        elif result.warning == "no_walkable":
            # Stable keys (no clock) so rotating does not re-trigger faster
            # than the configured cooldown.
            stop_prompt = self.scheduler.request(
                Prompt("seg:no_walkable", self.no_walkable_text, 90), now
            )
            if stop_prompt is not None:
                prompts.append(stop_prompt)
            rotate_prompt = self.scheduler.request(
                Prompt("seg:no_walkable_rotate", self.rotate_text, 90), now
            )
            if rotate_prompt is not None:
                prompts.append(rotate_prompt)
        return SegmentationDecision(
            result=result,
            command_clock=result.seg_clock,
            command_text=text,
            override_used=override_used,
            prompts=tuple(prompts),
        )

    def observe(
        self,
        frame_id: str,
        frame: Optional[np.ndarray] = None,
        captured_at_s: Optional[float] = None,
    ) -> dict:
        """Fetch a mask without a tracked command.

        Used when the navigation pose is unavailable: the provider is started
        and masks are produced/visualized, but no clock direction is evaluated
        and no voice prompt is emitted.
        """
        mask, source_frame = self._get_mask_and_frame(frame_id, frame, captured_at_s)
        if mask is None:
            return {"status": "starting", "mask_ready": False}
        if self.visualizer is not None and source_frame is not None:
            save_mask = getattr(self.visualizer, "save_mask", None)
            if save_mask is not None:
                try:
                    save_mask(source_frame, self.avoidance.mapper, mask)
                except Exception as exc:  # Visualization must never break navigation.
                    LOGGER.warning("segmentation visualization failed: %s", exc)
        return {"status": "pose_unavailable", "mask_ready": True}

    def close(self) -> None:
        provider_close = getattr(self.provider, "close", None)
        if provider_close is not None:
            provider_close()
        if self.visualizer is not None:
            visualizer_close = getattr(self.visualizer, "close", None)
            if visualizer_close is not None:
                visualizer_close()
