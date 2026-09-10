"""Pure prompt scheduling; actual TTS playback remains an adapter concern."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from collections.abc import Callable


@dataclass(frozen=True)
class Prompt:
    key: str
    text: str
    priority: int


class VoicePromptScheduler:
    def __init__(self, cooldown_s: float = 10.0):
        if cooldown_s < 0:
            raise ValueError("cooldown_s cannot be negative")
        self.cooldown_s = cooldown_s
        self._last_played: dict[str, float] = {}
        self._anchor_events: set[tuple[str, str]] = set()

    def request(self, prompt: Prompt, now_s: float) -> Prompt | None:
        previous = self._last_played.get(prompt.key)
        if previous is not None and now_s - previous < self.cooldown_s:
            return None
        self._last_played[prompt.key] = now_s
        return prompt

    def anchor_prompt(self, anchor_id: str, phase: str, text: str, now_s: float) -> Prompt | None:
        if phase not in ("advance", "arrival"):
            raise ValueError("anchor phase must be advance or arrival")
        event = (anchor_id, phase)
        if event in self._anchor_events:
            return None
        prompt = self.request(Prompt(f"anchor:{anchor_id}:{phase}", text, 10 if phase == "arrival" else 5), now_s)
        if prompt:
            self._anchor_events.add(event)
        return prompt

    def reset_route(self) -> None:
        self._anchor_events.clear()


class VoicePlaybackWorker:
    """Run a blocking TTS callable away from the navigation loop."""

    def __init__(self, speaker: Callable[[str], object], max_queue: int = 16):
        self._speaker = speaker
        self._queue: queue.PriorityQueue[tuple[int, int, Prompt]] = queue.PriorityQueue(max_queue)
        self._sequence = 0
        self._closed = False
        self.errors: list[Exception] = []
        self._thread = threading.Thread(target=self._run, name="memory-nav-voice", daemon=True)
        self._thread.start()

    def submit(self, prompt: Prompt) -> bool:
        if self._closed:
            return False
        self._sequence += 1
        try:
            self._queue.put_nowait((-prompt.priority, self._sequence, prompt))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            try:
                _priority, _sequence, prompt = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._closed:
                    return
                continue
            try:
                self._speaker(prompt.text)
            except Exception as exc:  # Playback failures must not kill navigation.
                self.errors.append(exc)
            finally:
                self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._thread.join(timeout=2.0)
