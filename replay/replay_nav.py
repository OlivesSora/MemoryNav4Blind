"""Online GPS+IMU route matcher using the existing hardware adapters."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from dataclasses import asdict
import cv2
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.models import ReferencePoint
from memory_nav.recording.session_writer import validate_route_id
from memory_nav.replay.deviation import DeviationMonitor
from memory_nav.replay.matcher import RouteMatcher
from memory_nav.replay.guidance import GuidanceController
from memory_nav.recording.anchor_collector import load_anchors
from memory_nav.interaction.voice_prompt import VoicePlaybackWorker
from memory_nav.trajectory.coordinate import LocalFrame
from memory_nav.replay.anchor_matcher import XFeatAnchorMatcher
from memory_nav.config import PROJECT_ROOT
from memory_nav.vio import OnlineVioAligner, VioSubscriber
from memory_nav.interaction.voice_prompt import Prompt
from memory_nav.trajectory.coordinate import wrap_to_180


def make_command(heading_deg: float, target_heading_deg: float, distance_m: float) -> tuple[str, str]:
    """Convert a look-ahead bearing into a clock-direction instruction."""
    relative = wrap_to_180(target_heading_deg - heading_deg)
    hour = round(relative / 30.0) % 12
    clock = str(hour or 12)
    distance = max(0.0, float(distance_m))
    if abs(relative) <= 22.5:
        return clock, f"{clock}点钟方向直行约{distance:.0f}米" if distance > 0.5 else f"{clock}点钟方向原地停住"
    if abs(relative) > 120:
        return clock, f"原地转至{clock}点钟方向，然后前进约{distance:.0f}米"
    return clock, f"{clock}点钟方向前进约{distance:.0f}米"


LOGGER = logging.getLogger(__name__)


def validate_ready_route(route_dir: Path) -> None:
    manifest_path = route_dir / "manifest.json"
    quality_path = route_dir / "quality_report.json"
    reference_path = route_dir / "reference_trajectory.json"
    missing = [path.name for path in (manifest_path, quality_path, reference_path) if not path.is_file()]
    if missing:
        raise ValueError(f"route is incomplete; missing: {', '.join(missing)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "READY" or not quality.get("ready"):
        reasons = quality.get("failure_reasons") or [f"manifest state={manifest.get('state')}"]
        raise ValueError(f"route is not ready: {'; '.join(str(reason) for reason in reasons)}")


class ReplayRunner:
    def __init__(self, route_dir: Path, config: dict, gps=None, imu=None, speaker=None, camera=None, visual_matcher=None, vio=None, voice_directions: bool = False):
        self.route_dir = route_dir
        route = json.loads((route_dir / "reference_trajectory.json").read_text(encoding="utf-8"))
        self.frame = LocalFrame(route["origin"]["longitude"], route["origin"]["latitude"])
        self.matcher = RouteMatcher([ReferencePoint(**point) for point in route["points"]], **config["matching"])
        self.monitor = DeviationMonitor(**config["deviation"])
        self.arrival_distance_m = float(config["anchors"]["arrival_distance_m"])
        anchors_path = route_dir / "anchors.json"
        anchors = load_anchors(anchors_path) if anchors_path.is_file() else []
        self.guidance = GuidanceController(
            anchors,
            advance_distance_m=float(config["anchors"]["advance_distance_m"]),
            arrival_distance_m=self.arrival_distance_m,
            cooldown_s=float(config["voice"]["cooldown_s"]),
        )
        self.voice_worker = None if speaker is None else VoicePlaybackWorker(speaker)
        self.voice_directions = voice_directions
        self.camera = camera
        self.visual_matcher = visual_matcher
        if gps is None or imu is None:
            from utils.gps import GPS
            from utils.imu import IMU

            gps = gps or GPS()
            imu = imu or IMU()
        self.gps = gps
        self.imu = imu
        self.vio = vio
        self.vio_aligner = None if vio is None else OnlineVioAligner(
            minimum_pairs=int(config["vio"]["online_minimum_pairs"]),
            max_rms_m=float(config["vio"]["maximum_alignment_rms_m"]),
        )
        self.running = True
        self._last_spoken_clock: int | None = None
        self._direction_min_clocks = max(
            1, int(config["voice"].get("direction_change_min_clocks", 2))
        )

    def step(self) -> dict | None:
        get_sample = getattr(self.gps, "get_sample", None)
        rich_sample = get_sample() if get_sample is not None else None
        if rich_sample is not None:
            coordinate = [rich_sample["longitude"], rich_sample["latitude"]]
            gps_state = rich_sample.get("state")
            source = rich_sample.get("source", "unknown")
        elif get_sample is not None:
            coordinate, gps_state, source = None, None, "unknown"
        else:
            coordinate, gps_state = self.gps.get_gps()
            source = "rtk" if gps_state == "good" else "phone"
        heading = self.imu.get_heading()
        if coordinate is None or heading is None:
            update = self.monitor.update(None, "lost")
            guidance = self.guidance.update(None, update.state.value, None, True, time.monotonic())
            self._play(guidance.prompts)
            return {"navigation_state": update.state.value, "match_quality": "lost", "pause_progress": True, "prompts": [asdict(prompt) for prompt in guidance.prompts]}
        east, north = self.frame.to_local(float(coordinate[0]), float(coordinate[1]))
        vio_sample = None if self.vio is None else self.vio.latest()
        if self.vio_aligner is not None:
            self.vio_aligner.add(vio_sample, east, north)
            projected = self.vio_aligner.project(vio_sample)
            if projected is not None:
                east, north = projected
        match = self.matcher.match(east, north, float(heading))
        target_s = min(match.matched_s_m + 8.0, self.matcher.points[-1].s_m)
        target = self.matcher.points[-1]
        for low, high in zip(self.matcher.points, self.matcher.points[1:]):
            if low.s_m <= target_s <= high.s_m:
                ratio = (target_s - low.s_m) / max(high.s_m - low.s_m, 1e-9)
                target_heading = (low.heading_deg + ratio * ((high.heading_deg - low.heading_deg + 180) % 360 - 180)) % 360
                break
        else:
            target_heading = target.heading_deg
        command_clock, command_text = make_command(float(heading), target_heading, target_s - match.matched_s_m)
        update = self.monitor.update(match.cross_track_error_m, match.match_quality, source)
        if match.match_quality == "good" and self.matcher.is_complete(self.arrival_distance_m):
            update = self.monitor.complete()
        guidance = self.guidance.update(match.matched_s_m, update.state.value, match.cross_track_error_m, update.pause_progress, time.monotonic())
        self._play(guidance.prompts)
        emitted_prompts = list(guidance.prompts)
        if self.voice_directions and self.voice_worker is not None and update.state.value in ("normal", "recovering", "degraded"):
            cur_clock = int(command_clock)
            if self._last_spoken_clock is None:
                clock_changed = True
            else:
                delta = abs(cur_clock - self._last_spoken_clock)
                clock_changed = min(delta, 12 - delta) >= self._direction_min_clocks
            if clock_changed:
                direction_prompt = self.guidance.scheduler.request(
                    Prompt(f"direction:{command_clock}", command_text, 2), time.monotonic()
                )
                if direction_prompt:
                    self._play((direction_prompt,))
                    emitted_prompts.append(direction_prompt)
                    self._last_spoken_clock = cur_clock
        visual_result = self._verify_anchor(guidance.visual_anchor)
        return {
            **match.to_dict(),
            "navigation_state": update.state.value,
            "pause_progress": update.pause_progress,
            "gps_source": source,
            "next_anchor_id": guidance.next_anchor_id,
            "distance_to_next_anchor_m": guidance.distance_to_next_anchor_m,
            "prompts": [asdict(prompt) for prompt in emitted_prompts],
            "visual_anchor": visual_result,
            "vio_state": None if self.vio_aligner is None else self.vio_aligner.state,
            "vio_alignment_rms_m": None if self.vio_aligner is None else self.vio_aligner.rms_m,
            "position_source": "vio" if self.vio_aligner is not None and self.vio_aligner.transform is not None else "gps",
            "target_heading_deg": target_heading,
            "command_clock": command_clock,
            "command": command_text,
        }

    def _play(self, prompts) -> None:
        if self.voice_worker is not None:
            for prompt in prompts:
                if not self.voice_worker.submit(prompt):
                    LOGGER.warning("voice queue full; dropped prompt %s", prompt.key)

    def _verify_anchor(self, anchor) -> dict | None:
        if anchor is None or self.camera is None or self.visual_matcher is None:
            return None
        try:
            reference_path = (self.route_dir / anchor.image_path).resolve()
            if self.route_dir.resolve() not in reference_path.parents:
                raise ValueError("anchor image escapes route directory")
            reference = cv2.imread(str(reference_path))
            if reference is None:
                raise FileNotFoundError(reference_path)
            current_rgb, _frame = self.camera.capture_frame()
            if current_rgb is None:
                raise RuntimeError("camera frame unavailable")
            current_bgr = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
            result = self.visual_matcher.match(reference, current_bgr)
            return {"anchor_id": anchor.anchor_id, **asdict(result), "error": None}
        except Exception as exc:
            LOGGER.warning("visual anchor verification failed for %s: %s", anchor.anchor_id, exc)
            return {"anchor_id": anchor.anchor_id, "matched": False, "error": str(exc)}

    def close(self) -> None:
        if self.voice_worker is not None:
            self.voice_worker.close()
        release = getattr(self.camera, "release", None)
        if release:
            release()
        stop = getattr(self.imu, "stop", None)
        if stop:
            stop()
        if self.vio is not None:
            self.vio.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--voice", action="store_true", help="play prompts through utils.voice")
    parser.add_argument("--visual-anchors", action="store_true", help="verify anchor images with the local XFeat model")
    parser.add_argument("--vio-topic", help="subscribe to live VINS odometry on this ROS 2 topic")
    parser.add_argument("--ros-image-topic", help="subscribe to the VINS ROS image topic for visual anchors")
    parser.add_argument("--imu-port", default="/dev/imu")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    config = load_config(args.config)
    route_dir = Path(config["routes_dir"]) / validate_route_id(args.route_id)
    try:
        validate_ready_route(route_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    speaker = None
    if args.voice:
        from utils.voice import TextToSpeechPlayer

        speaker = TextToSpeechPlayer(output_dir=str(route_dir / "tts_cache")).say
    camera = visual_matcher = vio = None
    if args.visual_anchors and args.ros_image_topic and args.vio_topic:
        vio = VioSubscriber(args.vio_topic, image_topic=args.ros_image_topic)
        vio.start()
        camera = vio
        visual_matcher = XFeatAnchorMatcher(PROJECT_ROOT)
    elif args.visual_anchors:
        from utils.glasses_camera import Camera

        camera = Camera()
        visual_matcher = XFeatAnchorMatcher(PROJECT_ROOT)
    if args.vio_topic and vio is None:
        vio = VioSubscriber(args.vio_topic)
        vio.start()
    if args.vio_topic:
        # VINS bridge owns /dev/imu and publishes /imu0; subscribe instead of
        # reopening the serial port (avoids double-read contention).
        from memory_nav.vio import ImuSubscriber
        imu = ImuSubscriber("/imu0")
    else:
        from utils.imu import IMU
        imu = IMU(port=args.imu_port)
    runner = ReplayRunner(route_dir, config, imu=imu, speaker=speaker, camera=camera, visual_matcher=visual_matcher, vio=vio, voice_directions=args.voice)
    signal.signal(signal.SIGINT, lambda *_: setattr(runner, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(runner, "running", False))
    try:
        while runner.running:
            print(json.dumps(runner.step(), ensure_ascii=False), flush=True)
            time.sleep(args.interval)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
