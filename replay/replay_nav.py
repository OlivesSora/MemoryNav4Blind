"""Online GPS+IMU route matcher using the existing hardware adapters."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
import cv2
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.models import ReferencePoint
from memory_nav.recording.session_writer import validate_route_id
from memory_nav.replay.matcher import RouteMatcher
from memory_nav.replay.guidance import GuidanceController
from memory_nav.recording.anchor_collector import load_anchors
from memory_nav.interaction.voice_prompt import VoicePlaybackWorker
from memory_nav.trajectory.coordinate import LocalFrame
from memory_nav.replay.anchor_matcher import XFeatAnchorMatcher
from memory_nav.config import PROJECT_ROOT
from memory_nav.vio import OnlineVioAligner, VioSubscriber
from memory_nav.interaction.voice_prompt import Prompt
from memory_nav.trajectory.coordinate import heading_from_delta, wrap_to_180


def make_command(heading_deg: float, target_heading_deg: float, distance_m: float) -> tuple[str, str]:
    """Convert a look-ahead bearing into a clock-direction instruction."""
    relative = wrap_to_180(target_heading_deg - heading_deg)
    hour = round(relative / 30.0) % 12
    clock = str(hour or 12)
    if abs(relative) <= 22.5:
        return clock, f"{clock}点钟方向直行"
    if abs(relative) > 120:
        return clock, f"原地转至{clock}点钟方向，然后前进"
    return clock, f"{clock}点钟方向前进"


def make_gps_action(heading_deg: float, target_heading_deg: float) -> str:
    relative = wrap_to_180(target_heading_deg - heading_deg)
    if abs(relative) <= 22.5:
        return "直行"
    if abs(relative) > 120:
        return "掉头"
    return "右转" if relative > 0 else "左转"


LOGGER = logging.getLogger(__name__)
BLIND_NAV_PROJECT_DIR = os.getenv(
    "BLIND_NAV_PROJECT_DIR", "/home/wheeltec/projects/blind-nav-server"
)


def create_hardware_agent_client():
    """Create the shared hardware-gateway client used by visual navigation."""
    if BLIND_NAV_PROJECT_DIR not in sys.path:
        sys.path.insert(0, BLIND_NAV_PROJECT_DIR)
    from src.hardware_agent_client import HardwareAgentClient

    return HardwareAgentClient()


class HardwareGatewayGPS:
    """Synchronous GPS adapter backed by the shared hardware gateway."""

    def __init__(self, hardware_agent) -> None:
        self._hardware_agent = hardware_agent

    def get_gps(self):
        return asyncio.run(self._hardware_agent.get_gps())


class HardwareGatewayIMU:
    """Synchronous heading adapter backed by the shared hardware gateway."""

    def __init__(self, hardware_agent) -> None:
        self._hardware_agent = hardware_agent

    def get_heading(self):
        return asyncio.run(self._hardware_agent.get_heading())

    def stop(self) -> None:
        pass


class HardwareGatewayCamera:
    """Camera adapter matching the local Camera.capture_frame interface."""

    def __init__(self, hardware_agent) -> None:
        self._hardware_agent = hardware_agent

    def capture_frame(self):
        return asyncio.run(self._hardware_agent.get_state_image(angle=0))

    def release(self) -> None:
        pass


class FollowGPSLogger:
    """Append follow samples in the outdoor_nav ``output.jsonl`` format."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def append(
        self,
        coordinate: list[float] | None,
        heading_deg: float | None,
        command: str | None = None,
        command_clock: str | None = None,
        gps_action: str | None = None,
        target_heading_deg: float | None = None,
        projected_position: tuple[float, float] | None = None,
        target_position: tuple[float, float] | None = None,
        match=None,
        voice_queued: bool = False,
        segmentation: dict | None = None,
    ) -> None:
        record = {
            "log_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "camera_time_stamp": None,
            "image_path": None,
            "model_output_raw": None,
            "guide": command,
            "played": False,
            "voice_queued": voice_queued,
            "high_level_hint": None,
            "gps_action": gps_action,
            "raw_gps_action": gps_action,
            "command_clock": command_clock,
            "route_instruction": command,
            "angle_diff": None if match is None else match.heading_error_deg,
            "distance": None if match is None else match.distance_m,
            "raw_imu_bearing": heading_deg,
            "cur_bearing": heading_deg,
            "heading_correction": 0.0,
            "heading_correction_updated": False,
            "gps_course_bearing": None,
            "gps_heading_track_distance": 0.0,
            "target_bearing": target_heading_deg,
            "current_pos": coordinate,
            "projected_pos": None if projected_position is None else list(projected_position),
            "lookahead_point": None if target_position is None else list(target_position),
            "route_progress": None if match is None else match.matched_s_m,
            "cross_track_error": None if match is None else match.cross_track_error_m,
            "segmentation": segmentation,
        }
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


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
    def __init__(self, route_dir: Path, config: dict, gps=None, imu=None, speaker=None, camera=None, visual_matcher=None, vio=None, voice_directions: bool = False, follow_log_path: Path | None = None, segmentation_guard=None, segmentation_observe_always: bool = False):
        self.route_dir = route_dir
        route = json.loads((route_dir / "reference_trajectory.json").read_text(encoding="utf-8"))
        self.frame = LocalFrame(route["origin"]["longitude"], route["origin"]["latitude"])
        self.matcher = RouteMatcher([ReferencePoint(**point) for point in route["points"]], **config["matching"])
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
        self.follow_logger = None if follow_log_path is None else FollowGPSLogger(follow_log_path)
        self.vio_aligner = None if vio is None else OnlineVioAligner(
            minimum_pairs=int(config["vio"]["online_minimum_pairs"]),
            max_rms_m=float(config["vio"]["maximum_alignment_rms_m"]),
        )
        self.segmentation_guard = segmentation_guard
        self.segmentation_observe_always = segmentation_observe_always
        self._frame_counter = 0
        self.running = True

    def _capture_frame(self):
        """Capture one camera frame, returning ``None`` on any failure."""
        if self.camera is None:
            return None
        try:
            captured, _frame = self.camera.capture_frame()
            return captured
        except Exception as exc:  # Frame capture must never break navigation.
            LOGGER.warning("segmentation frame capture failed: %s", exc)
            return None

    def _observe_segmentation(self, frame_id: str) -> dict | None:
        observe = getattr(self.segmentation_guard, "observe", None)
        if observe is None:
            return None
        return observe(frame_id, self._capture_frame())

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
            segmentation_status = None
            if self.segmentation_guard is not None and self.segmentation_observe_always:
                self._frame_counter += 1
                segmentation_status = self._observe_segmentation(f"frame_{self._frame_counter}")
            guidance = self.guidance.update(None, "following", None, False, time.monotonic())
            self._play(guidance.prompts)
            if self.follow_logger is not None:
                self.follow_logger.append(coordinate, heading, segmentation=segmentation_status)
            result = {"navigation_state": "following", "match_quality": "unavailable", "pause_progress": False, "prompts": [asdict(prompt) for prompt in guidance.prompts]}
            if self.segmentation_guard is not None and self.segmentation_observe_always:
                result["segmentation"] = segmentation_status
            return result
        east, north = self.frame.to_local(float(coordinate[0]), float(coordinate[1]))
        vio_sample = None if self.vio is None else self.vio.latest()
        if self.vio_aligner is not None:
            self.vio_aligner.add(vio_sample, east, north)
            projected = self.vio_aligner.project(vio_sample)
            if projected is not None:
                east, north = projected
        match = self.matcher.match(east, north, float(heading))
        points = self.matcher.points
        nearest_index = min(
            range(len(points)),
            key=lambda i: (points[i].east_m - east) ** 2 + (points[i].north_m - north) ** 2,
        )
        target_index = min(nearest_index + 1, len(points) - 1)
        target = points[target_index]
        target_east, target_north = target.east_m, target.north_m
        delta_east = target_east - east
        delta_north = target_north - north
        if delta_east == 0 and delta_north == 0:
            target_heading = target.heading_deg
        else:
            target_heading = heading_from_delta(delta_east, delta_north)
        command_clock, command_text = make_command(
            float(heading), target_heading, math.hypot(delta_east, delta_north)
        )
        segmentation_result = None
        segmentation_prompts: tuple = ()
        if self.segmentation_guard is not None:
            self._frame_counter += 1
            frame_id = f"frame_{self._frame_counter}"
            captured = self._capture_frame()
            decision = self.segmentation_guard.apply(
                command_clock, command_text, frame_id, captured, time.monotonic()
            )
            if decision is not None:
                segmentation_result = decision.result.to_dict()
                command_clock = str(decision.command_clock)
                command_text = decision.command_text
                segmentation_prompts = decision.prompts
        gps_action = make_gps_action(float(heading), target_heading)
        navigation_state = "completed" if self.matcher.is_complete(self.arrival_distance_m) else "following"
        pause_progress = navigation_state == "completed"
        guidance = self.guidance.update(match.matched_s_m, navigation_state, match.cross_track_error_m, pause_progress, time.monotonic())
        self._play(guidance.prompts)
        emitted_prompts = list(guidance.prompts)
        for prompt in segmentation_prompts:
            self._play((prompt,))
            emitted_prompts.append(prompt)
        if self.voice_directions and self.voice_worker is not None and navigation_state != "completed":
            direction_prompt = self.guidance.scheduler.request(
                Prompt(f"direction:{command_clock}", command_text, 2), time.monotonic()
            )
            if direction_prompt:
                self._play((direction_prompt,))
                emitted_prompts.append(direction_prompt)
        visual_result = self._verify_anchor(guidance.visual_anchor)
        if self.follow_logger is not None:
            self.follow_logger.append(
                coordinate,
                float(heading),
                command_text,
                command_clock,
                gps_action,
                target_heading,
                self.frame.to_geodetic(match.east_m, match.north_m),
                self.frame.to_geodetic(target_east, target_north),
                match,
                bool(emitted_prompts),
                segmentation_result,
            )
        return {
            **match.to_dict(),
            "navigation_state": navigation_state,
            "pause_progress": pause_progress,
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
            "segmentation": segmentation_result,
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
        if self.follow_logger is not None:
            self.follow_logger.close()
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
    parser.add_argument("--voice", action="store_true", help="play prompts through the hardware gateway")
    parser.add_argument("--visual-anchors", action="store_true", help="verify anchor images with the local XFeat model")
    parser.add_argument("--vio-topic", help="subscribe to live VINS odometry on this ROS 2 topic")
    parser.add_argument("--ros-image-topic", help="subscribe to the VINS ROS image topic for visual anchors")
    parser.add_argument("--follow-log", type=Path, help="write following GPS samples in outdoor_nav output.jsonl format")
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
    follow_log_path = args.follow_log or (
        route_dir / "follow_output" / f"follow-{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "output.jsonl"
    )
    hardware_agent = create_hardware_agent_client()
    gps = HardwareGatewayGPS(hardware_agent)
    imu = HardwareGatewayIMU(hardware_agent)
    speaker = None
    if args.voice:
        speaker = lambda text: asyncio.run(
            hardware_agent.play_text(text, need_feedback=False)
        )
    camera = visual_matcher = vio = None
    if args.visual_anchors and args.ros_image_topic and args.vio_topic:
        vio = VioSubscriber(args.vio_topic, image_topic=args.ros_image_topic)
        vio.start()
        camera = vio
        visual_matcher = XFeatAnchorMatcher(PROJECT_ROOT)
    elif args.visual_anchors:
        camera = HardwareGatewayCamera(hardware_agent)
        visual_matcher = XFeatAnchorMatcher(PROJECT_ROOT)
    if args.vio_topic and vio is None:
        vio = VioSubscriber(args.vio_topic)
        vio.start()
    runner = ReplayRunner(route_dir, config, gps=gps, imu=imu, speaker=speaker, camera=camera, visual_matcher=visual_matcher, vio=vio, voice_directions=args.voice, follow_log_path=follow_log_path)
    LOGGER.info("recording follow GPS to %s", follow_log_path)
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
