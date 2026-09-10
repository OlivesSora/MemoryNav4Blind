"""List, confirm or reject an anchor candidate for a built route."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from memory_nav.config import load_config
from memory_nav.recording.anchor_collector import AnchorCandidate, load_anchors, save_anchors
from memory_nav.recording.session_writer import atomic_write_json, validate_route_id


def update_anchor(anchors: list[AnchorCandidate], anchor_id: str, prompt_text: str | None = None, reject: bool = False) -> list[AnchorCandidate]:
    matches = [index for index, anchor in enumerate(anchors) if anchor.anchor_id == anchor_id]
    if len(matches) != 1:
        raise ValueError(f"anchor not found or duplicated: {anchor_id}")
    if reject:
        return [anchor for anchor in anchors if anchor.anchor_id != anchor_id]
    if prompt_text is None or not prompt_text.strip():
        raise ValueError("confirmed anchor requires non-empty prompt text")
    result = list(anchors)
    result[matches[0]] = replace(result[matches[0]], prompt_text=prompt_text.strip(), confirmed=True)
    return result


def refresh_route_state(route_dir: Path, anchors: list[AnchorCandidate]) -> str:
    manifest_path = route_dir / "manifest.json"
    quality_path = route_dir / "quality_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    pending = sum(not anchor.confirmed for anchor in anchors)
    state = "ERROR" if not quality.get("ready") else "REVIEW_REQUIRED" if pending else "READY"
    manifest.update({"state": state, "anchor_count": len(anchors), "pending_anchor_count": pending})
    atomic_write_json(manifest_path, manifest)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--anchor-id")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prompt", help="confirm using this spoken prompt")
    action.add_argument("--reject", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    route_dir = Path(config["routes_dir"]) / validate_route_id(args.route_id)
    anchors_path = route_dir / "anchors.json"
    if not anchors_path.is_file():
        parser.error(f"anchors file does not exist: {anchors_path}")
    anchors = load_anchors(anchors_path)
    if args.anchor_id is None:
        for anchor in anchors:
            print(f"{anchor.anchor_id}\ts={anchor.s_m:.1f}\tkind={anchor.kind}\tconfirmed={anchor.confirmed}\timage={anchor.image_path or '-'}\tprompt={anchor.prompt_text or '-'}")
        return 0
    if args.prompt is None and not args.reject:
        parser.error("--anchor-id requires --prompt or --reject")
    try:
        anchors = update_anchor(anchors, args.anchor_id, args.prompt, args.reject)
    except ValueError as exc:
        parser.error(str(exc))
    save_anchors(anchors_path, anchors)
    state = refresh_route_state(route_dir, anchors)
    print(f"route state: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
