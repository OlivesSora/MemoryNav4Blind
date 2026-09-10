#!/usr/bin/env bash
# Import an L4 outdoor_nav output.jsonl into a MemoryNav route, build the
# reference trajectory, then optionally start live GPS+IMU following.
#
# This wraps the pure GPS+IMU path: no VINS/VIO and no visual anchor matching.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

INPUT=""
ROUTE_ID=""
SKIP=0
CONFIG=""
FOLLOW=0
FORCE=0
IMU_PORT="/dev/imu"
VOICE=0

usage() {
  cat <<'EOF'
Usage: prepare_and_follow_gps_route.sh --input FILE --route-id ID [options]

Trims an L4 outdoor_nav output.jsonl, imports it as a MemoryNav route
(GPS+IMU only), builds the reference trajectory, and can start live following.

Options:
  --input FILE       L4 output.jsonl to import (required)
  --route-id ID      MemoryNav route id (required)
  --skip N           Leading lines to drop before import (default: 0)
  --config FILE      MemoryNav config override
  --force            Delete an existing route directory before importing
  --follow           Start live GPS+IMU following after building
  --voice            Enable voice prompts while following
  --imu-port DEVICE  IMU serial device for following (default: /dev/imu)
  --help, -h         Show this help

Examples:
  # Only prepare the route
  prepare_and_follow_gps_route.sh \
    --input /path/output.jsonl --route-id suishi-2 --skip 70 \
    --config memory_nav/config/suishi-2.yaml

  # Prepare then follow live (no camera, no visual anchors)
  prepare_and_follow_gps_route.sh \
    --input /path/output.jsonl --route-id suishi-2 --skip 70 \
    --config memory_nav/config/suishi-2.yaml --follow --voice
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --input) INPUT="$2"; shift 2 ;;
    --route-id) ROUTE_ID="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --follow) FOLLOW=1; shift ;;
    --voice) VOICE=1; shift ;;
    --imu-port) IMU_PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$INPUT" ]] || { echo "--input is required" >&2; usage >&2; exit 2; }
[[ -n "$ROUTE_ID" ]] || { echo "--route-id is required" >&2; usage >&2; exit 2; }
[[ -f "$INPUT" ]] || { echo "input log not found: $INPUT" >&2; exit 2; }
[[ "$SKIP" =~ ^[0-9]+$ ]] || { echo "--skip must be a non-negative integer" >&2; exit 2; }

IMPORT_ARGS=(--input "$INPUT" --route-id "$ROUTE_ID" --skip "$SKIP")
BUILD_ARGS=(--route-id "$ROUTE_ID")
REPLAY_ARGS=(--route-id "$ROUTE_ID" --imu-port "$IMU_PORT")
if [[ -n "$CONFIG" ]]; then
  [[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
  IMPORT_ARGS+=(--config "$CONFIG")
  BUILD_ARGS+=(--config "$CONFIG")
  REPLAY_ARGS+=(--config "$CONFIG")
  ROUTES_DIR="$(python3 -c "from memory_nav.config import load_config; print(load_config('$CONFIG')['routes_dir'])")"
else
  ROUTES_DIR="$(python3 -c "from memory_nav.config import load_config; print(load_config()['routes_dir'])")"
fi
[[ "$VOICE" -eq 1 ]] && REPLAY_ARGS+=(--voice)

ROUTE_DIR="${ROUTES_DIR}/${ROUTE_ID}"
if [[ -e "$ROUTE_DIR" ]]; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "Removing existing route: $ROUTE_DIR"
    rm -rf -- "$ROUTE_DIR"
  else
    echo "route already exists: $ROUTE_DIR (use --force to replace)" >&2
    exit 2
  fi
fi

echo "=== IMPORTING GPS LOG ==="
python3 -m memory_nav.tools.import_gps_log "${IMPORT_ARGS[@]}"

echo "=== BUILDING REFERENCE TRAJECTORY ==="
python3 -m memory_nav.trajectory.build_reference "${BUILD_ARGS[@]}"

if [[ "$FOLLOW" -eq 1 ]]; then
  echo "=== STARTING LIVE GPS+IMU FOLLOW ==="
  exec python3 -m memory_nav.replay.replay_nav "${REPLAY_ARGS[@]}"
fi

echo "=== ROUTE READY ==="
echo "Follow live with:"
printf '  %q' "$PROJECT_ROOT/memory_nav/scripts/start_replay.sh"
printf ' %q' "${REPLAY_ARGS[@]}"
printf '\n'
