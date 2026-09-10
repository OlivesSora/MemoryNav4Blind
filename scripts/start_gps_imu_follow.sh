#!/usr/bin/env bash
# Follow a built MemoryNav reference route with GPS + IMU only.
#
# The GPS service is assumed to already be running in another process; this
# script only launches the online replay matcher with live GPS + IMU. No
# VINS/VIO and no visual anchor matching is used.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUTE_ID=""
REPLAY_ARGS=()

usage() {
  cat <<'EOF'
Usage: start_gps_imu_follow.sh --route-id ID [options]

Follows a ready MemoryNav route using live GPS + IMU only. Visual anchors and
VINS/VIO are disabled. The GPS service must already be running.

Options:
  --route-id ID      MemoryNav route id (required)
  --config FILE      MemoryNav config override
  --interval S       Replay loop period in seconds (default: 0.2)
  --voice            Enable voice prompts
  --imu-port DEVICE  IMU serial device (default: /dev/imu)
  --help, -h         Show this help

Examples:
  start_gps_imu_follow.sh --route-id suishi-2 \
    --config memory_nav/config/suishi-2.yaml --voice
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --route-id) ROUTE_ID="$2"; REPLAY_ARGS+=(--route-id "$2"); shift 2 ;;
    --config) REPLAY_ARGS+=(--config "$2"); shift 2 ;;
    --interval) REPLAY_ARGS+=(--interval "$2"); shift 2 ;;
    --voice) REPLAY_ARGS+=(--voice); shift ;;
    --imu-port) REPLAY_ARGS+=(--imu-port "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ROUTE_ID" ]] || { echo "--route-id is required" >&2; usage >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }

echo "=== FOLLOWING ROUTE $ROUTE_ID (GPS + IMU) ==="
echo "Press Ctrl+C to stop."
exec "$PYTHON_BIN" -m memory_nav.replay.replay_nav "${REPLAY_ARGS[@]}"
