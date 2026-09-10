#!/usr/bin/env bash
# Start a new VINS session and follow a built MemoryNav route.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VINS_ROOT="${VINS_ROOT:-/home/wheeltec/projects/VINS-Fusion-ROS2-humble-arm}"
VINS_ARGS=()
REPLAY_ARGS=()

usage() {
  cat <<'EOF'
Usage: follow_reference_trajectory.sh --route-id ID [options]

Starts VINS, performs live GPS-to-VIO alignment, then follows the route.
MemoryNav options: --route-id ID --config FILE --interval S --voice --visual-anchors
VINS options:      --vins-imu-port DEVICE --vins-loop --vins-v4l2 --vins-device DEVICE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --route-id|--config|--interval) REPLAY_ARGS+=("$1" "$2"); shift 2 ;;
    --voice|--visual-anchors) REPLAY_ARGS+=("$1"); shift ;;
    --vins-imu-port) VINS_ARGS+=(--imu-port "$2"); REPLAY_ARGS+=(--imu-port "$2"); shift 2 ;;
    --vins-loop) VINS_ARGS+=(--loop); shift ;;
    --vins-v4l2) VINS_ARGS+=(--v4l2); shift ;;
    --vins-device) VINS_ARGS+=(--device "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ " ${REPLAY_ARGS[*]} " == *" --route-id "* ]] || { echo "--route-id is required" >&2; exit 2; }
[[ -x "$VINS_ROOT/scripts/run_live_vins.sh" && -f "$VINS_ROOT/install/setup.bash" ]] || { echo "VINS workspace is not built: $VINS_ROOT" >&2; exit 2; }

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1090
source "$VINS_ROOT/install/setup.bash"
set -u
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# run_live_vins.sh re-sources ROS setup under `set -u`; give its child shell
# defined (possibly empty) ament/colcon vars so those sources do not abort.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
export AMENT_PREFIX_PATH="${AMENT_PREFIX_PATH:-}"
export AMENT_PYTHON_EXECUTABLE="${AMENT_PYTHON_EXECUTABLE:-}"
export COLCON_TRACE="${COLCON_TRACE:-}"
export COLCON_PREFIX_PATH="${COLCON_PREFIX_PATH:-}"
export COLCON_PYTHON_EXECUTABLE="${COLCON_PYTHON_EXECUTABLE:-}"

"$VINS_ROOT/scripts/run_live_vins.sh" "${VINS_ARGS[@]}" &
VINS_PID=$!
cleanup() { kill -INT "$VINS_PID" 2>/dev/null || true; wait "$VINS_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "Following route; vio/visual anchors engage as VINS publishes /odometry & /cam0/image_raw."
python -m memory_nav.replay.replay_nav "${REPLAY_ARGS[@]}" --voice --vio-topic /odometry --ros-image-topic /cam0/image_raw --visual-anchors
