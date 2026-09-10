#!/usr/bin/env bash
# Record GPS/IMU/VINS and build one ready-to-follow reference route.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VINS_ROOT="${VINS_ROOT:-/home/wheeltec/projects/VINS-Fusion-ROS2-humble-arm}"
ROUTE_ID=""
CONFIG=""
VINS_ARGS=()
RECORDER_ARGS=()

usage() {
  cat <<'EOF'
Usage: generate_reference_trajectory.sh --route-id ID [options]

Records GPS, IMU and VINS odometry. Press Ctrl+C at the route end; the script
then builds the fused reference trajectory automatically.

MemoryNav options: --route-id ID --config FILE --camera --poll-hz N --gps-interval S
VINS options:      --vins-imu-port DEVICE --vins-loop --vins-v4l2 --vins-device DEVICE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --route-id) ROUTE_ID="$2"; RECORDER_ARGS+=("$1" "$2"); shift 2 ;;
    --config) CONFIG="$2"; RECORDER_ARGS+=("$1" "$2"); shift 2 ;;
    --camera|--poll-hz|--gps-interval) RECORDER_ARGS+=("$1"); [[ "$1" == "--camera" ]] || { RECORDER_ARGS+=("$2"); shift; }; shift ;;
    --vins-imu-port) VINS_ARGS+=(--imu-port "$2"); RECORDER_ARGS+=(--imu-port "$2"); shift 2 ;;
    --vins-loop) VINS_ARGS+=(--loop); shift ;;
    --vins-v4l2) VINS_ARGS+=(--v4l2); shift ;;
    --vins-device) VINS_ARGS+=(--device "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$ROUTE_ID" ]] || { echo "--route-id is required" >&2; exit 2; }
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

VINS_PID=""
cleanup() {
  [[ -z "$VINS_PID" ]] || kill -INT "$VINS_PID" 2>/dev/null || true
  [[ -z "$VINS_PID" ]] || wait "$VINS_PID" 2>/dev/null || true
}
trap cleanup EXIT

"$VINS_ROOT/scripts/run_live_vins.sh" "${VINS_ARGS[@]}" &
VINS_PID=$!

echo "=== RECORDING STARTED ==="
echo "route: $ROUTE_ID"
echo "Recording begins immediately; vio/camera are captured as VINS publishes /odometry & /cam0/image_raw."
echo "Press Ctrl+C only after reaching the route end."
python -m memory_nav.recording.recorder "${RECORDER_ARGS[@]}" --vio-topic /odometry --ros-image-topic /cam0/image_raw
echo "=== RECORDING STOPPED; FINALIZING SENSORS ==="
cleanup
VINS_PID=""
echo "=== BUILDING REFERENCE TRAJECTORY ==="
BUILD_ARGS=(--route-id "$ROUTE_ID")
[[ -z "$CONFIG" ]] || BUILD_ARGS+=(--config "$CONFIG")
exec python -m memory_nav.trajectory.build_reference "${BUILD_ARGS[@]}"
