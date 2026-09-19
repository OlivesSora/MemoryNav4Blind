#!/usr/bin/env bash
# Follow a MemoryNav route with online CAT-Seg walkable-segmentation avoidance.
#
# Does NOT start VINS. By default it runs without VINS: frames, GPS and heading
# come from the hardware gateway (client.py), so only the hardware gateway must
# be running. Pass --vio-topic to use VINS (/odometry, /cam0/image_raw) instead;
# then VINS must also be running. The CAT-Seg worker is started automatically by
# the provider in the catseg conda environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPLAY_ARGS=()
LOG_DIR=""

usage() {
  cat <<'EOF'
Usage: follow_reference_trajectory_seg.sh --route-id ID [options]

Runs the segmentation-aware follow loop. Does NOT start VINS. By default no VINS
is used (hardware-gateway camera); pass --vio-topic to use VINS, in which case
VINS and the hardware gateway must already be running.

MemoryNav options:   --route-id ID --config FILE --interval S --voice --visual-anchors
Segmentation options: --seg-device cuda|cpu --seg-vis-dir DIR --seg-max-hz N
                      --seg-input-scale F --seg-min-size-test N --seg-radius-ratio F
                      --seg-worker-python PATH --seg-walkable-names NAMES --seg-show
VINS options:         --vio-topic TOPIC --ros-image-topic TOPIC --no-vio

Logging options:      --log-dir DIR

  --log-dir collects every log under DIR (default
  <project>/memory_nav/analysis/test_<MMDD>). One timestamped run produces:
    DIR/follow-<ts>/output.jsonl   follow samples (new file, never appended)
    DIR/seg_online/catseg-<ts>.log CAT-Seg worker stderr
    DIR/run-<ts>.log               wrapper stdout/stderr
  --follow-log / --seg-socket / --seg-vis-dir still override their defaults.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --route-id|--config|--interval|--vio-topic|--ros-image-topic|--follow-log|\
    --seg-device|--seg-vis-dir|--seg-max-hz|--seg-input-scale|--seg-min-size-test|\
    --seg-worker-python|--seg-catseg-dir|--seg-config|--seg-weights|\
    --seg-walkable-names|--seg-socket|--seg-cooldown|--seg-vis-interval|--seg-radius-ratio)
      REPLAY_ARGS+=("$1" "$2"); shift 2 ;;
    --voice|--visual-anchors|--seg-online|--seg-show|--no-vio)
      REPLAY_ARGS+=("$1"); shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ " ${REPLAY_ARGS[*]} " == *" --route-id "* ]] || { echo "--route-id is required" >&2; exit 2; }

has_arg() { [[ " ${REPLAY_ARGS[*]} " == *" $1 "* ]]; }

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/memory_nav/analysis/test_$(date +%m%d)}"
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="${PROJECT_ROOT}/${LOG_DIR}"
fi
LOG_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
has_arg --follow-log || REPLAY_ARGS+=(--follow-log "${LOG_DIR}/follow-${LOG_TS}/output.jsonl")
has_arg --seg-socket || REPLAY_ARGS+=(--seg-socket "${LOG_DIR}/seg_online/catseg-${LOG_TS}.sock")
has_arg --seg-vis-dir || REPLAY_ARGS+=(--seg-vis-dir "${LOG_DIR}/seg_online")
RUN_LOG="${LOG_DIR}/run-${LOG_TS}.log"

FINAL_ARGS=(--seg-online)
if has_arg --vio-topic && ! has_arg --no-vio; then
  has_arg --ros-image-topic || FINAL_ARGS+=(--ros-image-topic /cam0/image_raw)
fi
has_arg --voice || FINAL_ARGS+=(--voice)
FINAL_ARGS+=("${REPLAY_ARGS[@]}")

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "logs: run=${RUN_LOG}" >&2
exec > >(tee -a "$RUN_LOG") 2>&1
exec python -m memory_nav.replay.replay_nav_seg "${FINAL_ARGS[@]}"
