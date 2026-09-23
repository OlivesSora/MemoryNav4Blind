#!/usr/bin/env bash
# Record camera + IMU, align timestamps, and export a Basalt ROS1 bag.
set -euo pipefail
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$KIT/../../.." && pwd)"
PYTHON="/home/wheeltec/anaconda3/envs/blind/bin/python"
BAG_PYTHON="$KIT/.venv-bag/bin/python"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo '用法：bash memory_nav/calibration/Basalt/collect_ready_bag.sh [秒数，默认120]'
  echo '自动采集、保留 Android 采样时间并整理 IMU、生成 aligned.bag；不会运行 Basalt 优化。'
  exit 0
fi
if [[ $# -gt 1 || ! "${1:-120}" =~ ^[0-9]+$ ]]; then
  echo '秒数必须为整数，例如 120。' >&2; exit 2
fi
DURATION="${1:-120}"
if (( ${#DURATION} > 6 )); then echo '采集秒数过大。' >&2; exit 2; fi
DURATION=$((10#$DURATION))
if (( DURATION < 30 )); then echo '至少采集30秒，建议120秒。' >&2; exit 2; fi
[[ -x "$PYTHON" && -x "$BAG_PYTHON" ]] || {
  echo "缺少 Python 环境，请参照 $KIT/TIME_ALIGNMENT_ZH.md。" >&2; exit 1;
}
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c 'import cv2, numpy, fastapi, uvicorn'
"$BAG_PYTHON" -c 'import cv2; from rosbags.rosbag1 import Writer'
mkdir -p "$PROJECT/calibration_data"
RUN="$(mktemp -d "$PROJECT/calibration_data/basalt_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
trap 'echo "本次未完成；已采集的数据保留在：$RUN。请查看报错，不要将未完成输出用于标定。" >&2' ERR
printf '采集目录：%s\n采集时长：%s秒。现在开启眼镜连续上传，并按文档移动相机与IMU。\n' "$RUN" "$DURATION"
"$PYTHON" -m memory_nav.calibration.collect_calibration \
  --output "$RUN/raw" --duration "$DURATION" --imu-source android 2>&1 | tee "$RUN/collect.log"
"$PYTHON" -m memory_nav.calibration.align_timestamps \
  "$RUN/raw" "$RUN/aligned" --camera-time-source android --trim-to-overlap \
  2>&1 | tee "$RUN/align.log"
"$BAG_PYTHON" "$KIT/convert_to_bag.py" \
  "$RUN/aligned" "$RUN/aligned.bag" --mode camera-imu 2>&1 | tee "$RUN/bag.log"
printf '\n采集、对齐、转换成功。\n后续标定输入：%s/aligned.bag\n对齐报告：%s/aligned/alignment_report.json\n' "$RUN" "$RUN"
