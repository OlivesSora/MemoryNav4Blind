#!/usr/bin/env bash
# Start the hardware gateway, device client, GPS, and MemoryNav in tmux.
set -Eeuo pipefail

DEVICE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_DIR="${SERVER_DIR:-$(dirname -- "$DEVICE_DIR")/blind-nav-server}"
SESSION_NAME="${SESSION_NAME:-memory-nav-seg}"
HARDWARE_READY_TIMEOUT="${HARDWARE_READY_TIMEOUT:-180}"

usage() {
    cat <<'EOF'
用法：
  bash memory_nav/scripts/quick_start_seg.sh --route-id ID --config FILE [--attach]

--config 的相对路径以 blind-nav 项目根目录为基准。
默认在后台启动 tmux 会话 memory-nav-seg；--attach 会在启动后进入会话。

环境变量：
  SESSION_NAME             tmux 会话名（默认：memory-nav-seg）
  SERVER_DIR               硬件网关项目目录
  HARDWARE_READY_TIMEOUT   等待设备连接的秒数（默认：180）
EOF
}

route_id=""
config=""
attach=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --route-id|--config)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "$1 需要一个值" >&2
                exit 2
            fi
            if [[ "$1" == --route-id ]]; then
                route_id="$2"
            else
                config="$2"
            fi
            shift 2
            ;;
        --attach) attach=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$route_id" || -z "$config" ]]; then
    echo "--route-id 和 --config 均为必填参数" >&2
    usage >&2
    exit 2
fi
if [[ "$config" != /* ]]; then
    config="$DEVICE_DIR/$config"
fi
if [[ ! -f "$config" ]]; then
    echo "配置文件不存在：$config" >&2
    exit 1
fi
if [[ ! -d "$SERVER_DIR" ]]; then
    echo "硬件网关目录不存在：$SERVER_DIR" >&2
    exit 1
fi
if ! [[ "$HARDWARE_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "HARDWARE_READY_TIMEOUT 必须是正整数" >&2
    exit 2
fi
for command in adb tmux python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "缺少命令：$command" >&2
        exit 1
    fi
done
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux 会话已存在：$SESSION_NAME（tmux attach -t $SESSION_NAME）" >&2
    exit 1
fi

echo "检查 Rokid ADB 连接…"
adb start-server >/dev/null
device_serial="$(adb devices | awk '$2 == "device" {print $1; exit}')"
if [[ -z "$device_serial" ]]; then
    echo "未发现已授权的 Rokid 设备，请先连接并授权 USB 调试。" >&2
    exit 1
fi
adb -s "$device_serial" reverse tcp:9989 tcp:9989

session_created=false
cleanup() {
    if [[ "$session_created" == true ]]; then
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    fi
}
trap cleanup ERR

tmux new-session -d -s "$SESSION_NAME" -n navigation -c "$SERVER_DIR" \
    'exec python3 -m src.hardware_gateway'
session_created=true
gateway_pane="$(tmux display-message -p -t "$SESSION_NAME:navigation.0" '#{pane_id}')"
gps_pane="$(tmux split-window -d -P -F '#{pane_id}' -h -t "$gateway_pane" -c "$DEVICE_DIR" \
    'exec python3 utils/gps_server.py')"
device_pane="$(tmux split-window -d -P -F '#{pane_id}' -v -t "$gateway_pane" -c "$DEVICE_DIR" \
    'sleep 3; exec python3 client.py')"
nav_pane="$(tmux split-window -d -P -F '#{pane_id}' -v -t "$gps_pane" -c "$DEVICE_DIR" \
    'exec bash')"

tmux select-pane -t "$gateway_pane" -T '硬件网关'
tmux select-pane -t "$device_pane" -T '设备客户端'
tmux select-pane -t "$gps_pane" -T 'GPS 服务'
tmux select-pane -t "$nav_pane" -T '记忆导航（等待设备）'
tmux set-option -t "$SESSION_NAME:navigation" pane-border-status top
tmux set-option -t "$SESSION_NAME:navigation" pane-border-format ' #{pane_title} '

echo "等待硬件网关与设备客户端连接（最长 ${HARDWARE_READY_TIMEOUT} 秒）…"
deadline=$((SECONDS + HARDWARE_READY_TIMEOUT))
connected=false
while (( SECONDS < deadline )); do
    if python3 -c '
import json
from urllib.request import urlopen
payload = json.load(urlopen("http://127.0.0.1:7001/health", timeout=2))
raise SystemExit(0 if payload.get("hardware_connected") else 1)
' >/dev/null 2>&1; then
        connected=true
        break
    fi
    sleep 1
done
if [[ "$connected" != true ]]; then
    echo "设备连接超时，请检查硬件网关和设备客户端。" >&2
    false
fi

if ! python3 -c '
import json
from urllib.request import urlopen
payload = json.load(urlopen("http://127.0.0.1:9000/gps", timeout=2))
raise SystemExit(0 if payload.get("gps_data") else 1)
' >/dev/null 2>&1; then
    echo "警告：GPS 服务暂无有效定位；请连接手机定位或检查 RTK，导航会提示停止前进。" >&2
fi

printf -v nav_command 'exec bash %q --route-id %q --config %q --seg-backend tensorrt --seg-device cuda' \
    "$DEVICE_DIR/memory_nav/scripts/follow_reference_trajectory_seg.sh" "$route_id" "$config"
tmux send-keys -l -t "$nav_pane" "$nav_command"
tmux send-keys -t "$nav_pane" Enter
tmux select-pane -t "$nav_pane" -T '记忆导航'
trap - ERR

echo "启动完成：tmux 会话 '$SESSION_NAME'。"
echo "进入会话：tmux attach -t $SESSION_NAME"
echo "停止服务：tmux kill-session -t $SESSION_NAME"
if [[ "$attach" == true ]]; then
    exec tmux attach -t "$SESSION_NAME"
fi
