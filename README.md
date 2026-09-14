# MemoryNav：参考轨迹生成与真机跟随

MemoryNav 是盲人导航实验的路线记忆与跟随模块。它把一次人工行走过程中的 GPS、N100 IMU、VINS-Fusion 视觉里程计和相机图像保存为可复现的路线；下一次行走时，用当前 GPS/IMU/VIO 与历史参考轨迹匹配，输出当前位置、偏离程度、下一步方向和语音提示。

本 README 面向**真机实操**，给出从零到"生成参考轨迹 → 可视化 → 跟随"的完整流程与当前配置含义。

## 1. 数据流与接口

```text
眼镜 ──推流──> :9989(唯一接收端= VINS glasses_imu_bridge)
                ├─ /imu0   (N100 IMU, 含真实航向)
                └─ /cam0/image_raw
VINS-Fusion(消费 /imu0 + /cam0/image_raw) ──> /odometry
GPS 服务 ──> http://localhost:9000/gps
                                    │
                       MemoryNav 记录 / 匹配 / 指导
                                    │
                       utils.voice.TextToSpeechPlayer.say
```

| 项目 | 默认值 |
|---|---|
| 包名 / 目录 | `memory_nav`（注意拼写，非 momery） |
| Conda 环境 | `blind`（脚本不再自动 activate，请先进入该环境） |
| 项目根目录 | `/home/wheeltec/projects/blind-nav` |
| VINS 工作区 | `/home/wheeltec/projects/VINS-Fusion-ROS2-humble-arm` |
| GPS 服务 | `http://localhost:9000/gps` |
| IMU 串口 | `/dev/imu`（仅无 VINS 时由 recorder/replay 直读） |
| 图像话题 | `/cam0/image_raw`（由 VINS bridge 发布） |
| VINS 里程计话题 | **`/odometry`**（VINS-Fusion 实际发布名，非 `/vins_estimator/odometry`） |
| 路线目录 | `memory_nav/routes/<route-id>` |

### IMU 数据来源（重要，避免双开串口）
- 带 VINS（recorder/replay 传了 `--vio-topic`）：VINS bridge 已独占 `/dev/imu` 并把 IMU 发到 `/imu0`。MemoryNav 的 recorder/replay **改订阅 `/imu0`**，不再自己打开 `/dev/imu`，避免两个进程读同一串口互相饿死。
- 无 VINS（如纯 `--camera` 录制）：recorder 才直读 `/dev/imu`（`utils.imu.IMU`）。

## 2. 环境与一次性前置

> 这些是让 VINS bridge 能在 `blind` 下正常运行的**已应用**修复，operator 只需知悉，无需重复操作；若日后重装 VINS 需再次确认。

- 全程在 **`blind`** conda 环境运行（其 `cv2 5.0.0` 正常）。系统 python 的 cv2 会在 `import cv2` 时因 GStreamer 崩溃。
- VINS `glasses_imu_bridge` 入口 shebang 需为 `#!/usr/bin/env python3`（已改）。
- `bridge_node.py` 手动构造 `sensor_msgs/Image`，不再用 `cv_bridge.cv2_to_imgmsg`（cv_bridge 按 opencv4 类型码编码，与 opencv5 不兼容）。同时把 N100 真实航向写入 `/imu0.orientation_covariance[0]`。
- N100 帧头的 CRC-8 已修正为 FDILink 正确算法（LSB 先行、poly `0x8C`），recorder 的 `utils/imu.py` 与 bridge 的 `n100_imu.py` 同步。
- 启动脚本 `generate_/follow_reference_trajectory.sh` 会 `source /opt/ros/humble/setup.bash`、`source $VINS_ROOT/install/setup.bash`，并导出 ament/colcon 变量（供 `run_live_vins.sh` 在其内部 `set -u` 下重 source 时不报 unbound）。它们**不会** `conda activate`——请确认你已处于 `blind`。

开始前检查：

```bash
conda activate blind
curl -sS http://localhost:9000/gps        # 应有 gps_data/gps_state
ls -l /dev/imu                            # 应在（无 VINS 直读时）
python3 -c "import memory_nav; print(memory_nav.__file__)"
```

先起 VINS（或直接跑下面的一键脚本）后，可用下面命令确认话题有数据：

```bash
ros2 topic hz /imu0 --window 10            # 期望 ~100Hz
ros2 topic echo /odometry --once          # VINS 出位姿后应能打印
ros2 topic echo /cam0/image_raw --once    # 眼镜推流时应能打印
```

> 眼镜必须把画面 POST 到本机 `9989`（`adb reverse tcp:9989 tcp:9989` 已设）。**不要**在跑 MemoryNav/VINS 的同时单独跑 `utils/glasses_camera.py`（会和 bridge 抢 9989）。

## 3. 真机一次完整流程（推荐路径）

```bash
cd /home/wheeltec/projects/blind-nav
conda activate blind

# A. 生成一条参考路线（VINS + GPS + IMU + 相机锚点）
bash memory_nav/scripts/generate_reference_trajectory.sh \
  --route-id liupu-yuanji \
  --config memory_nav/config/memory_nav.yaml \
  --vins-imu-port /dev/imu

# B. 可视化确认生成的参考轨迹
python3 -m memory_nav.analysis.plot_reference \
  --route-id liupu-yuanji --config memory_nav/config/memory_nav.yaml
#   → 输出 memory_nav/routes/liupu-yuanji/reference_trajectory.png

# C. 跟随这条路线（VINS + 视觉锚点 + 语音）
bash memory_nav/scripts/follow_reference_trajectory.sh \
  --route-id liupu-yuanji \
  --config memory_nav/config/memory_nav.yaml \
  --vins-imu-port /dev/imu
```

下面逐段详述。

## 4. 一键生成参考轨迹

```bash
bash memory_nav/scripts/generate_reference_trajectory.sh --route-id liupu-yuanji --vins-imu-port /dev/imu
```

可选参数：

```text
--config FILE          使用同一份 YAML（强烈建议显式传，与跟随一致）
--poll-hz N            轮询频率，默认 100
--gps-interval S       GPS 采样间隔，默认 1 秒
--vins-v4l2            用 USB/V4L2 相机（默认眼镜）
--vins-device DEVICE   指定 V4L2 设备
--vins-loop            启用 VINS loop fusion
```

### 流程与注意事项（重要，和旧版不同）
- **没有 odometry 门控**：脚本启动 VINS 后**立刻打印 `=== RECORDING STARTED ===` 并开始记录**（VINS/相机数据在话题发布后自动补录）。
- recorder 的 preflight 只把 **imu + gps 设为必需**；相机（`/cam0/image_raw`）暂不可用只告警、不中断——桥起来出图后锚点/图像自然开始写入。
- 请让设备**持续运动**（尤其 VINS 需要视差来初始化并发布 `/odometry`）。若无运动，`vio.jsonl` 可能为 0。
- 到达路线终点按 **Ctrl+C**。recorder 优雅停止后脚本会执行 `BUILDING REFERENCE TRAJECTORY`。
  - 若 Ctrl+C 把整个 bash 一起打断、没看到 `=== BUILDING REFERENCE TRAJECTORY ===`，可事后单独补建（见下）。
  - 若想**重跑录制**：先删旧目录，否则 recorder 会报 `route already exists`。

生成成功后检查：

```text
memory_nav/routes/liupu-yuanji/
├── manifest.json            # state 应为 READY
├── raw/{gps,imu,vio,events}.jsonl
├── anchors/                 # candidate_XXXX.jpg 锚点图
├── anchors.json             # 锚点（含 point_index / image_path）
├── reference_trajectory.json
└── quality_report.json      # ready 应为 true
```

只有 `manifest.state == READY` 且 `quality_report.ready == true` 才能被跟随。

### 事后补建 / 配置变更后重建
参考轨迹由 `build_reference` 从 `raw/*` 生成。若上次中断或改了 `resample_spacing_m`/锚点参数，需**先删旧 `anchors.json`**（否则不会被刷新）再重建：

```bash
rm -f memory_nav/routes/<route-id>/anchors.json
python3 -m memory_nav.trajectory.build_reference \
  --route-id <route-id> --config memory_nav/config/memory_nav.yaml
```

`anchors/*.jpg` 与 `raw/events.jsonl` 会被继续引用，不需要删。

## 5. 一键跟随参考轨迹

```bash
bash memory_nav/scripts/follow_reference_trajectory.sh \
  --route-id liupu-yuanji --vins-imu-port /dev/imu
```

该入口默认启用 VINS、GPS/VIO 在线对齐、视觉锚点验证与 TTS。可选参数：

```text
--config FILE          与生成阶段同一份配置
--interval S           跟随循环周期，默认 0.2 秒
--vins-v4l2 / --vins-device / --vins-loop
```

每行输出一条 JSON，常用字段：

```text
matched_s_m            当前在参考路线上的进度（米）
cross_track_error_m    横向偏移（米）
heading_error_deg      航向误差（度）
match_quality          good / degraded / lost
navigation_state       normal / recovering / deviated / complete
vio_state              warming_up / aligned / degraded / lost
position_source        gps / vio
next_anchor_id         下一个锚点
command_clock          例如 2
command                例如“2点钟方向前进约8米”
```

> `vio_used`/`position_source=vio` 依赖 VINS 在录制与跟随期间能初始化并对齐。单目 VINS 有尺度/漂移，若 `vio_alignment_rms_m` 很大会被拒，位置源会回落到 `gps`（这是预期降级，不影响跟随主体）。

## 6. 方向指令与语音节奏

方向 = “当前 IMU 航向” 与 “参考路线前方切线航向” 的差，转成时钟方向：

```text
12点钟方向直行约8米
2点钟方向前进约8米
原地转至6点钟方向，然后前进约8米
```

语音由 `utils.voice.TextToSpeechPlayer` 在后台队列播放，不阻塞匹配。为**避免太频繁**：
- 同一条提示受 `voice.cooldown_s`（默认 10s）冷却；
- 转向/方向播报只在**时钟方向相对上次已播变化 ≥ `voice.direction_change_min_clocks`（默认 2 格 ≈ 60°）**才说（见 `replay_nav.py`）。这样直线段不会重复唠叨，只有真正转向才提示。
- 偏离/恢复、锚点到达等状态提示另算。

语音需要百度 TTS 凭证有效、网络可用、系统有音频输出。合成缓存存在路线目录 `tts_cache/`。

## 7. 视觉锚点

- 生成阶段从 `/cam0/image_raw` 取清晰帧做锚点。当前配置只产生 **start / turn(转向≥35°) / end** 锚点（`anchors.max_spacing_m` 调大后不再按距离插间隔锚点）。
- 跟随阶段用本地 XFeat 把当前帧与历史锚点比对做视觉校验；失败不会中断，会自动回到 GPS+IMU/VIO。

只想要 GPS+IMU（不带视觉）时可直接用 Python 入口：

```bash
python3 -m memory_nav.replay.replay_nav \
  --route-id liupu-yuanji --voice --vio-topic /odometry
```

## 8. 可视化参考轨迹

把生成的参考轨迹（及锚点）画成 PNG：

```bash
python3 -m memory_nav.analysis.plot_reference \
  --route-id <route-id> --config memory_nav/config/memory_nav.yaml
```

输出：`memory_nav/routes/<route-id>/reference_trajectory.png`（经纬度折线 + 锚点叠加，start/turn/end 不同颜色/标记）。可加 `--out /path.png` 覆盖输出路径。

## 9. 保存跟随日志做复盘

```bash
mkdir -p memory_nav/hardware_reports
bash memory_nav/scripts/follow_reference_trajectory.sh \
  --route-id liupu-yuanji --vins-imu-port /dev/imu \
  | tee memory_nav/hardware_reports/liupu-yuanji-follow.jsonl
```

## 10. 其他入口

```bash
# 只录 GPS/IMU/相机，不起 VINS（recorder 此时直读 /dev/imu）
bash memory_nav/scripts/start_recording.sh --route-id office-a --camera

# 对已有 raw 构建参考轨迹
bash memory_nav/scripts/build_reference.sh --route-id office-a

# 基础在线跟随（无 VINS/视觉）
bash memory_nav/scripts/start_replay.sh --route-id office-a --interval 0.2

# 标定、硬件验收、离线重放
bash memory_nav/scripts/start_calibration.sh --help
bash memory_nav/scripts/hardware_acceptance.sh --help
bash memory_nav/scripts/replay_offline.sh --help

# 跟随suishi2接口：
# 先启动硬件网关：
# 终端1：
cd /home/wheeltec/projects/blind-nav-server
python3 -m src.hardware_gateway
# 终端2：
cd /home/wheeltec/projects/blind-nav
python3 client.py
# 终端3：
cd /home/wheeltec/projects/blind-nav
python utils/gps_server.py

# 记忆路线跟随启动命令：
2python3 -m memory_nav.replay.replay_nav   --route-id suishi-2   --config memory_nav/config/suishi-2.yaml   --voice
# 跟随 GPS 日志默认写入：
# memory_nav/routes/suishi-2/follow_output/follow-时间戳/output.jsonl
# 如需指定路径，追加：--follow-log /path/to/output.jsonl
```

> 生成、构建、审核、跟随必须用**同一份配置**，尤其 `routes_dir`、坐标系、匹配阈值。

## 11. 配置要点（当前默认值）

默认配置：`memory_nav/config/memory_nav.yaml`。

| 键 | 当前值 | 含义 |
|---|---|---|
| `trajectory.resample_spacing_m` | `5.0` | 参考点重采样间距（越小点越密） |
| `trajectory.smoothing_window` | `5` | 平滑窗口 |
| `trajectory.max_speed_m_s` | `4.0` | 速度去重上限 |
| `quality.minimum_route_length_m` | `3.0` | 路线长度下限（≤此值不 READY） |
| `vio.topic` | `/odometry` | VINS 里程计话题 |
| `vio.maximum_alignment_rms_m` | `5.0` | VIO↔GPS 对齐最大 RMS |
| `matching.*` | — | 匹配窗口 / good/lost 阈值 |
| `deviation.*` | — | 偏离 / 恢复阈值 |
| `anchors.max_spacing_m` | `1000.0` | 调大=**关闭间隔锚点**，只留 start/turn/end |
| `anchors.turn_angle_deg` | `35.0` | 转向锚点阈值 |
| `anchors.arrival_distance_m` | `2.0` | 到达判定距离 |
| `voice.cooldown_s` | `10.0` | 同一提示冷却 |
| `voice.direction_change_min_clocks` | `2` | 方向变化 ≥2 格才播报 |

> 为短/单次测试（如 ~1m 路线）可以复制配置并放宽 `quality.minimum_route_length_m`（例如 `0.5`），生成/跟随都传同一份副本。

## 12. 故障排查

| 现象 | 处理 |
|---|---|
| bridge 报 cv2/GStreamer 崩溃 | 确认在 `blind` 环境跑、bridge shebang 为 `env python3` |
| VINS 无位姿 / `vio.jsonl=0` | 录制时让设备持续运动，VINS 需视差初始化；`ros2 topic hz /imu0` 看 IMU 是否 ~100Hz |
| IMU 数据断续 / 掉帧 | 确认 recorder/replay 在带 VINS 时订阅 `/imu0` 而非双开 `/dev/imu` |
| GPS 预检失败 | `curl http://localhost:9000/gps`；若终端有 http_proxy 请 `unset http_proxy https_proxy` |
| 眼镜画面收不到 | 确认 `adb reverse tcp:9989 tcp:9989`、眼镜在向 9989 推流、且无别的进程占 9989 |
| 路线不是 READY | 看 `quality_report.json` 的 `failure_reasons`；太短则放宽 `minimum_route_length_m` |
| 没有语音 | 查百度凭证、网络、PyAudio、默认音频输出 |
| 语音太频繁 | 调大 `voice.cooldown_s` 或 `voice.direction_change_min_clocks` |
| 录制 `route already exists` | 先删 `memory_nav/routes/<id>` 再录 |
| 参考轨迹没更新（改配置后） | `rm -f routes/<id>/anchors.json` 后重新 `build_reference` |
| 结束后 VINS 仍跑 | 再按一次 Ctrl+C，或 `ps aux \| grep run_live_vins` |

## 13. 开发检查

```bash
python3 -m unittest discover -s memory_nav/tests -q
python3 -m compileall -q memory_nav
for script in memory_nav/scripts/*.sh; do bash -n "$script"; done
git diff --check
```
