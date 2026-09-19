# MemoryNav：参考轨迹生成与真机跟随

MemoryNav 是盲人导航实验的路线记忆与跟随模块。它把一次人工行走过程中的 GPS、N100 IMU、VINS-Fusion 视觉里程计和相机图像保存为可复现的路线；下一次行走时，用当前 GPS/IMU/VIO 与历史参考轨迹匹配，输出当前位置、偏离程度、下一步方向和语音提示。

在此基础上，MemoryNav 还提供一个**可选**的「分割映射避障」层：把跟踪输出的时钟方向映射到相机图像底部中点的半圆上，与 CAT-Seg 分割出的可通行区域比对；当跟踪方向被判定为不可通行时，回退到可通行的方向并语音提示。

本 README 面向**真机实操**，给出从零到「生成参考轨迹 → 可视化 → 跟随 →（可选）分割避障」的完整流程与当前配置含义。

---

## 1. 功能概览

| 能力 | 入口 | 说明 |
|---|---|---|
| 路线录制 | `generate_reference_trajectory.sh` / `start_recording.sh` | 同步记录 GPS/IMU/VIO/相机锚点 |
| 参考轨迹生成 | `build_reference.sh` | 清洗、融合、平滑、重采样、锚点、质量报告 |
| 在线跟随 | `follow_reference_trajectory.sh` / `replay_nav` | GPS+IMU+VIO 匹配历史轨迹，输出时钟方向指令 |
| 视觉锚点校验 | `replay_nav --visual-anchors` | 本地 XFeat 比对历史锚点图，失败不阻塞 |
| 语音提示 | `--voice` | 方向、锚点、偏离/恢复、分割避障提示 |
| 离线分析 | `analysis/run_analysis.py` | 记忆/测试轨迹对比、锚点分割、指标与可视化 |
| 分割映射避障 | `follow_reference_trajectory_seg.sh` / `replay_nav_seg` | 见 [第 16 节](#16-分割映射避障walkable-segmentation) |

---

## 2. 目录与模块结构

```text
memory_nav/
├── __init__.py
├── config.py                     # 配置加载与校验（不依赖 cwd）
├── models.py                     # GPSSample / IMUSample / VIOSample / ReferencePoint / MatchResult
├── vio.py                        # VIO/IMU 的 ROS2 适配、OnlineVioAligner
├── config/
│   ├── memory_nav.yaml           # 默认配置
│   └── kalibr/                   # 标定相关
├── calibration/
│   ├── collect_calibration.py    # 采集 Kalibr/AprilGrid 数据
│   ├── import_calibration.py
│   ├── validate_calibration.py
│   └── run_kalibr.py
├── recording/
│   ├── recorder.py               # 录制状态机与主循环
│   ├── session_writer.py         # 异步落盘、route_id 校验
│   ├── anchor_collector.py       # 锚点候选与 anchors.json
│   └── preflight.py              # 录制前传感器检查
├── trajectory/
│   ├── coordinate.py             # GCJ-02 ↔ 局部 EN、wrap_to_180、heading_from_delta
│   ├── processing.py             # 清洗/融合/平滑/重采样/质量报告
│   └── build_reference.py        # 从 raw/* 生成参考轨迹
├── replay/
│   ├── matcher.py                # 带进度约束的最近线段投影匹配
│   ├── deviation.py              # 偏离/恢复状态机
│   ├── guidance.py               # 锚点/状态提示调度
│   ├── anchor_matcher.py         # XFeat 视觉锚点匹配
│   ├── replay_nav.py             # 在线跟随主入口
│   └── replay_nav_seg.py         # 分割感知跟随入口（新增，可选）
├── interaction/
│   └── voice_prompt.py           # Prompt / VoicePromptScheduler / VoicePlaybackWorker
├── segmentation/                 # 分割映射避障（新增，可选）
│   ├── geometry.py               # 时钟↔图像半圆几何、净空距离
│   ├── avoidance.py              # 可行性判定、交集/回退
│   ├── guard.py                  # 把评估结果转为方向覆盖 + 语音
│   ├── providers.py              # CachedMaskProvider / SequenceMaskProvider
│   ├── online_provider.py        # CatSegWorkerProvider（自动拉起 worker）
│   ├── online_visualizer.py      # 在线标注帧保存
│   ├── catseg_worker.py          # catseg 环境常驻 CAT-Seg 服务（socket）
│   ├── protocol.py               # socket 长度前缀 pickle 协议
│   ├── mask_loader.py            # mask 读写/缩放
│   ├── visualize.py              # 半圆/扇区/箭头标注
│   ├── precompute_masks.py       # 离线批量生成 walkable mask
│   └── validate_offline.py       # 离线逐帧校验 + 可视化
├── analysis/                     # 离线分析（记忆 vs 测试）
├── tools/                        # import_gps_log / replay_offline / hardware_acceptance / confirm_anchor
├── tests/                        # 单元测试
├── scripts/                      # Bash 启动脚本（见第 12 节）
└── routes/<route-id>/            # 运行数据
```

约束：MemoryNav 的 Python/Bash 代码统一放在 `memory_nav/`；`utils/` 只做硬件适配；包内用包导入，不依赖 cwd。

---

## 3. 数据流与接口

### 3.1 路线记忆与跟随

```text
眼镜 ──推流──> :9989(唯一接收端= VINS glasses_imu_bridge)
                ├─ /imu0   (N100 IMU, 含真实航向)
                └─ /cam0/image_raw
VINS-Fusion(消费 /imu0 + /cam0/image_raw) ──> /odometry
硬件网关(:7001) ──> GPS / 航向 / 语音 / 相机(get_state_image)
                                    │
                       MemoryNav 记录 / 匹配 / 指导
                                    │
                       utils.voice.TextToSpeechPlayer.say
```

### 3.2 分割避障（可选）

```text
相机帧(RGB) ──> SegmentationGuard ──> MaskProvider ──> (在线) catseg worker 子进程
                     │                                      │ Unix socket
                     │<──────────── walkable mask ──────────┘
                     ▼
             SegmentationAvoidance.evaluate
                     │  clear_distances / walkable_hours / 回退
                     ▼
        覆盖方向 command_clock + 语音 prompts + 标注帧保存
```

### 3.3 关键接口

| 项目 | 默认值 |
|---|---|
| 包名 / 目录 | `memory_nav`（注意拼写，非 momery） |
| Conda 环境 | `blind`（脚本不再自动 activate，请先进入该环境） |
| 分割环境 | `catseg`（仅 CAT-Seg 推理用，脚本自动调用其 python） |
| 项目根目录 | `/home/wheeltec/projects/blind-nav` |
| 硬件网关 | `http://127.0.0.1:7001`（`python -m src.hardware_gateway`） |
| 物理端桥接 | `python client.py`（连网关 7002） |
| GPS 服务 | `http://localhost:9000/gps`（`python utils/gps_server.py`） |
| VINS 工作区 | `/home/wheeltec/projects/VINS-Fusion-ROS2-humble-arm` |
| IMU 串口 | `/dev/imu`（仅无 VINS 时由 recorder/replay 直读） |
| 图像话题 | `/cam0/image_raw`（由 VINS bridge 发布） |
| VINS 里程计话题 | **`/odometry`**（VINS-Fusion 实际发布名，非 `/vins_estimator/odometry`） |
| 路线目录 | `memory_nav/routes/<route-id>` |

### 3.4 IMU 数据来源（重要，避免双开串口）

- 带 VINS（recorder/replay 传了 `--vio-topic`）：VINS bridge 已独占 `/dev/imu` 并把 IMU 发到 `/imu0`。MemoryNav 的 recorder/replay **改订阅 `/imu0`**，不再自己打开 `/dev/imu`，避免两个进程读同一串口互相饿死。
- 无 VINS（如纯 `--camera` 录制）：recorder 才直读 `/dev/imu`（`utils.imu.IMU`）。

---

## 4. 环境与一次性前置

> 这些是让 VINS bridge 能在 `blind` 下正常运行的**已应用**修复，operator 只需知悉，无需重复操作；若日后重装 VINS 需再次确认。

- 全程在 **`blind`** conda 环境运行（其 `cv2 5.0.0` 正常）。系统 python 的 cv2 会在 `import cv2` 时因 GStreamer 崩溃。
- CAT-Seg 依赖 detectron2，只在 **`catseg`** conda 环境可用；`catseg` 通过 `.pth` 复用 `blind` 的 torch。**不需要**在 `blind` 里装 detectron2。
- VINS `glasses_imu_bridge` 入口 shebang 需为 `#!/usr/bin/env python3`（已改）。
- `bridge_node.py` 手动构造 `sensor_msgs/Image`，不再用 `cv_bridge.cv2_to_imgmsg`（cv_bridge 按 opencv4 类型码编码，与 opencv5 不兼容）。同时把 N100 真实航向写入 `/imu0.orientation_covariance[0]`。
- N100 帧头的 CRC-8 已修正为 FDILink 正确算法（LSB 先行、poly `0x8C`），recorder 的 `utils/imu.py` 与 bridge 的 `n100_imu.py` 同步。
- 启动脚本 `generate_/follow_reference_trajectory*.sh` 会 `source /opt/ros/humble/setup.bash`、`source $VINS_ROOT/install/setup.bash`，并导出 ament/colcon 变量。它们**不会** `conda activate`——请确认你已处于 `blind`。

开始前检查：

```bash
conda activate blind
curl -sS http://localhost:9000/gps        # 应有 gps_data/gps_state
curl -sS http://127.0.0.1:7001/health     # 在线跟随/GPS/IMU 需要硬件网关
ls -l /dev/imu                            # 应在（无 VINS 直读时）
python3 -c "import memory_nav; print(memory_nav.__file__)"
```

先起 VINS（或直接跑一键脚本）后，可用下面命令确认话题有数据：

```bash
ros2 topic hz /imu0 --window 10            # 期望 ~100Hz
ros2 topic echo /odometry --once          # VINS 出位姿后应能打印
ros2 topic echo /cam0/image_raw --once    # 眼镜推流时应能打印
```

> 眼镜必须把画面 POST 到本机 `9989`（`adb reverse tcp:9989 tcp:9989` 已设）。**不要**在跑 MemoryNav/VINS 的同时单独跑 `utils/glasses_camera.py`（会和 bridge 抢 9989）。

### 4.1 在线跟随的进程前置

在线跟随依赖硬件网关提供 GPS/IMU/语音，VINS 提供 `/odometry` 与 `/cam0/image_raw`。标准启动顺序（另开终端）：

```bash
# 终端1：硬件网关
cd /home/wheeltec/projects/blind-nav-server && python3 -m src.hardware_gateway
# 终端2：物理端桥接（设备需在线）
cd /home/wheeltec/projects/blind-nav && python3 client.py
# 终端3：GPS 服务
cd /home/wheeltec/projects/blind-nav && python utils/gps_server.py
# 终端4：VINS（分割避障需要图像，强烈建议启动）
/home/wheeltec/projects/VINS-Fusion-ROS2-humble-arm/scripts/run_live_vins.sh
```

> 网关未起时，`replay_nav.step` 取 GPS 会抛 `HardwareGatewayError` 并使跟随进程退出。请先确认 `curl http://127.0.0.1:7001/health` 返回 `hardware_connected: true`。

---

## 5. 真机一次完整流程（推荐路径）

```bash
cd /home/wheeltec/projects/blind-nav
conda activate blind

# 录制：--camera 用硬件网关相机抓锚点候选；不传则仅 GPS+IMU
bash memory_nav/scripts/start_recording.sh --route-id office-a --camera

# Ctrl+C 结束后，对已有 raw 构建参考轨迹
bash memory_nav/scripts/build_reference.sh --route-id office-a

# A. 生成一条参考路线（VINS + GPS + IMU + 相机锚点）
bash memory_nav/scripts/generate_reference_trajectory.sh \
  --route-id suishi-2 \
  --config memory_nav/config/memory_nav.yaml \
  --vins-imu-port /dev/imu

# B. 可视化确认生成的参考轨迹
python3 -m memory_nav.analysis.plot_reference \
  --route-id suishi-2 --config memory_nav/config/memory_nav.yaml
#   → 输出 memory_nav/routes/suishi-2/reference_trajectory.png

# C. 跟随这条路线（VINS + 视觉锚点 + 语音）
bash memory_nav/scripts/follow_reference_trajectory.sh \
  --route-id suishi-2 \
  --config memory_nav/config/memory_nav.yaml \
  --vins-imu-port /dev/imu

# D.（可选）分割感知跟随：假定 VINS 与硬件网关已在运行，脚本不启动 VINS
bash memory_nav/scripts/follow_reference_trajectory_seg.sh \
  --route-id suishi-2 \
  --config memory_nav/config/memory_nav.yaml \
  --seg-device cuda --seg-vis-dir memory_nav/analysis/output/seg_online
```

下面逐段详述。

---

## 6. 一键生成参考轨迹

```bash
bash memory_nav/scripts/generate_reference_trajectory.sh --route-id suishi-2 --vins-imu-port /dev/imu
```

可选参数：

```text
--config FILE          使用同一份 YAML（强烈建议显式传，与跟随一致）
--poll-hz N            轮询频率，默认 100
--gps-interval S       GPS 采样间隔，默认 1 秒
--camera               录制锚点图像候选（无 VINS 直读 /dev/imu 时）
--vins-v4l2            用 USB/V4L2 相机（默认眼镜）
--vins-device DEVICE   指定 V4L2 设备
--vins-loop            启用 VINS loop fusion
```

### 6.1 流程与注意事项（重要，和旧版不同）
- **没有 odometry 门控**：脚本启动 VINS 后**立刻打印 `=== RECORDING STARTED ===` 并开始记录**（VINS/相机数据在话题发布后自动补录）。
- recorder 的 preflight 只把 **imu + gps 设为必需**；相机（`/cam0/image_raw`）暂不可用只告警、不中断——桥起来出图后锚点/图像自然开始写入。
- 请让设备**持续运动**（尤其 VINS 需要视差来初始化并发布 `/odometry`）。若无运动，`vio.jsonl` 可能为 0。
- 到达路线终点按 **Ctrl+C**。recorder 优雅停止后脚本会执行 `BUILDING REFERENCE TRAJECTORY`。
  - 若 Ctrl+C 把整个 bash 一起打断、没看到 `=== BUILDING REFERENCE TRAJECTORY ===`，可事后单独补建（见 6.3）。
  - 若想**重跑录制**：先删旧目录，否则 recorder 会报 `route already exists`。

### 6.2 生成成功后检查

```text
memory_nav/routes/suishi-2/
├── manifest.json            # state 应为 READY
├── raw/{gps,imu,vio,events}.jsonl
├── anchors/                 # candidate_XXXX.jpg 锚点图
├── anchors.json             # 锚点（含 point_index / image_path）
├── reference_trajectory.json
└── quality_report.json      # ready 应为 true
```

只有 `manifest.state == READY` 且 `quality_report.ready == true` 才能被跟随。

> 不使用 VINS 录制时 `raw/vio.jsonl` 为空（无 `/odometry`）属正常，位置源只有 `gps`。

### 6.3 事后补建 / 配置变更后重建

参考轨迹由 `build_reference` 从 `raw/*` 生成。若上次中断或改了 `resample_spacing_m`/锚点参数，需**先删旧 `anchors.json`**（否则不会被刷新）再重建：

```bash
rm -f memory_nav/routes/<route-id>/anchors.json
python3 -m memory_nav.trajectory.build_reference \
  --route-id <route-id> --config memory_nav/config/memory_nav.yaml
```

`anchors/*.jpg` 与 `raw/events.jsonl` 会被继续引用，不需要删。

### 6.4 不使用 VINS 录制（只用 GPS+IMU，可带相机锚点）

不启动 VINS，`recorder` 不传 `--vio-topic` 时会**直读 `/dev/imu`**，只用 GPS + IMU（可选相机锚点）录制原始数据：

```bash
# 录制：--camera 用硬件网关相机抓锚点候选；不传则仅 GPS+IMU
bash memory_nav/scripts/start_recording.sh --route-id office-a --camera

# Ctrl+C 结束后，对已有 raw 构建参考轨迹
bash memory_nav/scripts/build_reference.sh --route-id office-a
```

要点：

- 前置：处于 `blind` 环境、GPS 服务在跑、硬件网关在跑（`--camera` 时）、`/dev/imu` 存在。
- 此时 recorder **不要**与其他进程（如 VINS bridge）抢同一串口 `/dev/imu`。
- 产物中 `raw/vio.jsonl` 为空，位置源只有 `gps`（`position_source=gps`），属正常。
- 跟随这条路线用纯 GPS+IMU 入口，见 [9.1](#91-只用-gpsimu-跟踪不带视觉vins)（`start_gps_imu_follow.sh`）。

---

## 7. 一键跟随参考轨迹

```bash
bash memory_nav/scripts/follow_reference_trajectory.sh \
  --route-id suishi-2 --vins-imu-port /dev/imu
```

该入口默认启用 VINS、GPS/VIO 在线对齐、视觉锚点验证与 TTS。可选参数：

```text
--config FILE          与生成阶段同一份配置
--interval S           跟随循环周期，默认 0.2 秒
--voice                启用语音
--visual-anchors       启用视觉锚点
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
segmentation           分割避障结果（仅 replay_nav_seg，见 16.8）
```

> `vio_used`/`position_source=vio` 依赖 VINS 在录制与跟随期间能初始化并对齐。单目 VINS 有尺度/漂移，若 `vio_alignment_rms_m` 很大会被拒，位置源会回落到 `gps`（这是预期降级，不影响跟随主体）。

---

## 8. 方向指令与语音节奏

方向 = “当前 IMU 航向” 与 “参考路线前方切线航向” 的差，转成时钟方向：

```text
12点钟方向直行约8米
2点钟方向前进约8米
原地转至6点钟方向，然后前进约8米
```

语音由 `utils.voice.TextToSpeechPlayer` 在后台队列播放，不阻塞匹配。为**避免太频繁**：
- 同一条提示受 `voice.cooldown_s`（默认 10s）冷却；
- 方向播报的 key 为 `direction:<钟点>`，因此**钟点变化会立即播报**，而同一钟点在 `voice.cooldown_s` 内不重复（见 `replay_nav.py`）。直线段保持同一钟点，不会反复唠叨。
- 偏离/恢复、锚点到达等状态提示另算。
- 分割避障的提示由 `SegmentationGuard` 自己的调度器按 `--seg-cooldown`（默认 5s）冷却，见 16.6。

语音需要百度 TTS 凭证有效、网络可用、系统有音频输出。合成缓存存在路线目录 `tts_cache/`。

---

## 9. 视觉锚点

- 生成阶段从 `/cam0/image_raw` 取清晰帧做锚点。当前配置只产生 **start / turn(转向≥35°) / end** 锚点（`anchors.max_spacing_m` 调大后不再按距离插间隔锚点）。
- 跟随阶段用本地 XFeat 把当前帧与历史锚点比对做视觉校验；失败不会中断，会自动回到 GPS+IMU/VIO。

### 9.1 只用 GPS+IMU 跟踪（不带视觉/VINS）

不需要视觉锚点，也不需要 VINS/VIO，只用实时 GPS + IMU 匹配历史轨迹。GPS 服务需已在运行：

```bash
# 一键脚本（推荐）
bash memory_nav/scripts/start_gps_imu_follow.sh \
  --route-id suishi-2 --voice

# 等价的 Python 入口（不带 --visual-anchors / --vio-topic）
python3 -m memory_nav.replay.replay_nav \
  --route-id suishi-2 --voice
```

不传 `--vio-topic` 时 `replay_nav` 不订阅 VINS，位置源只会是 `gps`；不传 `--visual-anchors` 时不做视觉校验。带 VINS 的版本（VIO+视觉锚点）见第 7 节。

---

## 10. 可视化参考轨迹

把生成的参考轨迹（及锚点）画成 PNG：

```bash
python3 -m memory_nav.analysis.plot_reference \
  --route-id <route-id> --config memory_nav/config/memory_nav.yaml
```

输出：`memory_nav/routes/<route-id>/reference_trajectory.png`（经纬度折线 + 锚点叠加，start/turn/end 不同颜色/标记）。可加 `--out /path.png` 覆盖输出路径。

---

## 11. 保存跟随日志做复盘

```bash
mkdir -p memory_nav/hardware_reports
bash memory_nav/scripts/follow_reference_trajectory.sh \
  --route-id suishi-2 --vins-imu-port /dev/imu \
  | tee memory_nav/hardware_reports/suishi-2-follow.jsonl
```

跟随 GPS 日志默认写入 `memory_nav/routes/<id>/follow_output/follow-<时间戳>/output.jsonl`，可用 `--follow-log /path/to/output.jsonl` 指定。

---

## 12. 其他入口与脚本清单

### 12.1 常用命令

```bash
# 只录 GPS/IMU/相机，不起 VINS（recorder 此时直读 /dev/imu）
bash memory_nav/scripts/start_recording.sh --route-id office-a --camera

# 对已有 raw 构建参考轨迹
bash memory_nav/scripts/build_reference.sh --route-id office-a

# 基础在线跟随（无 VINS/视觉）
bash memory_nav/scripts/start_replay.sh --route-id office-a --interval 0.2

# GPS+IMU 跟随（无 VINS/视觉，GPS 服务需已运行）
bash memory_nav/scripts/start_gps_imu_follow.sh --route-id suishi-2 --voice

# 从 L4 output.jsonl 导入路线并构建，可选直接跟随
bash memory_nav/scripts/prepare_and_follow_gps_route.sh \
  --input /path/output.jsonl --route-id suishi-2 --skip 70 --follow

# 标定、硬件验收、离线重放
bash memory_nav/scripts/start_calibration.sh --help
bash memory_nav/scripts/hardware_acceptance.sh --help
bash memory_nav/scripts/replay_offline.sh --help
```

### 12.2 脚本职责一览

| 脚本 | Python 入口 | 职责 |
|---|---|---|
| `generate_reference_trajectory.sh` | `recording.recorder` + `trajectory.build_reference` | 起 VINS 并录制，Ctrl+C 后自动构建参考轨迹 |
| `follow_reference_trajectory.sh` | `replay.replay_nav` | 起 VINS 并跟随（VIO+视觉锚点+语音） |
| `follow_reference_trajectory_seg.sh` | `replay.replay_nav_seg` | 分割感知跟随；**不启动 VINS**，worker 自动拉起 |
| `precompute_walkable_masks.sh` | `segmentation.precompute_masks` | 用 catseg 环境批量生成 walkable mask |
| `start_recording.sh` | `recording.recorder` | 只录制原始数据 |
| `start_replay.sh` | `replay.replay_nav` | 基础在线跟随 |
| `start_gps_imu_follow.sh` | `replay.replay_nav` | 纯 GPS+IMU 跟随 |
| `build_reference.sh` | `trajectory.build_reference` | 从 raw 构建参考轨迹 |
| `replay_offline.sh` | `tools.replay_offline` | 用历史 JSONL 离线复现匹配 |
| `start_calibration.sh` | `calibration.collect_calibration` | 采集 Kalibr/AprilGrid 数据 |
| `hardware_acceptance.sh` | `tools.hardware_acceptance` | 实机验收与 JSON 证据 |
| `import_gps_log.sh` | `tools.import_gps_log` | 导入 L4 output.jsonl 为 MemoryNav 路线 |
| `prepare_and_follow_gps_route.sh` | 组合 import+build+follow | 一条命令导入/构建/跟随 |

> 生成、构建、审核、跟随必须用**同一份配置**，尤其 `routes_dir`、坐标系、匹配阈值。

---

## 13. 配置要点（当前默认值）

默认配置：`memory_nav/config/memory_nav.yaml`。

| 键 | 当前值 | 含义 |
|---|---|---|
| `coordinate_system` | `GCJ-02` | 坐标系（必须 GCJ-02） |
| `routes_dir` | `memory_nav/routes` | 路线根目录（相对项目根解析） |
| `trajectory.resample_spacing_m` | `5.0` | 参考点重采样间距（越小点越密） |
| `trajectory.smoothing_window` | `5` | 平滑窗口 |
| `trajectory.max_speed_m_s` | `4.0` | 速度去重上限 |
| `trajectory.duplicate_distance_m` | `0.05` | 重复点阈值 |
| `vio.topic` | `/odometry` | VINS 里程计话题 |
| `vio.maximum_alignment_rms_m` | `5.0` | VIO↔GPS 对齐最大 RMS |
| `vio.online_minimum_pairs` | `3` | 在线对齐最少配对数 |
| `quality.minimum_gps_samples` | `5` | 最少 GPS 样本 |
| `quality.minimum_route_length_m` | `3.0` | 路线长度下限（≤此值不 READY） |
| `quality.maximum_gps_gap_s` | `5.0` | 允许最大 GPS 间隙 |
| `quality.maximum_rejection_ratio` | `0.5` | 最大剔除比例 |
| `matching.*` | `{}` | 匹配窗口 / good/lost 阈值（走 `RouteMatcher` 默认） |
| `deviation.warning_m` | `3.0` | 横向告警阈值 |
| `deviation.severe_m` | `5.0` | 严重偏离阈值 |
| `deviation.phone_scale` | `2.0` | 手机 GPS 阈值放大 |
| `deviation.enter_samples` / `recover_samples` | `3` / `3` | 进入/恢复确认样本数 |
| `anchors.max_spacing_m` | `1000.0` | 调大=**关闭间隔锚点**，只留 start/turn/end |
| `anchors.turn_angle_deg` | `35.0` | 转向锚点阈值 |
| `anchors.advance_distance_m` | `8.0` | 提前提示距离 |
| `anchors.arrival_distance_m` | `2.0` | 到达判定距离 |
| `anchors.blur_threshold` | `100.0` | 锚点清晰度阈值 |
| `anchors.auto_confirm` | `true` | 自动确认锚点 |
| `voice.cooldown_s` | `10.0` | 同一提示冷却（方向 key 含钟点，故钟点变化即播） |
| `segmentation.*` | 见 16.7 | 分割映射避障（可选） |

> 为短/单次测试（如 ~1m 路线）可以复制配置并放宽 `quality.minimum_route_length_m`（例如 `0.5`），生成/跟随都传同一份副本。

---

## 14. 故障排查

| 现象 | 处理 |
|---|---|
| bridge 报 cv2/GStreamer 崩溃 | 确认在 `blind` 环境跑、bridge shebang 为 `env python3` |
| VINS 无位姿 / `vio.jsonl=0` | 录制时让设备持续运动，VINS 需视差初始化；`ros2 topic hz /imu0` 看 IMU 是否 ~100Hz |
| IMU 数据断续 / 掉帧 | 确认 recorder/replay 在带 VINS 时订阅 `/imu0` 而非双开 `/dev/imu` |
| GPS 预检失败 | `curl http://localhost:9000/gps`；若终端有 http_proxy 请 `unset http_proxy https_proxy` |
| 跟随第一步崩溃 / `无法连接硬件网关` | 先起 `python -m src.hardware_gateway` 与 `python client.py`，确认 `/health` |
| 眼镜画面收不到 | 确认 `adb reverse tcp:9989 tcp:9989`、眼镜在向 9989 推流、且无别的进程占 9989 |
| 路线不是 READY | 看 `quality_report.json` 的 `failure_reasons`；太短则放宽 `minimum_route_length_m` |
| 没有语音 | 查百度凭证、网络、PyAudio、默认音频输出 |
| 语音太频繁 | 调大 `voice.cooldown_s`；分割提示调大 `--seg-cooldown` |
| 录制 `route already exists` | 先删 `memory_nav/routes/<id>` 再录 |
| 参考轨迹没更新（改配置后） | `rm -f routes/<id>/anchors.json` 后重新 `build_reference` |
| 结束后 VINS 仍跑 | 再按一次 Ctrl+C，或 `ps aux \| grep run_live_vins` |
| 分割无效果 / 一直无 mask | 见 16.10（GPU 显存、worker 连接超时、mask 目录） |

---

## 15. 开发检查

```bash
python3 -m unittest discover -s memory_nav/tests -q
python3 -m compileall -q memory_nav
for script in memory_nav/scripts/*.sh; do bash -n "$script"; done
git diff --check
```

---

## 16. 分割映射避障（walkable segmentation）

把跟踪输出的时钟方向（12 直行、3 右、9 左）映射到相机图像底部中点的一个固定长度半圆（覆盖 9-12-0-3 前进半区），与 CAT-Seg 可通行区域比对：方向可行则放行；被遮挡时回退到最接近指令的可行钟点，并可语音提示。核心算法在 `memory_nav/segmentation/`。

该功能完全**可选**，默认不启用；不启用时原跟随逻辑行为不变。

### 16.1 几何与判据

```text
            12 (图像上方, θ=0°)
     11            1
   10                2
  9  (左)      (右)   3
            ● 原点在图像底部中点 (W/2, H)

半径 R = radius_ratio · H（默认 0.25·H）
内半径 = inner_radius_ratio · H（默认 0.05·H）
每个钟点 = ±half_width_deg（默认 ±15°）的楔形扇区
```

- 角度自图像正上方顺时针：`12→0°`、`3→+90°`、`9→-90°`；像素偏移 `dx=R·sin θ`、`dy=-R·cos θ`。
- **净空距离**：扇区内（内半径→外半径）到原点最近的非可通行像素的欧氏距离；扇区无遮挡时记为 `R`。
- **可行判定**：`obstacle_pixels ≤ max_obstacle_ratio · wedge_pixels`（默认容差 `0.005`=0.5%，即"几乎无遮挡"，用于过滤零星噪声像素；设为 `0` 即严格无遮挡）。
- **指令一致**：指令钟点 ±`command_tolerance_hours`（默认 1 格=30°）带内有可行钟点即通过；否则 `fallback_used`，回退到**角度最近**的可行钟点。
- 所有前进钟点都不可行 → `warning="no_walkable"`；指令为 4–8 点钟（后方）→ `out_of_scope`，不参与避障判定。

### 16.2 时钟方向 → 图像角度

| 钟点 | 角度 θ | 含义 |
|---:|---:|---|
| 12 | 0° | 正前方（图像上方） |
| 1 | +30° | 右前 |
| 2 | +60° | 右前 |
| 3 | +90° | 正右 |
| 9 | −90° | 正左 |
| 10 | −60° | 左前 |
| 11 | −30° | 左前 |

（4–8 点钟在后方，不在半圆内。）

### 16.3 模块与文件

| 文件 | 职责 |
|---|---|
| `geometry.py` | `SemicircleMapper`、`clock_to_angle_deg`、`hour_clearance`、`HourClearance` |
| `avoidance.py` | `SegmentationConfig`、`SegmentationResult`、`SegmentationAvoidance.evaluate` |
| `guard.py` | `SegmentationGuard`：查询 mask、评估、覆盖方向、生成语音提示、可选可视化回调 |
| `providers.py` | `CachedMaskProvider`（按帧名读 PNG）、`SequenceMaskProvider`（帧序列同步） |
| `online_provider.py` | `CatSegWorkerProvider`（自动拉起 worker、后台线程、限速复用最新 mask）、`NullMaskProvider` |
| `catseg_worker.py` | catseg 环境常驻服务：加载 CAT-Seg，socket 收帧回 mask |
| `protocol.py` | 4 字节长度前缀 + pickle 的 socket 帧协议 |
| `mask_loader.py` | mask 读写、`<stem>_walkable.png` 命名、缩放 |
| `visualize.py` | 半圆/扇区/箭头/净空刻度标注 |
| `online_visualizer.py` | `SegmentationFrameSaver`：限速保存标注帧 |
| `precompute_masks.py` | 离线批量生成 mask + manifest |
| `validate_offline.py` | 离线逐帧校验 + 指标 + 标注图 |

### 16.4 预计算可通行 mask（catseg 环境）

CAT-Seg 依赖 detectron2，只在 `catseg` conda 环境中可用；MemoryNav 运行在 `blind` 环境。因此先在 `catseg` 中批量生成 mask，再由 MemoryNav 读取：

```bash
# 用 catseg 环境对整目录帧生成 mask（默认 cuda；GPU 显存不足时用 --device cpu）
bash memory_nav/scripts/precompute_walkable_masks.sh \
  --frames /path/to/frames \
  --masks  /path/to/walkable_masks
# 可选：CATSEG_PYTHON=/path/to/catseg/bin/python 覆盖解释器
```

等价 Python 入口与参数：

```bash
~/anaconda3/envs/catseg/bin/python -m memory_nav.segmentation.precompute_masks \
  --frames /path/to/frames --masks /path/to/walkable_masks \
  --catseg-dir /home/wheeltec/projects/blind-nav-server/CAT-Seg \
  --config configs/vitb_384.yaml --weights model_base.pth \
  --device cuda --walkable-names pavement,road,stairs,floor-marble,floor-stone,floor-tile,floor-wood [--overwrite]
```

产物：`<stem>_walkable.png`（255=可通行）与 `manifest.json`（模型、权重、可通行类别、耗时）。可通行类别默认 `pavement,road,stairs,floor-marble,floor-stone,floor-tile,floor-wood`（来自 `CAT-Seg/datasets/coco.json`；室内硬地面用 `floor-*`，室外用 `pavement/road`）。

### 16.5 离线校验

对 `output.jsonl` + 帧 + mask 逐帧评估，输出 `seg_metrics.jsonl`、`summary.json` 与带半圆可视化的标注图：

```bash
python3 -m memory_nav.segmentation.validate_offline \
  --log    /path/to/output.jsonl \
  --frames /path/to/frames \
  --masks  /path/to/walkable_masks \
  --output /path/to/seg_validation \
  [--config memory_nav/config/memory_nav.yaml] \
  [--assoc sequential|path|frame] [--no-visualize]
```

- 指令钟点优先取 `command_clock` 字段，其次由 `angle_diff` 推导，再次解析 `guide` 文本（如“2点钟方向”）。
- `--assoc`：日志记录与帧的配对方式。`sequential` 按时间序插值（默认，适合帧数≠记录数）；`path`/`frame` 按 `image_path`/`camera_time_stamp` 匹配。

产物：
- `vis/<stem>_seg.jpg`：标注图。
- `seg_metrics.jsonl`：每帧 `command_clock/seg_clock/consistent/fallback_used/out_of_scope/clear_distances/obstacle_pixels/walkable_hours/intersection_hours/warning` 等。
- `summary.json`：`evaluated/consistent/fallback/out_of_scope/no_walkable/missing_mask/unresolved_command/consistency_rate_in_scope/fallback_rate_in_scope`。

### 16.6 在线跟随（分割感知版本）

`memory_nav.replay.replay_nav_seg` 是 `replay_nav` 的分割感知版本。它复用原有跟踪逻辑，只在计算完 `command_clock` 后追加一步避障判定：可行则保持指令；不可行则用可通行钟点覆盖指令并语音提示。**不启用分割时与 `replay_nav` 行为一致。**

支持两种 mask 来源：

#### A. 在线实时（`--seg-online`）

provider 自动拉起一个常驻 `catseg` 环境的 CAT-Seg worker 子进程，通过本地 Unix domain socket 传帧传 mask。推理在后台线程执行、限速复用最新 mask，**绝不阻塞 5Hz 的导航主循环**；worker 崩溃时静默降级（不做覆盖），不影响原有导航。

```bash
# 一键脚本（不启动 VINS，假定 VINS + 硬件网关已在运行）
bash memory_nav/scripts/follow_reference_trajectory_seg.sh \
  --route-id suishi-2 --config memory_nav/config/suishi-2.yaml \
  --seg-device cuda --seg-vis-dir memory_nav/analysis/output/seg_online

# 等价的 Python 入口
python3 -m memory_nav.replay.replay_nav_seg \
  --route-id suishi-2 --config memory_nav/config/suishi-2.yaml \
  --seg-online --vio-topic /odometry --ros-image-topic /cam0/image_raw \
  --seg-device cuda --seg-vis-dir memory_nav/analysis/output/seg_online --voice
```

- 相机优先用 VINS 的 `--ros-image-topic`（推送式、与硬件网关无争用）；没有 VINS 时回退硬件网关 `get_state_image`。
- 在线参数：

  | 参数 | 默认 | 含义 |
  |---|---|---|
  | `--seg-online` | 关 | 启用常驻 CAT-Seg worker |
  | `--seg-radius-ratio F` | 用配置 | 在线判定半径覆盖（CLI 优先） |
  | `--seg-worker-python PATH` | catseg python | worker 解释器 |
  | `--seg-catseg-dir DIR` | 配置 | CAT-Seg 项目目录 |
  | `--seg-config` / `--seg-weights` | `configs/vitb_384.yaml` / `model_base.pth` | 模型 |
  | `--seg-device cuda\|cpu` | `cuda` | 推理设备 |
  | `--seg-input-scale F` | `0.5` | 推理前下采样（省显存/提速） |
  | `--seg-min-size-test N` | `384` | 覆盖 `INPUT.MIN_SIZE_TEST` |
  | `--seg-max-hz N` | `1.0` | worker 最大推理频率 |
  | `--seg-walkable-names` | `pavement,road,stairs,floor-marble,floor-stone,floor-tile,floor-wood` | 可通行类别 |
  | `--seg-socket PATH` | 自动 | worker socket 路径 |
  | `--seg-vis-dir DIR` | 配置 | 标注帧输出目录 |
  | `--seg-vis-interval S` | `1.0` | 标注帧保存间隔 |
  | `--seg-show` | 关 | 有显示时额外开窗口 |
  | `--seg-cooldown S` | `5.0` | 分割语音冷却 |

- 语音提示：
  - `fallback`：`前方不可通行，建议向X点钟方向前进`（key 含钟点）。
  - `no_walkable`（所有前进钟点都不可行）：依次播报 `前方未检测到可通行区域，请停止前进` 与 `无可行区域，请旋转一下`；两条使用**稳定 key**（旋转不刷屏）并按 `--seg-cooldown` 冷却。文案可在 `segmentation.messages` 配置。

#### B. 离线预计算（`--walkable-mask-dir`）

读取 `precompute_masks` 生成的 mask PNG，可用 `--simulate-frames` 回放帧序列：

```bash
python3 -m memory_nav.replay.replay_nav_seg \
  --route-id suishi-2 \
  --walkable-mask-dir /path/to/walkable_masks \
  --voice
# 回放已录制帧序列（mask 与帧同步推进）：
#   追加 --simulate-frames /path/to/frames
```

### 16.7 配置

`memory_nav/config/memory_nav.yaml` 的 `segmentation` 段：

| 键 | 默认 | 含义 |
|---|---|---|
| `enabled` | `false` | 预留开关（当前以命令行是否传 seg 参数为准） |
| `mask_dir` | `null` | 离线 mask 目录（也可 `--walkable-mask-dir`） |
| `origin_x_ratio` / `origin_y_ratio` | `0.5` / `1.0` | 半圆原点在图像中的相对位置（底部中点） |
| `radius_ratio` | `0.25` | 半圆半径 / 图像高 |
| `inner_radius_ratio` | `0.05` | 内半径（忽略原点附近噪声） |
| `half_width_deg` | `15.0` | 每钟点扇区半角 |
| `max_obstacle_ratio` | `0.005` | 扇区允许的障碍像素占比（0=严格无遮挡） |
| `command_tolerance_hours` | `1` | 指令一致的角度容差（格） |
| `forward_hours` | `[9,10,11,12,1,2,3]` | 前进半区钟点 |
| `messages.no_walkable` | `前方未检测到可通行区域，请停止前进` | 无可通行文案 |
| `messages.no_walkable_rotate` | `无可行区域，请旋转一下` | 旋转提示文案 |
| `catseg_project_dir` | `/home/wheeltec/projects/blind-nav-server/CAT-Seg` | CAT-Seg 目录 |
| `catseg_config` / `catseg_weights` / `catseg_device` | `configs/vitb_384.yaml` / `model_base.pth` / `cuda` | 模型与设备 |

在线子段 `segmentation.online`：

| 键 | 默认 | 含义 |
|---|---|---|
| `worker_python` | `~/anaconda3/envs/catseg_seg/bin/python` | worker 解释器 |
| `socket_path` | `null` | 指定 socket，默认自动 `/tmp/memory_nav_catseg_<pid>_<id>.sock` |
| `device` | `cuda` | 推理设备 |
| `input_scale` | `0.5` | 推理前下采样 |
| `min_size_test` | `384` | `INPUT.MIN_SIZE_TEST` 覆盖 |
| `max_hz` | `1.0` | 最大推理频率 |
| `walkable_names` | `pavement,road,stairs,floor-marble,floor-stone,floor-tile,floor-wood` | 可通行类别 |
| `vis_dir` | `memory_nav/analysis/test_0917/seg_online` | 标注帧目录 |
| `vis_interval_s` | `1.0` | 标注帧间隔 |
| `radius_ratio` | `null` | **在线专用**半径覆盖（`null`=沿用基础值） |

覆盖优先级：`--seg-radius-ratio` > `segmentation.online.radius_ratio` > `segmentation.radius_ratio`。半径同时作用于判定与可视化。

### 16.8 输出字段

`replay_nav_seg` 每步 JSON 增加 `segmentation` 字段（无 mask 时为 `null`）：

```text
segmentation.command_clock      跟踪原始指令钟点
segmentation.seg_clock          避障后的最终钟点（= 覆盖后的 command_clock）
segmentation.consistent         指令与可通行区域是否一致
segmentation.fallback_used      是否发生回退
segmentation.out_of_scope       指令在后方（4–8 点），未参与判定
segmentation.clear_distances    {钟点: 净空距离(px)}
segmentation.obstacle_pixels    {钟点: 障碍像素数}
segmentation.walkable_hours     可行的前进钟点
segmentation.intersection_hours 指令容差带内可行的钟点
segmentation.warning            null / command_blocked / no_walkable / out_of_scope
```

跟随 GPS 日志（`output.jsonl`）同样包含 `segmentation` 字段。

### 16.9 可视化说明

`visualize.py` / 在线标注帧约定：

- 绿色扇区 = 可行（无遮挡或容差内）；红色扇区 = 不可行。
- **蓝色箭头** = 跟踪原始指令方向；**橙色整半径箭头** = 修改（回退）后的方向，仅在发生回退时绘制。
- 黄色圆点 = 各钟点的净空距离刻度；白色弧与辐条 = 判定半圆与钟点划分。
- 半透明绿色 = CAT-Seg walkable mask 原始叠加。
- 顶部文字：`cmd=… seg=… [status] r=…px`、`walkable=…`、`clear(px)=钟点:距离 …`。

### 16.10 性能、资源与已知限制

- **CPU**：实测约 **57s/帧**（模型加载约 100s，仅一次），不适合实时，只适合离线/验证。
- **GPU（Orin 统一内存）**：当前机器可用显存约 2.8GB，CAT-Seg ViT-B 在 `SIZE_DIVISIBILITY:384` 下最小输入约 384×384，激活显存需求超过可用值，推理会 OOM（`NvMapMemAlloc` / NVML 断言）。纯算力本身正常；瓶颈是内存。释放内存到 ≥4–5GB 后再复测。
- **在线 worker 连接超时**：`CatSegWorkerProvider` 默认 `connect_timeout_s=60`，而 worker 是**先加载模型后 bind socket**。GPU 加载约 18s 无碍；**CPU 加载约 100s 可能超过 60s 而连接失败**，此时分割静默失效。用 CPU 在线时需注意，或后续把 `connect_timeout_s` 配置化。
- **降级行为**：worker 崩溃 / 无 mask / 无相机帧时，`guard` 返回 `None`，导航按原逻辑继续，不做方向覆盖。
- **实时性**：mask 以 `max_hz`（默认 1Hz）更新，主循环（默认 5Hz）复用最新 mask，因此避障反应约 1s 级。
- **相机争用**：用 VINS `--ros-image-topic` 时与硬件网关无争用；回退到硬件网关相机时，`get_state_image` 与 `get_gps`/`get_heading` 共用网关锁，可能降低 GPS/IMU 调用频率。

### 16.11 测试

```bash
python3 -m unittest memory_nav.tests.test_segmentation -q
python3 -m unittest memory_nav.tests.test_online_segmentation -q
```

覆盖：时钟↔角度映射、净空距离、可行性/交集/回退/out_of_scope/no_walkable、socket 协议、provider 非阻塞与 RGB→BGR、可视化箭头颜色、在线半径覆盖优先级、no_walkable 双提示与稳定 key 冷却。
