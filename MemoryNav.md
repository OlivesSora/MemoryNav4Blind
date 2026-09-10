# MemoryNav：轨迹记录与重演规划

## 1. 背景与目标

MemoryNav 的目标是利用 IMU、GPS 和相机记录一次真实行走过程，生成稳定、可复用的历史参考轨迹，并在后续重演时持续判断用户相对历史路线的位置、方向和进度，通过锚点图片和语音提供导航提示。

系统分为四个相互独立的阶段：

1. Camera+IMU 标定：获得相机内参、Camera-IMU 外参和时间偏移。
2. 路线录制：同步记录 GPS、IMU、必要的相机图像及定位质量。
3. 离线处理：清洗、融合和平滑传感器数据，生成参考轨迹和锚点。
4. 路线重演：将当前位置连续匹配到历史轨迹，计算偏航和偏移，触发视觉校验及语音提示。

首版只支持按录制方向单向重演。反向路线应作为另一条路线单独录制，避免反向提示、锚点视角和交叉路口方向产生歧义。

## 2. 总体原则

- MemoryNav 新增的 Python 代码、配置、测试工具和 Bash 启动脚本必须统一放在 `memory_nav/` 目录下，不在项目根目录散落新的导航脚本。
- `utils/imu.py`、`utils/gps.py`、相机和语音模块作为现有硬件适配层供 MemoryNav 导入；除非底层接口确实缺少必要数据，否则 MemoryNav 的业务逻辑不得写入 `utils/`。
- GPS 提供全局位置，IMU 提供航向和转动信息；不长期积分 IMU 加速度作为绝对位置，避免累计漂移。
- RTK 数据使用高权重；手机 GPS 或质量较差的数据保留但降低权重。
- 锚点图片用于确认路线进度和进行小范围纠偏，不单独承担连续定位。
- 所有传感器数据必须带有统一时钟下的采样时间，不能用相机帧计数代替时间戳。
- 同时保存原始数据和处理结果。算法升级后应能从原始数据重新生成参考轨迹。
- 坐标、角度、正负方向、数据来源和标定版本必须写入路线元数据，避免隐式约定。

总体数据流：

```text
IMU / GPS / Camera
        │
        ▼
统一时间戳与质量检查
        │
        ▼
原始录制会话落盘
        │
        ▼
轨迹清洗、融合、平滑、锚点确认
        │
        ▼
版本化参考路线
        │
        ▼
在线匹配、偏差计算、视觉校验、语音提示
```

## 3. Camera+IMU 时空联合标定

### 3.1 标定目标

使用 Kalibr + AprilGrid 完成：

- 相机焦距、主点和畸变参数。
- Camera 到 IMU 的刚体外参（旋转和平移）。
- Camera 与 IMU 的时间偏移。
- 标定结果的重投影误差和 IMU 残差评估。

普通路线数据可以验证标定后的时间、航向和图像运动是否一致，但不能可靠替代专用标定数据。标定时应单独创建会话，连续采集 AprilGrid 图像、陀螺仪和加速度计数据，并充分激励三个轴的旋转与平移。

### 3.2 标定采集要求

- 固定 Camera 和 IMU 的安装位置，标定后不得改变相对姿态。
- 使用实际运行分辨率采集清晰的 AprilGrid 图像。
- 标定板在画面中心、边缘和不同距离均应出现。
- 进行缓慢且丰富的 roll、pitch、yaw 运动，避免只绕单轴旋转。
- 图像和 IMU 使用同一个主机单调时钟记录接收时间，同时保存设备原始时间戳。
- 标定会话记录设备序列号、相机分辨率、IMU 安装偏角和软件版本。

### 3.3 标定产物

标定结果保存为版本化 YAML，至少包括：

- 相机模型、内参、畸变模型和畸变参数。
- `T_camera_imu` 或明确命名的等价变换矩阵，并注明变换方向。
- Camera-IMU 时间偏移及符号定义。
- 适用的设备、分辨率、采集日期和标定工具版本。
- 重投影误差、IMU 残差及是否通过质量检查。

路线清单必须引用具体的标定版本。标定质量不合格或设备配置不匹配时，不允许开始正式路线录制。

## 4. 传感器接口与时间同步

### 4.1 IMU

当前 `utils/imu.py` 已能提供航向、角速度、姿态四元数、加速度、磁场、设备时间和解析诊断信息。路线录制需要在此基础上增加非丢样的数据订阅或有界缓冲接口，避免低频轮询只能取得“最后一个样本”。

每条 IMU 样本至少包含：

```json
{
  "monotonic_ns": 0,
  "device_time_s": 0.0,
  "heading_deg": 0.0,
  "angular_velocity_rad_s": [0.0, 0.0, 0.0],
  "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
  "accel_m_s2": [0.0, 0.0, 0.0],
  "mag": [0.0, 0.0, 0.0],
  "heading_ready": true
}
```

IMU 诊断信息应定期写入日志，包括有效帧数、丢帧数、CRC 错误、最新 AHRS 数据年龄和航向是否就绪。

### 4.2 GPS

当前 `utils/gps.py` 只返回坐标和 `gps_state`。后续接口还应返回：

- 主机采样时间和服务端定位时间。
- `source`：`rtk` 或 `phone`。
- `state`、样本新鲜度和数据是否可用于融合。
- 可获得时记录 fix quality、水平精度、卫星数等质量信息。

坐标顺序统一为 `[longitude, latitude]`。当前 GPS 服务输出 GCJ-02，因此路线元数据必须明确记录 `coordinate_system: GCJ-02`，不得与 WGS84 数据直接混用。

建议的 GPS 样本结构：

```json
{
  "monotonic_ns": 0,
  "server_time_ns": 0,
  "longitude": 0.0,
  "latitude": 0.0,
  "coordinate_system": "GCJ-02",
  "source": "rtk",
  "state": "good",
  "age_s": 0.0,
  "accuracy_m": null,
  "fix_quality": null
}
```

### 4.3 Camera

当前相机接口返回图像和帧计数。后续应返回图像及帧元数据：

- `monotonic_ns`：该帧在主机侧的采集/接收时间。
- 相机端时间戳（若设备能够提供）。
- 帧号、宽高、图像格式和清晰度分数。
- 标定采集时保存连续帧；路线录制时只保存锚点候选及必要的前后帧。

### 4.4 统一时间基准

- 会话内统一采用 `time.monotonic_ns()`，避免系统时间校正导致时间倒退。
- 同时在会话清单中记录 UTC 开始和结束时间，便于日志追踪。
- 设备时间戳保持原值，离线处理中估计设备时钟到主机单调时钟的映射。
- Camera-IMU 使用标定得到的时间偏移；GPS 按定位生成时间优先、主机接收时间兜底。
- 记录数据排队和落盘应异步执行，不能阻塞 IMU 串口读取线程。

## 5. 代码与脚本组织

所有为 MemoryNav 新增的实现均放在 `memory_nav/`。建议目录如下，实际开发时应保持模块职责一致：

```text
memory_nav/
├── MemoryNav.md
├── __init__.py
├── config/
│   ├── memory_nav.yaml
│   └── kalibr/
├── calibration/
│   ├── collect_calibration.py
│   ├── import_calibration.py
│   └── validate_calibration.py
├── recording/
│   ├── recorder.py
│   ├── session_writer.py
│   └── anchor_collector.py
├── trajectory/
│   ├── coordinate.py
│   ├── fusion.py
│   ├── smoothing.py
│   └── build_reference.py
├── replay/
│   ├── matcher.py
│   ├── deviation.py
│   ├── anchor_matcher.py
│   └── replay_nav.py
├── interaction/
│   └── voice_prompt.py
├── tools/
│   ├── inspect_session.py
│   └── replay_offline.py
├── tests/
├── scripts/
│   ├── start_calibration.sh
│   ├── start_recording.sh
│   ├── build_reference.sh
│   ├── start_replay.sh
│   └── replay_offline.sh
└── routes/
```

目录约束：

- Python 包内部使用包导入，禁止依赖启动时的当前工作目录拼接模块路径。
- 运行数据默认写入 `memory_nav/routes/`，并允许通过配置或命令行参数覆盖数据根目录。
- 所有 Bash 入口放在 `memory_nav/scripts/`，脚本自行解析所在目录，将项目根目录加入 `PYTHONPATH`，因此可以从任意当前目录启动。
- Bash 脚本使用 `set -euo pipefail`，检查 Python 环境、配置文件、设备和必要参数，失败时返回非零状态。
- Bash 脚本只负责环境准备和启动，不在脚本中实现轨迹算法或复制 Python 业务逻辑。
- Python 模块应同时支持 `python -m memory_nav.<module>` 调用，便于测试和排错。
- 路线 ID、配置文件和日志级别通过命令行参数传入；禁止在 Python 或 Bash 中硬编码具体路线名和绝对项目路径。

启动脚本职责：

| 脚本 | Python 入口 | 职责 |
| --- | --- | --- |
| `start_calibration.sh` | `calibration.collect_calibration` | 检查相机/IMU 后采集 Kalibr 数据 |
| `start_recording.sh` | `recording.recorder` | 创建路线会话并同步记录传感器和锚点候选 |
| `build_reference.sh` | `trajectory.build_reference` | 清洗、融合、平滑并生成参考轨迹和质量报告 |
| `start_replay.sh` | `replay.replay_nav` | 加载已就绪路线并启动在线重演与语音提示 |
| `replay_offline.sh` | `tools.replay_offline` | 使用历史原始数据离线复现匹配和偏离计算 |

脚本的统一调用形式建议为：

```bash
bash memory_nav/scripts/start_recording.sh \
  --route-id route_001 \
  --config memory_nav/config/memory_nav.yaml

bash memory_nav/scripts/build_reference.sh --route-id route_001
bash memory_nav/scripts/start_replay.sh --route-id route_001
```

## 6. 路线数据组织

每条路线使用独立目录，例如：

```text
routes/<route_id>/
├── manifest.json
├── raw/
│   ├── imu.jsonl
│   ├── gps.jsonl
│   └── events.jsonl
├── calibration/
│   └── calibration.yaml
├── anchors/
│   ├── anchor_0001.jpg
│   └── anchor_0001_thumb.jpg
├── anchors.json
├── reference_trajectory.json
└── quality_report.json
```

`manifest.json` 至少记录：

- `schema_version`、`route_id`、路线名称和录制方向。
- UTC/单调时钟起止时间。
- 坐标系、局部坐标原点和轴方向。
- IMU、GPS、Camera 设备及软件版本。
- Camera+IMU 标定文件和版本。
- 轨迹处理参数、匹配参数和语音参数。
- 数据是否完整、是否已完成离线处理、是否允许重演。

JSONL 用于持续追加原始数据，进程异常退出时仍可保留此前样本。会话正常结束后再生成不可变的参考轨迹和质量报告。

## 7. 路线录制

### 7.1 录制状态机

```text
IDLE → SENSOR_CHECK → RECORDING → FINALIZING → READY
                         │              │
                         └── ERROR ─────┘
```

- `SENSOR_CHECK`：检查 IMU 航向已稳定、GPS 样本新鲜、相机可用、标定配置匹配且存储空间充足。
- `RECORDING`：持续写入 IMU/GPS，并生成锚点候选。
- `FINALIZING`：完成数据校验、轨迹融合、平滑、锚点确认和质量报告。
- `READY`：质量检查通过，可用于路线重演。
- 发生相机短暂断流时可继续记录位置；IMU 或 GPS 长时间失效时标记数据缺口并提示记录者。

### 7.2 自动锚点候选

以下场景自动生成候选：

- 路线起点和终点。
- 航向累计变化超过阈值的明显转弯。
- 路口、停留点或记录者明显减速的位置。
- 距离上一个锚点超过配置的最大间距。
- 记录者手动触发的位置。

候选图片先进行模糊检测。图片不清晰、曝光异常或视角不合适时应提示重新采集。自动候选必须由记录者确认、删除或补充提示文本后，才能进入正式参考路线。

## 8. 历史参考轨迹生成

### 8.1 坐标转换与清洗

- 以首个可信 GPS 点为局部原点，将 GCJ-02 经纬度转换为局部东-北平面坐标，单位为米。
- 原始经纬度必须保留，局部坐标只用于距离、匹配和平滑计算。
- 删除重复点、时间倒退、过期样本、不可能速度造成的跳点和明显离群点。
- RTK 样本赋高权重；手机 GPS 或 `poor` 状态降低权重。
- 长时间无有效 GPS 时形成轨迹缺口，不使用 IMU 加速度强行补齐长距离绝对位置。

### 8.2 GPS+IMU 融合

- GPS 约束二维位置，IMU 航向约束路线方向和转弯区段。
- 根据相邻可信 GPS 点的运动方向，离线估计 IMU 航向与轨迹切线之间的固定零偏。
- 低速或静止时不使用 GPS 差分方向校正航向，因为位置噪声会导致方向不稳定。
- IMU 角速度用于识别转弯和约束短时间方向变化；加速度仅用于运动状态判断或短时辅助。
- 输出每个融合点的定位来源、质量等级和是否位于数据缺口附近。

### 8.3 平滑与重采样

- 先完成离群点剔除，再进行鲁棒位置滤波。
- 按累计弧长等距重采样，避免 GPS 采样疏密影响最近点匹配。
- 使用保形的局部平滑或带约束样条降低抖动。
- 转弯点、起终点和人工确认锚点作为约束点，禁止平滑切角或跨越实际道路。
- 从平滑后的相邻点计算单位切线、参考航向和曲率。

每个参考点至少包含：

```json
{
  "index": 0,
  "s_m": 0.0,
  "east_m": 0.0,
  "north_m": 0.0,
  "longitude": 0.0,
  "latitude": 0.0,
  "heading_deg": 0.0,
  "curvature": 0.0,
  "quality": "good",
  "anchor_id": null
}
```

### 8.4 质量报告

处理完成后生成：

- 路线总长度和有效持续时间。
- RTK、手机 GPS 和无定位区段的占比。
- IMU 丢帧率、GPS 最大间隙和相机锚点有效率。
- 平滑前后的平均/最大位置变化。
- IMU 航向与轨迹切线的一致性。
- 未解决的数据缺口、异常区段和是否允许正式重演。

## 9. 路线重演

### 9.1 在线定位输入

重演以当前 GPS 位置和 IMU 航向为主要输入。当前位置转换到与参考路线相同的局部坐标系。每次计算均检查样本时间和质量；陈旧数据不能驱动路线进度更新。

### 9.2 连续最近点匹配

不能直接在整条轨迹上选择欧氏距离最小的单个点，否则在平行道路、折返和交叉路口容易跳段。采用“空间候选 + 路线进度连续性”的最近线段投影：

1. 首次定位在路线起点附近建立初始进度。
2. 后续以上次匹配进度为中心，只搜索有限的后退/前进窗口。
3. 将当前位置正交投影到候选参考线段。
4. 综合投影距离、当前航向与线段方向差、进度变化合理性选择最佳线段。
5. 匹配成功后更新沿线进度；默认进度不回退，只允许小范围抖动修正。
6. 匹配质量低时保持上一次可信进度，不推进锚点。

可使用 KD-Tree 获取空间候选，但最终结果必须经过进度窗口、方向和状态机约束。

### 9.3 偏移与偏航定义

匹配结果统一输出：

- `matched_s_m`：当前位置投影到路线后的累计里程。
- `cross_track_error_m`：有符号横向偏移。
- `heading_error_deg`：当前 IMU 航向与轨迹切线航向之差。
- `distance_to_next_anchor_m`：距离下一个有效锚点的沿线距离。
- `match_quality`：`good`、`degraded` 或 `lost`。
- `navigation_state`：正常、偏离、恢复中、完成等状态。

局部东-北坐标中，路线线段从 `P0` 指向 `P1`。横向偏移符号由二维叉积确定：正值表示位于路线前进方向左侧，负值表示右侧。航向误差归一化到 `[-180°, 180°]`：

```text
heading_error = wrap_to_180(current_heading - reference_heading)
```

### 9.4 偏离与恢复

- 首版建议在可信 RTK 下，横向偏移超过 3 m 进入告警，超过 5 m 判为严重偏离。
- 使用手机 GPS 或较差定位时适当放宽阈值，且需要更多连续样本确认。
- 进入偏离状态后暂停推进锚点和普通转向提示，播报偏离方向与距离。
- 只在当前路线进度附近持续尝试重新捕获，禁止跳到远处的平行或交叉轨迹段。
- 连续若干个可信样本回到路线走廊后进入恢复状态，再恢复普通导航。
- GPS 丢失和用户真实偏离必须区分：定位不可用时提示定位异常，不误报用户走错。

所有距离阈值、连续确认次数、前后搜索窗口和恢复条件均应配置化。

## 10. 锚点图片与视觉校验

每个正式锚点保存：

- 锚点 ID、类型、路线进度和参考点索引。
- GPS 坐标、局部坐标和参考航向。
- 原始图片、缩略图、采集时间和清晰度分数。
- 中文提示文本、提前播报距离和是否需要到点确认。
- 可选的局部特征缓存及特征提取器版本。

重演接近锚点时，采集当前图像并复用仓库中的 XFeat 与历史图片匹配：

- 仅在路线进度和航向已进入锚点窗口后启动视觉匹配，避免全库检索误匹配。
- 对匹配数量、内点比例和几何一致性进行联合判断，不能只使用原始匹配数量。
- 匹配成功时确认到达锚点，或将路线进度约束到锚点附近。
- 图片只校正路线进度，不直接覆盖当前 GPS 地理坐标。
- 匹配失败时继续使用 GPS+IMU；失败本身不阻塞安全和偏离提示。
- 对光照变化、动态遮挡和重复建筑纹理保留降级路径。

## 11. 语音提示

语音播放复用 `utils/voice.py` 中的 `TextToSpeechPlayer.say()`，分为：

1. 提前提示：接近转弯或重要锚点时播放，例如“前方路口左转”。
2. 到点确认：GPS/IMU 进度或视觉锚点确认后播放，例如“已到达路口，请左转”。
3. 异常提示：偏离、定位丢失、传感器异常和恢复路线。

提示调度要求：

- 每个锚点记录是否已提前提示、是否已到点确认。
- 使用滞回和冷却时间，避免阈值附近抖动造成重复播放。
- 偏离提示按状态变化和冷却周期播放，不得每个采样周期播报。
- 高优先级安全/偏离提示可以打断低优先级普通提示。
- TTS 或音频设备不可用时写入错误日志，但不能阻塞定位线程。

## 12. 配置建议

所有数值均应放入路线或系统配置，不在算法中硬编码。首版建议值用于初始实测，最终以场地测试结果为准：

| 配置项 | 首版建议 | 说明 |
| --- | ---: | --- |
| 参考轨迹重采样间距 | 0.5 m | 兼顾匹配精度与数据量 |
| 自动锚点最大间距 | 20 m | 没有语义事件时的兜底锚点 |
| RTK 横向告警 | 3 m | 需连续样本确认 |
| RTK 严重偏离 | 5 m | 暂停普通路线提示 |
| 锚点提前提示距离 | 8 m | 可按步速与场景调整 |
| 锚点到达距离 | 2 m | 与视觉确认联合使用 |
| 普通提示冷却 | 10 s | 防止重复播放 |

手机 GPS 阈值不能直接复用 RTK 阈值，应根据其 `accuracy_m` 动态放宽；无法取得精度时使用保守的独立配置。

## 13. 待办事项

### P0：数据基础

- [ ] 在 `memory_nav/` 下建立 Python 包、配置、测试、启动脚本和运行数据目录；后续新增文件不得散落到项目根目录。
- [ ] 实现统一的命令行参数和配置加载模块，供录制、处理、重演及 Bash 入口复用。
- [ ] 编写 `start_calibration.sh`、`start_recording.sh`、`build_reference.sh`、`start_replay.sh` 和 `replay_offline.sh`。
- [ ] 定义路线目录、JSON/JSONL/YAML 数据模型和 `schema_version`。
- [ ] 统一坐标系、角度、四元数顺序、横向偏移符号和时钟定义。
- [ ] 为 IMU 增加逐样本缓冲/订阅能力和录制诊断信息。
- [ ] 扩展 GPS 服务及客户端接口，返回时间、来源、新鲜度和精度元数据。
- [ ] 为相机帧增加真实采集时间及清晰度元数据。
- [ ] 实现异步会话写入器、正常结束和异常恢复机制。

### P1：标定与路线生成

- [ ] 实现 Kalibr + AprilGrid 标定数据采集工具。
- [ ] 生成 Kalibr 输入配置，导入并版本化保存标定结果。
- [ ] 实现标定质量检查和设备/分辨率匹配校验。
- [ ] 实现路线录制状态机和录制前传感器检查。
- [ ] 实现 GPS 清洗、局部坐标转换、GPS+IMU 离线融合。
- [ ] 实现保形平滑、等距重采样、参考航向和曲率计算。
- [ ] 生成路线质量报告，阻止低质量路线进入正式重演。

### P2：重演导航

- [ ] 实现带进度约束的最近线段投影匹配。
- [ ] 实现沿线进度、横向偏移、偏航和匹配质量计算。
- [ ] 实现正常、降级、偏离、恢复、完成状态机。
- [ ] 实现 GPS 中断、手机定位降级和平行/交叉路线防跳转逻辑。
- [ ] 提供离线重放和轨迹可视化工具，便于调整匹配参数。

### P3：锚点与交互

- [ ] 实现转弯、起终点、固定间距和手动锚点候选生成。
- [ ] 实现锚点图片质量检查和人工确认流程。
- [ ] 使用 XFeat 实现锚点局部特征匹配和几何验证。
- [ ] 实现锚点进度校正、匹配失败降级和误匹配保护。
- [ ] 实现提前提示、到点确认、偏离告警及语音冷却队列。

## 14. 测试与验收

### 14.1 单元测试

- 经纬度到局部东-北坐标的距离和方向正确。
- `wrap_to_180` 在 0°/360° 边界返回正确偏航。
- 线段投影、左右偏移符号和累计里程计算正确。
- 轨迹窗口能阻止平行道路、交叉点和折返路线发生远距离跳转。
- 平滑算法不移动受保护的转弯点和锚点，不产生自交或切角。
- 乱序、重复、过期和缺失传感器样本能被识别。
- 锚点和语音触发具备滞回、去重和冷却效果。

### 14.2 集成测试

- RTK fixed/float、手机 GPS 降级、GPS 中断和恢复。
- IMU 丢帧、CRC 错误、航向未就绪和串口中断。
- 相机断流、模糊图片、动态分辨率和采集超时。
- 正常结束、强制退出和部分会话数据恢复。
- 直线、急转弯、U 型路线、平行道路、交叉路线和静止漂移。
- 用户左右偏移、短暂偏离、严重偏离和重新回线。
- 锚点在光照变化、遮挡、重复纹理和视觉匹配失败时正确降级。
- TTS 网络失败或音频设备不可用时导航主循环继续运行。
- 所有 Bash 启动脚本均可从项目根目录和其他当前目录调用，参数缺失或设备检查失败时正确返回非零状态。
- 仓库根目录和 `utils/` 中不存在新增的 MemoryNav 业务脚本。

### 14.3 实地验收指标

每条测试路线至少统计：

- 参考轨迹横向误差和航向误差。
- 重演匹配进度的连续性和错误跳段次数。
- 偏离检测距离、检测延迟和恢复成功率。
- 锚点触发成功率、误触发率和视觉确认耗时。
- 语音提示提前量、重复播放次数和漏播次数。
- RTK、手机 GPS、IMU 与相机各自的有效数据比例。

测试路线通过质量阈值后才能标记为 `READY`。阈值应根据第一轮实测数据固化到配置和验收报告中。

### 14.4 实机验收命令与证据

实际完成状态以同目录的 `Todo.md` 为准。本节命令均可从任意目录执行，报告会按场景原子更新，失败返回状态码 2。

```bash
# 三设备同时冒烟；相机端需持续向 9989 端口发送图像
memory_nav/scripts/hardware_acceptance.sh \
  --report memory_nav/hardware_reports/site-01.json \
  smoke --duration 30

# GPS 来源切换日志：每行 JSON 至少包含 source
memory_nav/scripts/hardware_acceptance.sh \
  --report memory_nav/hardware_reports/site-01.json \
  sequence --scenario gps_fallback --input gps-transition.jsonl \
  --field source --expect rtk phone rtk

# 相机断流恢复日志：每行 JSON 至少包含 camera_state
memory_nav/scripts/hardware_acceptance.sh \
  --report memory_nav/hardware_reports/site-01.json \
  sequence --scenario camera_recovery --input camera-transition.jsonl \
  --field camera_state --expect streaming lost streaming

# 在线重演输出：验证正常、偏离、恢复和重新正常
memory_nav/scripts/hardware_acceptance.sh \
  --report memory_nav/hardware_reports/site-01.json \
  sequence --scenario deviation_recovery --input replay-result.jsonl \
  --field navigation_state --expect normal deviated recovering normal
```

命令只根据实际采样或输入日志判定，不允许用人工勾选替代证据。最终验收还应保留原始 JSONL、累计报告和对应路线 manifest。

## 15. 当前约束与默认决策

- MemoryNav 的新增 Python 代码、配置、测试、工具和 Bash 启动脚本全部位于 `memory_nav/`。
- 现有 `utils/` 模块只承担传感器和音频硬件适配；轨迹业务逻辑位于 `memory_nav/`。
- Camera+IMU 采用 Kalibr + AprilGrid 进行时空联合标定。
- 重演以 GPS+IMU 为连续定位主源，锚点图片用于验证和路线进度纠偏。
- 锚点采用“自动候选 + 人工确认”。
- 路线只支持录制方向；反向导航单独录制。
- 用户明显偏离后告警并暂停普通指引，回到路线走廊后自动恢复。
- `gps_state == "good"` 当前代表有效 RTK；手机定位不得按 RTK 精度处理。
- `utils/gps_server.py` 中的 GPS 坐标当前为 GCJ-02，所有后续模块必须保持一致或显式转换。
