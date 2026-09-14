# 周总结 · MemoryNav 离线/在线导航与跟随跟踪

日期：2026-09-10 ~ 2026-09-12
路线：`suishi-2`（参考轨迹约 296 m，61 个重采样点）

## 1. 项目整体实现

MemoryNav 是一条"先录制记忆轨迹、后盲人重演跟随"的导航流水线：

1. **录制**：GPS + IMU 采集，`trajectory/processing.py` 做去噪、重采样（5 m）与平滑，产出 `reference_trajectory.json`（原点 + 局部 EN 坐标 + 朝向 + 曲率）。
2. **锚点提取**：在起点 / 转弯（`turn_angle_deg=35°`）/ 终点处自动打锚点（`suishi-2/anchors.json`，共 8 个），可选图像锚点用于视觉确认。
3. **重演跟随**：`replay/replay_nav.py` 在线把 GPS/IMU 采样匹配回参考轨迹，按"钟表方向"播报可执行动作（`12点直行 / 9点钟方向前进 / 掉头`）。
4. **离线分析**：`analysis/` 对录制数据做锚点切段、1 Hz 时钟指令、跟随能力量化（横向/朝向误差）与可视化。

本目录 `test0910/` 为本周新增：用 `analysis/visualize_follow.py` 把 6 条 `follow_output/*/output.jsonl` 各自匹配回 `source/output_trimmed.jsonl`，输出逐条跟踪可视化（`route_overlay.png`、`follow_metrics.png`、`follow_comparison.html`、`match_metrics.jsonl`、`summary.json`）及总览 `overview.png` / `index.html`。

## 2. 视觉里程计（VINS）接入

- **订阅**：`memory_nav/vio.py::VioSubscriber` 后台 ROS 线程订阅 VINS `/odometry`（`nav_msgs/Odometry`，BEST_EFFORT），可选订阅 VINS 图像 topic 供视觉锚点用；延迟导入 `rclpy`，保证离线工具无 ROS 也能跑。
- **对齐**：`OnlineVioAligner` 累积"VIO 采样 ↔ 同时刻 GPS(EN)"配对（最少 3 对、1.5 s 内、RMS ≤ 5 m），用 `estimate_vio_transform`（SVD 最小二乘，**只估旋转+平移、不估尺度**）求 VINS 世界系 → 局部 EN 的刚体变换。
- **融合**：`ReplayRunner.step()` 中若 VIO 已对齐且新鲜（`state == "aligned"`），用 `transform.apply(x,y)` 把 VIO 位置投影到 EN 替代 GPS，`position_source` 记为 `"vio"`，否则回落 `"gps"`。
- **状态机**：`warming_up`（配对不足）→ `aligned`（可用）→ `degraded`（RMS 超标）→ `lost`（采样超时）。

## 3. GPS + IMU 匹配跟踪的贡献

- **硬件网关适配**：`HardwareGatewayGPS/IMU/Camera` 把共享 gateway 的 `get_gps()/get_heading()/capture_frame()` 包装成统一接口，与 VINS bridge 并行读 `/imu0` 而不抢串口。
- **匹配**：`replay/matcher.py::RouteMatcher` 把当前位置投影到最近参考段，输出 `matched_s_m / cross_track_error_m / heading_error_deg`。
- **引导**：`replay/guidance.py::GuidanceController` 按"距离/到达"去重生成锚点预告、到达、偏离告警等播报；`replay_nav.py::make_command/make_gps_action` 把"目标朝向−当前朝向"量化成 30° 一格的钟表指令。
- **可视化锚点确认**：`replay/anchor_matcher.py` 用本地 XFeat + MAGSAC RANSAC 做锚点图重识别，输出 inlier 比例做到达校验。
- **可复现日志**：`FollowGPSLogger` 按 L4 `output.jsonl` 字段落盘跟随样本，支撑离线分析。

## 4. 当前发现的问题

### 4.1 指令方向：参考朝向 vs 当前位置→前瞻点方位角

`replay_nav.py:243/249` 中 `target_heading` 取的是**参考轨迹在 `matched_s+8m` 处的朝向**，`make_command(heading, target_heading)` 据此给指令；而 `target_east/target_north`（前瞻点坐标）算出来仅用于写日志。离线 `analysis/core.py:744` 用的是 `bearing_to(tx,ty,heading)`（当前位置→前瞻点的方位角）。

后果：用户横向偏离时，系统只让用户"面朝参考朝向直行"，与路线平行而永不收敛（僵硬跟随）。实例：`follow-162722` 起点横向偏 25 m，`cur_bearing=105°`、`target_bearing=28°`，播报"9点钟方向前进"；而当前位置→前瞻点的真实方位角约 **280°**，正确动作应是掉头向西回到路线。

### 4.2 固定 8 m 前瞻点，转角处指向建筑后

`replay_nav.py:237` 的 `+8.0 m` 硬编码、与曲率无关。转角锚点在 `s≈75/105/110/115/265/280`，绕楼直角转弯时 `matched_s+8m` 的前瞻点已转到楼另一侧，直线指向前瞻点/参考朝向会穿楼（"直接指向建筑物后目标点"）。

### 4.3 匹配"直接指向最近点"，无进度约束

`matcher.py:66-69` `_candidate_indices()` 返回 `range(len(points)-1)`，`forward_window_m/backward_window_m/initial_search_m` 全部是死参数；`match()` 永远取**全局最近段**。拐角或偏离时 `matched_s` 会直接跳到最近点（可能退到楼后/上一圈），进度不单调，导致后续前瞻点与指令错误。

### 4.5 其他

- VIO 对齐只估刚体（无尺度），长距离/漂移下需定期用 GPS 重对齐，目前 `OnlineVioAligner` 一旦 aligned 不再刷新。
- 视觉相似度对相机视角/朝向敏感，全局 DINOv2 特征在同地点但不同朝向时相似度偏低。

## 5. 下一步

1. 把 `replay_nav.py` 的方向指令改为 `heading_from_delta(target_east - east, target_north - north)`（对齐 `core.py`）。
2. 前瞻距离随曲率收缩：转角锚点附近缩小前瞻，或以"下一个锚点"作为中间目标。
3. 恢复 `RouteMatcher` 的进度窗口匹配，并按横向/朝向误差给出真实 `match_quality`，接入 `DeviationMonitor`。
