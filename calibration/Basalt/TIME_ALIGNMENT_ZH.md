> 新版眼镜内置 Android IMU 请使用 [最简采集步骤](QUICK_COLLECT_ZH.md)。采集和整理默认已切换 Android 同时钟模式，绝不能用 HTTP 到达时间替换采样时间。以下是旧外置 N100 流程的历史说明；其中主机时间对齐命令不适用于新接口。

# Basalt 数据时间对齐：Orin 接收时间方案

按当前选择，默认使用 **Orin 完整收到图像 HTTP 请求体时的 `time.monotonic_ns()`**。记录点位于 JSON、Base64、JPEG 解码和处理队列之前；不是眼镜曝光时间，也不是系统日历时间。眼镜继续使用原来的 `/upload_image` 接口，无需添加时间戳或授时协议。

IMU 保留 N100 原始微秒时间及 Orin 串口接收时间，离线用每秒低延迟样本拟合比例和偏移，将 IMU 时间映射到同一 Orin 单调时钟。保留全部原始 IMU 样本，不插值、不降采样。图像与元数据在同一把锁内读取，防止错配。

接收时间包含曝光、编码和网络传输延迟。此方案可以生成同一时间轴上的标定输入，但不能消除可变上传延迟；Basalt 的时偏优化只能处理其中的常量偏移。报告明确标记 `host_receive_aligned_requires_basalt_time_offset`，不会声称已经硬件同步。

## 采集、对齐、生成 bag

使用 `blind` 环境。先停止占用 9989 或 `/dev/imu` 的其他采集程序，然后在同一个终端执行：

```bash
conda activate blind
cd /home/wheeltec/projects/blind-nav
KIT="$PWD/memory_nav/calibration/Basalt"
RUN="$PWD/calibration_data/basalt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

bash memory_nav/scripts/start_calibration.sh \
  --output "$RUN/raw" --duration 120 --image-fps 30

python -m memory_nav.calibration.align_timestamps \
  "$RUN/raw" "$RUN/aligned" --camera-time-source host-receive --trim-to-overlap

"$KIT/.venv-bag/bin/python" "$KIT/convert_to_bag.py" \
  "$RUN/aligned" "$RUN/aligned.bag" --mode camera-imu
```

这里不需要 `--require-device-timestamps` 或 `--experimental-host-time`。`host-receive` 是默认值，显式写出便于复现。项目中已准备 `.venv-bag`；如重建环境，使用 `python -m venv --system-site-packages "$KIT/.venv-bag"`，再在该环境中安装 `requirements-bag.txt`。

眼镜保持连续上传，固定分辨率。`--image-fps` 是读取上限，不能提高真实上传帧率。板固定，刚性相机–IMU 组件做三轴旋转和平移，首尾静置。不要用 `--width/--height` 切换成一次性拍照。尽量保证网络稳定、队列不积压。

## 文件及检查

原始 `frames.jsonl` 中 `host_receive_ns` 是本次选用的图像时间；`monotonic_ns` 仍保存旧的解码后时间供诊断。`imu.jsonl` 保存 `device_timestamp_us` 和主机接收时间。旧采集数据若缺少 `host_receive_ns` 会拒绝对齐，需重新录制。

输出 `aligned` 包含：

- `frames.jsonl`：原元数据加 `calibrated_ns`；它逐帧精确等于 `host_receive_ns`。
- `imu.jsonl`：原元数据加映射后的 `calibrated_ns`。
- `dataset/cam0/<统一时间ns>.png` 和 `dataset/imu0.csv`。
- `alignment_report.json`：使用的时间来源、IMU 时钟漂移与拟合残差、帧数、裁剪数量及限制。

IMU 拟合需要至少 6 个锚点、至少 20 秒跨度；图像与 IMU 的重叠也需至少 20 秒。默认漂移上限 500 ppm、IMU 锚点残差上限 3 ms。它们是拟合检查门槛，不是相机与 IMU 实际同步精度。接收模式下 `--max-camera-error-ms` 不约束未知网络延迟。

时间重复或倒退、IMU 重启、非有限数值、串口错误计数增长、缓冲溢出、HTTP 队列满及解码错误会拒绝输出。`--trim-to-overlap` 只裁剪 IMU 时间覆盖范围外的图像；报告记录裁剪数量。不覆盖原数据或已有输出目录。bag 中消息时间与 header 时间均使用 `calibrated_ns`，转换器校验对齐报告与记录的关联。

生成 bag 后按 `BASALT_CALIBRATION_ZH.md` 运行内参和相机–IMU 标定，联合标定使用 `aligned.bag`。对至少两段实际序列比较时偏与外参稳定性；尚未运行真实数据的 Basalt 优化，不能以软件测试代替实测精度。

## 可选：未来使用曝光时间

保留了 `camera_sender_sync.py`、`/time_sync` 和 `/time_sync/sample`，但当前流程不使用它们。以后若发送端能提供同一设备时钟域的真实曝光时间和四时间戳授时，可用采集选项 `--require-device-timestamps`、对齐选项 `--camera-time-source device` 切换。该模式要求 `camera_timestamp_ns`、`camera_clock_id`、`source_frame_id`、`timestamp_semantics`；曝光开始时间还需 `exposure_time_ns`。具体请求格式见参考客户端源码。

## 已验证范围

自动测试覆盖接收模式无发送端元数据/无授时文件的导出、设备时钟模式、时钟重启与丢样拒绝、HTTP 接口兼容、图像元数据原子读取，以及 bag 时间戳和消息头精确一致。测试使用合成数据和进程内 HTTP 请求；未启动或重启现场传感器，未测量实际网络延迟。

## 串口序号计数与实际 IMU 丢样

`imu.dropped_frames` 是所有串口包的序号推算计数，不能直接当作 TYPE_IMU 丢样数量。该计数增长时，现改为在报告 `diagnostics` 中保留起止值、增量和警告；计数倒退仍拒绝导出。实际 TYPE_IMU 必须时间严格递增，所有采样间隔位于本次中位间隔的 0.5～1.5 倍内；缺失一个常规样本、时间重置或中途采样率变化会被拒绝。原有 CRC、队列与缓冲溢出检查保留。

此检查假设单次录制采样率固定，不能识别持续均匀抽样或证明设备配置采样率。包序号异常的硬件原因仍未确认，不修改原始诊断或冒充零丢帧。报告 `imu_continuity` 给出本次实际频率及最小/最大间隔。
