# 眼镜内置 Android IMU：最简采集步骤

适用于新版 utils/glasses_camera.py 的 HTTP 图像＋整批 IMU 接口；不再启动外置 N100 的 utils/imu.py。

1. 眼镜 USB 接到 Jetson；停止占用 9989 的 server.py、glasses_camera.py 或其他相机程序。
2. Jetson 终端执行：

```bash
conda activate blind
cd /home/wheeltec/projects/blind-nav
adb devices
adb reverse tcp:9989 tcp:9989
adb reverse --list
bash memory_nav/calibration/Basalt/collect_ready_bag.sh 120
```

3. 立即在眼镜 Self CMD 长按硬件键开始上传。A4 标定板保持固定，相机和新 IMU 刚性固定在一起；让标定板在画面中可见，缓慢改变距离和朝向，并绕三个方向转动，避免模糊。120 秒结束后自动整理并生成 bag，再停止眼镜上传。
4. 只有终端显示“采集、对齐、转换成功”时，使用输出目录的 aligned.bag 进行后续 Basalt 标定，继续选用实测黑色标签边长 24 mm 对应的 A4 JSON。

输出目录：`/home/wheeltec/projects/blind-nav/calibration_data/basalt_日期_随机后缀/`

- `raw/`：所有收到的 BGR 图像保存为 PNG、原始逐条 IMU、图像元数据和 collection_report.json。
- `aligned/`：同一 Android 时钟下整理后的数据及 alignment_report.json。
- `aligned.bag`：ROS1，图像 `/cam0/image_raw`，IMU `/imu0`，可用于后续 Basalt 标定。
- `collect.log`、`align.log`、`bag.log`：排查日志。

最新图像仍实时覆盖 `/home/wheeltec/projects/blind-nav/calibration_data/tmp.jpg`。

USB 拔插、眼镜或 ADB 重启后重新执行 adb reverse。采集中 App 重启会使 session_id 改变，本次采集将失败，请重新采集。无需另外运行 HTTP 服务。

## 时间处理

相机使用 frame_timestamp_ns，陀螺仪使用 timestamp_ns，均保持原始整数纳秒。received_at_unix_ns 仅保存作诊断；不用于对齐，不计算主机时钟偏移。两者为同一 Android 时钟是发送端提供的协议保证，接收程序无法独立证明硬件时间戳精度。

加速度的真实时刻是 accel_timestamp_ns，不能直接当作陀螺仪时刻。原始数据完整保留；整理时去除相同的重复样本，按真实加速度时间线性插值到陀螺仪时刻。只裁剪无法插值的边界和超出 IMU 覆盖区间的图像，不外推，不补造丢失的陀螺仪样本。裁剪数量写入 alignment_report.json。

程序拒绝明显 IMU 缺口（超过 50 ms 或该传感器中位采样间隔的 5 倍，取较小值）、会话混合、缺失时间戳、冲突重复值、已检测的接收/记录丢包。检测不能证明每个发送端样本都已上传；App 仍须保证整批上传。需要至少 20 秒有效图像/IMU 重叠，建议采集 120 秒。

更换 IMU 后必须重新标定相机—IMU 外参及 IMU 参数；不要沿用原外置 N100 的结果。第一行曝光时间被原样保留，滚动快门及剩余传感器时延仍需结合实际标定残差评估。

旧 N100 数据的历史时间整理方式仍可显式选择 --camera-time-source host-receive 或 device；不要用于本次 Android 数据。
