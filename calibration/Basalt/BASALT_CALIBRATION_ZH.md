# blind-nav：Basalt 相机内参与相机–IMU 外参标定流程

> **时间对齐已实现，默认采用 Orin 收到图像请求体时的时间。** 请先按同目录 `TIME_ALIGNMENT_ZH.md` 采集、对齐并生成 bag，替代下文第 3、6 节的早期手工准备流程。无需修改眼镜发送端；网络可变延迟仍保留，不能视为曝光同步。

编写日期：2026-09-19。对象：`smil_orin_16g_540`（`wheeltec@192.168.3.150`），项目 `/home/wheeltec/projects/blind-nav`。已通过 SSH 只读检查代码；远程系统为 Ubuntu 22.04.5 / aarch64。用户已确认：眼镜相机与 N100 IMU 刚性固定，运动中相对位姿不变。

当前已更新远程相机、IMU 和采集代码，提供默认接收时间对齐与 bag 导出。尚未启动传感器或运行 Basalt 优化，没有产生实测标定参数。

## 1. 文件与打印

- 首选 `aprilgrid_6x6_A3_35mm.pdf` 与同名 JSON：A3 竖向，标签黑色外边长 **35 mm**，相邻标签间距 **10.5 mm**，6×6 主标签区域 **262.5×262.5 mm**。
- 备选 `aprilgrid_6x6_A4_24mm.pdf` 与同名 JSON：A4 竖向，标签边长 **24 mm**，间距 **7.2 mm**，主标签区域 **180×180 mm**。适合较近距离采集。
- 两者均为 **tag36h11、双层黑边（black border=2）、ID 0–35**。ID 0 在左下，沿行向右递增，下一行向上；ID 35 在右上。角落的小黑方块用于角点对称，不是额外标签，也不计入标签边长。排列与 Kalibr 生成器 `rotation=2` 一致。
- `convert_to_bag.py`：将现有采集器的 JSONL 与 PNG 转为 Basalt 可读取的单目 ROS1 bag。

Basalt 的 AprilTag 检测器明确使用 `tagCodes36h11` 和 `blackTagBorder(2)`。普通单层黑边的 AprilTag 图不可直接代替本板。[检测器源码](https://github.com/VladyslavUsenko/basalt/blob/master/thirdparty/apriltag/src/apriltag.cpp)

打印步骤：

1. 用 PDF 阅读器选择相应纸张、竖向、**实际大小 / 100%**，关闭“适合页面”、缩放、镜像、多页合一。不要从截图打印。
2. 采用清晰黑白打印，建议 600 dpi 或更高。贴在平整、坚硬、不反光的基板上，避免皱褶、弯曲、反光覆膜；胶带不要覆盖图案。
3. 用尺或卡尺检查页底 100 mm 标尺，并在多个位置量标签外黑边、间距和横纵主区域总长。A3 标签应为 35 mm，A4 应为 24 mm。
4. 工程建议：横纵比例误差控制在约 0.2% 内；若出现不同方向缩放、局部翘曲或裁切，重新打印。若只是均匀缩放，可按实测值修改 `tagSize`（单位 m）和 `tagSpacing=实测间距/实测边长`，并保存新的配置和测量记录。外参平移的尺度依赖这些物理尺寸。
5. 两种板只选一种参与同一组数据。不要把 A3 缩放成 A4 后仍使用 A3 JSON。不要把 Basalt 仓库自带的 88 mm 配置用于本板。

A3 JSON 内容如下；A4 仅 `tagSize` 为 `0.024`：

```json
{"tagCols":6,"tagRows":6,"tagSize":0.035,"tagSpacing":0.3}
```

`tagSize` 指标签整个外黑边长，不是中心 6×6 编码区；`tagSpacing` 是无量纲比值，不是米。Basalt 的角点几何由这四项构造。[AprilGrid 实现](https://github.com/VladyslavUsenko/basalt/blob/master/src/calibration/aprilgrid.cpp)

## 2. 已核对的设备接口及限制

### IMU：`utils/imu.py`

- 读取 WHEELTEC N100 FDILink，默认 `/dev/imu`、921600 baud。串口波特率不是 IMU 采样率；实际频率必须测量。
- 标定应使用 TYPE_IMU (`0x40`) 的原始陀螺仪和加速度，字段为 `gyro_rad_s`、`accel_m_s2`。不要使用航向角、AHRS 四元数或手机方向代替。
- `drain_imu_samples()` 可取出完整样本队列；每条含 `device_time_s` 与 `monotonic_ns`。前者来自设备微秒时间戳换算，后者在主机解析时生成。
- `get_imu()` 只有最近值；低频轮询会漏样。现有采集器优先使用 `drain_imu_samples()`，这一点可复用。
- `heading_offset_deg=-20` 只影响航向，不作用于原始 gyro/accel；不可把它当作相机–IMU 外参，也不应再把该偏角叠加到原始标定数据上。
- 构造器会等待 AHRS 航向稳定，不成功会持续重试。开始采集前先静置；初始化卡住时检查 AHRS 数据和串口，而不是假定已在记录。
- 正式采集前用六面静置与各轴转动核对单位和符号。静止加速度模应接近 9.81 m/s²，陀螺仪接近零；加速度应保留重力对应的比力，不要减重力或转到世界坐标。代码字段名不能替代实测核验。

### 相机：`utils/glasses_camera.py`

- 这是 HTTP 图像接收器，FastAPI 监听 `0.0.0.0:9989`，眼镜需向 `/upload_image` 上传图像；不是本地 USB 相机直接采集。
- 默认参数写着 640×480、30 fps，但服务端不能保证眼镜实际以该分辨率/帧率输出。当前代码保留上传图像实际尺寸。
- 接收队列长度 10，满时返回 `queue_full`；图像在出队、解码、可选模糊过滤之后才写入 `frame_monotonic_ns`。
- 上传协议没有曝光时间、源帧序号、曝光时长字段。`ImageRequest.imu` 虽存在，当前图像队列并未传递它，不能认定它提供了同步。
- `capture_frame_with_metadata()` 的普通路径先取图，再单独取元数据，线程间存在图像与时间戳错配窗口；`frame_count` 与图像也没有统一原子快照。正式采集接口需要一次性返回同一帧的全部数据。
- `save_pic=True` 保存的相对时间以第一帧为零，没有可直接与 IMU 对齐的统一时基；不要直接拿该时间当同步时间戳。

### 已有采集器

`memory_nav/calibration/collect_calibration.py` 和启动脚本 `memory_nav/scripts/start_calibration.sh` 已存在。输出包括 `frames.jsonl`、`imu.jsonl`、`dataset/cam0/<ns>.png`、`dataset/imu0.csv`。IMU JSONL 保存设备时间，CSV 使用主机解析时间。

目前它适合内参素材采集与流程试跑。仍需注意相机元数据竞态、采集开始前的 IMU 缓存、实际帧率和队列溢出；它不会自动实现时钟同步。长时间噪声记录也不宜直接复用这个逐帧落盘方案。

## 3. 正式外参标定前的数据条件

1. **固定成像模式**：使用与导航一致的分辨率、镜头、裁切和缩放；能关闭时关闭电子防抖、动态裁切、自动变焦、美颜等改变几何的处理，并锁定焦距/对焦。曝光尽量短、光照充分。不能控制的处理应记录并验证其稳定性。
2. **相机曝光时间**：在眼镜端取得传感器帧时间，说明它代表曝光开始、中点还是结束；能取得曝光时长时统一到曝光中点。同时记录源帧序号、曝光、分辨率和接收时间。
3. **统一时钟**：优先硬件同步；不同设备时钟需建立经过验证的映射 `t_common = a*t_device + b`，校正偏移与漂移。IMU 保留原始整数微秒更好，避免不必要的浮点时间转换。相机和 IMU 不能各自减首帧后假定同步。
4. **保留时间证据**：所有转换使用整数纳秒，并保留原设备时间和接收时间。建议另存 `calibrated_ns` 表示已经验证的统一采样时刻；本交付转换器认识这个字段，但不会替你估计它。
5. **逐帧一致性**：相机图像、源帧号和曝光时间组成同一条队列记录；在同一把锁内复制/取出，避免分别读取“最新帧”和“最新时间”。记录丢帧和队列溢出。
6. **IMU 完整性**：完整保存 `drain_imu_samples()`，记录诊断计数；开始会话前清理启动缓存，结束时保留覆盖最后一张图像的 IMU 样本。避免多进程同时读取串口或抢占相机服务端口。

当前 HTTP 到达/解码时间包含编码、网络、排队和解码时延，且随帧变化。Basalt 的相机时间偏移是一个常量参数，不能消除这种抖动。若眼镜端拿不到采样时间，建议先完成内参，或改用可提供时间信息的采集链路；使用主机时间得到的外参只能作为实验结果。

若相机是 rolling shutter，快速转动会产生逐行曝光误差。本流程不提供行读出时间标定；先核实传感器及读出方式，缩短曝光、避免运动过快，并检查运动相关残差。短曝光并不会消除整帧逐行读出时差。

## 4. Basalt 运行环境

建议 Orin 负责采集，带桌面和 OpenGL 的 Ubuntu 工作站负责离线标定。当前上游 Linux 发布包面向 amd64，不能把它直接装到 aarch64 Orin 上。Orin 若自行构建需验证 ARM 依赖，16 GB 内存建议从 `-j2` 开始；本次未进行编译验证。[官方安装说明](https://github.com/VladyslavUsenko/basalt#installation)

下面是当前上游 CMake preset 的源码构建路径，需先有 CMake ≥3.24、Ninja、C/C++ 编译器及 vcpkg 所需系统依赖。Ubuntu 22.04 系统默认 CMake 可能低于要求。若已有可用 Basalt，可跳过构建：

```bash
git clone --recursive https://github.com/VladyslavUsenko/basalt.git
cd basalt
git rev-parse HEAD > basalt-commit.txt
git submodule update --init --recursive
cmake --version
cmake --preset relwithdebinfo
cmake --build --preset relwithdebinfo -j2
ctest --preset relwithdebinfo
find build/relwithdebinfo -type f -name 'basalt_calibrate*'
```

通过最后一行定位两个可执行文件，将所在目录加入 PATH 或使用完整路径；不要假设所有版本都在 `build/` 顶层。先检查：

```bash
basalt_calibrate --help
basalt_calibrate_imu --help
```

本流程按已核对的上游源码使用 `--dataset-type bag`、`--aprilgrid`、`--result-path`、`--cam-types` 和 `--cache-name`。安装时保存实际 commit、命令帮助和构建日志，避免不同版本混用。GUI 需要图形会话；普通无显示 SSH 不足以运行 GUI。首轮不使用 `--no-gui`：当前相机程序会额外运行 vignette，IMU 程序会自动开启时偏、IMU scale 和 mocap 阶段，不适合作为本设备的无脑替代流程。[相机入口](https://github.com/VladyslavUsenko/basalt/blob/master/src/calibrate.cpp)、[IMU 入口](https://github.com/VladyslavUsenko/basalt/blob/master/src/calibrate_imu.cpp)

## 5. 采集动作与命令

固定标定板，让相机–IMU 整体运动。保持机械安装与使用时一致，预热并等待输出稳定。不要只动板、不动 IMU 来录外参数据。

### 5.1 内参序列 cam01

建议 2–3 分钟，保留约 100–300 张清晰且姿态多样的有效图像。目标是覆盖图像中央、四角和边缘，包含近/中/远距离以及不同俯仰、偏航、滚转。可在各姿态短暂停留，利于当前到达时间采集链路。不要只正对板，也不要长期只占据中心小区域。

距离以图像为准：让标定板占据不同面积；每个标签至少有足够清晰的像素，工程起点可取边长约 25–30 像素以上。边缘采集允许部分板出画，但画内应保留多个完整标签；优先保持至少 6 个清晰完整标签。Basalt 当前检测器的有效观测门槛为 4 个标签。

在 Orin 上，以项目原有可运行 `cv2/serial/fastapi` 的 Python 环境执行；需要让眼镜端同时正常上传：

```bash
cd /home/wheeltec/projects/blind-nav
SESSION_ROOT="$PWD/calibration_data/basalt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_ROOT"
bash memory_nav/scripts/start_calibration.sh \
  --output "$SESSION_ROOT/cam01" --duration 150 --image-fps 5
```

`--image-fps` 是采集器读取上限，不是眼镜实际帧率。初次先做 10 秒试录，确认尺寸、清晰度及时间单调后再录完整序列。内参阶段同时采到的 IMU 不会用于内参求解。

**不要在连续外参采集时传 `--width/--height`**：现有实现会反复请求一次性指定分辨率图像并等待返回，改变采集节奏。先在眼镜发送端固定连续流模式，再录制。若需要其他分辨率的内参，应单独配置、单独采集，不混合分辨率。

### 5.2 动态序列 imu01 / imu02

完成第 3 节改造后，使用保留相同命令行的改进采集器，或等价的逐帧采集工具，录制两段独立序列。当前原始采集器按下面命令运行仅能试跑链路：

```bash
bash memory_nav/scripts/start_calibration.sh \
  --output "$SESSION_ROOT/imu01" --duration 120 --image-fps 30
```

动作建议：

1. 起始静止 5–10 秒。
2. 保持板可见，分别绕三轴平稳往复转动，幅度约 ±20°–45°，不要只绕一个轴。
3. 加入左右、上下、前后的往复平移与加减速，避免全程纯旋转或恒速平移。运动应足以激励参数，同时让标签边界保持清晰。
4. 做一段组合运动，最后再静止 5–10 秒。
5. 换不同动作节奏录制 imu02，用于独立重复性检查。以上幅度和时长为工程起点，不是 Basalt 的硬性阈值。

图像实际采样建议约 20–30 Hz 或以上、IMU 约 100–200 Hz 或以上，并以硬件实际能力核验。若只有低帧率网络上传，不要用插帧、复制帧或伪造均匀时间戳补足频率；应改善采集链路、调整运动速度并验证可观测性。

### 5.3 录制后先检查

- 图像尺寸唯一且匹配使用模式；画面无镜像、动态变焦，标签边缘清楚。
- 相机和 IMU 时间分别严格递增，无重复、重启归零和异常跳跃。检查平均频率、间隔分位数和最大间隔，不能只看总数量。
- 查看 `get_diagnostics()` 的 CRC 错误、丢帧、`imu_sample_buffer_dropped` 是否增长；记录 HTTP `queue_full` 次数和真实源帧序号缺口。当前采集器并未自动保存这些诊断，正式采集应补上。
- IMU 覆盖所有图像时间。首尾不足时显式裁掉无 IMU 覆盖的图像，并保留原始数据；不要凭空补样。
- 静止段单位正确、运动段三轴有激励且不饱和；同步质量应有原始时间证据，不能以“同一台主机的 monotonic”作为充分条件。

## 6. 转换为单目 ROS1 bag

核对到的 EuRoC 读取器将相机数量写死为 2。这里采用 bag 自动识别单个 `sensor_msgs/Image` 话题，避免复制图像伪造第二路相机。不要把现有 `dataset/` 直接传给 `--dataset-type euroc`。[EuRoC 读取器](https://github.com/VladyslavUsenko/basalt/blob/master/include/basalt/io/dataset_io_euroc.h)、[bag 读取器](https://github.com/VladyslavUsenko/basalt/blob/master/include/basalt/io/dataset_io_rosbag.h)

可在离线工作站新建独立 Python 环境，不需要 ROS master 或完整 ROS 安装：

```bash
python3 -m venv .venv-bag
source .venv-bag/bin/activate
python -m pip install rosbags==0.11.5 numpy opencv-python-headless
```

将交付文件放入一个目录，在该目录执行；下面 `DATA` 改为本机收到的采集根目录：

```bash
DATA=/absolute/path/to/calibration_data/session
mkdir -p "$DATA/bags"
python convert_to_bag.py "$DATA/cam01" "$DATA/bags/cam01.bag" \
  --mode intrinsics
```

已完成曝光时间对齐时，给 `frames.jsonl` 和 `imu.jsonl` 的每一条记录增加整数纳秒字段 `calibrated_ns`，表示同一时钟的实际采样时刻；保留其他字段。然后：

```bash
python convert_to_bag.py "$DATA/imu01" "$DATA/bags/imu01.bag" \
  --mode camera-imu --timestamp-key calibrated_ns
```

这一步**不会产生或验证时钟映射**，只使用你提供的时间字段。仅测试 Basalt 界面和数据通路时，可显式用以下实验命令，输出必须标记为实验，不作为正式外参：

```bash
python convert_to_bag.py "$DATA/imu01" "$DATA/bags/imu01_host_experiment.bag" \
  --mode camera-imu --experimental-host-time
```

转换器输出 `/cam0/image_raw`（`mono8`）和可选 `/imu0`（`sensor_msgs/Imu`）。`header.stamp` 与 bag 记录时间一致；gyro 为 rad/s、accel 为 m/s²，orientation 标记为不可用。它拒绝混合分辨率、重复源帧号、非递增时间、缺图、非有限 IMU 值、IMU 覆盖不足及覆盖已有 bag。它不能检测所有图像–时间戳错配，也不估计曝光、漂移、丢帧或单位比例。

## 7. 相机内参标定

在有图形桌面的标定机上，设置绝对路径。这里以 A3 板为例，A4 必须切换成对应 JSON：

```bash
KIT=/absolute/path/to/this/package
DATA=/absolute/path/to/calibration_data/session
RESULT="$DATA/results_ds_01"
GRID="$KIT/aprilgrid_6x6_A3_35mm.json"
mkdir -p "$RESULT"
basalt_calibrate \
  --dataset-path "$DATA/bags/cam01.bag" --dataset-type bag \
  --aprilgrid "$GRID" --result-path "$RESULT/" \
  --cam-types ds --cache-name cam01
```

此处 `ds`（Double Sphere）是待验证的起点，不表示已确认眼镜镜头模型。当前入口支持 `ds / kb4 / eucm / pinhole`。单路图像只填一个模型；如需比较 `kb4`，在独立结果目录重做并比较独立数据上的误差、边缘表现。`pinhole` 不是 OpenCV 的 `pinhole+radtan`，不能默认它包含普通镜头的径向畸变。

GUI 顺序：`load_dataset` → `detect_corners`（等后台完成）→ `init_cam_intr` → `init_cam_poses` → `init_cam_extr` → `init_opt` → `optimize` 至收敛 → `save_calib`。可勾选 `opt_until_converge`。检查标签 ID、检测角点与重投影是否重合，关注边缘区域。单目 `init_cam_extr` 仅建立参考坐标，不是求出相机–IMU 外参。

保存后文件为 `$RESULT/calibration.json`，先备份：

```bash
cp "$RESULT/calibration.json" "$RESULT/calibration_intrinsics.json"
```

首轮不做 `compute_vign`。不要把暂时保存的单目单位外参矩阵误当作真实 IMU 外参。更换序列时改变 `--cache-name`；更换标定板尺寸、分辨率或相机模型时使用新的结果目录，避免旧缓存被当成新结果。[官方标定流程](https://github.com/VladyslavUsenko/basalt/blob/master/doc/Calibration.md)

## 8. 相机–IMU 联合标定

### 8.1 先确定 IMU 噪声参数

Basalt 的四个参数是连续时间量：gyro 白噪声 `rad/s/√Hz`、accel 白噪声 `m/s²/√Hz`、gyro bias random walk `rad/s²/√Hz`、accel bias random walk `m/s³/√Hz`。应来自 N100 对应设置的可靠规格或静止数据 Allan 分析。bias instability 不等同于 bias random walk，不能直接填入。

若需要实测，独立长时间记录预热后的静止 IMU，保持实际带宽、量程和采样频率，保存温度和原始时间；白噪声与慢漂移拟合需要不同时间尺度。几分钟静止片段可做单位和短时噪声检查，不能替代慢漂移估计。本次未测得 N100 噪声参数，不提供伪装成实测值的默认数字。[Kalibr 噪声定义及估计](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)

### 8.2 加载相机结果并优化

沿用上一步 `RESULT`，让 Basalt 从同一目录加载相机 `calibration.json`。先在当前终端将以下四个变量设为已核实的数值；检查语句会阻止漏填时直接套用程序默认值：

```bash
: "${GYRO_NOISE:?请设置已核实的 gyro noise density}"
: "${ACCEL_NOISE:?请设置已核实的 accel noise density}"
: "${GYRO_BIAS:?请设置已核实的 gyro bias random walk}"
: "${ACCEL_BIAS:?请设置已核实的 accel bias random walk}"

basalt_calibrate_imu \
  --dataset-path "$DATA/bags/imu01.bag" --dataset-type bag \
  --aprilgrid "$GRID" --result-path "$RESULT/" --cache-name imu01 \
  --gyro-noise-std "$GYRO_NOISE" --accel-noise-std "$ACCEL_NOISE" \
  --gyro-bias-std "$GYRO_BIAS" --accel-bias-std "$ACCEL_BIAS"
```

GUI 依次加载数据、检测角点、初始化相机位姿，点击 `init_cam_imu`、`init_opt`，再优化至收敛。初始保持内参不变（`opt_intr` 关闭）、以角点残差为主（`opt_corners` 开启），不启用 mocap。稳定后才打开 `opt_cam_time_offset` 微调常量时间偏移。只有数据激励足够、单位与模型可靠时，才考虑进一步开启 `opt_imu_scale`。每次增加自由度后重新检查误差和结果稳定性，不以误差下降作为唯一成功标准。

点击 `save_calib` 会写回同目录 `calibration.json`。另存 `calibration_cam_imu_imu01.json`，并保留内参备份。独立的 imu02 应从相同内参初值开始，用新目录、不同缓存名重新标定，比较两组外参。不要让一个未收敛结果成为下一次的唯一初值。

## 9. 结果含义与验收

### 坐标方向

Basalt `T_i_c[0]` 把相机坐标转换到 IMU 坐标：

```text
p_i = R_i_c * p_c + t_i_c
T_c_i = inverse(T_i_c)
      = [ R_i_c^T, -R_i_c^T*t_i_c ; 0 0 0 1 ]
```

平移单位为 m；它表示相机原点在 IMU 系中的位置。若下游要的是 Kalibr 风格 `T_cam_imu`，通常需要逆矩阵，并再次核对该下游的定义。按 JSON 的字段名读取四元数，注意库的 `wxyz/xyzw` 差异。相机内参结果中的单位 `T_i_c` 不能替代最终联合标定结果。[Basalt 投影中的变换用法](https://github.com/VladyslavUsenko/basalt/blob/master/src/calibration/cam_calib.cpp)

`cam_time_offset_ns` 单位是 ns。不同软件的时偏正负约定可能相反；下游接入前，应在实际安装版本中查看它加到哪个测量时刻，并用已知转动检查，不能照搬其他工具的符号。不要一边修改原始时间戳、一边又重复应用同一偏移。

Basalt 的 DS 参数不是 OpenCV 的 K+D 格式；保存相机模型名称、全部参数和分辨率，并使用对应投影/反投影实现。不要只复制 `fx/fy/cx/cy` 丢弃 DS 的 `xi/alpha`。项目现有 `import_calibration.py` 面向原流程，不应假定可直接导入 Basalt JSON。

### 建议的工程验收

- 内参：有效角点覆盖全画幅和多种倾斜；查看重投影残差分布、边缘误差和异常帧，而非只看总优化目标。清晰数据可先把重投影误差约 0.5 px 量级作为排查目标，明显超过 1 px 时检查板面、图像处理、模型与模糊。这不是 Basalt 保证或统一通过线。
- 外参：IMU 原始曲线与 spline 预测在动态段一致，静止段和动态段没有系统性错位；三轴残差无明显周期结构，常量时偏在不同序列中稳定。
- 几何合理：`R^T R≈I`、`det(R)≈1`，相机到 IMU 距离与安装尺寸一致。若标定出米级距离而真实只有厘米级，先检查板尺寸、时间与运动激励。
- 重复性：同一安装分别标定 imu01/imu02，比较旋转差、平移差和时偏。可用约 1°、1 cm 作为初步调查界线，再按导航误差预算收紧；短基线设备通常需要更严格。结果随动作明显变化时不能直接上线。
- 留出第三段独立数据做固定参数验证，检查重投影与惯性一致性，并在导航中验证实际效果。调优用过的数据不再算独立验证。

归档：原始图像/IMU、原始及统一时间、时钟映射方法、噪声参数来源、实测板尺寸、对应 JSON、Basalt commit、运行命令、日志、所有初值与最终结果、安装照片、传感器设置、验证记录。拆装、镜头模式变化或支架变形后重新评估。

## 10. 故障定位与本交付验证范围

- **检测不到板**：检查 tag36h11、双层黑边、无镜像、实际像素尺寸、曝光/反光；先看原始图像是否能检测，不要先调优化权重。
- **内参初始化失败**：增加完整板、倾斜和边缘观测，确认模型与唯一分辨率，等待角点检测结束。
- **相机–IMU 初始化失败**：检查三轴转动、原始 gyro 单位、时间重叠及轴方向；没有足够运动不能靠反复点击修复。
- **旋转看似合理、平移不稳**：检查平移加速度激励、板尺度、刚性与时间同步。
- **时偏每次不同**：优先排查网络排队、时钟漂移和元数据错配，而不是盲目放宽阈值。
- **相机采集没有图**：确认眼镜向 9989 上传，检查端口是否被另一个 Camera 实例占用；仅启动 Python 接收器不会让眼镜自动发送。
- **启动卡住**：IMU 构造器等待 AHRS 航向稳定；先静置并检查串口。

本次已完成：两张矢量 PDF 的纸张/版面检查；36 个标签码、方向与 Kalibr 生成方式比对；PDF 渲染后全部 36 个 ID 的 OpenCV 解码；转换器合成数据 ROS1 写入/回读、时间戳与单位保留检查，以及重复时间和不合适主机时间的拒绝检查。

本次未完成：实物打印测量、真实镜头成像检测、Basalt 二进制运行、设备实时帧率/噪声/同步测量、实际内外参求解。这些应在取得数据后按上述步骤执行，不代表本交付已经得到可用于导航的参数。

代码检查快照 SHA-256（用于核对后续代码是否改变）：

```text
utils/imu.py
0eff76fa5c97f1ad9ddd02020e6ad38a3a5cc6a5f8c061afb6af57c76427179e
utils/glasses_camera.py
8a8d99081d51a9e6d7324f9e274f93a64a5ac4c51839b67091b61bdcdd74a1a1
memory_nav/calibration/collect_calibration.py
83113ac08204c1143e616f35d020664b149ad6e64ab75f7dccb31352295337f1
```

标定板生成依据：[Kalibr 官方生成器](https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_create_target_pdf)。交付图案为矢量单元格绘制，标签码与该生成器及 OpenCV AprilTag 字典交叉校验。上游 master 可变，执行时应以你保存的安装版本和帮助输出为准。
