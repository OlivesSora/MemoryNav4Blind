# MemoryNav 离线轨迹锚点分析（记忆轨迹 vs 测试轨迹）

本工具对**已录制好的离线数据**做四件事：

1. **自适应切段 / 锚点提取**：沿轨迹按“转弯 / 视觉突变 / 距离过长”三种条件自动设置锚点，并保存每个锚点的**原始图像、语义文本（复用 `output.jsonl` 的 guide / route_instruction / gps_action）和视觉特征（DINOv2 全局嵌入）**。
2. **盲人可执行的时钟指令**：对每个 1Hz 关键点输出类似 `9点钟方向前进约X米` 的动作指令。
3. **跟随能力量化**：把测试轨迹逐点匹配回记忆（参考）轨迹，输出横向误差、朝向误差、视觉相似度等匹配指标。
4. **可视化**：生成两个轨迹各自的离线 HTML 页面（原图 + 语义 + 特征 + 指令），以及一张两者叠加 + 跟随指标的页面。

代码位于 `memory_nav/analysis/`，全部离线运行，依赖：

- `torch`（CPU 即可）、`torchvision`、`cv2`、`numpy`、`PIL`、`matplotlib`
- 缓存的 **DINOv2 ViT-B/14** 权重（`~/.cache/torch/hub/facebookresearch_dinov2_main` + `dinov2_vitb14_pretrain.pth`）
- 复用 `memory_nav` 的 `LocalFrame`、`RouteMatcher` 等模块

## 1. 使用方法

```bash
python3 -m memory_nav.analysis.run_analysis \
  --memory-gps /home/wheeltec/projects/L4/reFineWithMacroAndMicroLocal/outdoor_nav_output/outdoor_nav-20260821_153545-zixu-yuanji/output.jsonl \
  --test-gps   /home/wheeltec/projects/L4/reFineWithMacroAndMicroLocal/outdoor_nav_output/outdoor_nav-20260821_161908-liupu-yuanji/output.jsonl \
  --memory-frames /home/wheeltec/projects/blind-nav/video_output/20260821_153336_frames-zixu-yuanji \
  --test-frames   /home/wheeltec/projects/blind-nav/video_output/20260821_161505_frames-liupu-yuanji \
  --fps 15 \
  --map /home/wheeltec/projects/L4/reFineWithMacroAndMicroLocal/navigation_map.html \
  --out memory_nav/analysis/output
```

参数说明：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fps` | 自动（校准） | 帧率。**优先用 GPS 日志 `output.jsonl` 中 `image_path` 记录的帧号与 `log_time` 做帧↔GPS 标定**（实测该帧与帧文件夹内 `frame_号-1` 逐像素一致）；无标定点时才按“帧数 / (GPS 结束 − 帧目录开始)”推断 |
| `--tau-theta` | 30 | 转弯锚点阈值 `τ_θ`（航向变化，度） |
| `--tau-vis` | 0.15 | 视觉突变阈值 `τ_vis`（DINOv2 余弦距离，相对上一锚点，且需前进 ≥3m） |
| `--tau-s` | 20 | 距离锚点阈值 `τ_s`（米，长直路强制补锚点） |
| `--visual-window` | 2 | 视觉变化窗口（秒） |
| `--out` | 必填 | 输出目录 |
| `--vision-model` | `Qwen/Qwen3-VL-8B-Instruct` | 用于从锚点图像提取文字的大模型（ModelScope 视觉语言模型） |
| `--api-key-file` | L4 `config copy/api_keys.json` | ModelScope API 密钥 JSON 数组文件 |
| `--map` | 可选 | 生成基于 `navigation_map.html` 的叠加页 `analysis_map.html` |
| `--skip-vis` | 关 | 跳过可视化 |

## 2. 自适应切段条件

在时间 `t` 设置锚点，满足**任一**条件即触发（且首末点必为锚点）：

- **转弯（曲率大）**：`|Δθ_t| = |wrap_to_180(θ_t − θ_{t−1})| > τ_θ = 30°`
- **视觉突变（进入新区域）**：`d_vis(t, t_prev_anchor) = 1 − cosine(emb_t, emb_prev_anchor) > τ_vis`（相对上一锚点外观变化；要求前进 ≥3m，避免原地/车辆造成的抖动）
- **距离过长（长直路强制补锚点）**：`s_t − s_{t_last_anchor} > τ_s = 20m`

视觉特征 `emb` 用 **DINOv2 ViT-B/14** 的 CLS-token（768 维，L2 归一化）表示整帧外观。

## 3. 时钟指令（盲人可执行动作）

把“目标朝向 − 当前朝向”`Δθ` 按 30° 量化成时钟方向：

| 时钟 | 含义 |
|---|---|
| 12 点 | 正前方（直行） |
| 1 / 2 / 3 点 | 右转 30° / 60° / 90°（3 点 = 右 90°） |
| 11 / 10 / 9 点 | 左转 30° / 60° / 90°（9 点 = 左 90°） |
| 6 点 | 回头 180° |

指令示例：`直行约32米`、`9点钟方向前进约20米`、`原地转至3点钟方向，然后前进约5米`。每 1Hz 关键点输出一条，写入 `*_commands.jsonl`。

## 4. 输出文件

| 文件 | 内容 |
|---|---|
| `test_trajectory.html` | 测试轨迹页：**叠加记忆参考轨迹**，每个 1Hz 测试点标注依据参考轨迹的跟随动作（`X点钟方向前进约X米`）+ 匹配指标（匹配质量 / 横向误差 / 朝向误差 / 视觉相似 / 最近参考锚点）。**点击测试锚点**：在参考轨迹上用蓝色圆点标注该锚点匹配到的最近点，并用橙色箭头显示该锚点的建议方向 |
| `memory_trajectory.html` | 记忆轨迹页：锚点（原图/语义/特征/指令）+ 1Hz 指令表 |
| `analysis.html` | 双轨迹叠加离线页 + 测试跟随指标表（无需网络，图片已内嵌 base64） |
| `analysis_map.html` | 若提供 `--map`，把锚点叠加到 Leaflet 底图 |
| `routes_and_anchors.png` / `follow_metrics.png` / `commands.png` | matplotlib 静态图 |
| `memory_anchors.jsonl` / `test_anchors.jsonl` | 锚点：id、时间、坐标、s_m、触发原因、语义文本、指令、**建议方向(参考轨迹)**、**图像文本(VLM)**、DINOv2 特征（768 维） |
| `memory_commands.jsonl` / `test_commands.jsonl` | 1Hz 时钟指令 |
| `match_metrics.jsonl` | 测试跟随指标：横向误差、朝向误差、匹配质量、视觉相似度、最近锚点、跟随指令 |
| `summary.json` | 运行汇总（含 `text_extraction` 状态与视觉模型） |
| `anchors/<id>.jpg` + `.npy` | 每个锚点原图与嵌入向量 |

每个锚点含两个新增字段：

- **`suggested_direction`**（仅测试锚点）：追踪原始参考轨迹给出的建议方向，如 `向10点钟方向前进7米`、`向前直行3米`。
- **`image_text`**：用视觉大模型 `Qwen/Qwen3-VL-8B-Instruct`（ModelScope）从锚点原始图像提取的文本（如路牌、店名、招牌），无文字则留空。

## 5. 当前数据结果（zixu-yuanji 记忆 / liupu-yuanji 测试）

| 项目 | 记忆 (memory) | 测试 (test) |
|---|---|---|
| GPS 记录数 | 130 | 107 |
| 1Hz 关键点 | 103 | 94 |
| 锚点数 | 41 | 47 |
| 触发原因 | 转弯 16、视觉 27、间距 4、起/终 | 转弯 9、视觉 37、起/终 |
| 对齐方式 | 校准（77 点） | 校准（81 点） |
| 标定帧率 | 10.98 | 4.93 |
| 帧号范围 | ~256–4404 | 1601–3347 |
| 帧号重复数 | 0 | 0 |
| 含图像文本(VLM)锚点数 | 14 | 20 |
| 测试锚点含建议方向 | — | 47 / 47 |

> **帧↔GPS 对齐（关键）**：直接用 GPS 日志 `output.jsonl` 里的 `image_path` 帧号与 `log_time` 做标定——已实测该帧与帧文件夹内 `frame_<号-1>.jpg` 逐像素一致（diff=0），因此 `锚点时间 → 帧号` 由这些真实对应点插值/外推得到，**不再依赖猜测的帧率或目录名**。校正后每个锚点都映射到互不重复且与位置相符的帧；两条轨迹在相同地点的帧视觉相似度由 ~0.07 提升到 **0.55**，证明锚点图像已正确对应。
>
> 早期误用 15 帧/秒时约一半关键点被钳制到同一张最后一帧，导致锚点图像错误；随后按“帧数/(GPS 时长)”推断也因目录名并非真实录制起点而仍偏差较大。现采用 `image_path` 逐点标定后彻底修正。

**跟随能力（测试 vs 记忆）**：94 个测试点中 7 个 `good`、10 个 `degraded`、77 个 `lost`；横向误差中位数约 **13 m**、90 分位约 53 m；朝向误差均值约 38°；与最近锚点的视觉相似度均值约 **0.55**。

> 说明：测试轨迹起点、终点与记忆轨迹基本重合（相差 <1.5 m），但**中段明显偏离**（横向偏差可达 10–50 m），因此大部分测试点被判为 `lost`，视觉相似度也很低。这反映出**本次测试数据并未与记忆轨迹紧密贴合**，本工具正确地把这一差异量化了出来。若用真正重合的两次行走（同一路径、相近视角）重跑，`good` 占比和视觉相似度会显著上升。视觉相似度较低也与两次行走的相机视角/头部朝向不同有关。
>
> **闭环处理**：记忆参考轨迹含**闭环绕路**（同一区域在 `s≈90–112m` 与 `s≈152–174m` 被重复走过，弧长约 84m），而测试轨迹直行无闭环。若不处理，进度匹配会**卡在环路处**（`matched_s` 恒为 82.9m，方向错误）。现按参考轨迹中“空间相近但弧长差距最大”的距离**自适应放大匹配前视窗口**（本数据≈124m），使匹配跳过环路冗余弧、沿测试实际前进方向单调推进（`matched_s` 0→303.6m），建议方向随之正确。

## 6. 复现

```bash
cd /home/wheeltec/projects/blind-nav
python3 -m memory_nav.analysis.run_analysis \
  --memory-gps ... --test-gps ... --memory-frames ... --test-frames ... \
  --out memory_nav/analysis/output
```

输出的 HTML 用浏览器直接打开即可（已内嵌图片，无需联网）。