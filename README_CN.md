# fusion-vision

Fusion 单仓的**确定性图像合成算子库**。把平面素材层（印花图、字幕、贴纸）通过纯几何 + 图层算子合成到基底照片（样机图、封面）上。**零生成式 AI**：相同输入字节 + 相同参数 → 逐像素一致的可复现输出。

- [定位](#定位)
- [安装](#安装)
- [快速上手](#快速上手)
- [API](#api-1)
- [算子说明](#算子说明)
- [调参指南](#调参指南)
- [错误处理](#错误处理)
- [确定性与 Golden 回归](#确定性与-golden-回归)
- [性能与内存](#性能与内存)
- [嵌入规则（fusion-flow / worker）](#嵌入规则fusion-flow--worker)
- [故障排查](#故障排查)
- [开发](#开发-1)

## 定位

首个消费方：`business/ai-apparelmockup`（服装印花样机批量化）。第二消费方：`fusion-drama` 后处理（字幕烧录、封面合成）。两者都要求可复现：重跑必须产出字节级一致的图，且基底照片绝不能被破坏。生成式管线两条都保证不了；本库没有模型、没有无种子随机、并且有运行时红线断言（基底图不被改动），所以能保证。

## 安装

```bash
cd ~/fusion && source .venv/bin/activate
pip install -e "./fusion-vision[test]"
```

依赖仅 `opencv-python-headless` + `numpy` + `pillow`。严禁 torch/diffusers 等任何生成式依赖。Python ≥ 3.12。

## 快速上手

```python
import numpy as np
import fusion_vision as fv

template = fv.imread_rgb("tshirt_photo.png")  # (H, W, 3) uint8，基底照片
layer = fv.imread_rgba("print_design.png")  # (h, w, 4) uint8，平面素材
mask = fv.rasterize_polygon(template.shape, quad_pts)  # (H, W) uint8，允许印花的区域

cfg = fv.ComposeConfig(
    dst_quad=((480, 420), (1440, 420), (1440, 1240), (480, 1240)),  # 胸口区域四角
    blend_mode=fv.BlendMode.MULTIPLY,  # 白底印花：白色抠掉
    lighting_gamma=0.6,  # 吸收衣服的光影
    feather_px=2.0,  # 羽化蒙版边缘
)
out = fv.compose(template, layer, mask, cfg)  # (H, W, 3) uint8，基底绝不被改
fv.imwrite_rgb("mockup.png", out)
```

`compose()` 一旦发现图层透明区域的像素与模板不一致就抛 `BaseImageAlteredError` —— 每次成功调用都保证基底照片完好。

## API

```python
from fusion_vision import compose, ComposeConfig, BlendMode, Placement, NoiseSpec, qc
```

`ComposeConfig` 字段（冻结 dataclass，用 `cfg.copy(blend_mode=...)` 派生变体）：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `dst_quad` | 4×(x, y) | 必填 | 画布上的目标四角（任意凸四边形） |
| `src_quad` | 4×(x, y) 或 None | None | 素材内裁剪区；None = 整张素材 |
| `placement` | `Placement` | center, scale 1.0 | warp 后按锚点（"center" / "top_left"）缩放 |
| `blend_mode` | `BlendMode` | NORMAL | normal / multiply / linear_burn |
| `opacity` | float [0, 1] | 1.0 | 图层不透明度 |
| `lighting_gamma` | float [0, 2] | 0.6 | 0 表示关闭光照适配 |
| `sigma` | float | 8.0 | 亮度场高斯模糊 σ |
| `feather_px` | float ≥ 0 | 0.0 | 蒙版羽化半径 |
| `noise` | `NoiseSpec` 或 None | None | 带种子的颗粒：`{strength: (0, 0.5], seed: int}` |
| `assert_base_intact` | bool | True | 红线断言；False 会打 WARNING，仅限测试 |

管线顺序：`warp_to_quad` → `place` → `feather` → `apply_mask` → 光照（`luminance_field` + `adapt` + `grain`）→ `blend` → uint8 量化 → 红线断言。

## 算子说明

所有算子都可单独导入，供高级管线使用。

### `warp.warp_to_quad(layer_rgba, dst_quad, canvas_shape, src_quad=None) -> (H, W, 4) float32`

把素材透视变换到 `dst_quad`。重采样策略（保护文字边缘）：

- 缩小超过 1.15×：先用 `cv2.resize` INTER_AREA 预缩小，再双线性 warp。`cv2.warpPerspective` 本身**不做**真正的面积平均 —— INTER_AREA 在那里退化为点采样，2px 细条纹会混叠成实心块。
- 放大超过 1.15×：LANCZOS4。注意 LANCZOS4 会在 quad 边界外渗出约 8px 核响应（OpenCV BORDER_CONSTANT 标准行为）；alpha 下游会 clip 到 [0, 1]。
- 其余：INTER_LINEAR。

Quad 语义：覆盖 (x0, y0)-(x1, y1) 的 quad 覆盖像素 x ∈ [x0, x1)、y ∈ [y0, y1)。退化（零面积 / 共线 / 自交）quad 抛 `DegenerateQuadError`。

### `mask` — `apply_mask`、`feather`、`rasterize_polygon`

- `apply_mask(rgba, mask_u8)`：alpha 乘以 mask/255。
- `feather(rgba_or_rgb, sigma_px)`：高斯边缘柔化。
- `rasterize_polygon(shape, points, feather_px=0.0)`：多边形转 uint8 蒙版。

### `lighting` — `luminance_field`、`adapt`、`grain`

- `luminance_field(template_rgb, mask, sigma)`：模板 BT.601 亮度 → 高斯模糊 → **在蒙版内** min-max 归一化 → (H, W) float32 ∈ [0, 1]。蒙版内最亮区域映射为 1.0（不衰减）；平坦模板或过小蒙版返回全 1 并打 WARNING。
- `adapt(layer_rgb_f32, field, gamma)`：`layer ×= field**gamma`。`gamma=0` 为恒等。
- `grain(shape, seed, strength)`：带种子的乘性噪声，系数 ∈ [1−strength, 1+strength]。`np.random.default_rng(seed)` —— 种子相同则字节相同。

### `blend.blend(base, layer_rgb, alpha, mode) -> float32`

`alpha` 取值 0-1。各模式输出 = `base·(1−α) + blended·α`：

- NORMAL：`blended = layer`
- MULTIPLY：`blended = base · layer / 255` —— 素材的白色像素取基底值（白底印花抠白）
- LINEAR_BURN：`blended = clip(base + layer − 255, 0, 255)` —— 强压暗，用于阴影叠加

未知模式抛 `UnsupportedModeError`。

### `qc` — 只返回指标，零业务语义

库不知道什么叫"合格"。阈值策略属于消费方（如 ai-apparelmockup `qc/rules.py`）。

- `check_overflow(out, template, mask, tolerance_px=2) -> (violated_px, bbox|None)`：蒙版（膨胀后）之外与模板不同的像素。
- `check_blank(out, template) -> (std, diff_px)`：平图检测。
- `check_color(source, printed, mask) -> (mean_dE, p95_dE)`：CIELAB ΔE；尺寸不一致自动 resize。
- `check_sharpness(image, mask=None) -> float`：蒙版内 Laplacian（ksize=3）方差。注意核响应会向区域边界外渗 1px。

### `io` — `imread_rgb`、`imread_rgba`、`imread_gray`、`imwrite_rgb`

基于 Pillow，路径进 / 数组出。`imread_rgb` 丢弃第 4 通道并打 WARNING。`max_dims` 防解压炸弹。所有消费方**只传本地路径** —— 库不抓 URL、不读环境变量。

## 调参指南

### 光照 γ（`lighting_gamma`）

印花吸收基底照片光影的强度。亮度场做了归一化：蒙版内最亮区域为 1.0。

| γ | 效果 | 场景 |
|---|---|---|
| 0.0 | 关闭 | 字幕、贴纸、平面叠加 |
| 0.4-0.6 | 中等吸收 | 服装样机默认值（褶皱能透出来） |
| 0.8-1.0 | 强吸收 | 光面材质、重阴影场景 |

`sigma` 决定亮度场的粗细：小 σ 保留细褶皱，大 σ 只留大光影。1000px 以上图从 8px 起步；大致按图宽 / 100 缩放。

### 混合模式选择

| 模式 | 素材白色像素 | 用途 |
|---|---|---|
| NORMAL | 不透明白 | 自带背景艺术的满幅设计 |
| MULTIPLY | 透明（取基底） | 白底印花设计 —— DTG/服装标准选择 |
| LINEAR_BURN | 向黑加深 | 阴影/明暗叠加 |

### 羽化

`feather_px` 1-3 可掩盖照片基底上的蒙版边缘痕迹。需要像素精确的字幕框时用 0。羽化只软化蒙版边缘；（未羽化的）alpha 之外的红线断言依然成立。

### 颗粒

`noise=NoiseSpec(strength=0.03, seed=<固定值>)` 加胶片颗粒，让平面印花更贴布。**种子必须固定并记录** —— 不带种子或每次随机的种子会破坏字节级确定性。strength 超过 0.1 肉眼可见；常用 0.02-0.05。

### 不透明度

< 1.0 用于水印式叠加。0.0 逐位返回模板（测试验证过的快速路径）。

## 错误处理

所有错误继承 `VisionError`（带 `detail` 字典供程序化处理）。全量捕获用 `VisionError`，精确捕获用子类：

| 异常 | 触发条件 |
|---|---|
| `InputShapeError` | dtype/形状/通道数错误，配置值非法 |
| `DegenerateQuadError` | quad 零面积、共线、自交 |
| `BaseImageAlteredError` | 红线断言失败：图层 alpha 之外的基底像素被改 |
| `UnsupportedModeError` | 未知混合模式字符串 |

`BaseImageAlteredError.detail` 带 `violation_count` 和前若干个 `first_violations` 坐标 —— 打进日志，那是排查线索。它一旦触发说明下游算子改写了共享内存，是调用方 bug，不是调参问题。

fusion-flow 异常映射（DAG worker 约定）：

| 库异常 | worker 行为 |
|---|---|
| `InputShapeError` / `DegenerateQuadError` / `UnsupportedModeError` | 不可重试：任务置失败，detail 透传给调用方 |
| `BaseImageAlteredError` | 不可重试 + CRITICAL 日志：确定性算子重试不会自愈 |
| 其他 `Exception` | 按 worker 策略重试 |

## 确定性与 Golden 回归

- 所有随机性必须显式带种子。库里不存在无种子随机。
- `tests/golden/` 用 sha256 哈希锁住 19 个输出用例（3 种基底几何 × 3 种混合模式 × 2 种 γ + 1 个带种子噪声用例），记录在 `golden.json`，按平台键（`{platform}-{machine}-cv{主.次}-np{主.次}`）索引。
- 平台键匹配时测试是严格的：输出 sha256 必须等于记录值。其他平台则与已提交的参考 PNG 比对，容差 max_abs_diff ≤ 1。
- **任何改变输出字节的算子改动都是 semver minor+ 升级**，需要走评审更新 `golden.json` + `reference/`。重新生成：

```bash
python tests/golden/update_golden.py
```

- `opencv-python-headless` 在 `pyproject.toml` 锁 patch 区间；OpenCV 升级若改变插值输出，golden 测试会按设计失败。

## 性能与内存

Apple Silicon 目标（CI 由 `tests/perf/test_perf_2400.py` 强制）：

- 2400px 合成（含种子噪声）< 2s
- 峰值内存 < 500MB：中间量为 float32，尽量单缓冲，并在红线断言前 `del` 释放

fusion-flow worker 内存约束：库会分配约 4 个画布大小的 float32 缓冲（warp / mask / 光照 / blend）加一个 uint8 输出。2400×2400 画布峰值约 110MB。按此规划 worker 池；2400px 任务不要与大图任务共享 worker。

仅 sRGB（M1 范围）。ICC / 广色域是调用方的活 —— 进 `compose()` 前自己转好。

## 嵌入规则（fusion-flow / worker）

本库是**纯库**：无端口、无服务、无配置文件、不读环境变量。嵌入规则：

1. **只传本地路径。** 所有 `imread_*` / `imwrite_*` 只收文件系统路径。库不下载、不抓 URL、不从环境读配置。AI 预处理（如抠图）由调用方完成；库只接受纯图。
2. **宿主进程里不要调 `cv2.setNumThreads()`** —— 库自己也不调（`tests/test_import_purity.py` 强制：`import fusion_vision` 前后 `cv2.getNumThreads()` 不变）。如果你全局设了，后果自负。
3. **日志**用 `logging.NullHandler` —— 库自己不配日志。宿主自行挂 handler：

```python
import logging

logging.getLogger("fusion_vision").addHandler(logging.StreamHandler())
logging.getLogger("fusion_vision").setLevel(logging.INFO)
```

4. **零业务语义。** 库永远不知道"胸口位"或"QC 合格"是什么意思。阈值策略留在消费方。

## 故障排查

**正常 compose 抛 `BaseImageAlteredError`**
有代码在 compose 前后改了模板数组。常见原因：调用方跨线程复用模板缓冲；预处理（resize、色彩转换）原地写入。在调用方修 —— 共享缓冲就在边界传 `template.copy()`。不要用 `assert_base_intact=False` 压掉（该标志仅限测试，且会打 WARNING）。

**缩小后细条纹 / 文字变成实心块**
你的自定义管线直接用 INTER_AREA 做 warp。`warp_to_quad` 已正确预缩小 —— 用它，别自己调 `cv2.warpPerspective`。

**放大后 quad 外约 8px 幽灵像素**
LANCZOS4 核配合 BORDER_CONSTANT 的预期渗出。alpha 下游有 clip，红线断言依然成立（α=0 混合是逐位精确的）。视觉上有影响就把 `dst_quad` 往安全区内收 8px。

**OpenCV/numpy 升级后 golden 测试挂了**
按设计。插值输出变了 → 输出字节变了。有意地升 opencv 版本锁，重新生成 golden（`python tests/golden/update_golden.py`），评审 diff，按 semver minor+ 发版。

**`luminance_field` 返回全 1 + WARNING**
模板蒙版内平坦（零动态范围）或蒙版不足 3 像素。不是错误 —— 只是没有可吸收的光照信息。

**输出像"双重曝光" / 二值乱码**
自定义管线的 alpha 量纲错了：`blend()` 要 0-1 的 alpha，不是 0-255。`compose()` 内部已转换；直接调 `blend()` 就要自己除以 255 并 clip。

## 开发

```bash
cd ~/fusion && source .venv/bin/activate
cd fusion-vision
pip install -e ".[test]"

python -m pytest tests/ -q                    # 全量（含 golden + perf）
python -m pytest tests/test_warp.py -v        # 单文件
python -m pytest tests/test_warp.py::test_v7_known_homography_corner_accuracy -v

ruff check . && ruff format --check .
```

布局：`src/fusion_vision/`（`warp` / `mask` / `lighting` / `blend` / `compose` / `qc` / `io` / `color` / `config` / `errors` / `_types`），`tests/`（单元 + `golden/` + `perf/`）。权威规格：`../architecture/fusion-vision-prd-1027.md`。代码风格：4 的倍数缩进、无 docstring、全路径有日志。
