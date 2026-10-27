# fusion-vision

Deterministic image-composition operator library for the Fusion monorepo. Composites a flat source layer (print design, subtitle, sticker) onto a base photo (mockup, cover) via pure geometric + layer operators. **Zero generative AI**: same input bytes + same params → pixel-identical output, auditable per-pixel.

- [Why](#why)
- [Install](#install)
- [Quick Start](#quick-start)
- [API](#api)
- [Operators](#operators)
- [Tuning Guide](#tuning-guide)
- [Error Handling](#error-handling)
- [Determinism & Golden Regression](#determinism--golden-regression)
- [Performance & Memory](#performance--memory)
- [Embedding Rules (fusion-flow / workers)](#embedding-rules-fusion-flow--workers)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Why

First consumer: `business/ai-apparelmockup` (apparel print mockup batching). Second: `fusion-drama` post-pass (subtitle burn-in, cover compositing). Both need reproducible output — a re-run must produce byte-identical images, and the base photo must never be damaged. Generative pipelines cannot promise either; this library can, because it contains no model, no randomness without a seed, and a runtime red-line assert that the base image is untouched.

## Install

```bash
cd ~/fusion && source .venv/bin/activate
pip install -e "./fusion-vision[test]"
```

Dependencies: `opencv-python-headless` + `numpy` + `pillow` only. Never torch/diffusers or any generative dependency. Python ≥ 3.12.

## Quick Start

```python
import numpy as np
import fusion_vision as fv

template = fv.imread_rgb("tshirt_photo.png")  # (H, W, 3) uint8, the base photo
layer = fv.imread_rgba("print_design.png")  # (h, w, 4) uint8, the flat artwork
mask = fv.rasterize_polygon(template.shape, quad_pts)  # (H, W) uint8, allowed print region

cfg = fv.ComposeConfig(
    dst_quad=((480, 420), (1440, 420), (1440, 1240), (480, 1240)),  # chest area corners
    blend_mode=fv.BlendMode.MULTIPLY,  # white-base prints key out white
    lighting_gamma=0.6,  # absorb the shirt's light/shadow
    feather_px=2.0,  # soften the mask edge
)
out = fv.compose(template, layer, mask, cfg)  # (H, W, 3) uint8, base never altered
fv.imwrite_rgb("mockup.png", out)
```

`compose()` raises `BaseImageAlteredError` if any pixel where the layer is transparent differs from the template — the base photo is guaranteed intact on every successful call.

## API

```python
from fusion_vision import compose, ComposeConfig, BlendMode, Placement, NoiseSpec, qc
```

`ComposeConfig` fields (frozen dataclass; use `cfg.copy(blend_mode=...)` for variants):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dst_quad` | 4×(x, y) | required | Destination corners on the canvas (any convex quad) |
| `src_quad` | 4×(x, y) or None | None | Source crop in the layer; None = whole layer |
| `placement` | `Placement` | center, scale 1.0 | Post-warp scale about anchor ("center" / "top_left") |
| `blend_mode` | `BlendMode` | NORMAL | normal / multiply / linear_burn |
| `opacity` | float [0, 1] | 1.0 | Layer opacity |
| `lighting_gamma` | float [0, 2] | 0.6 | 0 disables lighting adaptation |
| `sigma` | float | 8.0 | Gaussian blur σ for the luminance field |
| `feather_px` | float ≥ 0 | 0.0 | Mask feather radius |
| `noise` | `NoiseSpec` or None | None | Seeded grain: `{strength: (0, 0.5], seed: int}` |
| `assert_base_intact` | bool | True | Red-line assert; False logs a WARNING and is test-only |

Pipeline order: `warp_to_quad` → `place` → `feather` → `apply_mask` → lighting (`luminance_field` + `adapt` + `grain`) → `blend` → uint8 quantize → red-line assert.

## Operators

All operators are importable individually for advanced pipelines.

### `warp.warp_to_quad(layer_rgba, dst_quad, canvas_shape, src_quad=None) -> (H, W, 4) float32`

Perspective warp of the layer onto `dst_quad`. Resampling policy (protects text edges):

- Downscale > 1.15×: pre-shrink with `cv2.resize` INTER_AREA, then warp bilinearly. `cv2.warpPerspective` alone does **not** do true area averaging — INTER_AREA there falls back to point sampling and fine 2px stripes alias to solid blocks.
- Upscale > 1.15×: LANCZOS4. Note LANCZOS4 has an ~8px kernel bleed past the quad edge (standard OpenCV BORDER_CONSTANT behavior); alpha is clipped to [0, 1] downstream.
- Otherwise: INTER_LINEAR.

Quad semantics: a quad spanning (x0, y0)-(x1, y1) covers pixels x ∈ [x0, x1), y ∈ [y0, y1). Degenerate (zero area / collinear / self-intersecting) quads raise `DegenerateQuadError`.

### `mask` — `apply_mask`, `feather`, `rasterize_polygon`

- `apply_mask(rgba, mask_u8)`: multiplies alpha by mask/255.
- `feather(rgba_or_rgb, sigma_px)`: gaussian edge softening.
- `rasterize_polygon(shape, points, feather_px=0.0)`: polygon → uint8 mask.

### `lighting` — `luminance_field`, `adapt`, `grain`

- `luminance_field(template_rgb, mask, sigma)`: BT.601 luma of the template → gaussian blur → min-max normalize **inside the mask** → (H, W) float32 in [0, 1]. The brightest in-mask region maps to 1.0 (no attenuation); flat or tiny masks return all-ones with a WARNING.
- `adapt(layer_rgb_f32, field, gamma)`: `layer ×= field**gamma`. `gamma=0` is identity.
- `grain(shape, seed, strength)`: seeded multiplicative noise, factor ∈ [1−strength, 1+strength]. `np.random.default_rng(seed)` — same seed, same bytes.

### `blend.blend(base, layer_rgb, alpha, mode) -> float32`

`alpha` is 0-1. Per mode, output = `base·(1−α) + blended·α`:

- NORMAL: `blended = layer`
- MULTIPLY: `blended = base · layer / 255` — white pixels of the layer take the base's value (white-base prints key out)
- LINEAR_BURN: `blended = clip(base + layer − 255, 0, 255)` — darkens hard, for shadow overlays

Unknown modes raise `UnsupportedModeError`.

### `qc` — metrics only, zero business semantics

The library knows nothing about "pass/fail". Threshold policy lives in the consumer (e.g. ai-apparelmockup `qc/rules.py`).

- `check_overflow(out, template, mask, tolerance_px=2) -> (violated_px, bbox|None)`: pixels differing from the template outside (dilated) mask.
- `check_blank(out, template) -> (std, diff_px)`: flat-output detector.
- `check_color(source, printed, mask) -> (mean_dE, p95_dE)`: CIELAB ΔE; sizes auto-resized on mismatch.
- `check_sharpness(image, mask=None) -> float`: Laplacian (ksize=3) variance in-mask. Note the kernel response bleeds 1px past a region boundary.

### `io` — `imread_rgb`, `imread_rgba`, `imread_gray`, `imwrite_rgb`

Pillow-backed, path-in / array-out. `imread_rgb` drops a 4th channel with a WARNING. `max_dims` guards against decompression bombs. All consumers pass **payload paths only** — the library never fetches URLs or reads env vars.

## Tuning Guide

### Lighting γ (`lighting_gamma`)

How strongly the print absorbs the base photo's light/shadow. The luminance field is normalized so the brightest in-mask area is 1.0.

| γ | Effect | Use when |
|---|---|---|
| 0.0 | Off | Subtitles, stickers, flat overlays |
| 0.4-0.6 | Moderate absorption | Default for apparel mockups (wrinkles show through) |
| 0.8-1.0 | Strong absorption | Glossy materials, heavy shadow scenes |

`sigma` controls how coarse the luminance field is: small σ keeps fine wrinkles, large σ keeps only broad light gradients. Start at 8px for 1000px+ images; scale roughly with image width / 100.

### Blend mode selection

| Mode | Layer white pixels | Use for |
|---|---|---|
| NORMAL | Opaque white | Full-bleed designs with background art |
| MULTIPLY | Transparent (take base) | White-base print designs — the standard for DTG/apparel |
| LINEAR_BURN | Deepen toward black | Shadow/shading overlays |

### Feather

`feather_px` 1-3 hides mask edge artifacts on photographic bases. 0 for pixel-exact subtitle boxes. Feather only softens the mask edge; the red-line assert still guarantees untouched pixels outside the (un-feathered) alpha.

### Grain

`noise=NoiseSpec(strength=0.03, seed=<fixed>)` adds a film-grain texture that makes flat prints sit more naturally on fabric. **The seed must be fixed and recorded** — an unseeded or per-run-random seed breaks byte determinism. Strength above 0.1 is visible; 0.02-0.05 is typical.

### Opacity

Use < 1.0 for watermark-style overlays. 0.0 returns the template bit-identical (fast path verified by tests).

## Error Handling

All errors inherit `VisionError` (with a `detail` dict for programmatic handling). Catch `VisionError` for everything, or the specific subclass:

| Exception | Raised when |
|---|---|
| `InputShapeError` | Wrong dtype/shape/channel count, bad config values |
| `DegenerateQuadError` | Quad has zero area, collinear points, or self-intersects |
| `BaseImageAlteredError` | Red-line assert failed: base pixels changed outside layer alpha |
| `UnsupportedModeError` | Unknown blend mode string |

`BaseImageAlteredError.detail` carries `violation_count` and the first `first_violations` coordinates — log them, they are the debugging trail. If it fires, a downstream operator mutated shared memory; this is a caller bug, not a tuning problem.

fusion-flow exception mapping (DAG worker convention):

| Library exception | Worker behavior |
|---|---|
| `InputShapeError` / `DegenerateQuadError` / `UnsupportedModeError` | Non-retryable: mark task failed, surface detail to the caller |
| `BaseImageAlteredError` | Non-retryable + CRITICAL log: a deterministic operator never self-heals on retry |
| Any other `Exception` | Retryable per worker policy |

## Determinism & Golden Regression

- All randomness requires an explicit seed. Unseeded randomness does not exist in this library.
- `tests/golden/` locks 19 output cases (3 base geometries × 3 blend modes × 2 γ + 1 seeded-noise case) as sha256 hashes in `golden.json`, keyed per platform (`{platform}-{machine}-cv{major.minor}-np{major.minor}`).
- On a platform with a matching key, the test is strict: output sha256 must equal the recorded hash. On any other platform it compares against the committed reference PNG with tolerance max_abs_diff ≤ 1.
- **Any operator change that alters output bytes is a semver minor+ bump** and requires a reviewed `golden.json` + `reference/` update. To regenerate:

```bash
python tests/golden/update_golden.py
```

- Local determinism acceptance: `python tests/perf/repro_check.py [runs]` (default 100) re-composes the 2400px perf case and asserts every run is byte-identical (sha256), with timing stats.

- `opencv-python-headless` is locked to a patch range in `pyproject.toml`; an OpenCV upgrade that changes interpolation output will fail the golden test by design.

## Performance & Memory

Targets on Apple Silicon (enforced by `tests/perf/test_perf_2400.py` in CI):

- 2400px compose (with seeded noise) < 2s
- Peak memory < 500MB: intermediates are float32, single-buffer where possible, and freed (`del`) before the red-line assert

Memory constraint for fusion-flow workers: the library allocates ~4 float32 buffers of canvas size (warp / mask / lighting / blend) plus one uint8 output. For a 2400×2400 canvas that is ~110MB peak. Size worker pools accordingly; a 2400px job must not share a worker with another large-image task.

sRGB only (M1 scope). ICC profiles / wide gamut are the caller's job — convert before calling `compose()`.

## Embedding Rules (fusion-flow / workers)

This is a **pure library**: no port, no service, no config file, no env vars. Embedding rules:

1. **Payload paths only.** Every `imread_*` / `imwrite_*` takes a filesystem path. The library never downloads, never fetches URLs, never reads configuration from the environment. AI preprocessing (e.g. background removal) happens in the caller; the library only accepts pure images.
2. **Never call `cv2.setNumThreads()` in your host process** — the library never calls it either (enforced by `tests/test_import_purity.py`: `cv2.getNumThreads()` is unchanged across `import fusion_vision`). If you set it globally, you own the consequences for co-located workers.
3. **Logging** uses `logging.NullHandler` — the library never configures logging itself. Attach handlers to the `fusion_vision` logger namespace from the host:

```python
import logging

logging.getLogger("fusion_vision").addHandler(logging.StreamHandler())
logging.getLogger("fusion_vision").setLevel(logging.INFO)
```

4. **Zero business semantics.** The library never learns what "chest placement" or "QC pass" means. Keep threshold policy in the consumer.

## Troubleshooting

**`BaseImageAlteredError` on a normal compose**
Something mutated the template array before or during compose. Common causes: caller reused the template buffer across threads; a pre-pass (resize, color convert) wrote in-place. Fix in the caller — pass `template.copy()` at the boundary if the buffer is shared. Do not set `assert_base_intact=False` to silence it (that flag is test-only and logs a WARNING).

**Fine stripes / text turn to solid blocks after downscale**
You are hitting a custom pipeline that warps with INTER_AREA directly. `warp_to_quad` already pre-resizes correctly — use it instead of calling `cv2.warpPerspective` yourself.

**Ghost pixels ~8px outside the quad after upscale**
Expected LANCZOS4 kernel bleed with BORDER_CONSTANT. Alpha is clipped downstream and the red-line assert still holds (blend with α=0 is bit-exact). If it visually matters, pull `dst_quad` 8px inside the safe area.

**Golden test fails after an OpenCV/numpy upgrade**
By design. Interpolation output changed → output bytes changed. Bump the opencv pin deliberately, regenerate golden (`python tests/golden/update_golden.py`), review the diff, release as semver minor+.

**`luminance_field` returns all-ones + WARNING**
Template is flat in-mask (zero dynamic range) or the mask has < 3 pixels. Not an error — there is simply no lighting information to absorb.

**Output looks "double-exposed" / binary garbage**
An alpha scale mismatch in a custom pipeline: `blend()` expects alpha in [0, 1], not [0, 255]. `compose()` handles the conversion; if you call `blend()` directly, divide by 255 and clip.

## Development

```bash
cd ~/fusion && source .venv/bin/activate
cd fusion-vision
pip install -e ".[test]"

python -m pytest tests/ -q                    # full suite incl. golden + perf
python -m pytest tests/test_warp.py -v        # single file
python -m pytest tests/test_warp.py::test_v7_known_homography_corner_accuracy -v

ruff check . && ruff format --check .
```

Layout: `src/fusion_vision/` (`warp` / `mask` / `lighting` / `blend` / `compose` / `qc` / `io` / `color` / `config` / `errors` / `_types`), `tests/` (unit + `golden/` + `perf/`). Authoritative spec: `../architecture/fusion-vision-prd-1027.md`. Style: 4-space-multiple indent, no docstrings, logging on every path.
