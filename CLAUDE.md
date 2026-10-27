# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

M1 + M1.5 implemented: all operators, QC, golden regression, perf tests. Authoritative spec: `../architecture/fusion-vision-prd-1027.md`; architecture plan: `../architecture/fusion-vision-archi-plan-1027.md`. When code and PRD disagree, the PRD wins until it is formally revised.

## What fusion-vision Is

A **deterministic image-composition operator library** (pure library, no port, no service, no config file, reads no env vars) for the Fusion monorepo. It composites a flat source layer (print design, subtitle, sticker) onto a base photo (mockup, cover) via geometric + layer operators — **zero generative AI**. It is the opposite of `fusion-drama`/`fusion-comfyui` (generative, non-reproducible): same input bytes + params → pixel-identical output, auditable per-pixel.

First consumer: `business/ai-apparelmockup` (apparel print mockup batching). Second: `fusion-drama` post-pass (subtitle burn-in, cover compositing).

## Hard Rules (PRD red lines)

1. **Base image is never modified.** `compose()` enforces at runtime: pixels where `mask==0` must be bit-identical to the template, or `BaseImageAlteredError` is raised. `assert_base_intact=False` exists for test paths only — production use must log a warning.
2. **Determinism.** All randomness (grain noise) requires an explicit injected seed. Golden-sample regression (`tests/golden/`, sha256 in `golden.json`) locks output bytes; any operator change that alters output bytes = semver minor+ and a reviewed `golden.json` update. OpenCV version locked to a patch range.
3. **Zero business semantics.** The library knows nothing about "chest/back" placement, templates, or pass/fail thresholds. QC operators return numbers; threshold policy belongs to the consumer (e.g. ai-apparelmockup `qc/rules.py`).
4. **Dependencies allowed: `opencv-python-headless` + `numpy` + `pillow` only.** Never torch/diffusers/generative deps. AI preprocessing (e.g. background removal) is done by the caller; the library only accepts pure images.
5. **Never call `cv2.setNumThreads()` inside the library** — it mutates the host process's global OpenCV thread pool and breaks co-located components (e.g. fusion-flow workers). Enforced by an import-purity test: `cv2.getNumThreads()` must be unchanged before/after `import fusion_vision`.

## Layout (per PRD §6)

- `src/fusion_vision/warp.py` — `warp_to_quad`, `compute_homography`, `place`; resampling: LANCZOS4 upscale / pre-resize+INTER_LINEAR downscale (`cv2.warpPerspective` INTER_AREA is point sampling and aliases — never warp with INTER_AREA directly)
- `src/fusion_vision/mask.py` — `apply_mask`, `feather` (gaussian), `rasterize_polygon`, `dilate_mask`
- `src/fusion_vision/lighting.py` — `luminance_field` (in-mask normalized, brightest region = 1.0), `adapt` (`RGB ×= L^γ`), seeded `grain`
- `src/fusion_vision/blend.py` — NORMAL / MULTIPLY (white-base prints) / LINEAR_BURN; alpha is 0-1
- `src/fusion_vision/compose.py` — pipeline: warp → place → feather → mask → lighting → blend → quantize → red-line assert; `compose_layers` for sequential multi-layer
- `src/fusion_vision/qc.py` — `check_overflow` / `check_blank` / `check_color` (CIELAB ΔE) / `check_sharpness` (Laplacian variance); metrics only
- `src/fusion_vision/io.py` / `color.py` / `config.py` / `errors.py` / `_types.py`
- `tests/golden/` — 22 cases (3 geometries × 3 blend modes × 2 γ + 1 seeded-noise + 3 real-sample from consumer fixtures in `tests/samples/`), sha256-locked in `golden.json` per platform key; `update_golden.py` regenerates (semver minor+ + review required)
- `tests/perf/test_perf_2400.py` — 2400px < 2s benchmark

Errors all inherit `VisionError`: `InputShapeError`, `DegenerateQuadError`, `BaseImageAlteredError`, `UnsupportedModeError`.

## API Contract (PRD §7)

```python
from fusion_vision import compose, ComposeConfig, BlendMode, qc

cfg = ComposeConfig(
    dst_quad=...,
    placement=...,
    lighting_gamma=0.6,
    blend_mode=BlendMode.MULTIPLY,
    opacity=1.0,
    feather_px=0.0,
    noise=None,
    assert_base_intact=True,
)
out = compose(template_rgb, layer_rgba, mask_u8, cfg)  # -> (H,W,3) uint8
```

## Build & Test

```bash
cd ~/fusion && source .venv/bin/activate
cd fusion-vision
pip install -e ".[test]"

python -m pytest tests/ -q                 # full suite (unit + golden + perf)
python -m pytest tests/test_warp.py -v     # single file
python -m pytest tests/test_warp.py::test_v6_identity_quad_equals_direct_paste -v

ruff check . && ruff format --check .
```

After any change that alters output bytes: `python tests/golden/update_golden.py`, review the `golden.json` + `reference/` diff, treat as semver minor+.

Local determinism acceptance: `python tests/perf/repro_check.py` (100-run byte-identical sweep).

Non-functional targets: 2400px compose < 2s on Apple Silicon; peak memory < 500MB (single float32 intermediate, freed promptly); sRGB only in M1 (ICC/wide-gamut is caller's job).

## Repo Conventions

Per monorepo root CLAUDE.md: independent git repo (init at M1), 4-space-multiple indentation, no docstrings, logging in all code, setuptools backend with `[tool.setuptools.packages.find]`, ruff if configured.
