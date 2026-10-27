# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Pre-M1 greenfield. This directory contains no code yet — the authoritative spec is `../architecture/fusion-vision-prd-1027.md`. Every design decision below comes from that PRD; when code and PRD disagree, the PRD wins until it is formally revised.

## What fusion-vision Is

A **deterministic image-composition operator library** (pure library, no port, no service, no config file, reads no env vars) for the Fusion monorepo. It composites a flat source layer (print design, subtitle, sticker) onto a base photo (mockup, cover) via geometric + layer operators — **zero generative AI**. It is the opposite of `fusion-drama`/`fusion-comfyui` (generative, non-reproducible): same input bytes + params → pixel-identical output, auditable per-pixel.

First consumer: `business/ai-apparelmockup` (apparel print mockup batching). Second: `fusion-drama` post-pass (subtitle burn-in, cover compositing).

## Hard Rules (PRD red lines)

1. **Base image is never modified.** `compose()` enforces at runtime: pixels where `mask==0` must be bit-identical to the template, or `BaseImageAlteredError` is raised. `assert_base_intact=False` exists for test paths only — production use must log a warning.
2. **Determinism.** All randomness (grain noise) requires an explicit injected seed. Golden-sample regression (`tests/golden/`, sha256 in `golden.json`) locks output bytes; any operator change that alters output bytes = semver minor+ and a reviewed `golden.json` update. OpenCV version locked to a patch range.
3. **Zero business semantics.** The library knows nothing about "chest/back" placement, templates, or pass/fail thresholds. QC operators return numbers; threshold policy belongs to the consumer (e.g. ai-apparelmockup `qc/rules.py`).
4. **Dependencies allowed: `opencv-python-headless` + `numpy` + `pillow` only.** Never torch/diffusers/generative deps. AI preprocessing (e.g. background removal) is done by the caller; the library only accepts pure images.

## Planned Layout (per PRD §6)

- `warp.py` — perspective warp to quad (`warp_to_quad`), resampling policy: LANCZOS4 upscale / INTER_AREA downscale (protects text edges; 2400px e-commerce default)
- `mask.py` — `apply_mask`, `feather` (gaussian), `rasterize_polygon`
- `lighting.py` — `luminance_field` (template grayscale → blurred, normalized in-mask), `adapt` (`RGB ×= L^γ`, default γ=0.6), seeded `grain`; blend modes NORMAL / MULTIPLY (white-base prints) / LINEAR_BURN
- `compose.py` — pipeline orchestration: warp → place → mask → adapt → blend → red-line assert
- `qc.py` — `check_overflow` / `check_blank` / `check_color` (CIELAB ΔE) / `check_sharpness` (Laplacian variance); returns metrics only
- `tests/golden/` — 18+ fixture combos (3 base geometries × 3 blend modes × 2 γ), hash-locked

Errors all inherit `VisionError`: `InputShapeError`, `DegenerateQuadError`, `BaseImageAlteredError`, `UnsupportedModeError`.

## API Contract (PRD §7)

```python
from fusion_vision import compose, ComposeConfig, BlendMode, qc

cfg = ComposeConfig(dst_quad=..., placement=..., lighting_gamma=0.6,
                    blend_mode=BlendMode.MULTIPLY, opacity=1.0,
                    feather_px=0.0, noise=None, assert_base_intact=True)
out = compose(template_rgb, layer_rgba, mask_u8, cfg)   # -> (H,W,3) uint8
```

## Build & Test (once code exists)

Standard Fusion Python project conventions: `pip install -e ".[test]"` after activating the repo-root `.venv` (`cd ~/fusion && source .venv/bin/activate`). Tests via `pytest tests/`, single test `pytest tests/test_foo.py::test_bar -v`. Golden-sample hashes and the 2400px < 2s perf benchmark run in CI.

Non-functional targets: 2400px compose < 2s on Apple Silicon; peak memory < 500MB (single float32 intermediate, freed promptly); sRGB only in M1 (ICC/wide-gamut is caller's job).

## Repo Conventions

Per monorepo root CLAUDE.md: independent git repo (init at M1), 4-space-multiple indentation, no docstrings, logging in all code, setuptools backend with `[tool.setuptools.packages.find]`, ruff if configured.
