# fusion-drama Post-Pass Integration Evaluation (M2, archi plan §12.3)

Status: evaluation complete, ready for drama-side integration. No library change required — all three drama workflows map onto existing M1 operators.

## Boundary criterion (unchanged)

If the artifact must be reproducible and the base image must not be modified → fusion-vision. Otherwise (generative steps) → fusion-mlx / fusion-gateway. Drama keeps its pipeline framework; it only `import fusion_vision` for the compositing steps.

## Workflow mapping

| Drama workflow | fusion-vision call | Notes |
|---|---|---|
| Subtitle burn-in | `compose(video_frame_rgb, subtitle_rgba, full_mask, cfg)` with `blend_mode=NORMAL`, `lighting_gamma=0.0`, `feather_px=0.0` | Text edges are already anti-aliased in the RGBA; never feather. Per-frame call; homography is identity when `dst_quad` matches the layer bbox |
| Cover patch / sticker | `compose(cover_rgb, art_rgba, mask, cfg)` with `dst_quad` from the cover's perspective quad, `feather_px=1-3` | Warp handles the perspective; LANCZOS4 upscale keeps text sharp |
| Storyboard layout (N panels) | `compose_layers(base, [panel_1..panel_n], [mask_1..mask_n], [cfg_1..cfg_n])` | Sequential deterministic order; each layer red-line-asserted against the previous result |

## Deterministic-by-construction requirements for drama

1. Fixed seeds: if grain is used on covers, the seed must be derived from a stable input (e.g. content hash), never from `time`/`random`.
2. Frame-seek before compose: the decoded frame fed to `compose()` must come from a deterministic seek; decode jitter is drama's responsibility.
3. `assert_base_intact=True` stays on. Per-frame cost of the red-line assert is a linear scan (~ms at 1080p) — measured acceptable in `tests/perf/repro_check.py` timing budget.
4. `qc.check_sharpness` on burned subtitles gives a cheap per-frame regression signal (variance should not collapse).

## Integration checklist (drama side)

- [ ] Add `fusion-vision` to drama's dependency list (in-repo `pip install -e ../fusion-vision`)
- [ ] Convert frames to RGB (cv2 users get BGR — use `fv.imread_rgb` contract or swap channels explicitly)
- [ ] Map drama's pipeline errors: `VisionError` subclasses are non-retryable (see README fusion-flow exception table)
- [ ] Keep subtitle RGBA generation (rendering) upstream; fusion-vision only composites

## Explicitly out of scope

- Video encoding/decoding, timecode handling, audio — drama's domain.
- Generative frame content — stays on fusion-mlx via fusion-gateway (`:11432`).
- ICC/wide-gamut — caller converts before `compose()` (M1 boundary).
