# Perf target (PRD SS8): 2400px compose < 2s on Apple Silicon.
# Not a correctness test; generous margin to stay stable on shared runners.
import os
import time

import numpy as np

import fusion_vision as fv


def test_perf_2400_compose_under_2s():
    h = w = 2400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gray = np.clip(np.rint(140.0 + 40.0 * (xx / w) + 20.0 * np.sin(0.02 * yy)), 0, 255)
    template = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    lh = lw = 1200
    layer = np.zeros((lh, lw, 4), dtype=np.uint8)
    layer[..., 0] = 60
    layer[..., 1] = 180
    layer[..., 2] = 90
    layer[..., 3] = 255

    mask = fv.rasterize_polygon(
        (h, w, 3),
        ((120, 120), (w - 120, 120), (w - 120, h - 120), (120, h - 120)),
    )
    cfg = fv.ComposeConfig(
        dst_quad=((120, 120), (w - 120, 120), (w - 120, h - 120), (120, h - 120)),
        blend_mode=fv.BlendMode.MULTIPLY,
        lighting_gamma=0.6,
        sigma=15.0,
        noise={"strength": 0.03, "seed": 7},
    )

    # Budget is env-overridable: PRD target is 2s on a quiet Apple Silicon
    # dev machine, but GitHub-hosted runners measure ~2.5-3s for the same
    # code (slower cores + noisy neighbors), and a busy workstation skews
    # a single sample arbitrarily. Median of 3 + env budget keeps the gate
    # meaningful in both contexts.
    budget_s = float(os.getenv("FUSION_VISION_PERF_BUDGET_S", "2.0"))
    timings = []
    out = None
    for _ in range(3):
        started = time.perf_counter()
        out = fv.compose(template, layer, mask, cfg)
        timings.append(time.perf_counter() - started)
    elapsed = sorted(timings)[1]
    assert out.shape == (h, w, 3)
    assert elapsed < budget_s, (
        f"2400px compose took {elapsed:.3f}s (> {budget_s}s budget; "
        f"samples {[f'{t:.3f}' for t in timings]})"
    )
