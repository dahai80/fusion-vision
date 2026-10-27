# Local acceptance script (archi plan SS8 / milestone M1.5):
# run compose N times with identical inputs+params, assert byte-identical
# output every run (PRD red line 2: determinism). CI samples 3 runs;
# the full 100-run sweep lives here.
#
# Usage: python tests/perf/repro_check.py [runs]
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import fusion_vision as fv  # noqa: E402  (needs sys.path tweak above)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("repro_check")


def build_case(h=2400, w=2400):
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
    return template, layer, mask, cfg


def main(runs=100):
    template, layer, mask, cfg = build_case()
    import hashlib

    reference = None
    reference_sha = None
    timings = []
    for i in range(runs):
        started = time.perf_counter()
        out = fv.compose(template, layer, mask, cfg)
        timings.append(time.perf_counter() - started)
        sha = hashlib.sha256(out.tobytes()).hexdigest()
        if reference is None:
            reference = out
            reference_sha = sha
            logger.info("run 0 reference sha256=%s", sha)
        elif sha != reference_sha:
            logger.error("run %d diverged: sha256=%s (expected %s)", i, sha, reference_sha)
            return 1
        del out

    logger.info(
        "repro_check PASS: %d/%d runs byte-identical, sha256=%s, mean=%.3fs min=%.3fs max=%.3fs",
        runs,
        runs,
        reference_sha,
        float(np.mean(timings)),
        float(np.min(timings)),
        float(np.max(timings)),
    )
    return 0


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    sys.exit(main(runs))
