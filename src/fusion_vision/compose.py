import logging
import time

import numpy as np

from ._quantize import quantize
from ._types import (
    REDLINE_MAX_VIOLATIONS_IN_DETAIL,
    validate_layer,
    validate_mask,
    validate_template,
)
from .blend import blend
from .config import ComposeConfig
from .errors import BaseImageAlteredError
from .lighting import adapt, grain, luminance_field
from .mask import apply_mask, feather
from .warp import place, warp_to_quad

logger = logging.getLogger(__name__)


def compose(template, layer, mask, cfg):
    started = time.perf_counter()
    template = validate_template(template)
    layer = validate_layer(layer)
    mask = validate_mask(mask, template.shape)
    if not isinstance(cfg, ComposeConfig):
        raise TypeError("cfg must be a ComposeConfig")

    canvas_shape = template.shape
    warped = warp_to_quad(layer, cfg.dst_quad, canvas_shape)
    placed = place(warped, cfg.placement, canvas_shape)

    eff_mask = mask
    if cfg.feather_px > 0:
        eff_mask = feather(mask, cfg.feather_px)
    masked = apply_mask(placed, eff_mask)
    alpha = np.clip(masked[..., 3] * np.float32(cfg.opacity) / 255.0, 0.0, 1.0)
    rgb = masked[..., :3]

    if cfg.lighting_gamma > 0:
        field = luminance_field(template, eff_mask, cfg.sigma)
        rgb = adapt(rgb, field, cfg.lighting_gamma)
    if cfg.noise is not None:
        rgb = rgb * grain(canvas_shape, cfg.noise.seed, cfg.noise.strength)

    out_f = blend(template, rgb, alpha, cfg.blend_mode)
    out = quantize(out_f)
    del out_f, rgb, masked, warped, placed

    if cfg.assert_base_intact:
        _redline_assert(out, template, alpha)
    else:
        logger.warning("assert_base_intact=False is test-only; production use is forbidden")

    elapsed = time.perf_counter() - started
    logger.info(
        "compose shape=%s mode=%s gamma=%.2f opacity=%.2f redline=%s elapsed=%.3fs",
        out.shape,
        cfg.blend_mode.value,
        cfg.lighting_gamma,
        cfg.opacity,
        cfg.assert_base_intact,
        elapsed,
    )
    return out


def _redline_assert(out, template, alpha):
    untouched = alpha == 0.0
    if not np.array_equal(out[untouched], template[untouched]):
        diff = out[untouched] != template[untouched]
        violation_count = int(np.count_nonzero(np.any(diff, axis=-1)))
        rows, cols = np.where(np.any(out != template, axis=-1) & untouched)
        first = [
            (int(r), int(c))
            for r, c in zip(
                rows[:REDLINE_MAX_VIOLATIONS_IN_DETAIL],
                cols[:REDLINE_MAX_VIOLATIONS_IN_DETAIL],
                strict=False,
            )
        ]
        logger.error(
            "redline violated: base image altered outside layer alpha (%d px)",
            violation_count,
        )
        raise BaseImageAlteredError(
            "base image altered outside layer alpha",
            detail={
                "first_violations": first,
                "violation_count": violation_count,
            },
        )
