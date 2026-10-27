import logging

import cv2
import numpy as np

from ._types import GRAIN_STRENGTH_MAX, LUMINANCE_MIN_MASK_PIXELS

logger = logging.getLogger(__name__)

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luminance_field(template, mask, sigma=15.0):
    template_f = np.asarray(template, dtype=np.float32)
    gray = template_f @ _LUMA
    ksize = int(sigma * 6 + 1) | 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    inside = np.asarray(mask) > 0
    values = blurred[inside]
    if values.size < LUMINANCE_MIN_MASK_PIXELS:
        logger.warning(
            "luminance_field: mask has %d in-mask pixels (< %d); L=1.0, no lighting adaptation",
            values.size,
            LUMINANCE_MIN_MASK_PIXELS,
        )
        return np.ones(gray.shape, dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 0.0:
        logger.warning("luminance_field: zero dynamic range in mask (min=max=%.2f); L=1.0", vmin)
        return np.ones(gray.shape, dtype=np.float32)
    field = (blurred - vmin) / (vmax - vmin)
    field = np.clip(field, 0.0, 1.0).astype(np.float32)
    logger.debug("luminance_field sigma=%.2f range=[%.2f, %.2f]", sigma, vmin, vmax)
    return field


def adapt(layer_rgb, field, gamma=0.6):
    if gamma == 0.0:
        return layer_rgb
    factor = np.power(field, np.float32(gamma))[..., None]
    out = layer_rgb.astype(np.float32, copy=False) * factor
    logger.debug("adapt gamma=%.2f", gamma)
    return out


def grain(shape, seed, strength):
    if not (0 < strength <= GRAIN_STRENGTH_MAX):
        raise ValueError(f"strength must be in (0, {GRAIN_STRENGTH_MAX}]")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((int(shape[0]), int(shape[1]), 1)).astype(np.float32)
    factor = np.clip(1.0 + strength * noise, 1.0 - strength, 1.0 + strength)
    logger.debug("grain seed=%d strength=%.3f", seed, strength)
    return factor
