import logging

import numpy as np

from .config import BlendMode
from .errors import UnsupportedModeError

logger = logging.getLogger(__name__)


def blend(base, layer_rgb, alpha, mode):
    base_f = np.asarray(base, dtype=np.float32)
    layer_f = np.asarray(layer_rgb, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)[..., None]
    if mode == BlendMode.NORMAL:
        mixed = layer_f
    elif mode == BlendMode.MULTIPLY:
        mixed = base_f * layer_f / 255.0
    elif mode == BlendMode.LINEAR_BURN:
        mixed = np.clip(base_f + layer_f - 255.0, 0.0, 255.0)
    else:
        raise UnsupportedModeError(
            f"unsupported blend mode: {mode!r}",
            detail={"mode": str(mode), "supported": [m.value for m in BlendMode]},
        )
    out = base_f * (1.0 - a) + mixed * a
    logger.debug("blend mode=%s alpha_mean=%.3f", mode, float(a.mean()))
    return out
