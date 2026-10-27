import logging

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0
_M = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab_from_rgb(rgb):
    arr = np.asarray(rgb, dtype=np.float64)
    linear = _srgb_to_linear(arr)
    xyz = linear @ _M.T
    f = np.where(xyz / _WHITE > _EPS, np.cbrt(xyz / _WHITE), (_KAPPA * xyz / _WHITE + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)
    logger.debug("lab_from_rgb shape=%s", lab.shape)
    return lab


def delta_e76(lab_a, lab_b):
    d = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    return np.sqrt(np.sum(d * d, axis=-1))
