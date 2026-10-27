import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_mask(layer_rgba, mask):
    out = layer_rgba.astype(np.float32, copy=True)
    out[..., 3] *= np.asarray(mask, dtype=np.float32) / 255.0
    logger.debug("apply_mask done")
    return out


def feather(mask, radius):
    if radius <= 0:
        return np.asarray(mask).copy()
    radius = float(radius)
    ksize = int(radius * 2) | 1
    if ksize < 1:
        ksize = 1
    blurred = cv2.GaussianBlur(np.asarray(mask), (ksize, ksize), radius / 2.0)
    logger.debug("feather radius=%.2f ksize=%d", radius, ksize)
    return blurred


def rasterize_polygon(shape, points, feather_px=0.0):
    h, w = shape[:2]
    pts = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    hard = np.zeros((int(h), int(w)), dtype=np.uint8)
    cv2.fillPoly(hard, [pts], 255)
    if feather_px > 0:
        hard = feather(hard, feather_px)
    logger.debug("rasterize_polygon n_points=%d feather_px=%.2f", len(points), feather_px)
    return hard
