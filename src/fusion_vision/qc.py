import logging

import cv2
import numpy as np

from .color import delta_e76, lab_from_rgb
from .mask import dilate_mask

logger = logging.getLogger(__name__)


def check_overflow(out, template, mask, tolerance_px=2):
    out = np.asarray(out)
    template = np.asarray(template)
    mask = np.asarray(mask)
    diff = np.any(out != template, axis=-1)
    if int(max(0, tolerance_px)) > 0:
        allowed = dilate_mask(mask, tolerance_px) > 0
    else:
        allowed = mask > 0
    violated = diff & ~allowed
    violated_px = int(np.count_nonzero(violated))
    if violated_px == 0:
        logger.debug("check_overflow clean tolerance_px=%d", tolerance_px)
        return violated_px, None
    ys, xs = np.where(violated)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    logger.debug("check_overflow violated_px=%d bbox=%s", violated_px, bbox)
    return violated_px, bbox


def check_blank(out, template):
    out = np.asarray(out)
    template = np.asarray(template)
    gray = out.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    std = float(gray.std())
    diff_px = int(np.count_nonzero(np.any(out != template, axis=-1)))
    logger.debug("check_blank std=%.3f diff_px=%d", std, diff_px)
    return std, diff_px


def check_color(source_rgb, printed_rgb, mask):
    source = np.asarray(source_rgb, dtype=np.uint8)
    printed = np.asarray(printed_rgb, dtype=np.uint8)
    mask = np.asarray(mask)
    if source.shape[:2] != printed.shape[:2]:
        flag = cv2.INTER_AREA if source.size > printed.size else cv2.INTER_LINEAR
        source = cv2.resize(source, (printed.shape[1], printed.shape[0]), interpolation=flag)
        logger.debug("check_color resized source to %s", printed.shape[:2])
    inside = mask > 0
    if not inside.any():
        logger.warning("check_color: empty mask, returning zeros")
        return 0.0, 0.0
    delta = delta_e76(lab_from_rgb(source), lab_from_rgb(printed))[inside]
    mean = float(delta.mean())
    p95 = float(np.percentile(delta, 95))
    logger.debug("check_color delta_e_mean=%.3f p95=%.3f", mean, p95)
    return mean, p95


def check_sharpness(image, mask=None):
    arr = np.asarray(image)
    gray = arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    if mask is not None:
        inside = np.asarray(mask) > 0
        if not inside.any():
            logger.warning("check_sharpness: empty mask, variance over full image")
            return float(lap.var())
        var = float(lap[inside].var())
    else:
        var = float(lap.var())
    logger.debug("check_sharpness laplacian_var=%.2f", var)
    return var
