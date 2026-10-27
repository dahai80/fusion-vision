import logging

import numpy as np

from .errors import InputShapeError

logger = logging.getLogger(__name__)

IO_MAX_DIMS = 4096
RESAMPLE_AREA_RATIO = 1.15
QUAD_EPS_AREA_RATIO = 1e-6
GRAIN_STRENGTH_MAX = 0.5
REDLINE_MAX_VIOLATIONS_IN_DETAIL = 10
LUMINANCE_MIN_MASK_PIXELS = 3
GOLDEN_MAX_ABS_DIFF = 1


def validate_template(template):
    arr = np.asarray(template)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise InputShapeError(
            "template must be (H, W, 3)",
            detail={"which": "template", "expected": "(H, W, 3)", "actual": str(arr.shape)},
        )
    if arr.dtype != np.uint8:
        raise InputShapeError(
            "template must be uint8",
            detail={"which": "template", "expected": "uint8", "actual": str(arr.dtype)},
        )
    return arr


def validate_layer(layer):
    arr = np.asarray(layer)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise InputShapeError(
            "layer must be (h, w, 4) RGBA",
            detail={"which": "layer", "expected": "(h, w, 4)", "actual": str(arr.shape)},
        )
    if arr.dtype != np.uint8:
        raise InputShapeError(
            "layer must be uint8",
            detail={"which": "layer", "expected": "uint8", "actual": str(arr.dtype)},
        )
    return arr


def validate_mask(mask, template_shape):
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise InputShapeError(
            "mask must be (H, W)",
            detail={"which": "mask", "expected": "(H, W)", "actual": str(arr.shape)},
        )
    if arr.dtype != np.uint8:
        raise InputShapeError(
            "mask must be uint8",
            detail={"which": "mask", "expected": "uint8", "actual": str(arr.dtype)},
        )
    if arr.shape != template_shape[:2]:
        raise InputShapeError(
            "mask shape must match template canvas",
            detail={
                "which": "mask",
                "expected": str(template_shape[:2]),
                "actual": str(arr.shape),
            },
        )
    return arr


def to_float32(arr):
    return np.asarray(arr).astype(np.float32, copy=True)
