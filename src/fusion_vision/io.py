import logging
import os

import cv2
import numpy as np
from PIL import Image

from ._types import IO_MAX_DIMS
from .errors import InputShapeError

logger = logging.getLogger(__name__)


def _decode(path, max_dims):
    if not os.path.isfile(path):
        raise InputShapeError(
            "image file not found",
            detail={"which": "path", "actual": str(path)},
        )
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise InputShapeError(
            "image decode failed (corrupt or not a valid image)",
            detail={"which": "path", "actual": str(path)},
        )
    h, w = img.shape[:2]
    if max(h, w) > max_dims:
        raise InputShapeError(
            "image exceeds max dimensions",
            detail={
                "which": "path",
                "actual": f"{w}x{h}",
                "expected": f"max side <= {max_dims}",
            },
        )
    return img


def imread_rgb(path, max_dims=IO_MAX_DIMS):
    img = _decode(path, max_dims)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        logger.warning(
            "imread_rgb: %s has an alpha channel which will be dropped; "
            "use imread_rgba for layer assets",
            path,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.debug("imread_rgb %s shape=%s", path, rgb.shape)
    return rgb


def imread_rgba(path, max_dims=IO_MAX_DIMS):
    img = _decode(path, max_dims)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        bgra = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        logger.debug("imread_rgba %s expanded 3ch -> 4ch", path)
        return rgba
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    logger.debug("imread_rgba %s shape=%s", path, rgba.shape)
    return rgba


def imread_gray(path, max_dims=IO_MAX_DIMS):
    img = _decode(path, max_dims)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.debug("imread_gray %s shape=%s", path, img.shape)
    return img


def imwrite_png(path, arr, dpi=None):
    arr = np.asarray(arr)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))) or ".", exist_ok=True)
    image = Image.fromarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        image = Image.fromarray(arr, mode="RGBA")
    kwargs = {"dpi": (dpi, dpi)} if dpi is not None else {}
    image.save(str(path), format="PNG", **kwargs)
    logger.debug("imwrite_png %s shape=%s dpi=%s", path, arr.shape, dpi)
    return str(path)
