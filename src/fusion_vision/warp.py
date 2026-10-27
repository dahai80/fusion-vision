import logging

import cv2
import numpy as np

from ._types import QUAD_EPS_AREA_RATIO, RESAMPLE_AREA_RATIO
from .errors import DegenerateQuadError

logger = logging.getLogger(__name__)


def _signed_area(pts):
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return 0.5 * sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4):
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def check_quad(quad, side, ref_area):
    pts = [(float(p[0]), float(p[1])) for p in quad]
    area = abs(_signed_area(pts))
    eps = abs(ref_area) * QUAD_EPS_AREA_RATIO
    if area <= eps:
        raise DegenerateQuadError(
            f"{side} quad is degenerate (zero area)",
            detail={"side": side, "quad": pts, "area": area},
        )
    for i in range(4):
        if abs(_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])) <= eps:
            raise DegenerateQuadError(
                f"{side} quad has collinear points",
                detail={"side": side, "quad": pts, "area": area},
            )
    if not _segments_intersect(pts[0], pts[2], pts[1], pts[3]):
        raise DegenerateQuadError(
            f"{side} quad is self-intersecting",
            detail={"side": side, "quad": pts, "area": area},
        )
    return area


def compute_homography(src_quad, dst_quad, canvas_shape):
    src_area = check_quad(src_quad, "src", _quad_bbox_area(src_quad))
    check_quad(
        dst_quad,
        "dst",
        max(_quad_bbox_area(dst_quad), float(canvas_shape[0] * canvas_shape[1])),
    )
    src = np.array(src_quad, dtype=np.float32)
    dst = np.array(dst_quad, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst), src_area


def _quad_bbox_area(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def _pick_flag(src_area, dst_area):
    if src_area > dst_area * RESAMPLE_AREA_RATIO:
        return cv2.INTER_AREA
    if dst_area > src_area * RESAMPLE_AREA_RATIO:
        return cv2.INTER_LANCZOS4
    return cv2.INTER_LINEAR


def warp_to_quad(layer, dst_quad, canvas_shape, src_quad=None):
    h, w = canvas_shape[:2]
    if src_quad is None:
        lh, lw = layer.shape[:2]
        src_quad = ((0.0, 0.0), (float(lw), 0.0), (float(lw), float(lh)), (0.0, float(lh)))
    layer_f = layer.astype(np.float32, copy=False) if layer.dtype != np.float32 else layer
    m, src_area = compute_homography(src_quad, dst_quad, canvas_shape)
    dst_area = abs(_signed_area(dst_quad))
    flag = _pick_flag(src_area, dst_area)
    if flag == cv2.INTER_AREA:
        # cv2.warpPerspective does not do true area averaging (INTER_AREA
        # falls back to point sampling and aliases on fine patterns);
        # pre-shrink with cv2.resize INTER_AREA, then warp bilinearly
        f = float(np.sqrt(dst_area / src_area))
        lh, lw = layer_f.shape[:2]
        new_w = max(1, int(np.ceil(lw * f)))
        new_h = max(1, int(np.ceil(lh * f)))
        layer_f = cv2.resize(layer_f, (new_w, new_h), interpolation=cv2.INTER_AREA)
        src_quad = tuple((p[0] * f, p[1] * f) for p in src_quad)
        m, _ = compute_homography(src_quad, dst_quad, canvas_shape)
        flag = cv2.INTER_LINEAR
        logger.debug("warp_to_quad pre-resized layer to (%d, %d) with INTER_AREA", new_w, new_h)
    warped = cv2.warpPerspective(
        layer_f,
        m,
        (int(w), int(h)),
        flags=flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )
    logger.debug("warp_to_quad src_area=%.1f dst_area=%.1f flag=%s", src_area, dst_area, flag)
    return warped


def place(warped, placement, canvas_shape):
    if placement.scale == 1.0:
        return warped
    h, w = canvas_shape[:2]
    if placement.anchor == "center":
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = 0.0, 0.0
    s = placement.scale
    m = np.array([[s, 0.0, cx * (1.0 - s)], [0.0, s, cy * (1.0 - s)]], dtype=np.float32)
    flag = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
    placed = cv2.warpAffine(
        warped,
        m,
        (int(w), int(h)),
        flags=flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )
    logger.debug("place scale=%.3f anchor=%s", s, placement.anchor)
    return placed
