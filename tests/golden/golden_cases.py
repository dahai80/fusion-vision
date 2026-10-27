import io
import platform
import sys

import cv2
import numpy as np

import fusion_vision as fv

GEOMETRIES = ("flat", "curve", "wrinkle")
MODES = ("NORMAL", "MULTIPLY", "LINEAR_BURN")
GAMMAS = (0.6, 1.0)
NOISE_CASE = ("flat-normal-g0.6-noise", "flat", "NORMAL", 0.6, {"strength": 0.05, "seed": 7})
SIZE = (256, 256)
MARGIN = 48


def platform_key():
    cv_ver = ".".join(cv2.__version__.split(".")[:2])
    np_ver = ".".join(np.__version__.split(".")[:2])
    return f"{sys.platform}-{platform.machine()}-cv{cv_ver}-np{np_ver}"


def case_list():
    cases = []
    for geometry in GEOMETRIES:
        for mode in MODES:
            for gamma in GAMMAS:
                name = f"{geometry}-{mode.lower()}-g{gamma}"
                cases.append((name, geometry, mode, gamma, None))
    cases.append(NOISE_CASE)
    return cases


def _make_template(kind, seed=7):
    h, w = SIZE
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "flat":
        base = 140.0 + 40.0 * (xx / w)
    elif kind == "curve":
        base = 140.0 + 50.0 * np.sin(2.0 * np.pi * xx / w)
    elif kind == "wrinkle":
        base = 140.0 + 50.0 * np.sin(0.35 * xx) * np.cos(0.27 * yy)
    else:
        raise ValueError(f"unknown geometry kind: {kind}")
    rgb = np.stack([base, base * 0.95, base * 0.9], axis=-1)
    rgb += rng.normal(0, 1.5, rgb.shape).astype(np.float32)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _make_layer(seed=11):
    h, w = 96, 96
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0] = 60
    layer[..., 1] = 180
    layer[..., 2] = 90
    layer[..., 3] = 255
    for i in range(0, h, 16):
        layer[i : i + 6, :, 0:3] = 240
    for j in range(0, w, 24):
        layer[:, j : j + 6, 0:3] = 30
    rng = np.random.default_rng(seed)
    ys, xs = rng.integers(0, h, 12), rng.integers(0, w, 12)
    layer[ys, xs, 0:3] = 255
    return layer


def _make_mask():
    h, w = SIZE
    return fv.rasterize_polygon(
        (h, w, 3),
        ((MARGIN, MARGIN), (w - MARGIN, MARGIN), (w - MARGIN, h - MARGIN), (MARGIN, h - MARGIN)),
    )


def build_case(name, geometry, mode, gamma, noise):
    template = _make_template(geometry)
    layer = _make_layer()
    mask = _make_mask()
    h, w = SIZE
    cfg = fv.ComposeConfig(
        dst_quad=(
            (MARGIN, MARGIN),
            (w - MARGIN, MARGIN),
            (w - MARGIN, h - MARGIN),
            (MARGIN, h - MARGIN),
        ),
        blend_mode=fv.BlendMode[mode],
        lighting_gamma=gamma,
        sigma=8.0,
        noise=noise,
    )
    return fv.compose(template, layer, mask, cfg)


def png_bytes(out):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()
