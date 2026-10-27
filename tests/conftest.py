import numpy as np
import pytest

from fusion_vision import rasterize_polygon


def make_template(kind, size=(256, 256), seed=7):
    h, w = size
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


def make_layer(size=(96, 96), seed=11):
    h, w = size
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


def make_mask(canvas_shape, shrink=0):
    h, w = canvas_shape[:2]
    margin = 48 + shrink
    points = (
        (margin, margin),
        (w - margin, margin),
        (w - margin, h - margin),
        (margin, h - margin),
    )
    return rasterize_polygon(canvas_shape, points)


def make_empty_mask(canvas_shape):
    h, w = canvas_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2, w // 2] = 255
    return mask


def base_cfg(canvas_shape, **overrides):
    h, w = canvas_shape[:2]
    quad = ((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48))
    return quad, overrides


@pytest.fixture(scope="session")
def canvas_shape():
    return (256, 256, 3)


@pytest.fixture(scope="session")
def flat_pair():
    return make_template("flat"), make_layer(), make_mask((256, 256, 3))


def fixture_case(geometry, blend_mode, gamma, seed=7):
    template = make_template(geometry, seed=seed)
    mask = make_mask(template.shape)
    h, w = template.shape[:2]
    quad = ((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48))
    from fusion_vision import BlendMode, ComposeConfig

    cfg = ComposeConfig(
        dst_quad=quad,
        blend_mode=BlendMode[blend_mode],
        lighting_gamma=gamma,
        sigma=8.0,
    )
    return template, make_layer(), mask, cfg
