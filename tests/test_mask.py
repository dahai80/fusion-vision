import numpy as np
from conftest import make_layer

import fusion_vision as fv


def test_apply_mask_scales_alpha():
    layer = make_layer(size=(16, 16))
    mask = np.full((16, 16), 128, dtype=np.uint8)
    out = fv.apply_mask(layer, mask)
    assert out.dtype == np.float32
    assert np.allclose(out[..., 3], 255.0 * 128 / 255.0)
    assert np.array_equal(out[..., :3], layer[..., :3])


def test_v11_feather_produces_soft_edge():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 32:] = 255
    soft = fv.feather(mask, 4.0)
    assert soft.dtype == np.uint8
    assert np.all(soft[:, :20] == 0)
    assert np.all(soft[:, 50:] == 255)
    boundary = soft[:, 20:50]
    assert np.any((boundary > 0) & (boundary < 255))


def test_feather_zero_radius_is_copy():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    out = fv.feather(mask, 0)
    assert np.array_equal(out, mask)


def test_rasterize_polygon_fills_interior():
    mask = fv.rasterize_polygon((64, 64, 3), ((8, 8), (56, 8), (56, 56), (8, 56)))
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert np.all(mask[20:44, 20:44] == 255)
    assert np.all(mask[:5, :5] == 0)


def test_rasterize_polygon_with_feather():
    hard = fv.rasterize_polygon((64, 64, 3), ((8, 8), (56, 8), (56, 56), (8, 56)))
    soft = fv.rasterize_polygon((64, 64, 3), ((8, 8), (56, 8), (56, 56), (8, 56)), feather_px=3.0)
    assert np.any(soft < hard)
    assert np.all(soft[28:36, 28:36] == 255)
