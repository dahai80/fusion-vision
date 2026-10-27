import numpy as np
import pytest
from conftest import make_layer

import fusion_vision as fv


def test_v6_identity_quad_equals_direct_paste():
    template = np.full((64, 64, 3), 120, dtype=np.uint8)
    layer = make_layer(size=(32, 32))
    mask = np.full((64, 64), 255, dtype=np.uint8)
    quad = ((16, 16), (48, 16), (48, 48), (16, 48))
    cfg = fv.ComposeConfig(dst_quad=quad, lighting_gamma=0.0)
    out = fv.compose(template, layer, mask, cfg)
    diff = np.abs(out[16:48, 16:48].astype(np.int32) - layer[..., :3].astype(np.int32))
    assert diff.max() <= 1


def test_v7_known_homography_corner_accuracy():
    layer = make_layer(size=(40, 40))
    quad = ((10.0, 12.0), (50.0, 12.0), (50.0, 44.0), (10.0, 44.0))
    warped = fv.warp_to_quad(layer, quad, (64, 64, 3))
    assert warped.shape == (64, 64, 4)
    # quad (10,12)-(50,44) covers pixels x in [10,50), y in [12,44)
    for x, y in [(10, 12), (49, 12), (49, 43), (10, 43), (30, 28)]:
        assert warped[y, x, 3] >= 254.0
    # bilinear border interpolation gives partial alpha up to 1px past the
    # quad edge, so outside points need 2px clearance
    for x, y in [(8, 28), (51, 28), (30, 11), (30, 46)]:
        assert warped[y, x, 3] == 0


def test_v8_downscale_uses_area_average():
    layer = np.zeros((256, 256, 4), dtype=np.uint8)
    layer[:, ::2, 0:3] = 255
    layer[..., 3] = 255
    quad = ((0.0, 0.0), (64.0, 0.0), (64.0, 64.0), (0.0, 64.0))
    warped = fv.warp_to_quad(layer, quad, (64, 64, 3))
    region = warped[..., 0].astype(np.float32)
    assert abs(region.mean() - 127.5) < 8
    assert region.std() < 12


def test_v9_upscale_keeps_text_edges_sharp():
    layer = np.zeros((16, 16, 4), dtype=np.uint8)
    layer[..., 0:3] = 60
    layer[:, 8:, 0:3] = 240
    layer[..., 3] = 255
    quad = ((0.0, 0.0), (128.0, 0.0), (128.0, 128.0), (0.0, 128.0))
    warped = fv.warp_to_quad(layer, quad, (128, 128, 3))
    assert warped[64, 48, 0] < 100
    assert warped[64, 80, 0] > 200


def test_offcanvas_quad_border_fill_transparent():
    layer = make_layer(size=(32, 32))
    quad = ((48.0, 48.0), (96.0, 48.0), (96.0, 96.0), (48.0, 96.0))
    warped = fv.warp_to_quad(layer, quad, (64, 64, 3))
    # far from the quad: exact border fill; near the edge a ~8px lanczos
    # kernel bleed is expected OpenCV behavior
    assert np.all(warped[:40, :, 3] == 0)
    assert np.all(warped[:, :40, 3] == 0)
    assert warped[56, 56, 3] == 255


def test_compute_homography_identity():
    quad = ((0.0, 0.0), (32.0, 0.0), (32.0, 32.0), (0.0, 32.0))
    m, area = fv.compute_homography(quad, quad, (64, 64, 3))
    assert area == pytest.approx(1024.0)
    assert np.allclose(m, np.eye(3), atol=1e-5)


def test_place_identity_fastpath():
    warped = np.full((64, 64, 4), 128, dtype=np.float32)
    assert fv.place(warped, fv.Placement(), (64, 64, 3)) is warped
