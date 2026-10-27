import dataclasses

import numpy as np
import pytest
from conftest import make_layer, make_mask, make_template

import fusion_vision as fv

CANVAS = (64, 64, 3)
FULL_QUAD = ((8, 8), (56, 8), (56, 56), (8, 56))


def _base():
    return make_template("flat", size=(64, 64)), make_layer(size=(48, 48)), make_mask(CANVAS)


def test_v10_collinear_quad():
    template, layer, mask = _base()
    cfg = fv.ComposeConfig(dst_quad=((8, 8), (56, 8), (56, 8), (8, 56)))
    with pytest.raises(fv.DegenerateQuadError):
        fv.compose(template, layer, mask, cfg)


def test_v10_self_intersecting_quad():
    template, layer, mask = _base()
    cfg = fv.ComposeConfig(dst_quad=((8, 8), (8, 56), (56, 8), (56, 56)))
    with pytest.raises(fv.DegenerateQuadError):
        fv.compose(template, layer, mask, cfg)


def test_v10_zero_area_quad():
    template, layer, mask = _base()
    cfg = fv.ComposeConfig(dst_quad=((10, 10), (10, 10), (10, 10), (10, 10)))
    with pytest.raises(fv.DegenerateQuadError):
        fv.compose(template, layer, mask, cfg)


def test_src_quad_degenerate_also_rejected():
    template, layer, mask = _base()
    bad_src = ((0, 0), (48, 0), (48, 0), (0, 48))
    with pytest.raises(fv.DegenerateQuadError):
        fv.warp_to_quad(layer, FULL_QUAD, template.shape, src_quad=bad_src)


def test_template_shape_error():
    _, layer, mask = _base()
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD)
    with pytest.raises(fv.InputShapeError):
        fv.compose(np.zeros((64, 64), np.uint8), layer, mask, cfg)


def test_layer_shape_error():
    template, _, mask = _base()
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD)
    with pytest.raises(fv.InputShapeError):
        fv.compose(template, np.zeros((48, 48, 3), np.uint8), mask, cfg)


def test_mask_shape_mismatch_error():
    template, layer, _ = _base()
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD)
    with pytest.raises(fv.InputShapeError):
        fv.compose(template, layer, np.zeros((32, 32), np.uint8), cfg)


def test_mask_dtype_error():
    template, layer, _ = _base()
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD)
    with pytest.raises(fv.InputShapeError):
        fv.compose(template, layer, np.zeros((64, 64), np.float32), cfg)


def test_quad_count_error():
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=((0, 0), (1, 1), (2, 2)))


def test_opacity_range_error():
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=FULL_QUAD, opacity=1.5)


def test_noise_requires_seed():
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=FULL_QUAD, noise={"strength": 0.05})


def test_noise_seed_none_rejected():
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=FULL_QUAD, noise={"strength": 0.05, "seed": None})


def test_noise_seed_type_rejected():
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=FULL_QUAD, noise={"strength": 0.05, "seed": "42"})


def test_noise_strength_range():
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=FULL_QUAD, noise={"strength": 0.9, "seed": 1})


def test_unsupported_mode():
    template, layer, mask = _base()
    with pytest.raises(fv.UnsupportedModeError):
        fv.blend(template, layer[..., :3].astype(np.float32), layer[..., 3] / 255.0, "screen")


def test_frozen_config_rejects_assignment():
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.opacity = 0.5


def test_v_p0_4_ndarray_quad_immutable():
    quad = np.array([[8, 8], [56, 8], [56, 56], [8, 56]], dtype=np.int32)
    cfg = fv.ComposeConfig(dst_quad=quad)
    quad[0, 0] = 999
    assert cfg.dst_quad[0] == (8.0, 8.0)


def test_config_replace_helper():
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD)
    cfg2 = cfg.copy(opacity=0.5)
    assert cfg2.opacity == 0.5 and cfg.opacity == 1.0


def test_placement_dict_normalized():
    cfg = fv.ComposeConfig(dst_quad=FULL_QUAD, placement={"anchor": "top_left", "scale": 2.0})
    assert cfg.placement.anchor == "top_left"
    with pytest.raises(fv.InputShapeError):
        fv.ComposeConfig(dst_quad=FULL_QUAD, placement={"anchor": "middle"})


def test_cfg_type_guard():
    template, layer, mask = _base()
    with pytest.raises(TypeError):
        fv.compose(template, layer, mask, {"dst_quad": FULL_QUAD})
