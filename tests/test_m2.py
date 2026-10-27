import numpy as np
import pytest
from conftest import make_layer, make_mask, make_template

import fusion_vision as fv


def test_dilate_mask_expands_region():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[16, 16] = 255
    out = fv.dilate_mask(mask, 2)
    assert out.dtype == np.uint8
    assert out[16, 16] == 255
    assert out[14, 16] == 255
    assert out[18, 18] == 255
    assert out[13, 16] == 0
    assert np.count_nonzero(out) == 25  # 5x5 square


def test_dilate_mask_zero_radius_binarizes():
    mask = np.full((16, 16), 77, dtype=np.uint8)
    mask[0, 0] = 0
    out = fv.dilate_mask(mask, 0)
    assert set(np.unique(out)) == {0, 255}
    assert out[0, 0] == 0
    assert out[8, 8] == 255


def test_check_overflow_uses_dilate_mask():
    template = make_template("flat", size=(64, 64))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 32] = 255
    out = template.copy()
    out[30, 32] = (1, 2, 3)  # 2px away from mask, within tolerance_px=2
    violated, bbox = fv.qc.check_overflow(out, template, mask, tolerance_px=2)
    assert violated == 0
    out[27, 32] = (1, 2, 3)  # 5px away, outside tolerance
    violated, bbox = fv.qc.check_overflow(out, template, mask, tolerance_px=2)
    assert violated >= 1


def _two_layer_setup():
    template = make_template("flat", size=(128, 128))
    h, w = template.shape[:2]
    mask1 = make_mask(template.shape, shrink=32)
    mask2 = make_mask(template.shape, shrink=16)
    quad1 = ((32, 32), (w - 32, 32), (w - 32, h - 32), (32, h - 32))
    quad2 = ((16, 16), (w - 16, 16), (w - 16, h - 16), (16, h - 16))
    cfg1 = fv.ComposeConfig(dst_quad=quad1, lighting_gamma=0.0)
    cfg2 = fv.ComposeConfig(dst_quad=quad2, lighting_gamma=0.0)
    return (
        template,
        make_layer(size=(64, 64)),
        mask1,
        cfg1,
        make_layer(size=(64, 64), seed=12),
        mask2,
        cfg2,
    )


def test_compose_layers_matches_sequential_compose():
    template, l1, m1, c1, l2, m2, c2 = _two_layer_setup()
    expected = fv.compose(fv.compose(template, l1, m1, c1), l2, m2, c2)
    out = fv.compose_layers(template, [l1, l2], [m1, m2], [c1, c2])
    assert np.array_equal(out, expected)
    assert out.dtype == np.uint8


def test_compose_layers_deterministic():
    template, l1, m1, c1, l2, m2, c2 = _two_layer_setup()
    a = fv.compose_layers(template, [l1, l2], [m1, m2], [c1, c2])
    b = fv.compose_layers(template, [l1, l2], [m1, m2], [c1, c2])
    assert np.array_equal(a, b)


def test_compose_layers_redline_holds_per_layer():
    template, l1, m1, c1, l2, m2, c2 = _two_layer_setup()
    out = fv.compose_layers(template, [l1, l2], [m1, m2], [c1, c2])
    # outer region untouched by both layers stays bit-identical
    both_zero = (m1 == 0) & (m2 == 0)
    assert np.array_equal(out[both_zero], template[both_zero])


def test_compose_layers_length_mismatch_rejected():
    template, l1, m1, c1, _, _, _ = _two_layer_setup()
    with pytest.raises(fv.InputShapeError):
        fv.compose_layers(template, [l1], [m1, m1], [c1])
    with pytest.raises(fv.InputShapeError):
        fv.compose_layers(template, [], [], [])


def test_compose_config_accepts_blend_mode_str():
    quad = ((16, 16), (112, 16), (112, 112), (16, 112))
    cfg = fv.ComposeConfig(dst_quad=quad, blend_mode="multiply", lighting_gamma=0.0)
    assert cfg.blend_mode is fv.BlendMode.MULTIPLY
    cfg = fv.ComposeConfig(dst_quad=quad, blend_mode="LINEAR_BURN", lighting_gamma=0.0)
    assert cfg.blend_mode is fv.BlendMode.LINEAR_BURN
    assert (
        fv.compose(
            make_template("flat", size=(128, 128)),
            make_layer(size=(64, 64)),
            make_mask((128, 128, 3), shrink=16),
            cfg,
        ).dtype
        == np.uint8
    )


def test_compose_config_rejects_unknown_blend_mode_str():
    quad = ((16, 16), (112, 16), (112, 112), (16, 112))
    with pytest.raises(fv.InputShapeError) as exc_info:
        fv.ComposeConfig(dst_quad=quad, blend_mode="screen")
    assert exc_info.value.detail["which"] == "blend_mode"
