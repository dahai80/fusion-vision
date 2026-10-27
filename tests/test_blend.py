import numpy as np
import pytest

import fusion_vision as fv


def _base(value):
    return np.full((1, 1, 3), value, dtype=np.float32)


def test_normal_blend_formula():
    base = _base(100.0)
    layer = _base(200.0)
    alpha = np.full((1, 1), 0.5, dtype=np.float32)
    out = fv.blend(base, layer, alpha, fv.BlendMode.NORMAL)
    assert np.allclose(out, 150.0)


def test_v13_multiply_white_is_invisible():
    base = np.array([[[100.0, 150.0, 200.0]]], dtype=np.float32)
    layer = np.full((1, 1, 3), 255.0, dtype=np.float32)
    alpha = np.ones((1, 1), dtype=np.float32)
    out = fv.blend(base, layer, alpha, fv.BlendMode.MULTIPLY)
    assert np.allclose(out, base)


def test_multiply_partial_alpha():
    base = _base(200.0)
    layer = _base(100.0)
    alpha = np.full((1, 1), 0.5, dtype=np.float32)
    out = fv.blend(base, layer, alpha, fv.BlendMode.MULTIPLY)
    assert np.allclose(out, 200.0 * 0.5 + (200.0 * 100.0 / 255.0) * 0.5)


def test_linear_burn_formula_clips():
    base = np.array([[[100.0, 200.0, 250.0]]], dtype=np.float32)
    layer = np.array([[[100.0, 200.0, 100.0]]], dtype=np.float32)
    alpha = np.ones((1, 1), dtype=np.float32)
    out = fv.blend(base, layer, alpha, fv.BlendMode.LINEAR_BURN)
    assert np.allclose(out, [0.0, 145.0, 95.0])


def test_alpha_zero_leaves_base_untouched():
    base = _base(123.0)
    layer = _base(7.0)
    alpha = np.zeros((1, 1), dtype=np.float32)
    for mode in fv.BlendMode:
        out = fv.blend(base, layer, alpha, mode)
        assert np.allclose(out, base), mode


def test_unsupported_mode_raises():
    with pytest.raises(fv.UnsupportedModeError):
        fv.blend(_base(1.0), _base(1.0), np.ones((1, 1)), "screen")
