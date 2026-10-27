import logging
import sys

import numpy as np
import pytest

import fusion_vision as fv

compose_module = sys.modules["fusion_vision.compose"]


def _compose(template, layer, mask, cfg):
    return fv.compose(template, layer, mask, cfg)


def test_v1_untouched_region_bit_identical(flat_pair):
    template, layer, mask = flat_pair
    h, w = template.shape[:2]
    cfg = fv.ComposeConfig(dst_quad=((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48)))
    out = _compose(template, layer, mask, cfg)
    assert np.array_equal(out[mask == 0], template[mask == 0])
    assert out.dtype == np.uint8
    assert out.shape == template.shape


def test_v2_mutation_is_intercepted(flat_pair, monkeypatch):
    template, layer, mask = flat_pair
    h, w = template.shape[:2]
    cfg = fv.ComposeConfig(dst_quad=((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48)))

    def evil_blend(base, layer_rgb, alpha, mode):
        result = fv.blend(base, layer_rgb, alpha, mode)
        result[0, 0] = (1, 2, 3)
        return result

    monkeypatch.setattr(compose_module, "blend", evil_blend)
    with pytest.raises(fv.BaseImageAlteredError) as excinfo:
        _compose(template, layer, mask, cfg)
    assert excinfo.value.detail["violation_count"] >= 1
    assert (0, 0) in excinfo.value.detail["first_violations"]


def test_v5_assert_off_warns_and_completes(flat_pair, caplog):
    template, layer, mask = flat_pair
    h, w = template.shape[:2]
    cfg = fv.ComposeConfig(
        dst_quad=((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48)),
        assert_base_intact=False,
    )
    with caplog.at_level(logging.WARNING, logger="fusion_vision.compose"):
        out = _compose(template, layer, mask, cfg)
    assert out.shape == template.shape
    assert any("assert_base_intact=False" in r.message for r in caplog.records)


def test_redline_worst_path_full_canvas_untouched(flat_pair):
    template, layer, mask = flat_pair
    h, w = template.shape[:2]
    empty = np.zeros_like(mask)
    cfg = fv.ComposeConfig(dst_quad=((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48)))
    out = _compose(template, layer, empty, cfg)
    assert np.array_equal(out, template)


def test_opacity_zero_returns_template(flat_pair):
    template, layer, mask = flat_pair
    h, w = template.shape[:2]
    cfg = fv.ComposeConfig(
        dst_quad=((48, 48), (w - 48, 48), (w - 48, h - 48), (48, h - 48)),
        opacity=0.0,
    )
    out = _compose(template, layer, mask, cfg)
    assert np.array_equal(out, template)
