import logging

import numpy as np

import fusion_vision as fv


def _half_template(h=64, w=128):
    template = np.full((h, w, 3), 220, dtype=np.uint8)
    template[:, w // 2 :] = 40
    return template


def test_v12_dark_region_attenuated():
    template = _half_template()
    mask = np.full(template.shape[:2], 255, dtype=np.uint8)
    field = fv.luminance_field(template, mask, sigma=8.0)
    assert field.shape == template.shape[:2]
    assert np.all(field[:, 16] > 0.8)
    assert np.all(field[:, 112] < 0.2)
    layer = np.full((*template.shape[:2], 3), 200.0, dtype=np.float32)
    adapted = fv.adapt(layer, field, gamma=0.6)
    # brightest in-mask region normalizes to L=1.0: no attenuation there
    assert np.allclose(adapted[:, 16], layer[:, 16])
    # darkest region attenuates hard
    assert np.all(adapted[:, 112] < layer[:, 112] * 0.5)
    assert np.all(adapted[:, 16] > adapted[:, 112])


def test_luminance_field_small_mask_returns_ones(caplog):
    template = _half_template()
    mask = np.zeros(template.shape[:2], dtype=np.uint8)
    mask[0, 0] = 255
    mask[0, 1] = 255
    with caplog.at_level(logging.WARNING, logger="fusion_vision.lighting"):
        field = fv.luminance_field(template, mask)
    assert np.all(field == 1.0)
    assert any("in-mask pixels" in r.message for r in caplog.records)


def test_luminance_field_flat_template_returns_ones(caplog):
    template = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    with caplog.at_level(logging.WARNING, logger="fusion_vision.lighting"):
        field = fv.luminance_field(template, mask)
    assert np.all(field == 1.0)
    assert any("zero dynamic range" in r.message for r in caplog.records)


def test_adapt_gamma_zero_is_identity():
    layer = np.full((8, 8, 3), 200.0, dtype=np.float32)
    field = np.zeros((8, 8), dtype=np.float32)
    assert fv.adapt(layer, field, gamma=0.0) is layer


def test_grain_deterministic_and_bounded():
    g1 = fv.grain((32, 32), seed=42, strength=0.1)
    g2 = fv.grain((32, 32), seed=42, strength=0.1)
    g3 = fv.grain((32, 32), seed=43, strength=0.1)
    assert np.array_equal(g1, g2)
    assert not np.array_equal(g1, g3)
    assert g1.shape == (32, 32, 1)
    assert np.all(g1 >= 0.9 - 1e-6) and np.all(g1 <= 1.1 + 1e-6)


def test_grain_rejects_bad_strength():
    for bad in (0.0, -0.1, 0.9):
        try:
            fv.grain((4, 4), seed=1, strength=bad)
        except ValueError:
            continue
        raise AssertionError(f"strength={bad} should raise ValueError")
