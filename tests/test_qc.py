import numpy as np
from conftest import make_mask, make_template

import fusion_vision as fv


def test_check_overflow_clean():
    template = make_template("flat", size=(64, 64))
    mask = make_mask(template.shape)
    out = template.copy()
    violated, bbox = fv.qc.check_overflow(out, template, mask)
    assert violated == 0
    assert bbox is None


def test_v15_overflow_injected_outside_mask():
    template = make_template("flat", size=(64, 64))
    mask = make_mask(template.shape)
    out = template.copy()
    out[0, 0] = (1, 2, 3)
    violated, bbox = fv.qc.check_overflow(out, template, mask, tolerance_px=2)
    assert violated >= 1
    assert bbox == (0, 0, 0, 0)


def test_check_overflow_inside_mask_tolerated():
    template = make_template("flat", size=(128, 128))
    mask = make_mask(template.shape)
    out = template.copy()
    out[64, 64] = (1, 2, 3)
    violated, _ = fv.qc.check_overflow(out, template, mask)
    assert violated == 0


def test_check_blank_flat_output():
    template = make_template("flat", size=(64, 64))
    out = np.full((64, 64, 3), 128, dtype=np.uint8)
    std, diff_px = fv.qc.check_blank(out, template)
    assert std < 1.0
    assert diff_px == 64 * 64


def test_check_blank_structured_output():
    template = make_template("flat", size=(64, 64))
    out = make_template("wrinkle", size=(64, 64))
    std, diff_px = fv.qc.check_blank(out, template)
    assert std > 5.0
    assert diff_px > 0


def test_v14_check_color_identical_is_zero():
    source = np.full((32, 32, 3), [60, 180, 90], dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    mean, p95 = fv.qc.check_color(source, source.copy(), mask)
    assert mean == 0.0
    assert p95 == 0.0


def test_v14_check_color_small_shift_low_delta_e():
    source = np.full((32, 32, 3), [60, 180, 90], dtype=np.uint8)
    printed = np.full((32, 32, 3), [62, 180, 90], dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    mean, p95 = fv.qc.check_color(source, printed, mask)
    assert 0.0 < mean < 5.0
    assert p95 < 5.0


def test_v14_check_color_big_shift_high_delta_e():
    source = np.full((32, 32, 3), [60, 180, 90], dtype=np.uint8)
    printed = np.full((32, 32, 3), [200, 20, 20], dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    mean, _ = fv.qc.check_color(source, printed, mask)
    assert mean > 10.0


def test_check_color_size_mismatch_resizes():
    source = np.full((64, 64, 3), [60, 180, 90], dtype=np.uint8)
    printed = np.full((32, 32, 3), [62, 180, 90], dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    mean, _ = fv.qc.check_color(source, printed, mask)
    assert 0.0 < mean < 5.0


def test_check_color_empty_mask():
    source = np.full((8, 8, 3), 100, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mean, p95 = fv.qc.check_color(source, source.copy(), mask)
    assert mean == 0.0 and p95 == 0.0


def test_check_sharpness_sharp_vs_blurred():
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    blurred = fv.feather(sharp, 3.0)
    var_sharp = fv.qc.check_sharpness(sharp)
    var_blur = fv.qc.check_sharpness(blurred)
    assert var_sharp > 1000.0
    assert var_blur < var_sharp / 10.0


def test_check_sharpness_respects_mask():
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    # sharp content kept 4px away from the mask edge so the 3x3 Laplacian
    # kernel cannot bleed content into the outside region
    image[20:44, 20:44] = sharp[20:44, 20:44]
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    inside = fv.qc.check_sharpness(image, mask)
    outside = fv.qc.check_sharpness(image, 255 - mask)
    assert inside > 100.0
    assert outside < 1.0
