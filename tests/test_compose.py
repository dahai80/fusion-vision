import numpy as np
from conftest import fixture_case

import fusion_vision as fv


def test_v4_determinism_three_runs():
    template, layer, mask, cfg = fixture_case("wrinkle", "NORMAL", 0.6)
    outs = [fv.compose(template, layer, mask, cfg) for _ in range(3)]
    assert np.array_equal(outs[0], outs[1])
    assert np.array_equal(outs[1], outs[2])


def test_determinism_with_seeded_noise():
    template, layer, mask, cfg = fixture_case("flat", "MULTIPLY", 0.6)
    cfg = cfg.copy(noise={"strength": 0.05, "seed": 7})
    outs = [fv.compose(template, layer, mask, cfg) for _ in range(2)]
    assert np.array_equal(outs[0], outs[1])


def test_different_noise_seed_changes_output():
    template, layer, mask, cfg = fixture_case("flat", "NORMAL", 0.6)
    out_a = fv.compose(template, layer, mask, cfg.copy(noise={"strength": 0.1, "seed": 1}))
    out_b = fv.compose(template, layer, mask, cfg.copy(noise={"strength": 0.1, "seed": 2}))
    assert not np.array_equal(out_a, out_b)


def test_feather_path_end_to_end():
    template, layer, mask, cfg = fixture_case("curve", "NORMAL", 0.6)
    out = fv.compose(template, layer, mask, cfg.copy(feather_px=4.0))
    assert out.shape == template.shape
    assert out.dtype == np.uint8


def test_placement_scale_path_end_to_end():
    template, layer, mask, cfg = fixture_case("curve", "NORMAL", 0.6)
    out = fv.compose(
        template,
        layer,
        mask,
        cfg.copy(placement={"anchor": "center", "scale": 0.5}),
    )
    assert out.shape == template.shape
    assert not np.array_equal(out, template)


def test_all_geometry_blend_gamma_matrix_runs():
    for geometry in ("flat", "curve", "wrinkle"):
        for mode in ("NORMAL", "MULTIPLY", "LINEAR_BURN"):
            for gamma in (0.0, 0.6):
                template, layer, mask, cfg = fixture_case(geometry, mode, gamma)
                out = fv.compose(template, layer, mask, cfg)
                assert out.shape == template.shape
                assert out.dtype == np.uint8
