import logging

import numpy as np
import pytest
from conftest import make_layer, make_template

import fusion_vision as fv


def test_rgb_png_roundtrip(tmp_path):
    template = make_template("flat", size=(32, 32))
    path = str(tmp_path / "rgb.png")
    fv.imwrite_png(path, template)
    loaded = fv.imread_rgb(path)
    assert loaded.shape == (32, 32, 3)
    assert np.array_equal(loaded, template)


def test_rgba_png_roundtrip(tmp_path):
    layer = make_layer(size=(32, 32))
    path = str(tmp_path / "rgba.png")
    fv.imwrite_png(path, layer)
    loaded = fv.imread_rgba(path)
    assert loaded.shape == (32, 32, 4)
    assert np.array_equal(loaded, layer)


def test_imread_rgb_warns_and_drops_alpha(tmp_path, caplog):
    layer = make_layer(size=(16, 16))
    path = str(tmp_path / "alpha.png")
    fv.imwrite_png(path, layer)
    with caplog.at_level(logging.WARNING, logger="fusion_vision.io"):
        loaded = fv.imread_rgb(path)
    assert loaded.shape == (16, 16, 3)
    assert np.array_equal(loaded, layer[..., :3])
    assert any("alpha channel" in r.message for r in caplog.records)


def test_imread_rgba_expands_rgb(tmp_path):
    template = make_template("flat", size=(16, 16))
    path = str(tmp_path / "noalpha.png")
    fv.imwrite_png(path, template)
    loaded = fv.imread_rgba(path)
    assert loaded.shape == (16, 16, 4)
    assert np.array_equal(loaded[..., :3], template)
    assert np.all(loaded[..., 3] == 255)


def test_imread_gray(tmp_path):
    template = make_template("flat", size=(16, 16))
    path = str(tmp_path / "gray.png")
    fv.imwrite_png(path, template)
    gray = fv.imread_gray(path)
    assert gray.ndim == 2
    assert gray.shape == (16, 16)


def test_max_dims_rejected(tmp_path):
    template = make_template("flat", size=(32, 32))
    path = str(tmp_path / "big.png")
    fv.imwrite_png(path, template)
    with pytest.raises(fv.InputShapeError):
        fv.imread_rgb(path, max_dims=16)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(fv.InputShapeError):
        fv.imread_rgb(str(tmp_path / "nope.png"))


def test_corrupt_file_rejected(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not a png at all")
    with pytest.raises(fv.InputShapeError) as excinfo:
        fv.imread_rgb(str(path))
    assert "decode failed" in str(excinfo.value)


def test_imwrite_png_creates_dirs(tmp_path):
    template = make_template("flat", size=(8, 8))
    path = tmp_path / "deep" / "nested" / "out.png"
    fv.imwrite_png(str(path), template)
    assert path.is_file()
