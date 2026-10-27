import logging

from . import qc
from ._quantize import quantize
from .blend import blend
from .color import delta_e76, lab_from_rgb
from .compose import compose, compose_layers
from .config import BlendMode, ComposeConfig, NoiseSpec, Placement
from .errors import (
    BaseImageAlteredError,
    DegenerateQuadError,
    InputShapeError,
    UnsupportedModeError,
    VisionError,
)
from .io import imread_gray, imread_rgb, imread_rgba, imwrite_png
from .lighting import adapt, grain, luminance_field
from .mask import apply_mask, dilate_mask, feather, rasterize_polygon
from .warp import compute_homography, place, warp_to_quad

__version__ = "0.1.0"

__all__ = [
    "compose",
    "compose_layers",
    "ComposeConfig",
    "BlendMode",
    "Placement",
    "NoiseSpec",
    "warp_to_quad",
    "compute_homography",
    "place",
    "apply_mask",
    "dilate_mask",
    "feather",
    "rasterize_polygon",
    "luminance_field",
    "adapt",
    "grain",
    "blend",
    "quantize",
    "qc",
    "lab_from_rgb",
    "delta_e76",
    "imread_rgb",
    "imread_rgba",
    "imread_gray",
    "imwrite_png",
    "VisionError",
    "InputShapeError",
    "DegenerateQuadError",
    "BaseImageAlteredError",
    "UnsupportedModeError",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
