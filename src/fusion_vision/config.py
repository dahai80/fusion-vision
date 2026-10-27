import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum

from ._types import GRAIN_STRENGTH_MAX
from .errors import InputShapeError

logger = logging.getLogger(__name__)

VALID_ANCHORS = ("center", "top_left")


class BlendMode(StrEnum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    LINEAR_BURN = "linear_burn"


@dataclass(frozen=True)
class Placement:
    anchor: str = "center"
    scale: float = 1.0

    def __post_init__(self):
        if self.anchor not in VALID_ANCHORS:
            raise InputShapeError(
                "placement.anchor invalid",
                detail={"which": "placement.anchor", "supported": list(VALID_ANCHORS)},
            )
        if self.scale <= 0:
            raise InputShapeError(
                "placement.scale must be > 0",
                detail={"which": "placement.scale", "actual": self.scale},
            )


@dataclass(frozen=True)
class NoiseSpec:
    strength: float
    seed: int

    def __post_init__(self):
        if not (0 < self.strength <= GRAIN_STRENGTH_MAX):
            raise InputShapeError(
                "noise.strength out of range",
                detail={
                    "which": "noise.strength",
                    "expected": f"(0, {GRAIN_STRENGTH_MAX}]",
                    "actual": self.strength,
                },
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise InputShapeError(
                "noise.seed must be int",
                detail={"which": "noise.seed", "actual": repr(self.seed)},
            )
        if self.seed < 0:
            raise InputShapeError(
                "noise.seed must be >= 0",
                detail={"which": "noise.seed", "actual": self.seed},
            )


@dataclass(frozen=True)
class ComposeConfig:
    dst_quad: tuple
    placement: Placement = field(default_factory=Placement)
    lighting_gamma: float = 0.6
    sigma: float = 15.0
    blend_mode: BlendMode = BlendMode.NORMAL
    opacity: float = 1.0
    feather_px: float = 0.0
    noise: NoiseSpec = None
    assert_base_intact: bool = True

    def __post_init__(self):
        object.__setattr__(self, "dst_quad", _normalize_quad(self.dst_quad))
        if isinstance(self.placement, dict):
            object.__setattr__(self, "placement", Placement(**self.placement))
        if isinstance(self.noise, dict):
            if "seed" not in self.noise or self.noise.get("seed") is None:
                raise InputShapeError(
                    "noise requires an explicit int seed",
                    detail={"which": "noise", "actual": repr(self.noise)},
                )
            object.__setattr__(self, "noise", NoiseSpec(**self.noise))
        if self.lighting_gamma < 0:
            raise InputShapeError(
                "lighting_gamma must be >= 0",
                detail={"which": "lighting_gamma", "actual": self.lighting_gamma},
            )
        if self.sigma <= 0:
            raise InputShapeError(
                "sigma must be > 0",
                detail={"which": "sigma", "actual": self.sigma},
            )
        if not (0 <= self.opacity <= 1):
            raise InputShapeError(
                "opacity must be in [0, 1]",
                detail={"which": "opacity", "actual": self.opacity},
            )
        if self.feather_px < 0:
            raise InputShapeError(
                "feather_px must be >= 0",
                detail={"which": "feather_px", "actual": self.feather_px},
            )
        if not isinstance(self.blend_mode, BlendMode):
            raise InputShapeError(
                "blend_mode must be a BlendMode",
                detail={"which": "blend_mode", "actual": repr(self.blend_mode)},
            )

    def copy(self, **updates):
        logger.debug("ComposeConfig.copy updates=%s", sorted(updates))
        return replace(self, **updates)


def _normalize_quad(quad):
    try:
        pts = [(float(p[0]), float(p[1])) for p in quad]
    except (TypeError, ValueError, IndexError) as exc:
        raise InputShapeError(
            "dst_quad must be 4 (x, y) points",
            detail={"which": "dst_quad", "actual": repr(quad)},
        ) from exc
    if len(pts) != 4:
        raise InputShapeError(
            "dst_quad must have exactly 4 points (TL, TR, BR, BL)",
            detail={"which": "dst_quad", "actual": repr(quad)},
        )
    return tuple(pts)
