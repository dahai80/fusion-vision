import logging

logger = logging.getLogger(__name__)


class VisionError(Exception):
    def __init__(self, message, detail=None):
        self.detail = detail or {}
        super().__init__(message)
        logger.debug("%s detail=%s", message, self.detail)


class InputShapeError(VisionError):
    pass


class DegenerateQuadError(VisionError):
    pass


class UnsupportedModeError(VisionError):
    pass


class BaseImageAlteredError(VisionError):
    pass
