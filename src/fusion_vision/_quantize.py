import logging

import numpy as np

logger = logging.getLogger(__name__)


def quantize(arr):
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
