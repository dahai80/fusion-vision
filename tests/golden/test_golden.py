# Golden-sample regression: sha256 per platform, cross-arch tolerance mode.
# Rules (archi plan SS6):
#   - current platform has a hash entry -> strict sha256, byte-exact
#   - platform missing -> compare against reference PNG, max_abs_diff <= 1
#   - any operator change that alters output bytes = semver minor+ and a
#     reviewed golden.json update (run: python tests/golden/update_golden.py)
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import golden_cases as gc

GOLDEN_DIR = Path(__file__).parent
GOLDEN_JSON = GOLDEN_DIR / "golden.json"
REFERENCE_DIR = GOLDEN_DIR / "reference"
MAX_ABS_DIFF = 1

logger = logging.getLogger("fusion_vision.golden")


@pytest.fixture(scope="module")
def golden_index():
    with open(GOLDEN_JSON) as f:
        return json.load(f)


@pytest.mark.parametrize("name,geometry,mode,gamma,noise", gc.case_list())
def test_golden_case(name, geometry, mode, gamma, noise, golden_index):
    out = gc.build_case(name, geometry, mode, gamma, noise)
    payload = gc.png_bytes(out)
    digest = hashlib.sha256(payload).hexdigest()

    entry = golden_index["cases"].get(name)
    assert entry is not None, f"golden case {name!r} missing from golden.json"

    key = gc.platform_key()
    if key in entry["hashes"]:
        assert digest == entry["hashes"][key], (
            f"golden regression for {name!r} on {key}: "
            f"expected {entry['hashes'][key]}, got {digest}; "
            "if intentional, bump semver minor+ and run tests/golden/update_golden.py"
        )
        return

    ref_path = REFERENCE_DIR / f"{name}.png"
    assert ref_path.is_file(), (
        f"no hash for platform {key} and no reference PNG for {name!r}; "
        "run tests/golden/update_golden.py on this machine"
    )
    reference = np.asarray(Image.open(ref_path))
    diff = np.abs(out.astype(np.int32) - reference.astype(np.int32)).max()
    assert diff <= MAX_ABS_DIFF, f"cross-arch drift for {name!r}: max_abs_diff={diff}"
    logger.warning("golden: platform %s has no hash entry for %r; ran in tolerance mode", key, name)
