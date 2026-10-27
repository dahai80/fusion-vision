# Regenerate golden.json hash entries + reference PNGs for the current platform.
# Run: python tests/golden/update_golden.py
# Any change to output bytes requires a reviewed commit of golden.json +
# reference/ (semver minor+ per PRD red line 2).
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).parent))

import golden_cases as gc  # noqa: E402  (needs sys.path tweak above)

GOLDEN_JSON = Path(__file__).parent / "golden.json"
REFERENCE_DIR = Path(__file__).parent / "reference"


def main():
    REFERENCE_DIR.mkdir(exist_ok=True)
    key = gc.platform_key()
    try:
        with open(GOLDEN_JSON) as f:
            index = json.load(f)
    except FileNotFoundError:
        index = {"version": 1, "cases": {}}

    cases = index.setdefault("cases", {})
    for name, geometry, mode, gamma, noise in gc.case_list():
        out = gc.build_case(name, geometry, mode, gamma, noise)
        payload = gc.png_bytes(out)
        (REFERENCE_DIR / f"{name}.png").write_bytes(payload)
        import hashlib

        cases.setdefault(name, {"hashes": {}})["hashes"][key] = hashlib.sha256(payload).hexdigest()
        print(f"updated {name} ({key})")

    with open(GOLDEN_JSON, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"golden.json updated for platform {key}")


if __name__ == "__main__":
    main()
