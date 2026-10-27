import subprocess
import sys

CODE = (
    "import cv2; before = cv2.getNumThreads(); import fusion_vision; "
    "after = cv2.getNumThreads(); assert before == after, (before, after)"
)


def test_import_does_not_mutate_cv2_thread_count():
    result = subprocess.run(
        [sys.executable, "-c", CODE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
