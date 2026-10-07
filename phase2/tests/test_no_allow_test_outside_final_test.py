import re
from pathlib import Path

from btdl import config

# Actual code usage: allow_test=True as a keyword argument (preceded by "(" or
# "," and whitespace, not inside a string). Deliberately does not match the
# same text appearing inside an error-message string literal, e.g.
# dataset.py's "...requires allow_test=True ...".
_USAGE_PATTERN = re.compile(r"[(,]\s*allow_test\s*=\s*True\b")


def test_allow_test_true_only_appears_in_final_test_cli():
    src_dir = config.repo_root() / "phase2" / "src" / "btdl"
    allowed_file = src_dir / "cli" / "final_test.py"

    offenders = []
    for path in sorted(src_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _USAGE_PATTERN.search(line) and path != allowed_file:
                offenders.append(f"{path.relative_to(src_dir)}:{lineno}: {line.strip()}")

    assert not offenders, f"allow_test=True must only appear in cli/final_test.py: {offenders}"

    # Sanity: confirm it DOES appear in final_test.py (the test isn't vacuous).
    assert _USAGE_PATTERN.search(allowed_file.read_text())
