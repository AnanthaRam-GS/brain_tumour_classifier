import re
from pathlib import Path

from btdl import config

_IMPORT_PATTERN = re.compile(r"^\s*(import\s+skimage|from\s+skimage\b)")


def test_no_btdl_runtime_module_imports_skimage():
    """scikit-image is a parity-test-only optional dependency (phase2[parity]);
    no module under src/btdl may import it at runtime."""

    src_dir = config.repo_root() / "phase2" / "src" / "btdl"
    offenders = []
    for path in sorted(src_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _IMPORT_PATTERN.match(line):
                offenders.append(f"{path.relative_to(src_dir)}:{lineno}: {line.strip()}")

    assert not offenders, f"btdl runtime modules must not import skimage: {offenders}"
