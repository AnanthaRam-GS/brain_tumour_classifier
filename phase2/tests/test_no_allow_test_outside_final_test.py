import ast

from btdl import config


def _find_allow_test_true_calls(source: str):
    """AST-based (not regex): finds actual `allow_test=True` keyword
    arguments in function/method calls, ignoring docstrings/comments and
    any other textual mention of the string."""

    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "allow_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                matches.append(node.lineno)
    return matches


def test_allow_test_true_only_appears_in_final_test_cli():
    src_dir = config.repo_root() / "phase2" / "src" / "btdl"
    allowed_file = src_dir / "cli" / "final_test.py"

    offenders = {}
    for path in sorted(src_dir.rglob("*.py")):
        if path == allowed_file:
            continue
        matches = _find_allow_test_true_calls(path.read_text())
        if matches:
            offenders[str(path.relative_to(src_dir))] = matches

    assert not offenders, f"allow_test=True must only appear in cli/final_test.py: {offenders}"

    # Sanity: confirm it DOES appear in final_test.py (the test isn't vacuous).
    assert _find_allow_test_true_calls(allowed_file.read_text())
