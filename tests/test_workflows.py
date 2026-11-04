"""Workflow `run:` scripts must not interpolate event text directly."""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = sorted((Path(__file__).parent.parent / ".github" / "workflows").glob("*.yml"))

# Attacker-controllable event fields (commit messages, PR/issue text, branch names).
_UNTRUSTED = re.compile(
    r"\$\{\{\s*github\.(event\.[\w.]*(message|title|body|name|label|ref|email)\b|head_ref)"
)


def _run_scripts(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                yield value
            else:
                yield from _run_scripts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _run_scripts(item)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_untrusted_event_text_in_run(path):
    """Pass event text through `env:` so the shell never parses it as code."""
    workflow = yaml.safe_load(path.read_text())
    offending = [s for s in _run_scripts(workflow) if _UNTRUSTED.search(s)]
    assert not offending, f"{path.name}: move event text into env: {offending}"
