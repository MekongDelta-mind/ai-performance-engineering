from __future__ import annotations

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.generate_mcp_docs import END_MARKER, START_MARKER, render_mcp_tool_block


def _extract_block(content: str) -> str:
    start = content.index(START_MARKER) + len(START_MARKER)
    end = content.index(END_MARKER)
    return content[start:end].strip()


def test_mcp_tools_doc_is_current():
    doc_path = REPO_ROOT / "docs" / "mcp_tools.md"
    content = doc_path.read_text(encoding="utf-8")
    generated = render_mcp_tool_block().strip()
    existing = _extract_block(content)
    assert existing == generated, "docs/mcp_tools.md is out of date; run python -m scripts.generate_mcp_docs"
