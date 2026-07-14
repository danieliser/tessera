"""End-to-end coverage for the compact MCP explore tool."""

from __future__ import annotations

import json
from pathlib import Path

from fastmcp import Client

from tessera.indexer import IndexerPipeline
from tessera.server import create_server


async def test_explore_returns_bounded_code_and_graph_context(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(
        "def run() -> str:\n"
        "    return 'service'\n"
    )
    (tmp_path / "caller.py").write_text(
        "from service import run\n\n"
        "def handle_request() -> str:\n"
        "    return run()\n"
    )
    await IndexerPipeline(str(tmp_path), languages=["python"]).index_project()

    server = create_server(
        str(tmp_path), str(tmp_path / "global.db"), tool_profile="compact"
    )
    async with Client(server) as client:
        result = await client.call_tool("explore", {"query": "run service", "max_results": 3})

    payload = json.loads(result.content[0].text)
    assert payload["query"] == "run service"
    assert 1 <= len(payload["results"]) <= 3
    assert any(item["path"] == "service.py" for item in payload["results"])
    assert payload["symbols"]
    assert payload["files"]
    assert "next_steps" in payload
