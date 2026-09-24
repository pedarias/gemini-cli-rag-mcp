import contextlib
from datetime import timedelta
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from vector_store import Document

import gemini_cli_mcp as server


class ServerTests(unittest.TestCase):
    def test_query_returns_sources_and_never_prints_to_stdout(self):
        document = Document(page_content="Configure MCP.", metadata={
            "source": "docs/tools/mcp-server.md", "section": "MCP > Stdio",
            "source_url": "https://example.com/docs", "version": "v0.60.0", "commit": "a" * 40,
        })
        index = Mock()
        index.search.return_value = [(document, 0.9)]
        output = io.StringIO()
        with patch.object(server, "get_runtime", return_value=index), contextlib.redirect_stdout(output):
            result = server.gemini_cli_query_tool("Como configurar MCP?")
        self.assertEqual(output.getvalue(), "")
        self.assertIn("docs/tools/mcp-server.md", result)
        self.assertIn("MCP > Stdio", result)
        self.assertIn("v0.60.0", result)
        self.assertIn("https://example.com/docs", result)
        self.assertIn("Configure MCP.", result)

    def test_runtime_is_loaded_once(self):
        with patch.object(server, "_runtime", None), patch.object(server, "load_index") as loader:
            first = server.get_runtime()
            second = server.get_runtime()
        self.assertIs(first, second)
        loader.assert_called_once()

    def test_resource_uses_the_index_dataset_without_loading_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            rows = [{"page_content": "Current docs", "metadata": {"source": "docs/tools/mcp-server.md", "section": "MCP"}}]
            with patch.object(server, "index_directory", return_value=Path(directory)), patch.object(server, "validate_manifest"), patch.object(server, "load_docs", return_value=rows), patch.object(server, "get_runtime") as runtime:
                result = server.get_all_gemini_cli_docs()
            self.assertIn("Current docs", result)
            self.assertIn("docs/tools/mcp-server.md", result)
            runtime.assert_not_called()

    def test_resource_errors_are_not_returned_as_documentation(self):
        with patch.object(server, "validate_manifest", side_effect=ValueError("Index checksum mismatch")):
            with self.assertRaises(ValueError):
                server.get_all_gemini_cli_docs()


@unittest.skipUnless(os.environ.get("RUN_MCP_INTEGRATION") == "1", "Requires a built index and cached model weights")
class StdioIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_queries_and_resource(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parameters = StdioServerParameters(
            command=sys.executable, args=["-B", str(Path(server.__file__))],
            env={
                "RAG_INDEX_DIR": str(server.index_directory()), "RAG_MODEL": server.model_profile(),
                "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
            },
        )
        with tempfile.TemporaryFile(mode="w+") as errors:
            async with stdio_client(parameters, errlog=errors) as (reader, writer):
                async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=120)) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertIn("gemini_cli_query_tool", [tool.name for tool in tools.tools])
                    for query in ("Como configurar um servidor MCP?", "How do I ignore files?"):
                        result = await session.call_tool("gemini_cli_query_tool", {"query": query})
                        self.assertFalse(result.isError)
                        text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
                        self.assertIn("Source: docs/", text)
                        self.assertIn("https://github.com/google-gemini/gemini-cli/blob/", text)
                        self.assertIn("Version: v0.60.0", text)
                    invalid = await session.call_tool("gemini_cli_query_tool", {"query": " "})
                    self.assertTrue(invalid.isError)
                    resource_result = await session.read_resource("docs://gemini-cli/full")
                    self.assertIn("docs/tools/mcp-server.md", resource_result.contents[0].text)
            errors.seek(0)
            self.assertNotIn("Failed to parse JSONRPC", errors.read())


if __name__ == "__main__":
    unittest.main()
