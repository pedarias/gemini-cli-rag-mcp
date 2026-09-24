import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import extract


COMMIT = "a" * 40
COLLECTED_AT = "2026-09-22T12:00:00+00:00"


class ExtractionTests(unittest.TestCase):
    def documents(self, sources):
        return extract.extract_documents(
            sources, version="v1.2.3", commit=COMMIT, collected_at=COLLECTED_AT
        )

    def test_selects_user_documentation_only(self):
        paths = [
            "get-started/installation.md", "reference/configuration.md",
            "tools/mcp-server.md", "cli/gemini-md.md", "extensions/index.md",
            "hooks/reference.md", "resources/troubleshooting.md",
            "ide-integration/index.md", "admin/enterprise-controls.md",
            "core/subagents.md", "index.md", "get-started/authentication.mdx",
            "get-started/installation.mdx",
        ]
        excluded = [
            "contributing.md", "integration-tests.md", "local-development.md",
            "issue-and-pr-automation.md", "changelogs/latest.md",
            "core/architecture.md", "translations/pt-BR/index.md",
            "tools/image.png", "../tools/outside.md",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(extract.is_user_document(path))
        for path in excluded:
            with self.subTest(path=path):
                self.assertFalse(extract.is_user_document(path))

    def test_sections_keep_heading_hierarchy_and_provenance(self):
        text = "# Configuration\n\nOverview.\n\n## MCP\n\nSetup.\n\n### Stdio\n\nUse a subprocess.\n"
        documents = self.documents({"reference/configuration.md": text})
        self.assertEqual(len(documents), 3)
        metadata = documents[-1]["metadata"]
        self.assertEqual(metadata["title"], "Configuration")
        self.assertEqual(metadata["section"], "Configuration > MCP > Stdio")
        self.assertEqual(metadata["source"], "docs/reference/configuration.md")
        self.assertEqual(metadata["version"], "v1.2.3")
        self.assertEqual(metadata["commit"], COMMIT)
        self.assertEqual(metadata["collected_at"], COLLECTED_AT)
        self.assertEqual(metadata["language"], "en")
        self.assertEqual(metadata["line_start"], 9)
        self.assertEqual(metadata["line_end"], 11)
        self.assertEqual(
            metadata["source_url"],
            f"https://github.com/google-gemini/gemini-cli/blob/{COMMIT}/docs/reference/configuration.md#L9-L11",
        )
        self.assertEqual(metadata["document_hash"], hashlib.sha256(text.encode()).hexdigest())

    def test_sibling_sections_do_not_inherit_previous_subheadings(self):
        documents = self.documents({"tools/mcp-server.md": "# MCP\n## Local\n### Stdio\nA\n## Remote\nB\n"})
        self.assertEqual(documents[-1]["metadata"]["section"], "MCP > Remote")

    def test_code_fences_do_not_create_sections(self):
        text = "# Shell\n\n```bash\n# Not a heading\necho hello\n```\n\n~~~markdown\n## Also code\n~~~\n\n## Usage\nRun it.\n"
        documents = self.documents({"tools/shell.md": text})
        self.assertEqual(len(documents), 2)
        self.assertIn("# Not a heading", documents[0]["page_content"])
        self.assertIn("## Also code", documents[0]["page_content"])

    def test_shorter_fence_does_not_close_code_block(self):
        text = "# Shell\n````markdown\n```\n## Still code\n````\n## Usage\nRun it.\n"
        documents = self.documents({"tools/shell.md": text})
        self.assertEqual(len(documents), 2)
        self.assertIn("## Still code", documents[0]["page_content"])

    def test_frontmatter_is_not_indexed_and_line_numbers_are_preserved(self):
        text = "---\ntitle: Metadata\n---\n# Actual title\n\n## Usage\nText.\n"
        documents = self.documents({"cli/headless.md": text})
        self.assertEqual(documents[0]["metadata"]["line_start"], 4)
        self.assertEqual(documents[0]["metadata"]["title"], "Actual title")
        self.assertNotIn("title: Metadata", documents[0]["page_content"])

    def test_mdx_imports_are_not_indexed_but_tab_content_is_preserved(self):
        text = "import { Tabs, TabItem } from '@astrojs/starlight/components';\n\n# Authentication\n\n## API key\n<Tabs>\n<TabItem label=\"Linux\">\n```bash\nexport GEMINI_API_KEY=example\n```\n</TabItem>\n</Tabs>\n"
        documents = self.documents({"get-started/authentication.mdx": text})
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["metadata"]["line_start"], 3)
        self.assertNotIn("import {", "".join(doc["page_content"] for doc in documents))
        self.assertIn('label="Linux"', documents[1]["page_content"])
        self.assertIn("export GEMINI_API_KEY=example", documents[1]["page_content"])

    def test_local_reader_includes_mdx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "get-started").mkdir()
            (root / "get-started" / "installation.mdx").write_text("# Installation", encoding="utf-8")
            self.assertEqual(extract.read_local_docs(root), {"get-started/installation.mdx": "# Installation"})

    def test_tables_links_and_unicode_are_preserved(self):
        text = "# Config\n\n| Key | Value |\n| --- | --- |\n| mode | stdio |\n\n[Guide](../cli/index.md) — café\n"
        document = self.documents({"reference/configuration.md": text})[0]
        self.assertEqual(document["page_content"], text.strip())

    def test_headingless_content_preserves_indented_code(self):
        text = "    print('example')\n\nDocumentation.\n"
        document = self.documents({"cli/headless.md": text})[0]
        self.assertEqual(document["page_content"], text.rstrip("\n"))
        self.assertEqual(document["metadata"]["title"], "Headless")

    def test_existing_documentation_snapshot_can_be_processed(self):
        snapshot = Path(extract.__file__).with_name("gemini_cli_docs.txt").read_text(encoding="utf-8")
        parts = re.split(r"^--- Content from: gemini-cli/docs/(.+?) ---\s*$", snapshot, flags=re.MULTILINE)
        sources = dict(zip(parts[1::2], parts[2::2]))
        documents = extract.extract_documents(sources, collected_at=COLLECTED_AT)
        paths = {document["metadata"]["source"] for document in documents}
        self.assertGreater(len(paths), 10)
        self.assertIn("docs/tools/mcp-server.md", paths)
        self.assertNotIn("docs/integration-tests.md", paths)
        self.assertTrue(all(document["metadata"]["commit"] is None for document in documents))

    def test_ids_are_deterministic_and_content_sensitive(self):
        sources = {"tools/mcp-server.md": "# MCP\nSetup.\n"}
        first = self.documents(sources)
        self.assertEqual(first, self.documents(sources))
        changed = self.documents({"tools/mcp-server.md": "# MCP\nUpdated setup.\n"})
        self.assertNotEqual(first[0]["id"], changed[0]["id"])

    def test_documents_are_sorted_by_source(self):
        documents = self.documents({"tools/shell.md": "# Shell", "cli/headless.md": "# Headless"})
        self.assertEqual([d["metadata"]["source"] for d in documents], ["docs/cli/headless.md", "docs/tools/shell.md"])

    def test_local_documents_do_not_claim_an_official_revision(self):
        document = extract.extract_documents({"cli/headless.md": "# Headless"})[0]
        self.assertIsNone(document["metadata"]["commit"])
        self.assertIsNone(document["metadata"]["version"])
        self.assertIsNone(document["metadata"]["source_url"])

    def test_empty_or_excluded_corpus_fails(self):
        for sources in ({}, {"cli/headless.md": " \n"}, {"contributing.md": "# Internal"}):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                self.documents(sources)

    def test_missing_source_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                extract.read_local_docs(Path(directory) / "missing")

    def test_local_reader_skips_excluded_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            (docs / "cli").mkdir(parents=True)
            (docs / "cli" / "headless.md").write_text("# Headless", encoding="utf-8")
            (docs / "contributing.md").write_text("# Internal", encoding="utf-8")
            outside = root / "outside.md"
            outside.write_text("Not documentation", encoding="utf-8")
            (docs / "cli" / "outside.md").symlink_to(outside)
            self.assertEqual(extract.read_local_docs(docs), {"cli/headless.md": "# Headless"})

    def test_jsonl_writer_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "documents.jsonl"
            documents = self.documents({"cli/headless.md": "# Headless"})
            extract.write_documents(documents, output)
            self.assertEqual(json.loads(output.read_text()), documents[0])
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                extract.write_documents(documents, output)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_missing_source_exits_nonzero_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "documents.jsonl"
            result = subprocess.run(
                [sys.executable, "-B", str(Path(extract.__file__)), "--source-dir", str(Path(directory) / "missing"), "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_cli_local_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs" / "tools").mkdir(parents=True)
            (root / "docs" / "tools" / "mcp-server.md").write_text("# MCP\n\n## Stdio\nConfigure it.\n", encoding="utf-8")
            output = root / "documents.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                extract.main(["--source-dir", str(root / "docs"), "--output", str(output)])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["metadata"]["section"], "MCP > Stdio")


class OfficialSourceTests(unittest.TestCase):
    def archive(self, members):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            for path, content in members.items():
                encoded = content.encode()
                info = tarfile.TarInfo(path)
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
        return data.getvalue()

    def test_download_resolves_tag_and_reads_only_documentation(self):
        archive = self.archive({
            "gemini-snapshot/docs/tools/mcp-server.md": "# MCP",
            "gemini-snapshot/docs/contributing.md": "# Internal",
            "gemini-snapshot/packages/cli/README.md": "# Package",
        })

        def run(command, **kwargs):
            if command[-1] == ".sha":
                return subprocess.CompletedProcess(command, 0, stdout=f"{COMMIT}\n")
            kwargs["stdout"].write(archive)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(extract.subprocess, "run", side_effect=run) as mocked:
            sources, commit = extract.fetch_official_docs("v1.2.3")
        self.assertEqual(commit, COMMIT)
        self.assertEqual(sources, {"tools/mcp-server.md": "# MCP"})
        self.assertIn(f"repos/google-gemini/gemini-cli/tarball/{COMMIT}", mocked.call_args.args[0])

    def test_moving_refs_are_rejected_before_network_access(self):
        for ref in ("main", "latest", "v1.2.3-preview.1", "../../other"):
            with self.subTest(ref=ref), patch.object(extract.subprocess, "run") as mocked:
                with self.assertRaises(ValueError):
                    extract.fetch_official_docs(ref)
                mocked.assert_not_called()

    def test_download_errors_are_reported(self):
        error = subprocess.CalledProcessError(1, ["gh"], stderr="Bad credentials")
        with patch.object(extract.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Bad credentials"):
                extract.fetch_official_docs("v1.2.3")

    def test_invalid_revision_response_is_rejected(self):
        response = subprocess.CompletedProcess(["gh"], 0, stdout="not-a-commit\n")
        with patch.object(extract.subprocess, "run", return_value=response) as mocked:
            with self.assertRaisesRegex(ValueError, "valid commit"):
                extract.fetch_official_docs("v1.2.3")
            self.assertEqual(mocked.call_count, 1)

    def test_missing_github_cli_is_reported(self):
        with patch.object(extract.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Install GitHub CLI"):
                extract.fetch_official_docs("v1.2.3")

    def test_download_timeout_is_reported(self):
        error = subprocess.TimeoutExpired(["gh"], 60)
        with patch.object(extract.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                extract.fetch_official_docs("v1.2.3")

    def test_archive_traversal_is_rejected(self):
        data = io.BytesIO(self.archive({"snapshot/docs/tools/../../outside.md": "# Invalid"}))
        with self.assertRaises(ValueError):
            extract.read_archive_docs(data)


if __name__ == "__main__":
    unittest.main()
