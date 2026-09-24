from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import rag
from create_vectorstore import build_index, load_docs


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class FakeEncoder:
    def __init__(self, dimensions=384):
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 512
        self.dimensions = dimensions
        self.calls = []

    def get_embedding_dimension(self):
        return self.dimensions

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, 0 if "MCP" in text else 1] = 1.0
        return vectors


def record(text="# MCP\nConfigure the server.", section="MCP", source="docs/tools/mcp-server.md"):
    return {
        "id": hashlib.sha256((source + text).encode()).hexdigest(),
        "page_content": text,
        "metadata": {
            "schema_version": 1, "title": section, "section": section, "source": source,
            "source_url": "https://github.com/google-gemini/gemini-cli/blob/" + "a" * 40 + "/" + source,
            "version": "v0.60.0", "commit": "a" * 40, "line_start": 1, "line_end": 2,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        },
    }


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.config = rag.MODELS["e5-small"]
        self.encoder = FakeEncoder()
        self.embeddings = rag.LocalEmbeddings(self.config, encoder=self.encoder)

    def test_e5_uses_distinct_query_and_passage_prefixes(self):
        self.embeddings.embed_documents(["MCP configuration"])
        self.embeddings.embed_query("Como configurar MCP?")
        self.assertEqual(self.encoder.calls[0][0], ["passage: MCP configuration"])
        self.assertEqual(self.encoder.calls[1][0], ["query: Como configurar MCP?"])
        self.assertTrue(all(kwargs["normalize_embeddings"] for _, kwargs in self.encoder.calls))

    def test_rejects_silent_query_truncation(self):
        with self.assertRaisesRegex(ValueError, "token"):
            self.embeddings.embed_query("x" * 512)
        self.assertEqual(self.encoder.calls, [])

    def test_rejects_silent_document_truncation(self):
        with self.assertRaisesRegex(ValueError, "token"):
            self.embeddings.embed_documents(["x" * 512])
        self.assertEqual(self.encoder.calls, [])

    def test_model_dimension_must_match_configuration(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            rag.LocalEmbeddings(self.config, encoder=FakeEncoder(1024))

    def test_model_revisions_are_pinned(self):
        for config in rag.MODELS.values():
            self.assertRegex(config.revision, r"^[0-9a-f]{40}$")


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(rag.MODELS["e5-small"], max_tokens=96)
        self.embeddings = rag.LocalEmbeddings(self.config, encoder=FakeEncoder())

    def test_chunk_budget_includes_title_prefix_and_special_tokens(self):
        source = record("Texto multilíngue com configuração. " * 20)
        chunks = rag.split_documents([source], self.embeddings, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            tokens = self.embeddings.token_count(self.config.passage_prefix + chunk.page_content)
            self.assertLessEqual(tokens, self.config.max_tokens)
            self.assertEqual(chunk.metadata["source"], source["metadata"]["source"])
            self.assertEqual(chunk.metadata["section_id"], source["id"])
            self.assertEqual(chunk.metadata["source_url"], source["metadata"]["source_url"])
        self.assertEqual(len({chunk.metadata["chunk_id"] for chunk in chunks}), len(chunks))

    def test_chunks_cover_original_text_without_gaps(self):
        text = "\n\n".join(["Paragraph " + str(i) + ": " + "áβ中 " * 10 for i in range(8)])
        chunks = rag.split_documents([record(text)], self.embeddings, overlap_tokens=10)
        covered = set()
        previous_end = 0
        for chunk in chunks:
            start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
            self.assertLessEqual(start, previous_end)
            self.assertGreater(end, previous_end)
            self.assertIn(text[start:end], chunk.page_content)
            covered.update(range(start, end))
            previous_end = end
        self.assertEqual(covered, set(range(len(text))))

    def test_small_section_is_not_split(self):
        chunks = rag.split_documents([record()], self.embeddings, overlap_tokens=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("MCP", chunks[0].page_content)

    def test_impossible_title_budget_fails(self):
        with self.assertRaisesRegex(ValueError, "title"):
            rag.split_documents([record(section="x" * 200)], self.embeddings)

    def test_invalid_overlap_is_rejected(self):
        for overlap in (-1, self.config.max_tokens):
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                rag.split_documents([record()], self.embeddings, overlap_tokens=overlap)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "documents.jsonl"
        self.records = [record(), record("Use memory files.", section="Memory", source="docs/tools/memory.md")]
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.records) + "\n", encoding="utf-8")
        self.embeddings = rag.LocalEmbeddings(rag.MODELS["e5-small"], encoder=FakeEncoder())
        self.output = self.root / "index"

    def build(self):
        return build_index(self.dataset, self.output, "e5-small", embeddings=self.embeddings)

    def test_jsonl_loader_preserves_metadata(self):
        self.assertEqual(load_docs(self.dataset), self.records)

    def test_jsonl_loader_rejects_empty_or_malformed_input(self):
        for content in ("", "not json\n", '{"page_content": "text"}\n'):
            self.dataset.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_docs(self.dataset)

    def test_jsonl_loader_rejects_duplicate_ids(self):
        self.dataset.write_text((json.dumps(self.records[0]) + "\n") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_docs(self.dataset)

    def test_roundtrip_preserves_sources_and_limits_k(self):
        manifest = self.build()
        self.assertEqual(manifest["embedding"]["model_name"], "intfloat/multilingual-e5-small")
        self.assertEqual(manifest["chunks"], 2)
        index = rag.load_index(self.output, "e5-small", embeddings=self.embeddings)
        matches = index.search("MCP", k=5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0].metadata["source"], "docs/tools/mcp-server.md")
        self.assertAlmostEqual(matches[0][1], 1.0)
        self.assertEqual(rag.load_docs(self.output / "documents.jsonl"), self.records)

    def test_existing_index_is_never_overwritten(self):
        self.build()
        original = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)

    def test_wrong_model_fails_before_loading_weights(self):
        self.build()
        with patch.object(rag, "get_embeddings") as mocked:
            with self.assertRaisesRegex(ValueError, "incompatible"):
                rag.load_index(self.output, "bge-large-en")
            mocked.assert_not_called()

    def test_wrong_revision_or_prefix_is_rejected(self):
        self.build()
        path = self.output / "manifest.json"
        original = json.loads(path.read_text())
        for field, value in (("revision", "b" * 40), ("query_prefix", "wrong: ")):
            manifest = json.loads(json.dumps(original))
            manifest["embedding"][field] = value
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "incompatible"):
                rag.load_index(self.output, "e5-small", embeddings=self.embeddings)

    def test_corrupt_vectors_are_rejected(self):
        self.build()
        with (self.output / "vectors.parquet").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            rag.load_index(self.output, "e5-small", embeddings=self.embeddings)

    def test_changed_dataset_is_rejected(self):
        self.build()
        with (self.output / "documents.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            rag.load_index(self.output, "e5-small", embeddings=self.embeddings)

    def test_incomplete_index_does_not_download_a_model(self):
        self.output.mkdir()
        with patch.object(rag, "get_embeddings") as mocked:
            with self.assertRaises(FileNotFoundError):
                rag.load_index(self.output, "e5-small")
            mocked.assert_not_called()

    def test_empty_queries_and_invalid_k_are_rejected(self):
        self.build()
        index = rag.load_index(self.output, "e5-small", embeddings=self.embeddings)
        for query, k in (("", 3), ("   ", 3), ("MCP", 0), ("MCP", 11)):
            with self.subTest(query=query, k=k), self.assertRaises(ValueError):
                index.search(query, k)


if __name__ == "__main__":
    unittest.main()
