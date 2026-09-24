from dataclasses import replace
from types import SimpleNamespace
import unittest

from evaluate_retrieval import summarize, validate_chunks
from rag import LocalEmbeddings, MODELS, split_documents
from test_retrieval import FakeEncoder, record


class EvaluationTests(unittest.TestCase):
    def test_metrics_include_misses_and_are_grouped_by_language(self):
        results = [
            {"language": "pt", "rank": 1, "latency_ms": 10},
            {"language": "pt", "rank": None, "latency_ms": 20},
            {"language": "en", "rank": 3, "latency_ms": 30},
        ]
        metrics = summarize(results)
        self.assertAlmostEqual(metrics["all"]["source_hit_at_1"], 1 / 3)
        self.assertAlmostEqual(metrics["all"]["source_hit_at_5"], 2 / 3)
        self.assertAlmostEqual(metrics["all"]["source_mrr_at_5"], (1 + 1 / 3) / 3)
        self.assertEqual(metrics["pt"]["source_hit_at_5"], 0.5)
        self.assertEqual(metrics["pt"]["latency_median_ms"], 15)

    def test_chunk_audit_detects_missing_text(self):
        embeddings = LocalEmbeddings(replace(MODELS["e5-small"], max_tokens=96), encoder=FakeEncoder())
        records = [record("Long documentation paragraph. " * 20)]
        chunks = split_documents(records, embeddings, overlap_tokens=10)
        index = SimpleNamespace(store=SimpleNamespace(embeddings=embeddings, documents=chunks))
        self.assertLessEqual(validate_chunks(index, records), 96)
        index.store.documents = chunks[:-1]
        with self.assertRaisesRegex(ValueError, "omitted"):
            validate_chunks(index, records)


if __name__ == "__main__":
    unittest.main()
