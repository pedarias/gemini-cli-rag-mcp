import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from rag import DEFAULT_PROFILE, MODELS, ROOT, file_hash, load_docs, load_index


def validate_chunks(index, records):
    originals = {row["id"]: row for row in records}
    coverage = defaultdict(list)
    maximum = 0
    embeddings = index.store.embeddings
    for document in index.store.documents:
        metadata = document.metadata
        row = originals[metadata["section_id"]]
        start, end = metadata["char_start"], metadata["char_end"]
        expected = row["metadata"]["section"] + "\n\n" + row["page_content"][start:end]
        if not 0 <= start < end <= len(row["page_content"]) or document.page_content != expected:
            raise ValueError("Chunk content or offsets do not match the original section")
        if hashlib.sha256(document.page_content.encode()).hexdigest() != metadata["content_hash"]:
            raise ValueError("Chunk content checksum mismatch")
        length = embeddings.token_count(embeddings.config.passage_prefix + document.page_content)
        if length > embeddings.config.max_tokens:
            raise ValueError("A chunk exceeds the model token budget")
        maximum = max(maximum, length)
        coverage[row["id"]].append((start, end))
    for row in records:
        end = 0
        for start, next_end in sorted(coverage[row["id"]]):
            if start > end:
                raise ValueError("Chunking omitted part of a section")
            end = max(end, next_end)
        if end != len(row["page_content"]):
            raise ValueError("Chunking omitted the end of a section")
    return maximum


def summarize(results):
    groups = {}
    for language in ("all", "pt", "en"):
        rows = [row for row in results if language == "all" or row["language"] == language]
        if not rows:
            continue
        groups[language] = {
            "queries": len(rows),
            **{f"source_hit_at_{k}": sum(row["rank"] is not None and row["rank"] <= k for row in rows) / len(rows) for k in (1, 3, 5)},
            "source_mrr_at_5": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / len(rows),
            "latency_median_ms": float(np.median([row["latency_ms"] for row in rows])),
            "latency_p95_ms": float(np.percentile([row["latency_ms"] for row in rows], 95)),
        }
    return groups


def evaluate(index_dir, profile, cases_path):
    started = time.perf_counter()
    index = load_index(index_dir, profile)
    load_seconds = time.perf_counter() - started
    records = load_docs(Path(index_dir) / "documents.jsonl")
    maximum = validate_chunks(index, records)
    sources = {row["metadata"]["source"] for row in records}
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    missing = {case["source"] for case in cases} - sources
    if missing:
        raise ValueError(f"Evaluation labels are absent from this corpus: {sorted(missing)}")
    index.search("How do I configure Gemini CLI?", k=5)
    results = []
    for case in cases:
        for language in ("pt", "en"):
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                matches = index.search(case[language], k=5)
                timings.append((time.perf_counter() - started) * 1000)
            rank = next((i for i, (doc, _) in enumerate(matches, 1) if doc.metadata["source"] == case["source"]), None)
            results.append({
                "topic": case["topic"], "language": language, "query": case[language],
                "expected_source": case["source"], "rank": rank,
                "latency_ms": float(np.median(timings)),
                "matches": [{"source": doc.metadata["source"], "section": doc.metadata["section"], "similarity": score} for doc, score in matches],
            })
    return {
        "profile": profile, "embedding": index.manifest["embedding"],
        "dataset_sha256": index.manifest["dataset_sha256"], "cases_sha256": file_hash(cases_path),
        "source_versions": index.manifest["source_versions"], "chunks": index.manifest["chunks"],
        "max_passage_tokens_including_prefix": maximum, "model_and_index_load_seconds": load_seconds,
        "process_peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "metrics": summarize(results), "results": results,
        "limitations": "Small manually authored source-level smoke benchmark, not held-out section relevance or answer correctness. Latency is the median of three warm queries; peak RSS includes the entire Python process. BGE uses corrected chunking and its recommended query instruction, not the historical pipeline.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval on paired Portuguese and English questions (Linux).")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_PROFILE)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation_queries.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.exit(1, "Error: evaluation output already exists; choose a new path.\n")
    report = evaluate(args.index, args.model, args.cases)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
