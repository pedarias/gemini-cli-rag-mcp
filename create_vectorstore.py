import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from vector_store import VectorStore

from rag import DEFAULT_INDEX, DEFAULT_PROFILE, MODELS, ROOT, file_hash, get_embeddings, load_docs, split_documents


def build_index(input_file, output_dir, profile=DEFAULT_PROFILE, *, embeddings=None, overlap_tokens=40):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Index directory already exists: {output}. Choose a new output directory.")
    started = time.perf_counter()
    dataset_hash = file_hash(input_file)
    records = load_docs(input_file)
    embeddings = embeddings or get_embeddings(profile)
    if embeddings.config != MODELS[profile]:
        raise ValueError("Provided embeddings are incompatible with the selected model")
    splits = split_documents(records, embeddings, overlap_tokens=overlap_tokens)
    output.mkdir(parents=True, exist_ok=False)
    vector_path = output / "vectors.parquet"
    vectors = embeddings.embed_documents([doc.page_content for doc in splits])
    store = VectorStore(splits, vectors, embeddings)
    store.persist(vector_path)
    shutil.copyfile(input_file, output / "documents.jsonl")
    if file_hash(output / "documents.jsonl") != dataset_hash:
        raise ValueError("The input dataset changed while building the index; retry in a new directory")
    manifest = {
        "schema_version": 1, "profile": profile, "embedding": asdict(embeddings.config),
        "chunking": {"version": 1, "overlap_tokens": overlap_tokens, "include_section_title": True},
        "dataset_sha256": dataset_hash, "vectors_sha256": file_hash(vector_path),
        "sections": len(records), "chunks": len(splits),
        "documents": len({row["metadata"]["source"] for row in records}),
        "source_revisions": sorted({row["metadata"].get("commit") or "unverified-local" for row in records}),
        "source_versions": sorted({row["metadata"].get("version") or "unverified-local" for row in records}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds_including_model_load": time.perf_counter() - started,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a new, model-specific index from extracted JSONL documentation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory; existing directories are never overwritten.")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_PROFILE)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    args = parser.parse_args(argv)
    output = args.output or (DEFAULT_INDEX if args.model == DEFAULT_PROFILE else ROOT / "indexes" / args.model)
    try:
        manifest = build_index(args.input, output, args.model, overlap_tokens=args.overlap_tokens)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({"index": str(output), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
