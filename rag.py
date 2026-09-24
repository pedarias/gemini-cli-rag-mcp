from bisect import bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from vector_store import Document, VectorStore


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = "e5-small"
DEFAULT_INDEX = ROOT / "indexes" / "e5-small-v0.60.0"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    revision: str
    dimensions: int
    query_prefix: str
    passage_prefix: str
    max_tokens: int = 512
    normalize_embeddings: bool = True


MODELS = {
    "e5-small": ModelConfig(
        "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        384, "query: ", "passage: ",
    ),
    "bge-large-en": ModelConfig(
        "BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        1024, "Represent this sentence for searching relevant passages: ", "",
    ),
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_docs(file_path):
    records = []
    ids = set()
    with Path(file_path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                metadata = row["metadata"]
                if (
                    not isinstance(row["id"], str) or not row["id"]
                    or not isinstance(row["page_content"], str) or not row["page_content"].strip()
                    or not isinstance(metadata, dict) or metadata.get("schema_version") != 1
                    or any(not isinstance(metadata.get(key), str) or not metadata[key] for key in ("source", "title", "section"))
                ):
                    raise ValueError("Invalid document fields")
                content_hash = hashlib.sha256(row["page_content"].encode()).hexdigest()
                if metadata.get("content_hash") != content_hash:
                    raise ValueError("Content hash does not match")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL document at line {number}: {error}") from error
            if row["id"] in ids:
                raise ValueError(f"Duplicate document ID at line {number}")
            ids.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError("The documentation dataset is empty")
    return records


class LocalEmbeddings:
    def __init__(self, config, *, encoder=None, batch_size=16):
        self.config = config
        self.batch_size = batch_size
        if encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(max(1, min(int(os.environ.get("RAG_THREADS", "4")), os.cpu_count() or 4)))
            encoder = SentenceTransformer(
                config.model_name, revision=config.revision, device="cpu",
                trust_remote_code=False, token=False, model_kwargs={"use_safetensors": True},
            )
        if encoder.get_embedding_dimension() != config.dimensions:
            raise ValueError("The model embedding dimension does not match its configuration")
        if encoder.max_seq_length < config.max_tokens:
            raise ValueError("The configured token limit exceeds the model capacity")
        encoder.max_seq_length = config.max_tokens
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer

    def token_count(self, text):
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])

    def _encode(self, texts, prefix):
        inputs = [prefix + text for text in texts]
        if any(self.token_count(text) > self.config.max_tokens for text in inputs):
            raise ValueError(f"Input exceeds the {self.config.max_tokens}-token model limit; refusing truncation")
        return self.encoder.encode(
            inputs, prompt="", normalize_embeddings=self.config.normalize_embeddings,
            batch_size=self.batch_size, show_progress_bar=False,
        ).tolist()

    def embed_documents(self, texts):
        return self._encode(texts, self.config.passage_prefix)

    def embed_query(self, text):
        return self._encode([text], self.config.query_prefix)[0]


@lru_cache(maxsize=1)
def get_embeddings(profile):
    return LocalEmbeddings(MODELS[profile])


def split_documents(records, embeddings, overlap_tokens=40):
    config = embeddings.config
    if not 0 <= overlap_tokens < config.max_tokens:
        raise ValueError("Overlap must be nonnegative and smaller than the model token limit")
    chunks = []
    for row in records:
        text = row["page_content"]
        header = row["metadata"]["section"] + "\n\n"
        budget = config.max_tokens - embeddings.token_count(config.passage_prefix + header) - 2
        if budget < 16:
            raise ValueError(f"Section title leaves insufficient token budget: {row['metadata']['source']}")
        offsets = embeddings.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            truncation=False, verbose=False,
        )["offset_mapping"]
        if not offsets:
            continue
        starts = [start for start, _ in offsets]
        start = 0
        chunk_number = 0
        overlap = min(overlap_tokens, budget // 2)
        while start < len(offsets):
            end = min(start + budget, len(offsets))
            char_start = starts[start] if start else 0
            char_end = starts[end] if end < len(offsets) else len(text)
            if end < len(offsets):
                boundary = text.rfind("\n\n", (char_start + char_end) // 2, char_end)
                if boundary != -1:
                    candidate = bisect_right(starts, boundary)
                    if candidate > start:
                        end = candidate
            while end > start:
                char_end = starts[end] if end < len(offsets) else len(text)
                passage = header + text[char_start:char_end]
                if embeddings.token_count(config.passage_prefix + passage) <= config.max_tokens:
                    break
                end -= 1
            if end <= start or char_end <= char_start:
                raise ValueError(f"Cannot split section within token budget: {row['id']}")
            identity = f"{row['id']}:{config.revision}:{char_start}:{char_end}"
            chunks.append(Document(page_content=passage, metadata={
                **row["metadata"], "section_id": row["id"], "chunk_index": chunk_number,
                "chunk_id": hashlib.sha256(identity.encode()).hexdigest(),
                "char_start": char_start, "char_end": char_end,
                "section_content_hash": row["metadata"]["content_hash"],
                "content_hash": hashlib.sha256(passage.encode()).hexdigest(),
            }))
            chunk_number += 1
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
    if not chunks:
        raise ValueError("No indexable chunks were produced")
    return chunks


def validate_manifest(index_dir, profile):
    path = Path(index_dir)
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported index format; rebuild the index")
    if manifest.get("embedding") != asdict(MODELS[profile]):
        raise ValueError("Index and embedding configuration are incompatible; select the correct model or rebuild")
    if not isinstance(manifest.get("chunks"), int) or manifest["chunks"] < 1:
        raise ValueError("Invalid index chunk count")
    for filename, key in (("vectors.parquet", "vectors_sha256"), ("documents.jsonl", "dataset_sha256")):
        if file_hash(path / filename) != manifest.get(key):
            raise ValueError(f"Index checksum mismatch: {filename}")
    return manifest


@dataclass
class SearchIndex:
    store: VectorStore
    manifest: dict

    def search(self, query, k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        matches = self.store.similarity_search_with_score(query.strip(), k=min(k, self.manifest["chunks"]))
        return [(document, float(1.0 - distance)) for document, distance in matches]


def load_index(index_dir, profile=DEFAULT_PROFILE, *, embeddings=None):
    manifest = validate_manifest(index_dir, profile)
    embeddings = embeddings or get_embeddings(profile)
    if embeddings.config != MODELS[profile]:
        raise ValueError("Provided embeddings are incompatible with the index")
    store = VectorStore.load(Path(index_dir) / "vectors.parquet", embeddings)
    if len(store.documents) != manifest["chunks"]:
        raise ValueError("Index chunk count does not match the manifest")
    return SearchIndex(store, manifest)
