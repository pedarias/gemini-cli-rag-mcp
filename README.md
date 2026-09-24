# Gemini CLI RAG MCP

A local MCP documentation search server. It extracts versioned official Gemini CLI documentation, generates local embeddings, and returns relevant passages with source URLs. The connected MCP client generates the final answer; this server does not call a generative model.

## Pipeline

1. `extract.py` downloads an explicit official tag/commit, or reads a local docs directory. It selects user-facing `.md` and `.mdx` files and writes JSONL sections with provenance.
2. `create_vectorstore.py` consumes that JSONL and builds a new model-specific index directory.
3. `rag.py` shares pinned model configuration, token-aware chunking, input validation, and index compatibility checks between indexing and serving.
4. `vector_store.py` stores vectors and metadata in Parquet and performs exact cosine-neighbor search with scikit-learn. LangChain is not required.
5. `gemini_cli_mcp.py` exposes documentation search and a full-documentation resource over MCP stdio.

The default embedding candidate is `intfloat/multilingual-e5-small` (384 dimensions). `BAAI/bge-large-en-v1.5` remains available as the `bge-large-en` comparison profile (1024 dimensions). Both use pinned model revisions and a maximum of 512 input tokens.

## Requirements

- Python 3.11+; local verification uses the Conda environment `project_env` with Python 3.11.
- GitHub CLI (`gh`) with working authentication for official documentation downloads. Local extraction does not require it.
- Internet access for the initial model download. Embedding inference runs locally on CPU.
- Docker and Compose are optional. The Python server itself does not require Node.js or a Gemini CLI installation.

## Install dependencies

The direct dependencies are pinned in `requirements.txt`. `requirements.lock` contains the resolved Linux CPU dependencies and distribution hashes.

With an activated Python environment:

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --require-hashes -r requirements.lock
```

Alternatively, with uv and the existing Conda environment:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" --torch-backend cpu --require-hashes -r requirements.lock
```

The CPU-specific lock was generated with:

```bash
uv pip compile requirements.txt --python-version 3.11 --torch-backend cpu --exclude-newer 2026-09-15 --no-annotate --no-header --generate-hashes --output-file requirements.lock
```

## Extract documentation

Authenticate `gh` if needed, then choose an explicit stable tag:

```bash
python extract.py --ref v0.60.0 --output gemini_cli_documents-v0.60.0-r2.jsonl
```

The tag is resolved to a full commit before downloading. Moving branches and preview tags are rejected; an explicit full commit SHA is also accepted. Downloads are temporary and do not depend on the legacy `gemini-cli` submodule.

The `r2` dataset includes MDX installation and authentication guides that were absent from the first Markdown-only extraction. The v0.60.0 corpus contains 84 source documents and 1,310 sections.

For local input:

```bash
python extract.py --source-dir gemini-cli/docs --output gemini_cli_documents-local.jsonl
```

Local files are not assigned an unverified official revision or URL. Extraction and its tests use only the Python standard library.

The initial corpus includes installation, authentication, configuration, commands, tools, MCP, extensions, hooks, skills, sandboxing, and troubleshooting. Contributor workflows, internal development documentation, translations, and changelogs are excluded. Markdown headings define sections while code examples, tables, and links remain in the text. Single-line MDX component imports are omitted; component labels and content are retained as text. MDX is not rendered or executed.

Each JSONL row contains `id`, `page_content`, and `metadata`. Metadata includes the source file, heading hierarchy, line range, document/content hashes, collection timestamp, and official version/commit where verified. Existing output files are never overwritten. Missing or empty input and download errors exit nonzero.

## Build a new index

```bash
python create_vectorstore.py --input gemini_cli_documents-v0.60.0-r2.jsonl --model e5-small
```

The default destination is `indexes/e5-small-v0.60.0`. To compare the previous embedding model on the same corpus:

```bash
python create_vectorstore.py --input gemini_cli_documents-v0.60.0-r2.jsonl --model bge-large-en --output indexes/bge-large-en-v0.60.0
```

An index directory contains:

- `vectors.parquet`: normalized embeddings, chunk text, and source metadata.
- `documents.jsonl`: the original extracted sections used to build this index.
- `manifest.json`: model/revision, dimensions, query/document prefixes, chunking settings, source revisions, counts, and artifact checksums.

Existing directories are never overwritten. A failed build may leave an incomplete directory without a manifest; use a new destination for a retry. The historical `gemini_cli_docs.txt` and `gemini_cli_vectorstore.parquet` are not modified or used by the new server.

Chunking uses the selected model's tokenizer. The 512-token budget includes the section title, model-specific prefix, and special tokens. Long sections are divided with up to 40 tokens of overlap, preferring paragraph boundaries. Query and passage inputs that still exceed the model limit are rejected rather than silently truncated.

E5 uses `query: ` and `passage: ` prefixes. BGE uses its recommended retrieval instruction for queries. Model weights are loaded with remote code disabled and safetensors enabled. Changing the model or its preprocessing requires a new index, even if the embedding dimensions happen to match.

## Run the MCP server

The client starts the server as a stdio subprocess. Do not allocate a TTY for the server process. Model and index loading is lazy and reused across queries in that process; diagnostics go to stderr, not protocol stdout.

Settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_MODEL` | `e5-small` | Must match the index manifest. |
| `RAG_INDEX_DIR` | `indexes/e5-small-v0.60.0`, relative to this repository | Directory containing the complete index bundle. |
| `RAG_THREADS` | `4` | CPU thread count for PyTorch. |
| `HF_HUB_OFFLINE` | unset | Set to `1` after caching the model to prevent Hugging Face network access. |

For a client such as Gemini CLI, configure the appropriate `mcpServers` entry with absolute paths:

```json
{
  "mcpServers": {
    "gemini_docs": {
      "command": "/absolute/path/to/python-environment/bin/python",
      "args": ["/absolute/path/to/gemini-cli-rag-mcp/gemini_cli_mcp.py"],
      "env": {
        "RAG_MODEL": "e5-small",
        "RAG_INDEX_DIR": "/absolute/path/to/gemini-cli-rag-mcp/indexes/e5-small-v0.60.0"
      }
    }
  }
}
```

The existing MCP interfaces are retained:

- `gemini_cli_query_tool(query: str)`: returns three retrieved chunks, including source file, section, version, commit, URL, and cosine similarity. Similarity is not a confidence probability.
- `docs://gemini-cli/full`: returns the extracted sections from the same index bundle, without loading embedding weights.

The manifest and artifact checksums are validated before an index is opened. Missing, corrupted, and incompatible indexes fail explicitly; the server never silently falls back to the historical index. Restart the MCP process after changing its index or model settings.

Retrieval always returns nearest passages for a nonempty valid query. There is no calibrated relevance threshold yet; the client should not assume every returned passage answers the question. The full resource can be large, so search is preferable for focused questions.

### Docker

Build the index on the host first, then:

```bash
docker compose up -d --build
```

Compose mounts the E5 index read-only and provides a named Hugging Face model cache. No Anthropic or other LLM API key is needed. It refuses to create a missing host index directory automatically. The image excludes datasets, indexes, Git history, and dotenv files from its build context.

The existing on-demand `docker exec` workflow is preserved: Compose keeps a helper container alive, and the MCP client launches the server with:

```json
{
  "mcpServers": {
    "gemini_docs": {
      "command": "docker",
      "args": ["exec", "-i", "gemini-cli-mcp-container", "python", "gemini_cli_mcp.py"]
    }
  }
}
```

## Verification and model comparison

Offline unit tests, including mocked encoders and index round trips:

```bash
python -B -m unittest discover -v
```

Using the existing environment, prefix Python commands with `conda run -n project_env`. After building the default index and caching its model, run the real MCP stdio test:

```bash
RUN_MCP_INTEGRATION=1 python -B -m unittest -v test_mcp.StdioIntegrationTests
```

The evaluation fixture contains 15 topics with paired Portuguese and English questions (30 queries). Evaluate each model separately, without another indexing job competing for the CPU:

```bash
HF_HUB_OFFLINE=1 python evaluate_retrieval.py --index indexes/e5-small-v0.60.0 --model e5-small --output indexes/e5-small-v0.60.0/evaluation.json
HF_HUB_OFFLINE=1 python evaluate_retrieval.py --index indexes/bge-large-en-v0.60.0 --model bge-large-en --output indexes/bge-large-en-v0.60.0/evaluation.json
```

Evaluation checks coverage of every original section and the token budget of every chunk before querying. Reports include source hit rates at 1/3/5, source MRR at 5, warm query latency, model/index load time, and peak process RSS on Linux. It validates that every expected source exists in the selected corpus. Output reports are never overwritten.

This is a small, manually authored source-level smoke benchmark, not a held-out section-relevance benchmark or an evaluation of generated answers. BGE is evaluated with corrected preprocessing, not the original oversized-chunk pipeline. Model selection should account for these limitations, not just the headline score.
