

import logging
import os
from pathlib import Path
from threading import Lock

from mcp.server.fastmcp import FastMCP

from rag import DEFAULT_INDEX, DEFAULT_PROFILE, MODELS, load_docs, load_index, validate_manifest

# Define common path to the repo locally
PATH = os.path.dirname(os.path.abspath(__file__))

# Create an MCP server
mcp = FastMCP("Gemini-CLI-Docs-MCP-Server")
logger = logging.getLogger(__name__)
_runtime = None
_runtime_lock = Lock()


def index_directory():
    path = Path(os.environ.get("RAG_INDEX_DIR", str(DEFAULT_INDEX)))
    return path if path.is_absolute() else Path(PATH) / path


def model_profile():
    profile = os.environ.get("RAG_MODEL", DEFAULT_PROFILE)
    if profile not in MODELS:
        raise ValueError(f"Unknown RAG_MODEL: {profile}")
    return profile


def get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = load_index(index_directory(), model_profile())
        return _runtime


# Add a tool to query the Gemini CLI documentation
@mcp.tool()
def gemini_cli_query_tool(query: str):
    """
    Query the Gemini CLI documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    relevant_docs = get_runtime().search(query, k=3)
    logger.info("Retrieved %d documentation chunks", len(relevant_docs))
    results = []
    for number, (doc, similarity) in enumerate(relevant_docs, 1):
        metadata = doc.metadata
        results.append(
            f"==DOCUMENT {number}==\n"
            f"Source: {metadata['source']}\nSection: {metadata['section']}\n"
            f"Version: {metadata.get('version') or 'unverified-local'}\n"
            f"Commit: {metadata.get('commit') or 'unverified-local'}\n"
            f"URL: {metadata.get('source_url') or 'unavailable'}\n"
            f"Cosine similarity (not confidence): {similarity:.4f}\n\n{doc.page_content}"
        )
    return "\n\n".join(results)

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://gemini-cli/full")
def get_all_gemini_cli_docs() -> str:
    """
    Get all the Gemini CLI documentation from the active index's documents.jsonl,
    which contains a curated set of Gemini CLI documentation. This is useful
    for a comprehensive response to questions about Gemini CLI.

    Args: None

    Returns:
        str: The contents of the Gemini CLI documentation
    """

    # Local path to the Gemini CLI documentation
    doc_path = index_directory() / "documents.jsonl"
    validate_manifest(index_directory(), model_profile())
    documents = load_docs(doc_path)
    return "\n\n".join(
        f"=={row['metadata']['source']} | {row['metadata']['section']}==\n{row['page_content']}"
        for row in documents
    )

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

