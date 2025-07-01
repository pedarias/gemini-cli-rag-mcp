

import os
from mcp.server.fastmcp import FastMCP
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# Define common path to the repo locally
PATH = os.path.dirname(os.path.abspath(__file__))

# Create an MCP server
mcp = FastMCP("Gemini-CLI-Docs-MCP-Server")

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
    retriever = SKLearnVectorStore(
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"), 
        persist_path=os.path.join(PATH, "gemini_cli_vectorstore.parquet"), 
        serializer="parquet").as_retriever(search_kwargs={"k": 3}
        )

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://gemini-cli/full")
def get_all_gemini_cli_docs() -> str:
    """
    Get all the Gemini CLI documentation. Returns the contents of the file gemini_cli_docs.txt,
    which contains a curated set of Gemini CLI documentation. This is useful
    for a comprehensive response to questions about Gemini CLI.

    Args: None

    Returns:
        str: The contents of the Gemini CLI documentation
    """

    # Local path to the Gemini CLI documentation
    doc_path = os.path.join(PATH, "gemini_cli_docs.txt")
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

