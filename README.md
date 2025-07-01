# Gemini CLI RAG MCP

This project builds a standalone RAG service, transforming the static `gemini-cli` documentation into a dynamic and queryable tool. This tool exposes knowledge via a protocol (like MCP), making it accessible to any integrated client. Therefore, environments like gemini-cli, VS Code, or Cursor can provide developers with instant, accurate answers in natural language, directly within their workflow. Accelerating learning and letting you intuitively leverage the tool's full potential.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Run the MCP Service with Docker](#1-run-the-mcp-service-with-docker)
  - [2. Configure Gemini CLI](#2-configure-gemini-cli)
  - [3. Ask Questions](#3-ask-questions)
- [How It Works](#how-it-works)
  - [Data Extraction and Vectorization](#data-extraction-and-vectorization)
  - [MCP Server](#mcp-server)
  - [Gemini CLI Integration](#gemini-cli-integration)
- [Scripts](#scripts)
- [Dependencies](#dependencies)


## Project Overview

This project integrates a RAG pipeline and it consists of three main components:

1.  **Data Extraction and Processing**: Python scripts that extract content from all markdown files in the `gemini-cli/docs` directory and sub-directories, process it, and create a vector store.
2.  **MCP Server**: A Python-based MCP server that exposes the vector store as a queryable tool.
3.  **Gemini CLI/VSCode/ClaudeCode/Windsurf/Cursor...etc**: The official Gemini CLI, which can connect to the MCP server to answer questions about its documentation.

## Features

-   **RAG-based Q&A**: Ask questions about the Gemini CLI in natural language and get answers based on its official documentation.
-   **Local Vector Store**: The entire documentation is stored and indexed locally using `SKLearnVectorStore`.
-   **Extensible**: The MCP server can be easily extended with new tools and data sources.

## System Architecture

The system is composed of the following parts:

1.  **`extract.py`**: This script walks through the `gemini-cli/docs` directory, finds all `.md` files, and concatenates their content into a single `gemini_cli_docs.txt` file.
2.  **`create_vectorstore.py`**: This script loads the `gemini_cli_docs.txt` file, splits it into chunks, and creates a `gemini_cli_vectorstore.parquet` file using `HuggingFaceEmbeddings` and `SKLearnVectorStore`.
3.  **`gemini_cli_mcp.py`**: This script runs a `FastMCP` server that loads the vector store and exposes two endpoints:
    -   `gemini_cli_query_tool(query: str)`: A tool that takes a user query, retrieves relevant documents from the vector store, and returns them.
    -   `docs://gemini-cli/full`: A resource that returns the entire content of the `gemini_cli_docs.txt` file.
4.  **`gemini-cli/`**: The official Gemini CLI, which can be configured to use the MCP server.

## Getting Started

### Prerequisites

-   Python 3.13
-   [Node.js 18+](https://nodejs.org/en/download) 
-   An existing `gemini-cli` installation. If you don't have it, you can clone the official repository:
    ```bash
    git clone https://github.com/google-gemini/gemini-cli.git
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/gemini-cli-rag-mcp.git
    cd gemini-cli-rag-mcp
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the documentation data:**
    Run the `extract.py` script to gather all the markdown documentation into a single file.
    ```bash
    python extract.py
    ```

4.  **Create the vector store:**
    Run the `create_vectorstore.py` script to create the vector store from the documentation file.
    ```bash
    python create_vectorstore.py
    ```

## Usage
Before running with docker, try running the mcp in dev mode and test:
```bash
mcp dev gemini_cli_mcp.py
``` 
On ``Command`` field type 'python' and on ``Arguments`` type 'gemini_cli_mcp.py' and press Connect.

### 1. Run the MCP Service with Docker

The most efficient way to run the MCP server is with Docker Compose. This starts a container in the background and keeps it ready for Gemini CLI to connect to.

```bash
docker-compose up -d
```

The container will keep running, but the Python MCP script itself will only be executed on-demand by Gemini CLI.

### 2. Configure Gemini CLI

To make Gemini CLI aware of your local MCP server, you need to create a configuration file.

-  Inside the `.gemini` directory add the following content to the `settings.json` file:

    ```json
    {
      "mcpServers": {
        "local_rag_server": {
          "command": "docker",
          "args": [
            "exec",
            "-i",
            "gemini-cli-mcp-container",
            "python",
            "gemini_cli_mcp.py"
          ]
        }
      }
    }
    ```

This configuration tells Gemini CLI how to launch your MCP server using `docker exec`.
**Obs**: To use it in VSCode, go to `Settings` type 'mcp' and click on `settings.json`. Then put on Agent mode and ask copilot to implement the gemini-cli-mcp server (give the json above as context).

### 3. Ask Questions

After restarting terminal to changes make effect, simply run `gemini` from your terminal. It will automatically discover the `local_rag_server` and use its tools when needed.

**Example:**
> How do I customize my gemini-cli?

or something more specific:
> My gemini cli is not showing an interactive prompt when I run it on my build server, it just exits. I have a CI_TOKEN environment variable set. Why is this happening and how can I fix it?

## How It Works

### Data Extraction and Vectorization

The `extract.py` script recursively finds all markdown files in the `gemini-cli/docs` directory. It reads their content and combines it into a single text file, `gemini_cli_docs.txt`.

The `create_vectorstore.py` script then takes this text file and:
1.  Loads the document.
2.  Splits it into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`.
3.  Uses `HuggingFaceEmbeddings` (with the `BAAI/bge-large-en-v1.5` model) to create embeddings for each chunk.
4.  Stores these embeddings in a `SKLearnVectorStore`, which is persisted to `gemini_cli_vectorstore.parquet`.

### MCP Server

The `gemini_cli_mcp.py` script creates a `FastMCP` server. This server defines a tool, `gemini_cli_query_tool`, which can be called by the Gemini CLI or VSCode/Cursor/etc. When this tool is invoked, it:
1.  Loads the persisted `SKLearnVectorStore`.
2.  Uses the vector store as a retriever to find the most relevant document chunks for the given query.
3.  Returns the content of these chunks to the Gemini CLI.

### Gemini CLI Integration

The Gemini CLI is designed to be extensible through MCP servers. The CLI discovers available tools by connecting to servers defined in the `mcpServers` object in a `settings.json` file (either in the project's `.gemini` directory or in the user's home `~/.gemini` directory).

Gemini CLI supports three transport mechanisms for communication:

-   **Stdio Transport**: Spawns a subprocess and communicates with it over `stdin` and `stdout`. This is the method used in this project, with the `command` property in `settings.json`.
-   **SSE Transport**: Connects to a Server-Sent Events (SSE) endpoint, defined with a `url` property.
-   **Streamable HTTP Transport**: Uses HTTP streaming for communication, configured with an `httpUrl` property.

By using the `docker exec` command, we are leveraging the `stdio` transport to create a direct communication channel with the Python script inside the container.

## Scripts

-   **`extract.py`**: Extracts documentation from markdown files.
-   **`create_vectorstore.py`**: Creates the vector store.
-   **`gemini_cli_mcp.py`**: Runs the MCP server.

## Dependencies

### Python

The main Python dependencies are listed in `requirements.txt`:
-   `langchain`: For text splitting, vector stores, and embeddings.
-   `tiktoken`: For token counting.
-   `sentence-transformers`: For the embedding model.
-   `scikit-learn`: For the vector store.
-   `mcp`: For the MCP server.
-   `fastapi`: For the MCP server.

### Node.js

The project relies on the `gemini-cli` package and its dependencies. See `gemini-cli/package.json` for more details.
