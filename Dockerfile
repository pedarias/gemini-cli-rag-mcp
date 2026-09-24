# Use Python 3.13 slim image as a lean base
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt requirements.lock ./

# Install CPU-only PyTorch. sentence-transformers requires it, and this is an optimization to keep the image size small.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python dependencies
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy the rest of your application code
COPY . .

# Set the default command to run the MCP server
CMD ["python", "gemini_cli_mcp.py"]