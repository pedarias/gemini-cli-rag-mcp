
import os
import re
import tiktoken
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def count_tokens(text, model="cl100k_base"):
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def load_docs(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document from {file_path}")
    total_tokens = sum(count_tokens(doc.page_content) for doc in documents)
    print(f"Total tokens in loaded document: {total_tokens}")
    return documents

def split_documents(documents):
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks from document.")
    total_tokens = sum(count_tokens(doc.page_content) for doc in split_docs)
    print(f"Total tokens in split documents: {total_tokens}")
    return split_docs

def create_vectorstore(splits):
    print("Creating SKLearnVectorStore...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    persist_path = os.path.join(os.getcwd(), "gemini_cli_vectorstore.parquet")
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")
    vectorstore.persist()
    print(f"SKLearnVectorStore was persisted to {persist_path}")
    return vectorstore

if __name__ == "__main__":
    documents = load_docs("gemini_cli_docs.txt")
    split_docs = split_documents(documents)
    create_vectorstore(split_docs)

