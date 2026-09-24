from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors


@dataclass
class Document:
    page_content: str
    metadata: dict


class VectorStore:
    def __init__(self, documents, vectors, embeddings):
        matrix = np.asarray(vectors, dtype=np.float32)
        expected_shape = (len(documents), embeddings.config.dimensions)
        if not documents or matrix.shape != expected_shape:
            raise ValueError("Index vector dimensions or document count are invalid")
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
            raise ValueError("Index vectors must be finite and normalized")
        self.documents = documents
        self.vectors = matrix
        self.embeddings = embeddings
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix)

    def persist(self, path):
        table = pa.Table.from_pydict({
            "ids": [doc.metadata["chunk_id"] for doc in self.documents],
            "texts": [doc.page_content for doc in self.documents],
            "metadatas": [doc.metadata for doc in self.documents],
            "embeddings": self.vectors.tolist(),
        })
        with Path(path).open("xb") as stream:
            pq.write_table(table, stream)

    @classmethod
    def load(cls, path, embeddings):
        data = pq.read_table(path, columns=["texts", "metadatas", "embeddings"]).to_pydict()
        documents = [Document(text, metadata) for text, metadata in zip(data["texts"], data["metadatas"])]
        return cls(documents, data["embeddings"], embeddings)

    def similarity_search_with_score(self, query, *, k):
        vector = self.embeddings.embed_query(query)
        distances, indices = self.neighbors.kneighbors([vector], n_neighbors=k)
        return [(self.documents[index], float(distance)) for index, distance in zip(indices[0], distances[0])]
