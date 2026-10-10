
import faiss
import numpy as np
from embeddings import embed_texts, embed_query


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks):
        """Create a FAISS index from document chunks."""
        if not chunks:
            raise ValueError("No document chunks available to index.")

        texts = [chunk["text"] for chunk in chunks]
        vectors = embed_texts(texts)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.ascontiguousarray(vectors))
        self.chunks = chunks

    def search(self, question, top_k=3):
        """Find the most similar document chunks."""
        if self.index is None or not self.chunks:
            raise ValueError("Build the vector store before searching.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = embed_query(question).reshape(1, -1)
        scores, indices = self.index.search(
            np.ascontiguousarray(query_vector), 
            min(top_k, len(self.chunks))
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(score),
            })

        return results
