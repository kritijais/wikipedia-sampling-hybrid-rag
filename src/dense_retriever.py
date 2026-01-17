import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Dense Retriever using Sentence Transformers and FAISS
class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build FAISS index
    def build_index(self, chunks):
        embeddings = self.model.encode([c["text"] for c in chunks])
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.embeddings = embeddings

    # Search for top-k similar chunks
    def search(self, query, k=10):
        q_emb = self.model.encode([query])
        scores, idx = self.index.search(q_emb, k)
        return list(zip(idx[0], scores[0]))
