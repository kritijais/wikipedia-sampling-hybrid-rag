from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.rrf import rrf
from src.generator import Generator

class HybridRAG:
    """
    Hybrid RAG with reusable dense index
    """
    def __init__(
            self, 
            chunks, 
            dense_retriever=None,
            sparse_retriever=None, 
            mode="hybrid", 
            top_k=10, 
            top_n=8, 
            rrf_k=60):
        self.chunks = chunks
        self.mode = mode
        self.top_k = top_k
        self.top_n = top_n
        self.rrf_k = rrf_k

        print(f"Embedding {len(chunks)} chunks...")
        # -------------------------
        # Reuse dense index
        # -------------------------
        if dense_retriever is None:
            print("Building dense index (once)...")
            self.dense = DenseRetriever()
            self.dense.build_index(chunks)
        else:
            self.dense = dense_retriever

        # Sparse is cheap, OK to rebuild
        self.sparse = sparse_retriever or SparseRetriever(chunks)

        self.generator = Generator()

    def run(self, query):
        dense_hits = self.dense.search(query, self.top_k)
        sparse_hits = self.sparse.search(query, self.top_k)

        if self.mode == "dense":
            fused = dense_hits
        elif self.mode == "sparse":
            fused = sparse_hits
        else:
            fused = rrf(dense_hits, sparse_hits, self.rrf_k)

        contexts = [self.chunks[idx]["text"] for idx, _ in fused[:self.top_n]]
        answer = self.generator.generate(query, contexts)

        return {
            "answer": answer,
            "dense_hits": dense_hits,
            "sparse_hits": sparse_hits,
            "fused_hits": fused
        }
