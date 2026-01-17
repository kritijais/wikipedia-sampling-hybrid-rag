from rank_bm25 import BM25Okapi
import nltk

# Sparse Retriever using BM25
class SparseRetriever:
    def __init__(self, chunks):
        # tokenized = [nltk.word_tokenize(c["text"].lower()) for c in chunks]
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    # Search for top-k similar chunks
    def search(self, query, k=10):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
