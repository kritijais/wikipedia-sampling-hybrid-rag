# RRF implementation
def rrf(dense, sparse, k=60):
    scores = {}
    for rank, (idx, _) in enumerate(dense):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(sparse):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
