from bert_score import score

def mrr(ranks):
    valid = [r for r in ranks if r > 0]
    return sum(1 / r for r in valid) / len(valid) if valid else 0.0

def recall_at_k(ranks, k):
    valid = [r for r in ranks if r > 0]
    return sum(1 for r in valid if r <= k) / len(valid) if valid else 0.0

def bertscore(preds, refs):
    P, R, F1 = score(preds, refs, lang="en")
    return F1.mean().item()

def categorize_error(rank, qtype=None):
    if qtype == "unanswerable":
        return "unanswerable"
    if rank == 0:
        return "retrieval_failure"
    return "correct"
