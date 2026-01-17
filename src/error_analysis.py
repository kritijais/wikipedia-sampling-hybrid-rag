# # Classify every failed question into one dominant error category:
# retrieval_failure - Correct Wikipedia URL not retrieved
# context_failure - Correct URL retrieved, but not used in top-N chunks
# generation_failure - Correct context used, but answer is wrong / hallucinated
# success - Correct retrieval + acceptable answer
def categorize_error_detailed(
    rank,
    correct_url,
    top_context_urls,
    entity_coverage,
    judge_accuracy=None
):
    if rank == 0:
        return "retrieval_failure"

    if correct_url not in top_context_urls:
        return "context_failure"

    if entity_coverage < 0.3:
        return "generation_failure"

    if judge_accuracy is not None and judge_accuracy <= 2:
        return "generation_failure"

    return "success"
