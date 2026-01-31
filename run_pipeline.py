import json
import time
import os
import argparse
import hashlib
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.rag_pipeline import HybridRAG
from src.evaluation import mrr, recall_at_k, bertscore
from src.llm_judge import judge_answer
from src.report_generator import generate_html_report

from src.confidence_calibration import (
    compute_confidence,
    expected_calibration_error,
    confidence_correlation
)

from src.custom_metrics import (
    entity_coverage_score,
    is_hallucinated,
    answer_diversity,
    hallucination_rate
)

from src.error_analysis import categorize_error_detailed

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"

TOP_K = 10
TOP_N = 5
RRF_K = 60

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Hybrid RAG Evaluation Pipeline")
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()

if args.rebuild:
    print("Rebuilding dataset...")
    os.system("python src/collect_wikipedia.py")
    os.system("python src/preprocess.py")
    os.system("python src/question_generation.py")
    print("Rebuild completed\n")

# -----------------------------
# Utility
# -----------------------------
def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------
# Load data
# -----------------------------
with open(f"{DATA_DIR}/corpus_chunks.json") as f:
    chunks = json.load(f)

with open(f"{DATA_DIR}/questions_100.json") as f:
    questions = json.load(f)

dataset_version = {
    "corpus_checksum": checksum(f"{DATA_DIR}/corpus_chunks.json"),
    "questions_checksum": checksum(f"{DATA_DIR}/questions_100.json"),
    "num_chunks": len(chunks),
    "num_questions": len(questions),
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

print("Dataset Version:")
for k, v in dataset_version.items():
    print(f"  {k}: {v}")

print("Building shared dense index...")
shared_dense = DenseRetriever()
shared_dense.build_index(chunks)

shared_sparse = SparseRetriever(chunks)

# -----------------------------
# Evaluation
# -----------------------------
all_rows = []
metrics_by_mode = {}
llm_judge_by_mode = {}

for mode in ["dense", "sparse", "hybrid"]:
    print(f"\n==============================")
    print(f" Evaluating mode: {mode.upper()}")
    print(f"==============================")

    rag = HybridRAG(
        chunks,
        dense_retriever=shared_dense,
        sparse_retriever=shared_sparse,
        mode=mode,
        top_k=TOP_K,
        top_n=TOP_N,
        rrf_k=RRF_K
    )

    url_ranks = []
    pred_answers = []
    gt_answers = []

    confidence_scores = []
    correctness_labels = []
    hallucination_flags = []
    all_answers = []

    judge_scores = {
        "accuracy": [],
        "completeness": [],
        "relevance": [],
        "coherence": [],
        "explanation": []
    }

    for q in tqdm(questions, desc=mode.upper()):
        question = q["question"]
        gt_urls = q.get("source_urls", [])
        gt_answer = q["answer"]

        start = time.time()
        result = rag.run(question)
        latency = time.time() - start

        fused = result["fused_hits"]
        answer = result["answer"]

        # -----------------------------
        # URL-level ranking
        # -----------------------------
        ranked_urls = []
        for idx, _ in fused:
            u = chunks[idx]["url"]
            if u not in ranked_urls:
                ranked_urls.append(u)

        rank = 0
        for i, u in enumerate(ranked_urls, start=1):
            if u in gt_urls:
                rank = i
                break

        # -----------------------------
        # Context URLs
        # -----------------------------
        top_context_urls = [
            chunks[idx]["url"]
            for idx, _ in fused[:TOP_N]
        ]

        # -----------------------------
        # LLM-as-Judge (first 20)
        # -----------------------------
        judge_result = None
        if q["id"] < 20:
            judge_result = judge_answer(question, gt_answer, answer)

            # print(f"\n[LLM Judge] QID: {q['id']}")
            # print(f"Question: {question}")
            # print(f"Ground Truth Answer: {gt_answer}")
            # print(f"Predicted Answer: {answer}")
            # print(f"Judge Result: {judge_result}")

            if not isinstance(judge_result, dict):
                judge_result = {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "coherence": 0,
                    "explanation": "Invalid judge output",
                }

            for k in judge_scores:
                judge_scores[k].append(judge_result.get(k, 0))

        # -----------------------------
        # Novel Metrics
        # -----------------------------
        entity_cov = entity_coverage_score(gt_answer, answer)
        hallucinated = is_hallucinated(rank, entity_cov)

        # -----------------------------
        # Error Analysis
        # -----------------------------
        error_type = categorize_error_detailed(
            rank=rank,
            correct_url=gt_urls[0] if gt_urls else None,
            top_context_urls=top_context_urls,
            entity_coverage=entity_cov,
            judge_accuracy=judge_result["accuracy"] if judge_result else None
        )

        # -----------------------------
        # Confidence Calibration
        # -----------------------------
        confidence = compute_confidence(rank)
        correct = 1 if rank > 0 else 0

        # -----------------------------
        # Save row
        # -----------------------------
        row = {
            "mode": mode,
            "question_id": q["id"],
            "question": question,
            "question_type": q["question_type"],
            "rank_of_correct_url": rank,
            "generated_answer": answer,
            "latency_sec": round(latency, 3),
            "entity_coverage": round(entity_cov, 3),
            "hallucinated": int(hallucinated),
            "confidence": round(confidence, 3),
            "correct": correct,
            "error_type": error_type
        }

        if judge_result:
            row.update({
                "judge_accuracy": judge_result["accuracy"],
                "judge_completeness": judge_result["completeness"],
                "judge_relevance": judge_result["relevance"],
                "judge_coherence": judge_result["coherence"],
                "judge_explanation": judge_result["explanation"]
            })

        all_rows.append(row)

        if q["question_type"] != "unanswerable":
            url_ranks.append(rank)
            pred_answers.append(answer)
            gt_answers.append(gt_answer)

        confidence_scores.append(confidence)
        correctness_labels.append(correct)
        hallucination_flags.append(hallucinated)
        all_answers.append(answer)

    # -----------------------------
    # Metrics for mode
    # -----------------------------
    metrics_by_mode[mode] = {
        "MRR_URL_Level": round(mrr(url_ranks), 4),
        "Recall@5_URL": round(recall_at_k(url_ranks, 5), 4),
        "Recall@10_URL": round(recall_at_k(url_ranks, 10), 4),
        "BERTScore_F1": round(bertscore(pred_answers, gt_answers), 4),
        "ECE": round(
            expected_calibration_error(confidence_scores, correctness_labels), 4
        ),
        "Confidence_Correctness_Correlation": round(
            confidence_correlation(confidence_scores, correctness_labels), 4
        ),
        "Answer_Diversity": round(answer_diversity(all_answers), 4),
        "Hallucination_Rate": round(hallucination_rate(hallucination_flags), 4)
    }

    # llm_judge_by_mode[mode] = {
    #     k: round(sum(v) / len(v), 3) if v else 0.0
    #     for k, v in judge_scores.items()
    # }

    NUMERIC_METRICS = {
        'accuracy',
        'completeness',
        'relevance',
        'coherence'
    }

    llm_judge_by_mode[mode] = {}
    for k, v in judge_scores.items():
        if k in NUMERIC_METRICS:
            numeric_vals = [x for x in v if isinstance(x, (int, float))]
            llm_judge_by_mode[mode][k] = round(sum(numeric_vals) / len(numeric_vals), 3) if numeric_vals else 0.0

# -----------------------------
# Save Outputs
# -----------------------------
df = pd.DataFrame(all_rows)
df.to_csv(f"{RESULTS_DIR}/evaluation_results.csv", index=False)

with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
    json.dump(
        {
            "metrics_by_mode": metrics_by_mode,
            "dataset_version": dataset_version
        },
        f,
        indent=2
    )

with open(f"{RESULTS_DIR}/llm_judge_metrics.json", "w") as f:
    json.dump(llm_judge_by_mode, f, indent=2)

# -----------------------------
# Generate Evaluation Plots
# -----------------------------
print("\nGenerating evaluation plots...")
os.system("python src/plot_evaluation.py")

# -----------------------------
# HTML Report
# -----------------------------
generate_html_report(
    df=df,
    metrics={
        "metrics_by_mode": metrics_by_mode,
        "dataset_version": dataset_version
    },
    judge_metrics=llm_judge_by_mode,
    output_path=f"{RESULTS_DIR}/final_report.html"
)

print("\nPipeline completed successfully")
