import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_DIR = "results"
FIG_DIR = f"{RESULTS_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set(style="whitegrid")

# -----------------------------
# Load results
# -----------------------------
df = pd.read_csv(f"{RESULTS_DIR}/evaluation_results.csv")

with open(f"{RESULTS_DIR}/metrics_summary.json") as f:
    metrics = json.load(f)

judge_path = f"{RESULTS_DIR}/llm_judge_metrics.json"
judge_metrics = json.load(open(judge_path)) if os.path.exists(judge_path) else {}

# -----------------------------
# Helper
# -----------------------------
def save(fig, name):
    path = f"{FIG_DIR}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# =====================================================
# RETRIEVAL QUALITY
# =====================================================

# -----------------------------
# MRR by Retrieval Mode
# -----------------------------
mrr_by_mode = (
    df[df["rank_of_correct_url"] > 0]
    .assign(rr=lambda x: 1 / x["rank_of_correct_url"])
    .groupby("mode")["rr"]
    .mean()
)

fig = plt.figure()
mrr_by_mode.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.title("MRR by Retrieval Mode (URL Level)")
plt.ylabel("MRR")
plt.xlabel("Mode")
save(fig, "mrr_by_mode.png")

# -----------------------------
# Recall@K Curve (Global)
# -----------------------------
ks = [1, 3, 5, 10]
recall = {
    k: (df["rank_of_correct_url"] <= k).mean()
    for k in ks
}

fig = plt.figure()
plt.plot(list(recall.keys()), list(recall.values()), marker="o")
plt.title("Recall@K (URL Level)")
plt.xlabel("K")
plt.ylabel("Recall")
save(fig, "recall_at_k.png")

# =====================================================
# EFFICIENCY
# =====================================================

# -----------------------------
# Latency Distribution
# -----------------------------
fig = plt.figure()
sns.histplot(df["latency_sec"], bins=30, kde=True)
plt.title("Response Latency Distribution")
plt.xlabel("Latency (seconds)")
save(fig, "latency_distribution.png")

# =====================================================
# ERROR ANALYSIS
# =====================================================

# -----------------------------
# Error Type Distribution
# -----------------------------
fig = plt.figure()
df["error_type"].value_counts().plot(kind="bar")
plt.title("Error Type Distribution")
plt.xlabel("Error Type")
plt.ylabel("Count")
save(fig, "error_distribution.png")

# -----------------------------
# Error Type by Retrieval Mode
# -----------------------------
pivot_mode = pd.crosstab(df["mode"], df["error_type"])

fig = plt.figure(figsize=(8, 5))
sns.heatmap(pivot_mode, annot=True, fmt="d", cmap="Blues")
plt.title("Error Type by Retrieval Mode")
plt.xlabel("Error Type")
plt.ylabel("Mode")
save(fig, "error_by_mode_heatmap.png")

# -----------------------------
# Error Type by Question Type
# -----------------------------
pivot_qtype = (
    df[df["error_type"] != "success"]
    .pivot_table(
        index="question_type",
        columns="error_type",
        aggfunc="size",
        fill_value=0
    )
)

fig = plt.figure(figsize=(9, 5))
sns.heatmap(pivot_qtype, annot=True, fmt="d", cmap="Reds")
plt.title("Error Type by Question Type")
plt.ylabel("Question Type")
plt.xlabel("Error Type")
save(fig, "error_heatmap.png")

# -----------------------------
# Success Rate by Question Type
# -----------------------------
success = df.assign(success=lambda x: x["error_type"] == "success")
rate = success.groupby("question_type")["success"].mean()

fig = plt.figure()
rate.plot(kind="bar", color="#2ca02c")
plt.title("Success Rate by Question Type")
plt.ylabel("Success Rate")
plt.xlabel("Question Type")
save(fig, "success_by_question_type.png")

# -----------------------------
# Rank Histogram
# -----------------------------
fig = plt.figure()
sns.histplot(
    df[df["rank_of_correct_url"] > 0]["rank_of_correct_url"],
    bins=20
)
plt.title("Rank of Correct URL")
plt.xlabel("Rank")
save(fig, "rank_histogram.png")

# =====================================================
# GENERATION QUALITY
# =====================================================

# -----------------------------
# LLM-as-Judge Scores (by mode)
# -----------------------------
if judge_metrics:
    judge_df = pd.DataFrame(judge_metrics).T

    fig, ax = plt.subplots(figsize=(8, 4))
    judge_df.plot(kind="bar", ax=ax)
    ax.set_title("LLM-as-Judge Scores by Retrieval Mode")
    ax.set_ylabel("Score (1–5)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    save(fig, "llm_judge_scores.png")

# -----------------------------
# Entity Coverage Distribution
# -----------------------------
fig = plt.figure()
sns.histplot(df["entity_coverage"], bins=30, kde=True)
plt.title("Entity Coverage Distribution")
plt.xlabel("Entity Coverage")
save(fig, "entity_coverage_distribution.png")

# -----------------------------
# Hallucination Rate by Mode
# -----------------------------
halluc_by_mode = df.groupby("mode")["hallucinated"].mean()

fig = plt.figure()
halluc_by_mode.plot(kind="bar", color="#d62728")
plt.title("Hallucination Rate by Retrieval Mode")
plt.ylabel("Hallucination Rate")
plt.xlabel("Mode")
save(fig, "hallucination_rate_by_mode.png")

# =====================================================
# CONFIDENCE CALIBRATION
# =====================================================

# -----------------------------
# Confidence Calibration Curve
# -----------------------------
bins = np.linspace(0, 1, 11)
df["confidence_bin"] = pd.cut(df["confidence"], bins)

calib = df.groupby("confidence_bin").agg(
    avg_confidence=("confidence", "mean"),
    accuracy=("correct", "mean")
).dropna()

fig = plt.figure()
plt.plot(calib["avg_confidence"], calib["accuracy"], marker="o", label="Model")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
plt.xlabel("Predicted Confidence")
plt.ylabel("Empirical Accuracy")
plt.title("Confidence Calibration Curve")
plt.legend()
save(fig, "calibration_curve.png")

# =====================================================
# DIVERSITY
# =====================================================

# -----------------------------
# Answer Length & Diversity Proxy
# -----------------------------
df["answer_length"] = df["generated_answer"].str.split().apply(len)

fig = plt.figure()
sns.boxplot(x="mode", y="answer_length", data=df)
plt.title("Answer Length Distribution by Mode")
plt.ylabel("Answer Length (words)")
save(fig, "answer_length_by_mode.png")

print("\nAll evaluation plots generated successfully.")
