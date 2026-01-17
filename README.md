# Hybrid RAG System with Advanced Evaluation

## Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** system that combines:

* **Dense Retrieval** using Sentence Transformers + FAISS
* **Sparse Retrieval** using BM25
* **Reciprocal Rank Fusion (RRF)** for hybrid ranking
* **Open-source LLM (FLAN-T5)** for answer generation

The system is evaluated using **retrieval metrics, generation quality metrics, LLM-as-Judge scoring, confidence calibration, hallucination detection, adversarial testing, ablation studies, and error analysis**, with both **static reports** and an **interactive Streamlit dashboard**.

---


## High-Level Pipeline Flow

```
┌────────────────────────┐
│   Wikipedia Categories │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ collect_wikipedia.py   │
│ - Fixed + Random URLs  │
│ - Subcategories        │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ raw_corpus.json        │
│ (Clean Wikipedia Text) │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ preprocess.py          │
│ - Chunking (300/50)    │
│ - URL + Chunk IDs      │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ corpus_chunks.json     │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ question_generation.py │
│ - Factual              │
│ - Comparative          │
│ - Inferential          │
│ - Multi-hop            │
│ - Adversarial          │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ questions_100.json     │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│ run_pipeline.py        │
│ Dense / Sparse / Hybrid│
└────────────┬───────────┘
             │
             ▼
┌───────────────────────────────────────────────┐
│ Evaluation                                    │
│ - MRR (URL-level)                             │
│ - Recall@K                                   │
│ - BERTScore                                  │
│ - Confidence Calibration                     │
│ - Custom Metrics                             │
│ - LLM-as-Judge                               │
└────────────┬──────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────┐
│ Outputs                                      │
│ CSV | JSON | HTML | Figures | Dashboard       │
└───────────────────────────────────────────────┘
```

---

## Detailed Retrieval & Generation Flow

```
User Question
     │
     ▼
┌───────────────┐
│ Query Encoder │
│ (Sentence-BERT) 
└───────┬───────┘
        │
        ▼
┌────────────────────┐      ┌──────────────────┐
│ Dense Retriever    │      │ Sparse Retriever │
│ (FAISS)            │      │ (BM25)           │
└─────────┬──────────┘      └─────────┬────────┘
          │                             │
          └──────────┬──────────────────┘
                     ▼
             ┌────────────────┐
             │ RRF Fusion     │
             │ (Rank-level)   │
             └───────┬────────┘
                     ▼
             ┌────────────────┐
             │ Top-N Chunks   │
             └───────┬────────┘
                     ▼
             ┌────────────────┐
             │ Generator LLM  │
             │ (FLAN-T5)      │
             └───────┬────────┘
                     ▼
               Generated Answer
```

---

## Evaluation & Analysis Flow

```
Generated Answer
      │
      ├── Retrieval Metrics
      │   ├─ MRR (URL-level)
      │   ├─ Recall@5, Recall@10
      │
      ├── Generation Metrics
      │   ├─ BERTScore
      │   ├─ Answer Diversity
      │
      ├── Faithfulness
      │   ├─ Entity Coverage
      │   ├─ Hallucination Rate
      │
      ├── LLM-as-Judge
      │   ├─ Accuracy
      │   ├─ Completeness
      │   ├─ Relevance
      │   └─ Coherence
      │
      ├── Confidence Calibration
      │   ├─ Confidence Score
      │   ├─ ECE
      │   └─ Correlation
      │
      └── Error Analysis
          ├─ Retrieval Failure
          ├─ Generation Failure
          └─ Context Mismatch
```

---


## Project Structure

```
Group_07_Hybrid_RAG/
│
├── data/
│   ├── fixed_urls.json
│   ├── random_urls.json
│   ├── raw_corpus.json
│   ├── corpus_chunks.json
│   └── questions_100.json
│
├── src/
│   ├── collect_wikipedia.py
│   ├── preprocess.py
│   ├── question_generation.py
│   ├── dense_retriever.py
│   ├── sparse_retriever.py
│   ├── rrf.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   ├── llm_judge.py
│   ├── confidence_calibration.py
│   ├── custom_metrics.py
│   ├── error_analysis.py
│   ├── report_generator.py
│   ├── plot_evaluation.py
│
├── run_pipeline.py
├── app.py
├── results/
│   ├── evaluation_results.csv
│   ├── metrics_summary.json
│   ├── llm_judge_metrics.json
│   ├── final_report.html
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## System Architecture

**Hybrid RAG Pipeline**

1. **Query Input**
2. **Dense Retrieval** (Sentence-Transformers)
3. **Sparse Retrieval** (BM25)
4. **Reciprocal Rank Fusion (RRF)**
5. **Context Selection**
6. **FLAN-T5 Answer Generation**
7. **Evaluation & Logging**

Supports **Dense-only**, **Sparse-only**, and **Hybrid** modes for ablation studies.

---

## Dataset Construction

### Wikipedia Corpus

* Category-based random sampling
* Fixed seed URLs for stability
* Concurrent fetching with retry logic
* Minimum content filtering

### Preprocessing

* Chunk size: **300 tokens**
* Overlap: **50 tokens**
* Chunk-level metadata preserved

---

## Question Generation (100 Q&A)

Diverse evaluation set generated from the corpus:

| Type                         | Count |
| ---------------------------- | ----- |
| Factual                      | 30    |
| Comparative                  | 15    |
| Inferential                  | 10    |
| Multi-hop                    | 15    |
| Ambiguous                    | 10    |
| Negated                      | 10    |
| Paraphrased Adversarial      | 10    |
| Unanswerable / Hallucination | 10    |

Additional adversarial variations:

* Paraphrased questions
* Ambiguous wording
* Negation
* Hallucination traps

Each question includes:

* Ground truth answer
* Source URLs
* Source chunk IDs
* Question category

---

## Evaluation Metrics

### Mandatory Metric

**Mean Reciprocal Rank (MRR) – URL Level**

* Measures how quickly the correct Wikipedia page is retrieved.

---

## Additional Metrics (Justified)

### **Recall@5 and Recall@10 (URL Level)**

**Why chosen:**
Recall@K measures **retrieval completeness**—whether the system is able to retrieve the correct source document within the top-K results.
At the URL level, this metric evaluates **document discovery**, independent of chunk granularity, making it more meaningful for RAG systems.

**Method:**

For each question:

* Identify the rank of the first correct Wikipedia URL.
* Recall@K is computed as:

```
Recall@K = (Number of questions where rank ≤ K) / (Total number of questions)
```

Evaluated at **K = 5 and K = 10**.

**Interpretation:**

* High Recall@5 → system retrieves correct sources quickly
* High Recall@10 → system is robust even with deeper retrieval
* Low recall → retrieval failure, even if generation seems fluent

---

### **BERTScore (Generation Quality)**

**Why chosen:**
Exact string matching is insufficient for open-ended QA.
BERTScore evaluates **semantic similarity** between generated answers and ground-truth answers using contextual embeddings.

**Method:**

* Compute token-level cosine similarity using a pretrained BERT model.
* Precision, Recall, and F1 are computed.
* We report **BERTScore-F1**.

```
BERTScore_F1 = mean semantic similarity between generated and reference answers
```

**Interpretation:**

* High score → semantically correct and well-phrased answers
* Lower score → missing key concepts or incorrect meaning
* Useful even when wording differs from ground truth

---

### Additional Custom Metrics

#### **Entity Coverage Score**

**Why chosen:**
Evaluates factual grounding by measuring overlap between entities in ground truth and generated answers.

**Method:**

```
Entity Coverage = |Entities(prediction ∩ reference)| / |Entities(reference)|
```

**Interpretation:**

* High score → grounded, factual answers
* Low score → missing or hallucinated content

---

#### **Hallucination Rate**

**Why chosen:**
Directly measures unsafe or fabricated answers.

**Method:**
An answer is flagged hallucinated if:

* Retrieval failed AND
* Entity coverage is below a threshold

**Interpretation:**

* Lower is better
* Critical for trustworthiness

---

### **Latency (Efficiency Metric)**

**Why chosen:**
Practical RAG systems must balance **accuracy and responsiveness**.
Latency captures the **end-to-end response time** of retrieval + generation.

**Method:**

```
Latency = time(after answer generated) − time(query issued)
```

Measured per query and summarized using:

* Mean latency
* Distribution plots

**Interpretation:**

* Lower latency → better user experience
* High latency may indicate:

  * Expensive embedding
  * Large retrieval K
  * Slow generation

---

### **Answer Diversity**

**Why chosen:**
Ensures the system does not produce **overly repetitive or template-based answers**, especially across multiple questions.

**Method:**

Let `U` be the number of unique answers and `N` be total answers:

```
Answer Diversity = |unique generated answers| / |total answers|
```

**Interpretation:**

* High diversity → system adapts answers to different contexts
* Low diversity → over-reliance on generic responses
* Useful to detect mode collapse in generation

---

### **Confidence Calibration (Expected Calibration Error – ECE)**

**Why chosen:**
A trustworthy system should be **confident when correct** and **uncertain when wrong**.
ECE measures how well predicted confidence aligns with empirical correctness.

**Method:**

1. Assign a confidence score to each answer (based on retrieval rank).
2. Bin predictions into confidence intervals.
3. Compute:

```
ECE = Σ |accuracy(bin) − confidence(bin)| × (bin_size / total_samples)
```

**Interpretation:**

* ECE ≈ 0 → well-calibrated system
* High ECE → overconfidence or underconfidence
* Critical for decision-making and human-AI trust

---

### **Confidence–Correctness Correlation**

**Why chosen:**
Measures whether **higher confidence actually corresponds to correctness**, beyond calibration alone.

**Method:**

```
Correlation = Pearson(confidence scores, correctness labels)
```

Where:

* Correctness = 1 if correct URL retrieved, else 0

**Interpretation:**

* High positive correlation → confidence is meaningful
* Near zero or negative → confidence estimates are unreliable
* Helps validate confidence scoring design

---

## Summary of Metric Coverage

| Metric                 | Evaluates                 |
| ---------------------- | ------------------------- |
| MRR                    | Retrieval ranking quality |
| Recall@K               | Retrieval completeness    |
| BERTScore              | Semantic answer quality   |
| Entity Coverage        | Grounded factuality       |
| Hallucination Rate     | Safety & trust            |
| Latency                | Efficiency                |
| Answer Diversity       | Generative robustness     |
| ECE                    | Calibration               |
| Confidence Correlation | Confidence reliability    |

---

## LLM-as-Judge Evaluation

An LLM (FLAN-T5) evaluates generated answers on:

* Factual Accuracy
* Completeness
* Relevance
* Coherence

Scores (1–5) are averaged per retrieval mode and included in reports.

---

## Error Analysis

Failures are categorized as:

* Retrieval failure
* Generation failure
* Context mismatch
* Hallucination

Breakdowns provided:

* By question type
* By retrieval mode
* With example failure cases

---

## Advanced Experiments

✔ Adversarial Testing
✔ Dense vs Sparse vs Hybrid Ablation
✔ Confidence Calibration Curves
✔ Error Heatmaps
✔ Hallucination Detection
✔ LLM-as-Judge Scoring

---

## Visualizations (Auto-Generated)

* MRR comparison
* Recall@K curves
* Latency distribution
* Calibration curve
* Error distribution
* Error heatmap by question type
* LLM-as-Judge scores

Generated via:

```bash
python plot_evaluation.py
```

---

## Interactive Dashboard

Run the Streamlit app:

```bash
streamlit run app.py
```

Features:

* Real-time querying
* Dense / Sparse / Hybrid comparison
* Retrieval explanations
* Chunk-level inspection
* Latency tracking

---

## One-Command Pipeline

The entire Hybrid RAG system—from data ingestion to evaluation and reporting—can be executed using a **single command**, with optional rebuilding of the dataset.

---

### **Full Rebuild + Evaluation**

```bash
python run_pipeline.py --rebuild
```

**When to use:**

* First-time setup
* After modifying:

  * Wikipedia sampling logic
  * Chunking strategy
  * Question generation (new question types, adversarial cases)
* When dataset versions must be regenerated for reproducibility

**What it does:**

1. Collects a fresh Wikipedia corpus
2. Preprocesses and chunks documents
3. Regenerates 100+ diverse evaluation questions
4. Runs RAG evaluation (Dense / Sparse / Hybrid)
5. Computes all metrics
6. Generates reports and visualizations

**Outputs:**

* `results/evaluation_results.csv`
* `results/metrics_summary.json`
* `results/llm_judge_metrics.json`
* `results/final_report.html`
* `results/figures/*.png`
* Dataset checksums for version tracking

---

### **Evaluation Only (No Rebuild)**

```bash
python run_pipeline.py
```

**When to use:**

* After dataset is already built
* When experimenting with:

  * Retrieval parameters (K, N, RRF-k)
  * Dense vs Sparse vs Hybrid comparison
  * New evaluation metrics or plots
  * Confidence calibration and error analysis
* For rapid iteration and debugging

**Why this mode exists:**

* Avoids **expensive recomputation** (Wikipedia crawling, LLM-based question generation)
* Enables **fast experimentation** on a fixed dataset
* Ensures **fair comparison** across models using the same corpus and questions

**What it does:**

* Loads existing:

  * `corpus_chunks.json`
  * `questions_100.json`
* Runs evaluation only
* Updates metrics, plots, and reports

---

### Reproducibility & Dataset Versioning

Each run records:

* SHA-256 checksum of:

  * Corpus chunks
  * Question set
* Timestamp
* Number of chunks and questions

This guarantees:

* Exact reproducibility
* Traceable evaluation results
* Fair ablation studies across system variants

---

### Design Rationale

| Mode               | Purpose                             |
| ------------------ | ----------------------------------- |
| `--rebuild`        | Dataset regeneration, major changes |
| No flag            | Fast evaluation, parameter tuning   |
| Checksums          | Version control for data            |
| Single entry point | CI/CD & automation friendly         |

---

## Evaluation Report

Generated at:

```
results/final_report.html
```

Includes:

* Overall performance summary
* Metric justifications
* Ablation study results
* Error analysis with examples
* Visualizations
* Architecture description
* Innovation highlights

---

## Reproducibility & Versioning

* Dataset checksums logged
* Configuration fixed
* Deterministic evaluation
* Fully automated pipeline

---

## Conclusion

This project demonstrates a **rHybrid RAG system** with:

* Robust retrieval
* Grounded generation
* Deep evaluation
* Explainability
* Practical deployment via Streamlit

---
