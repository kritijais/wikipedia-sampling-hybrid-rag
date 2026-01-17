import pandas as pd
import json
from jinja2 import Template
from pathlib import Path

def generate_html_report(
    df: pd.DataFrame,
    metrics: dict,
    judge_metrics: dict,
    output_path: str
):
    """
    Generates a comprehensive HTML evaluation report for the Hybrid RAG system.
    """

    metrics_by_mode = metrics["metrics_by_mode"]
    dataset_version = metrics["dataset_version"]

    # -----------------------------
    # Overall averages (cross-mode)
    # -----------------------------
    overall_summary = {
        "MRR_avg": round(
            sum(m["MRR"] for m in metrics_by_mode.values()) / len(metrics_by_mode), 4
        ),
        "Entity_Coverage_avg": round(df["entity_coverage"].mean(), 4),
        "Hallucination_Rate_avg": round(df["hallucinated"].mean(), 4),
        "Avg_Latency_sec": round(df["latency_sec"].mean(), 3),
        "LLM_Judge_Accuracy_avg": round(
            sum(j["LLM_Judge_Accuracy"] for j in judge_metrics.values()) / len(judge_metrics), 3
        ),
        "LLM_Judge_Relevance_avg": round(
            sum(j["LLM_Judge_Relevance"] for j in judge_metrics.values()) / len(judge_metrics), 3
        )
    }

    # -----------------------------
    # Sample failures
    # -----------------------------
    failures = (
        df[df["error_type"] != "success"]
        .head(10)
        .to_dict(orient="records")
    )

    # -----------------------------
    # HTML Template
    # -----------------------------
    template = Template("""
    <html>
    <head>
        <title>Hybrid RAG – Evaluation Report</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            th, td { border: 1px solid #ccc; padding: 8px; font-size: 14px; }
            th { background-color: #f4f6f7; }
            .metric-box { background: #eef2f3; padding: 15px; margin: 10px 0; }
            img { max-width: 100%; margin-bottom: 20px; border: 1px solid #ccc; }
        </style>
    </head>

    <body>

    <h1>Hybrid RAG System – Evaluation Report</h1>

    <h2>1. Overall Performance Summary</h2>
    <div class="metric-box">
        <b>Average MRR (URL Level):</b> {{ summary.MRR_avg }}<br>
        <b>Average Entity Coverage:</b> {{ summary.Entity_Coverage_avg }}<br>
        <b>Average Hallucination Rate:</b> {{ summary.Hallucination_Rate_avg }}<br>
        <b>Average Latency (sec):</b> {{ summary.Avg_Latency_sec }}<br>
        <b>LLM Judge Accuracy (avg):</b> {{ summary.LLM_Judge_Accuracy_avg }}<br>
        <b>LLM Judge Relevance (avg):</b> {{ summary.LLM_Judge_Relevance_avg }}
    </div>

    <h2>2. Dataset Versioning</h2>
    <pre>{{ dataset_version }}</pre>

    <h2>3. Retrieval & Generation Metrics (Ablation Study)</h2>
    <pre>{{ metrics_by_mode }}</pre>

    <h2>4. LLM-as-Judge Evaluation</h2>
    <p>
    An LLM (FLAN-T5) was used as an automatic evaluator to assess:
    factual accuracy, completeness, relevance, and coherence.
    Scores are averaged over the first 20 questions per mode.
    </p>
    <pre>{{ judge_metrics }}</pre>

    <h2>5. Question-Level Results (Sample)</h2>
    {{ results_table }}

    <h2>6. Visualizations</h2>
    <img src="../figures/mrr_by_mode.png">
    <img src="../figures/recall_at_k.png">
    <img src="../figures/latency_distribution.png">
    <img src="../figures/calibration_curve.png">
    <img src="../figures/error_distribution.png">
    <img src="../figures/error_heatmap.png">

    <h2>7. Error Analysis – Failure Examples</h2>
    <table>
        <tr>
            <th>Question</th>
            <th>Type</th>
            <th>Error</th>
            <th>Generated Answer</th>
        </tr>
        {% for f in failures %}
        <tr>
            <td>{{ f.question }}</td>
            <td>{{ f.question_type }}</td>
            <td>{{ f.error_type }}</td>
            <td>{{ f.generated_answer }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>8. Architecture & Innovation</h2>
    <ul>
        <li>Hybrid Dense + Sparse Retrieval with RRF</li>
        <li>LLM-as-Judge evaluation</li>
        <li>Confidence calibration (ECE)</li>
        <li>Hallucination-aware metrics</li>
        <li>Adversarial & unanswerable question testing</li>
        <li>Interactive Streamlit dashboard</li>
    </ul>

    <h2>9. System Screenshots</h2>
    <p>(Add screenshots of Streamlit app, dashboard, error analysis)</p>

    </body>
    </html>
    """)

    html = template.render(
        summary=overall_summary,
        dataset_version=json.dumps(dataset_version, indent=2),
        metrics_by_mode=json.dumps(metrics_by_mode, indent=2),
        judge_metrics=json.dumps(judge_metrics, indent=2),
        failures=failures,
        results_table=df.head(30).to_html(index=False)
    )

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Evaluation report generated at {output_path}")
