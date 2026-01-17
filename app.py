import json
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.rag_pipeline import HybridRAG

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Hybrid RAG – Interactive Dashboard",
    layout="wide"
)

st.title("Hybrid RAG System – Interactive Evaluation Dashboard")

# -----------------------------
# Load evaluation data
# -----------------------------
@st.cache_data
def load_eval_data():
    df = pd.read_csv("results/evaluation_results.csv")

    with open("results/metrics_summary.json") as f:
        metrics = json.load(f)

    with open("results/llm_judge_metrics.json") as f:
        judge = json.load(f)

    return df, metrics, judge

df, metrics, judge_metrics = load_eval_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

mode_filter = st.sidebar.multiselect(
    "Retrieval Mode",
    ["dense", "sparse", "hybrid"],
    default=["hybrid"]
)

question_filter = st.sidebar.multiselect(
    "Question Type",
    sorted(df["question_type"].unique()),
    default=sorted(df["question_type"].unique())
)

filtered_df = df[
    df["mode"].isin(mode_filter) &
    df["question_type"].isin(question_filter)
]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Live RAG Demo",
    "Metrics Overview",
    "Error Analysis",
    "Confidence & Hallucination"
])

# =====================================================
# TAB 1: LIVE RAG SYSTEM
# =====================================================
with tab1:
    st.subheader("Live Hybrid RAG Query")

    @st.cache_resource
    def load_rag():
        with open("data/corpus_chunks.json") as f:
            chunks = json.load(f)
        return chunks, HybridRAG(chunks)

    chunks, rag = load_rag()

    rag.mode = st.selectbox("Retrieval Mode", ["dense", "sparse", "hybrid"])
    rag.top_k = st.slider("Top-K Retrieval", 5, 20, rag.top_k)
    rag.top_n = st.slider("Chunks for Generation", 3, 8, rag.top_n)

    query = st.text_input("Enter a question:")

    if st.button("Run RAG") and query.strip():
        start = time.time()
        result = rag.run(query)
        latency = time.time() - start

        st.markdown("### Generated Answer")
        st.write(result["answer"])
        st.success(f"Latency: {latency:.2f} sec")

        st.markdown("### Retrieved Chunks")
        for rank, (idx, score) in enumerate(result["fused_hits"][:rag.top_n], start=1):
            c = chunks[idx]
            with st.expander(f"Rank {rank} | Score {score:.4f}"):
                st.markdown(f"**Title:** {c['title']}")
                st.markdown(f"**URL:** {c['url']}")
                st.write(c["text"])

# =====================================================
# TAB 2: METRICS OVERVIEW
# =====================================================
with tab2:
    st.subheader("Performance Metrics by Retrieval Mode")

    metrics_df = pd.DataFrame(metrics["metrics_by_mode"]).T
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("### LLM-as-Judge Scores")
    st.dataframe(pd.DataFrame(judge_metrics).T, use_container_width=True)

    st.markdown("### MRR Comparison")
    fig, ax = plt.subplots()
    metrics_df["MRR_URL_Level"].plot(kind="bar", ax=ax)
    ax.set_ylabel("MRR")
    st.pyplot(fig)

# =====================================================
# TAB 3: ERROR ANALYSIS
# =====================================================
with tab3:
    st.subheader("Error Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Error Type Distribution")
        fig, ax = plt.subplots()
        filtered_df["error_type"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### Error Type by Question Type")
        pivot = pd.crosstab(
            filtered_df["question_type"],
            filtered_df["error_type"]
        )
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds", ax=ax)
        st.pyplot(fig)

    st.markdown("### Failure Examples")
    failed = filtered_df[filtered_df["error_type"] != "success"]
    st.dataframe(
        failed[[
            "question",
            "question_type",
            "error_type",
            "generated_answer"
        ]].head(10),
        use_container_width=True
    )

# =====================================================
# TAB 4: CONFIDENCE & HALLUCINATION
# =====================================================
with tab4:
    st.subheader("Confidence Calibration")

    bins = pd.cut(filtered_df["confidence"], bins=10)
    calib = filtered_df.groupby(bins).agg(
        avg_conf=("confidence", "mean"),
        acc=("correct", "mean")
    ).dropna()

    fig, ax = plt.subplots()
    ax.plot(calib["avg_conf"], calib["acc"], marker="o", label="Model")
    ax.plot([0,1], [0,1], "--", color="gray", label="Perfect")
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Hallucination Rate by Mode")
    fig, ax = plt.subplots()
    filtered_df.groupby("mode")["hallucinated"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Hallucination Rate")
    st.pyplot(fig)

    st.markdown("### Entity Coverage Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["entity_coverage"], kde=True, ax=ax)
    st.pyplot(fig)

st.success("Interactive dashboard ready")
