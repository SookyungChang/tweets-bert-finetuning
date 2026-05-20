# app.py — Gradio version for Hugging Face Spaces
import os
import sys
from pathlib import Path
import hdbscan
import gradio as gr
import pandas as pd
from huggingface_hub import snapshot_download
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent.parent.parent
sys.path.append(str(parent_dir))

from src.data.get_comments import get_comments, filter_english_comments
from src.inference.predictor_bert import Predictor
from src.config import urlConfig
from src.analysis.topic_modeling import get_topics
from src.database.storage import save_results, save_to_vectors
from src.analysis.rag import ask_to_llm


# ─────────────────────────────────────────
# Model loading (once at startup)
# ─────────────────────────────────────────

print("⏳ Loading BERT model...")
bert_path = snapshot_download(repo_id=urlConfig.trained_model_id)
bert_model = Predictor(bert_path)
SHARED_EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)  # This model will be used to do for BERTopic and Chroma
print("✅ BERT model loaded!")


@lru_cache(maxsize=8)
def load_hdbscan_model(min_cluster_size: int):
    """Cache HDBSCAN models by cluster size to avoid re-creating them."""
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True
    )


# ─────────────────────────────────────────
# Shared state (replaces st.session_state)
# ─────────────────────────────────────────

# Holds the last processed results so Tab 2 can access them.
# _session: dict = {}


# ─────────────────────────────────────────
# Helper — URL → video ID
# ─────────────────────────────────────────


def extract_video_id(raw: str) -> str:
    raw = raw.strip()
    if "youtube.com" in raw:
        from urllib.parse import urlparse, parse_qs

        params = parse_qs(urlparse(raw).query)
        return params.get("v", [raw])[0]
    if "youtu.be" in raw:
        return raw.split("youtu.be/")[-1].split("?")[0]
    return raw


# ─────────────────────────────────────────
# Tab 1 — Analyze Video
# ─────────────────────────────────────────


def analyze_video(video_input: str, session_state: dict, progress=None):
    """
    Fetch comments → sentiment → topics → save.

    Returns:
        status_md    : str            — status / error message
        metrics_md   : str            — markdown table (total / pos / neg)
        chart_df     : pd.DataFrame   — for gr.BarPlot
        pos_topics   : pd.DataFrame   — positive topic table
        neg_topics   : pd.DataFrame   — negative topic table
        video_id     : str            — pre-fills video ID in Tab 2
        session_state: dict           — Individual storage
    """
    EMPTY = (
        "",
        "",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        session_state,
    )  ## Error return

    if not video_input or not video_input.strip():
        video_input = "https://www.youtube.com/watch?v=XqYTfpxFuDM"
        video_id = extract_video_id(video_input)
        # return ("⚠️ Please enter a YouTube video ID or URL.",) + EMPTY[1:]
    else:
        video_id = extract_video_id(video_input)

    # ── Fetch comments ──────────────────
    if progress is not None:
        progress(0.1, desc="⏳ 1/4 Fetching YouTube comments...")
    try:
        df = get_comments(video_id, max_results=100, max_pages=15)
        df = filter_english_comments(df)
    except Exception as e:
        return (f"❌ Failed to fetch comments: {e}",) + EMPTY[1:]

    if df.empty:
        return ("⚠️ No English comments found for this video.",) + EMPTY[1:]

    # ── Sentiment ───────────────────────
    if progress is not None:
        progress(0.4, desc="🧠 2/4 Running BERT sentiment analysis...")
    try:
        df_sen = bert_model.predict_df(df)
    except Exception as e:
        return (f"❌ Sentiment prediction failed: {e}",) + EMPTY[1:]

    # ── Topic modeling ──────────────────
    if progress is not None:
        progress(0.6, desc="🔮 3/4 Topic modeling with BERTopic (UMAP/HDBSCAN)...")
    all_dfs = []
    topic_summary: dict = {}

    try:
        for label in df_sen["sentiment_label"].unique():
            count = len(df_sen[df_sen["sentiment_label"] == label])
            size = len(str(count)) + 1
            hdbscan_model = load_hdbscan_model(min_cluster_size=size)

            df_add, topic_model = get_topics(
                df_sen,
                sentiment_label=label,
                min_cluster_size=size,
                hdbscan_model=hdbscan_model,
                embedding_model=SHARED_EMBEDDING_MODEL,
            )

            label_name = "✅ Positive" if label == 1 else "❌ Negative"
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info["Topic"] != -1]
            topic_summary[label_name] = topic_info

            save_df = df_add.drop(columns=["language"], errors="ignore")
            save_results(
                video_id,
                save_df[save_df["topic"] != -1],
                topic_model.get_topic_info(),
                sentiment_label=label,
            )
            all_dfs.append(df_add)

        if progress is not None:
            progress(0.9, desc="💾 4/4 Building a Vector DB for RAG integration...")
        if 0 not in df_sen["sentiment_label"].values:
            save_to_vectors(1, embedding_model=SHARED_EMBEDDING_MODEL)
        elif 1 not in df_sen["sentiment_label"].values:
            save_to_vectors(0, embedding_model=SHARED_EMBEDDING_MODEL)
        else:
            save_to_vectors(embedding_model=SHARED_EMBEDDING_MODEL)

    except Exception as e:
        return (f"❌ Topic modeling failed: {e}",) + EMPTY[1:]

    df_full = pd.concat(all_dfs)

    if progress is not None:
        progress(1.0, desc="✅ Analysis completed.")

    # ── Persist for RAG tab ─────────────
    session_state["video_id"] = video_id
    session_state["topic_summary"] = topic_summary

    # ── Build outputs ───────────────────
    counts = df_full["sentiment_label"].value_counts()
    total = len(df_full)
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))

    metrics_md = (
        f"| 📊 Total | ❌ Negative | ✅ Positive |\n"
        f"|---------|------------|------------|\n"
        f"| **{total}** | **{neg}** | **{pos}** |"
    )

    chart_df = pd.DataFrame(
        {
            "Sentiment": ["Negative", "Positive"],
            "Count": [neg, pos],
        }
    )

    pos_topics = topic_summary.get("✅ Positive", pd.DataFrame())
    neg_topics = topic_summary.get("❌ Negative", pd.DataFrame())

    def trim_cols(tdf):
        if not tdf.empty:
            cols = [c for c in ["Topic", "Count", "Name"] if c in tdf.columns]
            return tdf[cols].reset_index(drop=True)
        return tdf

    status = f"✅ Processed **{total}** comments! Head over to the **💬 Ask AI** tab to query them."

    return (
        status,
        metrics_md,
        chart_df,
        trim_cols(pos_topics),
        trim_cols(neg_topics),
        session_state,
    )


# ─────────────────────────────────────────
# Tab 2 — Ask AI (RAG)
# ─────────────────────────────────────────


def get_topic_hint(sentiment_val, topic_id: int, session_state: dict) -> str:
    """Return topic name hint from the last processed video, if available."""
    topic_summary = session_state.get("topic_summary", {})
    label_name = "✅ Positive" if int(sentiment_val) == 1 else "❌ Negative"
    topic_info = topic_summary.get(label_name)
    if topic_info is not None and not topic_info.empty:
        match = topic_info[topic_info["Topic"] == int(topic_id)]
        if not match.empty:
            return f"📌 Topic {int(topic_id)}: **{match.iloc[0]['Name']}**"
    return ""


def ask_ai(video_id_rag: str, sentiment_val, topic_id: int, num_k: int, query: str):
    """Call RAG and return the answer + retrieved comments."""
    if not query or not query.strip():
        return "⚠️ Please enter a question.", ""

    try:
        response = ask_to_llm(
            query=query,
            sentiment_label=int(sentiment_val),
            topic=int(topic_id),
            num_k=int(num_k),
        )

        answer = response["answer"]

        context_parts = []
        for i, doc in enumerate(response["context"], 1):
            context_parts.append(
                f"**Comment {i}:**\n\n{doc.page_content}\n\n"
                f"*Metadata: {doc.metadata}*\n\n---"
            )
        context_md = (
            "\n".join(context_parts) if context_parts else "_No context retrieved._"
        )

        return answer, context_md

    except Exception as e:
        return f"❌ Error: {e}", ""


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="emerald",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
)

with gr.Blocks(theme=theme, title="🎬 YouTube Comment Analyzer") as demo:
    session_state = gr.State(value={})
    gr.Markdown(
        """
        # 🎬 YouTube Comment Sentiment & RAG Analyzer
        Analyze YouTube comments with **BERT** sentiment analysis, discover **topics**,
        and ask questions using **RAG**.
        """
    )

    with gr.Tabs():
        # ════════════════════════════════════════
        # Tab 1: Analyze Video
        # ════════════════════════════════════════
        with gr.Tab("📊 Analyze Video"):
            gr.Markdown("### Step 1 — Enter a YouTube video to process its comments.")

            with gr.Row():
                video_input = gr.Textbox(
                    label="YouTube Video ID or URL",
                    placeholder="e.g. XqYTfpxFuDM or https://www.youtube.com/watch?v=XqYTfpxFuDM",
                    scale=5,
                )
                analyze_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)

            status_out = gr.Markdown()
            metrics_out = gr.Markdown()

            chart_out = gr.BarPlot(
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_map={"Negative": "#e74c3c", "Positive": "#2ecc71"},
                title="Sentiment Distribution",
                height=300,
                visible=False,
                y_lim=[0, None],
            )

            gr.Markdown("#### 🔍 Topics Found")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("##### ✅ Positive Topics")
                    pos_topics_out = gr.DataFrame(interactive=False)
                with gr.Column():
                    gr.Markdown("##### ❌ Negative Topics")
                    neg_topics_out = gr.DataFrame(interactive=False)

            # Carries video_id to Tab 2
            # rag_vid_state = gr.State("")

            def run_analysis(video_input, current_session, progress=gr.Progress()):
                status, metrics, chart_df, pos_t, neg_t, updated_session = (
                    analyze_video(video_input, current_session, progress)
                )
                show_chart = not chart_df.empty
                return (
                    status,
                    metrics,
                    gr.BarPlot(value=chart_df, visible=show_chart, y_lim=[0, None]),
                    pos_t,
                    neg_t,
                    updated_session,
                )

            analyze_btn.click(
                fn=run_analysis,
                inputs=[video_input, session_state],
                show_progress="full",
                # show_progress_on=[analyze_btn],
                outputs=[
                    status_out,
                    metrics_out,
                    chart_out,
                    pos_topics_out,
                    neg_topics_out,
                    session_state,
                ],
            )

        # ════════════════════════════════════════
        # Tab 2: Ask AI (RAG)
        # ════════════════════════════════════════
        with gr.Tab("💬 Ask AI"):
            gr.Markdown("### Step 2 — Ask a question about the analyzed comments.")

            video_id_rag = gr.Textbox(
                label="Video ID",
                placeholder="Auto-filled after Tab 1, or enter manually",
            )

            with gr.Row():
                sentiment_sel = gr.Radio(
                    choices=[("✅ Positive", 1), ("❌ Negative", 0)],
                    value=1,
                    label="Sentiment",
                    scale=2,
                )
                topic_sel = gr.Number(
                    label="Topic ID",
                    value=0,
                    minimum=0,
                    maximum=20,
                    step=1,
                    info="Topic number from the topic breakdown in Tab 1",
                    scale=1,
                )

            topic_hint_out = gr.Markdown()

            num_k_slider = gr.Slider(
                minimum=1,
                maximum=30,
                value=15,
                step=1,
                label="Number of retrieved comments (k)",
                info="How many comments to retrieve from the vector store for context",
            )

            query_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Why are commenters feeling negative? What are they complaining about?",
                lines=3,
            )

            ask_btn = gr.Button("💬 Ask AI", variant="primary")

            gr.Markdown("#### 🧠 AI Answer")
            answer_out = gr.Markdown()

            with gr.Accordion("📄 Retrieved Comments (Context)", open=False):
                context_out = gr.Markdown()

            # ── Wiring ──────────────────────────

            # Update topic hint on sentiment / topic change
            for trigger in [sentiment_sel, topic_sel]:
                trigger.change(
                    fn=get_topic_hint,
                    inputs=[sentiment_sel, topic_sel, session_state],
                    outputs=[topic_hint_out],
                )

            # Auto-fill video ID from Tab 1
            session_state.change(
                fn=lambda session: session.get("video_id", ""),
                inputs=[session_state],
                outputs=[video_id_rag],
            )

            ask_btn.click(
                fn=ask_ai,
                inputs=[
                    video_id_rag,
                    sentiment_sel,
                    topic_sel,
                    num_k_slider,
                    query_input,
                ],
                outputs=[answer_out, context_out],
            )

            gr.Examples(
                examples=[
                    [1, 0, 15, "Why are people positive about this video?"],
                    [
                        0,
                        0,
                        15,
                        "What are the main complaints in the negative comments?",
                    ],
                ],
                inputs=[sentiment_sel, topic_sel, num_k_slider, query_input],
                label="Example questions",
            )

    gr.Markdown(
        """
        ---
        <div style='text-align:center; color:#aaa; font-size:13px;'>
        Built with 🤗 Gradio &nbsp;·&nbsp; BERT Sentiment &nbsp;·&nbsp; BERTopic &nbsp;·&nbsp; RAG
        </div>
        """
    )

is_hf_space = "SPACE_ID" in os.environ
if is_hf_space:
    demo.launch()
else:
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    pass
