# app.py — Gradio version for Hugging Face Spaces
from pathlib import Path
import sys
import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent.parent.parent
sys.path.append(str(parent_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import gradio as gr
from huggingface_hub import snapshot_download

from src.inference import predictor_base, predictor_bert
from src.data.get_comments import get_comments, filter_english_comments
from src.utils.youtube_analyzer import extract_video_id


# ─────────────────────────────────────────
# Model loading (once at startup)
# ─────────────────────────────────────────

print("⏳ Loading models...")
base_model = predictor_base.load_model()
bert_path = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
bert_model = predictor_bert.Predictor(bert_path)
print("✅ Models loaded!")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────


def sentiment_label(label: int) -> str:
    return "✅ Positive" if label == 1 else "❌ Negative"


def make_chart(df: pd.DataFrame):
    """Return a matplotlib Figure with the sentiment distribution bar chart."""
    label_names = {0: "Negative", 1: "Positive"}
    counts = df["sentiment_label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c" if i == 0 else "#2ecc71" for i in counts.index]
    bars = ax.bar(
        [label_names[i] for i in counts.index],
        counts.values,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )

    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Comments", fontsize=12)
    ax.set_title("Sentiment Distribution of YouTube Comments", fontsize=14, pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, counts.values.max() * 1.2)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# Tab 1 — Text Comparison
# ─────────────────────────────────────────


def predict_text(text: str):
    """Run both models on user text and return formatted results."""
    if not text or not text.strip():
        return "⚠️ Please enter some text.", "", "", ""

    try:
        base_result = predictor_base.predict(base_model, text)
        bert_result = bert_model.predict_text(text)

        base_label = sentiment_label(base_result["prediction"])
        base_conf = f"{round(base_result['confidence'] * 100, 1)}%"

        bert_label = sentiment_label(bert_result["prediction"])
        bert_conf = f"{round(bert_result['confidence'] * 100, 1)}%"

        return base_label, base_conf, bert_label, bert_conf

    except Exception as e:
        return f"Error: {e}", "", "", ""


# ─────────────────────────────────────────
# Tab 2 — YouTube Analyzer
# ─────────────────────────────────────────


def analyze_youtube(video_url: str):
    """Fetch comments, run BERT, return stats + chart."""
    if not video_url or not video_url.strip():
        return "⚠️ Please enter a YouTube URL or video ID.", "", "", None

    try:
        video_id = extract_video_id(video_url)

        df = get_comments(video_id, max_results=100, max_pages=3)
        df = filter_english_comments(df)

        if df.empty:
            return "⚠️ No English comments found for this video.", "", "", None

        df_predicted = bert_model.predict_df(df)

        total = len(df_predicted)
        counts = df_predicted["sentiment_label"].value_counts().sort_index()
        pos = counts.get(1, 0)
        neg = counts.get(0, 0)

        summary = f"📊 **{total}** English comments analyzed"
        pos_text = f"✅ {pos} Positive"
        neg_text = f"❌ {neg} Negative"
        fig = make_chart(df_predicted)

        return summary, pos_text, neg_text, fig

    except Exception as e:
        return f"❌ Error: {e}", "", "", None


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="emerald",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
)

with gr.Blocks(theme=theme, title="🧠 Sentiment Analysis") as demo:
    gr.Markdown(
        """
        # 🧠 Sentiment Analysis
        Compare a **Baseline (TF-IDF)** model vs a fine-tuned **BERT** model — or analyze
        the sentiment of an entire YouTube video's comment section.
        """
    )

    with gr.Tabs():
        # ── Tab 1: Text Comparison ──────────────────────────────
        with gr.Tab("✍️ Text Comparison"):
            gr.Markdown(
                "Enter any text below and compare how both models classify its sentiment."
            )

            with gr.Row():
                text_input = gr.Textbox(
                    label="Your Text",
                    placeholder="e.g. This product is absolutely amazing!",
                    lines=3,
                    scale=4,
                )

            compare_btn = gr.Button("⚡ Compare Models", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Baseline (TF-IDF)")
                    base_label_out = gr.Textbox(label="Prediction", interactive=False)
                    base_conf_out = gr.Textbox(label="Confidence", interactive=False)

                with gr.Column():
                    gr.Markdown("### 🤖 BERT")
                    bert_label_out = gr.Textbox(label="Prediction", interactive=False)
                    bert_conf_out = gr.Textbox(label="Confidence", interactive=False)

            compare_btn.click(
                fn=predict_text,
                inputs=[text_input],
                outputs=[base_label_out, base_conf_out, bert_label_out, bert_conf_out],
            )

            gr.Examples(
                examples=[
                    ["This movie was absolutely fantastic! Loved every second."],
                    ["Terrible experience. I would never recommend this to anyone."],
                    ["It was okay, nothing special but not bad either."],
                ],
                inputs=text_input,
                label="Try an example",
            )

        # ── Tab 2: YouTube Analyzer ─────────────────────────────
        with gr.Tab("🎬 YouTube Analyzer"):
            gr.Markdown(
                "Paste a YouTube URL or video ID. The app fetches up to **100 comments**, "
                "filters for English, and runs BERT sentiment analysis on all of them."
            )

            with gr.Row():
                yt_input = gr.Textbox(
                    label="YouTube URL or Video ID",
                    placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    scale=4,
                )

            analyze_btn = gr.Button("🔍 Analyze Comments", variant="primary")

            summary_out = gr.Markdown()

            with gr.Row():
                pos_out = gr.Textbox(label="Positive", interactive=False)
                neg_out = gr.Textbox(label="Negative", interactive=False)

            chart_out = gr.Plot(label="Sentiment Distribution")

            analyze_btn.click(
                fn=analyze_youtube,
                inputs=[yt_input],
                outputs=[summary_out, pos_out, neg_out, chart_out],
            )

    gr.Markdown(
        """
        ---
        <div style='text-align:center; color:#aaa; font-size:13px;'>
        Built with 🤗 Gradio · Models: TF-IDF Baseline + fine-tuned BERT
        </div>
        """
    )

demo.launch()

if __name__ == "__main__":
    pass
