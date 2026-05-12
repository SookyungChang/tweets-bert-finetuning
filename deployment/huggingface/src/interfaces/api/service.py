import pandas as pd
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.inference import predictor_base
from src.data.get_comments import get_comments, filter_english_comments


def predict_text_service(text: str, base_model, bert_model) -> tuple:
    """Run both models on a single text. Returns (base_result, bert_result)."""
    base_result = predictor_base.predict(base_model, text)
    bert_result = bert_model.predict_text(text)
    return base_result, bert_result


async def analyze_youtube_service(video_id: str, bert_model) -> dict:
    """
    Fetch comments, run BERT prediction, build chart.
    Returns dict with all data needed for the results page.
    """
    df = get_comments(video_id, max_results=100, max_pages=3)
    df = filter_english_comments(df)

    if df.empty:
        raise ValueError("No English comments found for this video.")

    df_predicted = bert_model.predict_df(df)

    total  = len(df_predicted)
    counts = df_predicted["sentiment_label"].value_counts().sort_index()
    pos    = int(counts.get(1, 0))
    neg    = int(counts.get(0, 0))
    chart  = make_chart(df_predicted)

    return {
        "video_id":  video_id,
        "total":     total,
        "pos":       pos,
        "neg":       neg,
        "chart_b64": chart
    }


def make_chart(df: pd.DataFrame) -> str:
    """Generate sentiment bar chart and return as base64 string."""
    label_names = {0: "Negative", 1: "Positive"}
    counts = df["sentiment_label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [label_names[i] for i in counts.index],
        counts.values,
        color=["#e74c3c", "#2ecc71"]
    )
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution of YouTube Comments")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode("utf-8")