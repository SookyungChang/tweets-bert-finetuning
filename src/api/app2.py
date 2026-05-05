from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (important for servers)
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from huggingface_hub import snapshot_download
from src.inference import predictor_bert
from src.data.get_comments import  get_comments, filter_english_comments

# --- Load model once at startup (not on every request) ---
app = FastAPI()
bert_path = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
bert = predictor_bert.Predictor(bert_path)

def make_chart(df: pd.DataFrame) -> str:
    """Generate a bar chart and return it as a base64 string to embed in HTML."""
    label_names = {0: "Negative", 1: "Positive"}
    counts = df["sentiment_label"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(
        [label_names[i] for i in counts.index],
        counts.values,
        color=["#e74c3c", "#2ecc71"]
    )
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution of YouTube Comments")

    # Convert chart to base64 so we can embed it directly in HTML
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the input form."""
    return """
    <html>
        <body style="font-family: Arial; max-width: 600px; margin: 60px auto; text-align: center;">
            <h2>🎬 YouTube Comment Sentiment Analyzer</h2>
            <form method="post" action="/analyze">
                <input
                    name="video_id"
                    placeholder="Enter YouTube Video ID (e.g. dQw4w9WgXcQ)"
                    style="width: 80%; padding: 10px; font-size: 16px;"
                    required
                />
                <br><br>
                <button type="submit" style="padding: 10px 30px; font-size: 16px;">
                    Analyze
                </button>
            </form>
        </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(video_id: str = Form(...)):
    """Fetch comments, run prediction, return chart."""
    try:
        # Step 1: Fetch and filter comments
        df = get_comments(video_id, max_results=100, max_pages=3)
        df = filter_english_comments(df)

        if df.empty:
            return "<h3>No English comments found for this video.</h3>"

        # Step 2: Run BERT sentiment prediction
        df_predicted = bert.predict_df(df)

        # Step 3: Build summary stats
        total = len(df_predicted)
        counts = df_predicted["sentiment_label"].value_counts().sort_index()
        pos = counts.get(1, 0)
        neg = counts.get(0, 0)

        # Step 4: Generate chart
        chart_b64 = make_chart(df_predicted)

        return f"""
        <html>
            <body style="font-family: Arial; max-width: 700px; margin: 60px auto; text-align: center;">
                <h2>Results for video: <code>{video_id}</code></h2>
                <p>📊 Analyzed <strong>{total}</strong> comments —
                   ✅ <strong>{pos} Positive</strong> &nbsp;|&nbsp;
                   ❌ <strong>{neg} Negative</strong></p>
                <img src="data:image/png;base64,{chart_b64}" width="500"/>
                <br><br>
                <a href="/">← Analyze another video</a>
            </body>
        </html>
        """

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>← Go back</a>"