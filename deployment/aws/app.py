# app.py
from pathlib import Path
import sys
import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent.parent
sys.path.append(str(parent_dir))

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from mangum import Mangum

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

from contextlib import asynccontextmanager
from src.inference import predictor_base, predictor_bert
from src.data.get_comments import get_comments, filter_english_comments
from src.utils.youtube_analyzer import extract_video_id


async def lifespan(app: FastAPI):
    # Everything before yield runs at STARTUP

    # --- Global models ---
    global base_model, bert_model
    base_model = predictor_base.load_model()
    bert_path = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
    bert_model = predictor_bert.Predictor(bert_path)
    print("✅ Models loaded!")
    
    yield  # ← app runs here
    
    # Everything after yield runs at SHUTDOWN
    print("🛑 Shutting down...")
    # clean up if needed (optional)

app = FastAPI(lifespan=lifespan)

# ─────────────────────────────────────────
# Shared styles
# ─────────────────────────────────────────

CSS = """
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 60px auto;
            text-align: center;
            background: #f9f9f9;
            color: #333;
        }
        h1 { font-size: 2em; margin-bottom: 5px; }
        h2 { color: #555; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        input[type=text] {
            width: 80%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        button {
            padding: 10px 30px;
            font-size: 16px;
            background: #2c3e50;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        button:hover { background: #1a252f; }
        .btn-choose {
            display: inline-block;
            margin: 10px;
            padding: 20px 40px;
            font-size: 18px;
            background: white;
            border: 2px solid #2c3e50;
            border-radius: 12px;
            text-decoration: none;
            color: #2c3e50;
            transition: all 0.2s;
        }
        .btn-choose:hover { background: #2c3e50; color: white; }
        .back { margin-top: 20px; display: block; color: #888; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th { background: #2c3e50; color: white; padding: 10px; }
        td { padding: 10px; border-bottom: 1px solid #eee; }
        .pos { color: #27ae60; font-weight: bold; }
        .neg { color: #e74c3c; font-weight: bold; }
    </style>
"""

# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────

def make_chart(df: pd.DataFrame) -> str:
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

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    """Landing page — choose which tool to use."""
    return f"""
    <html><head>{CSS}</head>
    <body>
        <h1>🧠 Sentiment Analysis</h1>
        <p>Choose what you want to do:</p>

        <div class="card">
            <a class="btn-choose" href="/text">
                ✍️ Text Comparison<br>
                <small style="font-size:13px; color:#888;">Compare Baseline vs BERT on your own text</small>
            </a>
            <a class="btn-choose" href="/youtube">
                🎬 YouTube Analyzer<br>
                <small style="font-size:13px; color:#888;">Analyze sentiment distribution of video comments</small>
            </a>
        </div>

        <p style="color:#aaa; font-size:13px;">API also available: <code>POST /predict_all</code></p>
    </body></html>
    """

# ── App 1: Text comparison ──────────────

@app.get("/text", response_class=HTMLResponse)
def text_home():
    """Input form for text comparison."""
    return f"""
    <html><head>{CSS}</head>
    <body>
        <h1>✍️ Text Sentiment Comparison</h1>
        <div class="card">
            <form method="post" action="/text/predict">
                <input type="text" name="text" placeholder="Enter your text here..." required />
                <br>
                <button type="submit">Compare Models</button>
            </form>
        </div>
        <a class="back" href="/">← Back to home</a>
    </body></html>
    """

@app.post("/text/predict", response_class=HTMLResponse)
def text_predict(text: str = Form(...)):
    """Run both models and show comparison table."""
    try:
        base_result = predictor_base.predict(base_model, text)
        bert_result = bert_model.predict_text(text)

        def label_html(label):
            name = "Positive" if label == 1 else "Negative"
            css = "pos" if label == 1 else "neg"
            return f'<span class="{css}">{name}</span>'

        def conf_bar(conf):
            pct = round(conf * 100, 1)
            return f'{pct}%'

        return f"""
        <html><head>{CSS}</head>
        <body>
            <h1>✍️ Results</h1>
            <div class="card">
                <p><strong>Input:</strong> "{text}"</p>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Label</th>
                        <th>Confidence</th>
                    </tr>
                    <tr>
                        <td>📊 Baseline (TF-IDF)</td>
                        <td>{label_html(base_result['prediction'])}</td>
                        <td>{conf_bar(base_result['confidence'])}</td>
                    </tr>
                    <tr>
                        <td>🤖 BERT</td>
                        <td>{label_html(bert_result['prediction'])}</td>
                        <td>{conf_bar(bert_result['confidence'])}</td>
                    </tr>
                </table>
            </div>
            <a class="back" href="/text">← Try another text</a>
            <a class="back" href="/">← Back to home</a>
        </body></html>
        """
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/text'>← Go back</a>"

# ── App 2: YouTube analyzer ─────────────

@app.get("/youtube", response_class=HTMLResponse)
def youtube_home():
    return f"""
    <html><head>{CSS}
    <style>
        #loading {{
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(255,255,255,0.95);
            z-index: 999;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        #loading.show {{
            display: flex;
        }}
        .spinner {{
            width: 60px;
            height: 60px;
            border: 6px solid #f0f0f0;
            border-top: 6px solid #2c3e50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }}
        @keyframes spin {{
            0%   {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .loading-text {{
            font-size: 18px;
            color: #2c3e50;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .loading-sub {{
            font-size: 13px;
            color: #aaa;
        }}
    </style>
    </head>
    <body>
        <!-- Loading overlay -->
        <div id="loading">
            <div class="spinner"></div>
            <p class="loading-text">⏳ Analyzing comments...</p>
            <p class="loading-sub">Fetching comments from YouTube and running BERT model</p>
            <p class="loading-sub">This may take 30–60 seconds on our server</p>
        </div>

        <h1>🎬 YouTube Sentiment Analyzer</h1>
        <div class="card">
            <form id="analyzeForm" method="post" action="/youtube/analyze"
                  onsubmit="showLoading()">
                <input
                    type="text"
                    name="video_id"
                    placeholder="Paste YouTube URL or Video ID (e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ)"
                    required
                />
                <br>
                <button type="submit">Analyze Comments</button>
            </form>
        </div>
        <a class="back" href="/">← Back to home</a>

        <script>
            function showLoading() {{
                // Show loading overlay when form is submitted
                document.getElementById('loading').classList.add('show');

                // Cycle through messages so it feels alive
                const messages = [
                    "⏳ Fetching YouTube comments...",
                    "🔍 Filtering English comments...",
                    "🤖 Running BERT sentiment model...",
                    "📊 Building your chart...",
                    "Almost there..."
                ];
                let i = 0;
                const textEl = document.querySelector('.loading-text');
                setInterval(() => {{
                    i = (i + 1) % messages.length;
                    textEl.textContent = messages[i];
                }}, 4000);  // change message every 4 seconds
            }}
        </script>
    </body></html>
    """

@app.post("/youtube/analyze", response_class=HTMLResponse)
async def youtube_analyze(video_id: str = Form(...)):
    """Fetch comments, run BERT, return chart."""
    try:
        video_id = extract_video_id(video_id)

        df = get_comments(video_id, max_results=100, max_pages=3)
        df = filter_english_comments(df)

        if df.empty:
            return f"<h3>No English comments found.</h3><a href='/youtube'>← Go back</a>"

        df_predicted = bert_model.predict_df(df)

        total = len(df_predicted)
        counts = df_predicted["sentiment_label"].value_counts().sort_index()
        pos = counts.get(1, 0)
        neg = counts.get(0, 0)
        chart_b64 = make_chart(df_predicted)

        return f"""
        <html><head>{CSS}</head>
        <body>
            <h1>🎬 Results</h1>
            <div class="card">
                <p><strong>Video ID:</strong> <code>{video_id}</code></p>
                <p>📊 <strong>{total}</strong> comments analyzed</p>
                <p>✅ <span class="pos">{pos} Positive</span>
                   &nbsp;|&nbsp;
                   ❌ <span class="neg">{neg} Negative</span></p>
                <img src="data:image/png;base64,{chart_b64}" width="500"/>
            </div>
            <a class="back" href="/youtube">← Analyze another video</a>
            <a class="back" href="/">← Back to home</a>
        </body></html>
        """
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/youtube'>← Go back</a>"

# ── API endpoint (JSON) ─────────────────

class TextRequest(BaseModel):
    text: str

@app.post("/predict_all")
def predict_all(request: TextRequest):
    base_result = predictor_base.predict(base_model, request.text)
    bert_result = bert_model.predict_text(request.text)
    return {"text": request.text, "baseline": base_result, "bert": bert_result}

# ── Lambda handler ──────────────────────
handler = Mangum(app)