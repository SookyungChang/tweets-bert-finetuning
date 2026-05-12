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


def home_page() -> str:
    return f"""
    <html><head>{CSS}</head>
    <body>
        <h1>🧠 Sentiment Analysis</h1>
        <p>Choose what you want to do:</p>
        <div class="card">
            <a class="btn-choose" href="/text">
                ✍️ Text Comparison<br>
                <small style="font-size:13px; color:#888;">
                    Compare Baseline vs BERT on your own text
                </small>
            </a>
            <a class="btn-choose" href="/youtube">
                🎬 YouTube Analyzer<br>
                <small style="font-size:13px; color:#888;">
                    Analyze sentiment distribution of video comments
                </small>
            </a>
        </div>
        <p style="color:#aaa; font-size:13px;">
            API also available: <code>POST /predict_all</code>
        </p>
    </body></html>
    """


def text_home_page() -> str:
    return f"""
    <html><head>{CSS}</head>
    <body>
        <h1>✍️ Text Sentiment Comparison</h1>
        <div class="card">
            <form method="post" action="/text/predict">
                <input type="text" name="text"
                    placeholder="Enter your text here..." required />
                <br>
                <button type="submit">Compare Models</button>
            </form>
        </div>
        <a class="back" href="/">← Back to home</a>
    </body></html>
    """


def text_results_page(text: str, base_result: dict, bert_result: dict) -> str:
    def label_html(label):
        name = "Positive" if label == 1 else "Negative"
        css  = "pos"      if label == 1 else "neg"
        return f'<span class="{css}">{name}</span>'

    def conf_str(conf):
        return f"{round(conf * 100, 1)}%"

    return f"""
    <html><head>{CSS}</head>
    <body>
        <h1>✍️ Results</h1>
        <div class="card">
            <p><strong>Input:</strong> "{text}"</p>
            <table>
                <tr>
                    <th>Model</th><th>Label</th><th>Confidence</th>
                </tr>
                <tr>
                    <td>📊 Baseline (TF-IDF)</td>
                    <td>{label_html(base_result["prediction"])}</td>
                    <td>{conf_str(base_result["confidence"])}</td>
                </tr>
                <tr>
                    <td>🤖 BERT</td>
                    <td>{label_html(bert_result["prediction"])}</td>
                    <td>{conf_str(bert_result["confidence"])}</td>
                </tr>
            </table>
        </div>
        <a class="back" href="/text">← Try another text</a>
        <a class="back" href="/">← Back to home</a>
    </body></html>
    """


def youtube_home_page() -> str:
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
        #loading.show {{ display: flex; }}
        .spinner {{
            width: 60px; height: 60px;
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
        .loading-text {{ font-size: 18px; color: #2c3e50; font-weight: bold; }}
        .loading-sub  {{ font-size: 13px; color: #aaa; }}
    </style>
    </head>
    <body>
        <div id="loading">
            <div class="spinner"></div>
            <p class="loading-text">⏳ Analyzing comments...</p>
            <p class="loading-sub">Fetching comments and running BERT model</p>
            <p class="loading-sub">This may take 30–60 seconds on our server</p>
        </div>

        <h1>🎬 YouTube Sentiment Analyzer</h1>
        <div class="card">
            <form id="analyzeForm" method="post" action="/youtube/analyze"
                  onsubmit="showLoading()">
                <input type="text" name="video_id"
                    placeholder="Paste YouTube URL or Video ID"
                    required />
                <br>
                <button type="submit">Analyze Comments</button>
            </form>
        </div>
        <a class="back" href="/">← Back to home</a>

        <script>
            function showLoading() {{
                document.getElementById('loading').classList.add('show');
                const messages = [
                    "⏳ Fetching YouTube comments...",
                    "🔍 Filtering English comments...",
                    "🤖 Running BERT sentiment model...",
                    "📊 Building your chart...",
                    "Almost there..."
                ];
                let i = 0;
                const el = document.querySelector('.loading-text');
                setInterval(() => {{
                    i = (i + 1) % messages.length;
                    el.textContent = messages[i];
                }}, 4000);
            }}
        </script>
    </body></html>
    """


def youtube_results_page(
    video_id: str,
    total: int,
    pos: int,
    neg: int,
    chart_b64: str
) -> str:
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


def error_page(message: str, back_url: str) -> str:
    return f"""
    <html><head>{CSS}</head>
    <body>
        <h1>❌ Error</h1>
        <div class="card">
            <p>{message}</p>
        </div>
        <a class="back" href="{back_url}">← Go back</a>
    </body></html>
    """