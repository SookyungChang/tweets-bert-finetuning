# streamlit_app.py
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from huggingface_hub import snapshot_download

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent.parent
sys.path.append(str(parent_dir))

from src.data.get_comments import get_comments, filter_english_comments
from src.inference.predictor_bert import Predictor
from src.config import urlConfig
from src.analysis.topic_modeling import get_topics
from src.database.storage import save_results, save_to_vectors
from src.analysis.rag import ask_to_llm

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="YouTube Comment Analyzer", page_icon="🎬", layout="wide")


# ─────────────────────────────────────────
# Load model once (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    bert_path = snapshot_download(repo_id=urlConfig.trained_model_id)
    return Predictor(bert_path)


# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
def process_video(video_id: str, bert_model) -> pd.DataFrame:
    """Fetch comments, predict sentiment, extract topics, save to DB."""
    with st.spinner("📥 Fetching YouTube comments..."):
        df = get_comments(video_id, max_results=100, max_pages=10)
        df = filter_english_comments(df)

    if df.empty:
        st.error("No English comments found for this video.")
        return None

    with st.spinner("🤖 Running BERT sentiment prediction..."):
        df_sen = bert_model.predict_df(df)

    with st.spinner("🔍 Extracting topics..."):
        all_dfs = []
        topic_summary = {}

        for label in df_sen["sentiment_label"].unique():
            count = len(df_sen[df_sen["sentiment_label"] == label])
            size = len(str(count)) + 2

            df_add, topic_model = get_topics(
                df_sen, sentiment_label=label, min_cluster_size=size
            )

            # Show topic summary
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

    st.success(f"✅ Processed {len(df_sen)} comments!")
    return pd.concat(all_dfs), topic_summary


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("🎬 YouTube Comment Sentiment & RAG Analyzer")
st.markdown(
    "Analyze YouTube comments with BERT sentiment analysis and ask questions using RAG."
)

bert_model = load_model()

# ── Tabs ─────────────────────────────────
tab1, tab2 = st.tabs(["📊 Analyze Video", "💬 Ask AI"])

# ────────────────────────────────────────
# Tab 1: Analyze Video
# ────────────────────────────────────────
with tab1:
    st.header("Step 1: Process a YouTube Video")

    video_id = st.text_input(
        "YouTube Video ID or URL",
        placeholder="e.g. SbNDmAJBtyU or https://www.youtube.com/watch?v=SbNDmAJBtyU",
    )

    if st.button("🚀 Analyze", type="primary"):
        if not video_id.strip():
            st.warning("Please enter a video ID or URL.")
        else:
            # Extract video ID from URL if needed
            if "youtube.com" in video_id:
                from urllib.parse import urlparse, parse_qs

                params = parse_qs(urlparse(video_id).query)
                video_id = params.get("v", [video_id])[0]
            elif "youtu.be" in video_id:
                video_id = video_id.split("youtu.be/")[-1].split("?")[0]

            result = process_video(video_id, bert_model)

            if result is not None:
                df_full, topic_summary = result

                # Sentiment distribution
                st.subheader("📊 Sentiment Distribution")
                counts = df_full["sentiment_label"].value_counts()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Comments", len(df_full))
                col2.metric("✅ Positive", int(counts.get(1, 0)))
                col3.metric("❌ Negative", int(counts.get(0, 0)))

                # Bar chart
                st.bar_chart(counts.rename(index={0: "Negative", 1: "Positive"}))

                # Topic breakdown
                st.subheader("🔍 Topics Found")
                for label_name, topic_info in topic_summary.items():
                    with st.expander(f"{label_name} Topics"):
                        st.dataframe(
                            topic_info[["Topic", "Count", "Name"]],
                            use_container_width=True,
                        )

                # Save video_id to session for Tab 2
                st.session_state["video_id"] = video_id
                st.session_state["topic_summary"] = topic_summary
                st.info("💬 Now go to the **Ask AI** tab to query these comments!")

# ────────────────────────────────────────
# Tab 2: Ask AI (RAG)
# ────────────────────────────────────────
with tab2:
    st.header("Step 2: Ask AI About the Comments")

    # Video ID input
    video_id_rag = st.text_input(
        "Video ID",
        value=st.session_state.get("video_id", ""),
        placeholder="Same video ID you analyzed in Tab 1",
    )

    col1, col2 = st.columns(2)

    with col1:
        sentiment_label = st.selectbox(
            "Sentiment",
            options=[0, 1],
            format_func=lambda x: "❌ Negative" if x == 0 else "✅ Positive",
        )

    with col2:
        topic = st.number_input(
            "Topic ID",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Topic number from the topic breakdown in Tab 1",
        )

    # Show topic words if available
    if "topic_summary" in st.session_state:
        label_name = "✅ Positive" if sentiment_label == 1 else "❌ Negative"
        topic_info = st.session_state["topic_summary"].get(label_name)
        if topic_info is not None:
            match = topic_info[topic_info["Topic"] == topic]
            if not match.empty:
                st.caption(f"📌 Topic {topic}: **{match.iloc[0]['Name']}**")

    num_k = st.slider(
        "Number of retrieved comments (k)",
        min_value=1,
        max_value=30,
        value=15,
        help="How many comments to retrieve from vector store for context",
    )

    query = st.text_area(
        "Your question",
        placeholder="e.g. Why are commenters feeling negative? What are they complaining about?",
        height=100,
    )

    if st.button("💬 Ask AI", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 Thinking..."):
                try:
                    response = ask_to_llm(
                        query=query,
                        sentiment_label=sentiment_label,
                        topic=topic,
                        num_k=num_k,
                    )

                    # AI Answer
                    st.subheader("🧠 AI Answer")
                    st.markdown(
                        f"""
                        <div style="background:#f0f9ff; padding:20px;
                                    border-radius:10px; border-left:4px solid #2c3e50;">
                            {response["answer"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Retrieved context
                    with st.expander("📄 Retrieved Comments (Context)"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Comment {i + 1}:**")
                            st.text(doc.page_content)
                            st.caption(f"Metadata: {doc.metadata}")
                            st.divider()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
