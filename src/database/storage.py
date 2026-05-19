# src/database/storage.py
import sqlite3
import pandas as pd
import json
from datetime import datetime, timezone
import os
from src.config import PathConfig

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_PATH = os.getenv("DB_PATH", PathConfig.DATA_PATH / "Results.db")
labels = {0: "negative", 1: "positive"}


def save_results(
    video_id: str, df: pd.DataFrame, topic_info: pd.DataFrame, sentiment_label: int
):
    """Save full prediction DataFrame to SQLite, skipping duplicate commentIds."""
    df = df.copy()
    df["video_id"] = video_id
    # df["sentiment_label"] = sentiment_label
    df["analyzed_at"] = datetime.now(timezone.utc).isoformat()

    # Serialize list columns in topic_info for SQLite compatibility
    topic_info = topic_info.copy()
    if "Representation" in topic_info.columns:
        topic_info["Representation"] = topic_info["Representation"].apply(json.dumps)
    if "Representative_Docs" in topic_info.columns:
        topic_info["Representative_Docs"] = topic_info["Representative_Docs"].apply(
            json.dumps
        )

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(
            f"comment_results_{labels[sentiment_label]}",
            con=conn,
            if_exists="replace",
            index=False,
        )

        topic_info.to_sql(
            f"topic_info_{labels[sentiment_label]}",
            con=conn,
            if_exists="replace",
            index=False,
        )

    print(f"✅ Saved {len(df)} comments for video {video_id}")


def save_to_vectors(if_only=None, embedding_model=None):
    if if_only == 1:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
            SELECT * FROM comment_results_positive
            """
            df = pd.read_sql(query, con=conn)

    elif if_only == 0:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
            SELECT * FROM comment_results_negative
            """
            df = pd.read_sql(query, con=conn)

    else:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
            SELECT * FROM comment_results_positive
            UNION ALL
            SELECT * FROM comment_results_negative
            """
            df = pd.read_sql(query, con=conn)

    existing_ids = set()
    documents = []

    for _, row in df.iterrows():
        if row["commentId"] not in existing_ids:
            doc = Document(
                page_content=row["text"],
                metadata={
                    "commentId": row["commentId"],
                    "author": row["author"],
                    "likeCount": row["likeCount"],
                    "publishedAt": row["publishedAt"],
                    "video_id": row["video_id"],
                    "sentiment_label": row["sentiment_label"],
                    "topic": row["topic"],
                },
            )
            documents.append(doc)
            existing_ids.add(row["commentId"])

    if documents:
        doc_ids = [doc.metadata["commentId"] for doc in documents]

        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
            )

        vectorstore = Chroma.from_documents(
            documents=documents,
            ids=doc_ids,
            embedding=embedding_model,
            persist_directory=str(PathConfig.VECTORSTORE_PATH),
        )
    else:
        print("No new unique documents to add.")

    return vectorstore
