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
        # table_exists = (
        #     conn.execute(
        #         "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='comment_results'"
        #     ).fetchone()[0]
        #     == 1
        # )

        # if not table_exists:
        #     df.to_sql("comment_results", con=conn, if_exists="append", index=False)
        # else:
        #     df.to_sql(
        #         "temp_comment_results", con=conn, if_exists="replace", index=False
        #     )
        #     conn.execute("""
        #         INSERT INTO comment_results
        #         SELECT * FROM temp_comment_results
        #         WHERE NOT EXISTS (
        #             SELECT 1 FROM comment_results
        #             WHERE comment_results.commentId = temp_comment_results.commentId
        #         )
        #     """)
        #     conn.execute("DROP TABLE temp_comment_results")

    print(f"✅ Saved {len(df)} comments for video {video_id}")


def save_to_vectors():
    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT * FROM comment_results_positive
        UNION ALL
        SELECT * FROM comment_results_negative
        """
        df = pd.read_sql(query, con=conn)

    documents = []
    for _, row in df.iterrows():
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

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        ),
        persist_directory=str(PathConfig.VECTORSTORE_PATH),
    )
    return vectorstore
