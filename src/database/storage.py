# src/database/storage.py
import sqlite3
import pandas as pd
import json
from datetime import datetime, timezone
import os

DB_PATH = os.getenv("DB_PATH", "data/Results.db")

def save_results(video_id: str, df: pd.DataFrame):
    """Save full prediction DataFrame to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    
    # Add metadata columns
    df = df.copy()
    df['video_id']    = video_id
    df['analyzed_at'] = datetime.now(timezone.utc).isoformat()
    
    # Save to table — creates table automatically if not exists
    df.to_sql(
        name='comment_results',
        con=conn,
        if_exists='append',   # ← adds to existing data
        index=False
    )
    conn.close()
    print(f"✅ Saved {len(df)} comments for video {video_id}")


def load_results(video_id: str) -> pd.DataFrame:
    """Load saved results for a specific video."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM comment_results WHERE video_id = '{video_id}'",
        conn
    )
    conn.close()
    return df


def load_topic_comments(video_id: str, sentiment: int, topic: int) -> list:
    """
    Load comments for a specific topic — this is what RAG will use.
    
    Args:
        video_id:  YouTube video ID
        sentiment: 0 = negative, 1 = positive
        topic:     topic number from BERTopic
    
    Returns:
        list of comment texts for that topic
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT text 
        FROM comment_results 
        WHERE video_id    = '{video_id}'
          AND sentiment_label = {sentiment}
          AND topic       = {topic}
        ORDER BY topic_probability DESC
    """, conn)
    conn.close()
    return df['text'].tolist()