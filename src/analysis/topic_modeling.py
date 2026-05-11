# src/analysis/topic_modeling.py

from bertopic import BERTopic
import pandas as pd

def get_topics(df: pd.DataFrame, sentiment_label: int):
    df_sentiment = df[df['sentiment_label'] == sentiment_label].copy()
    # ↑ add .copy() to avoid SettingWithCopyWarning

    # Guard: need minimum comments
    if len(df_sentiment) < 10:
        return df_sentiment, None

    topic_model = BERTopic(language="english")
    topics, probs = topic_model.fit_transform(df_sentiment["text"].tolist())
    # ↑ add .tolist() — BERTopic works better with lists than pandas Series

    df_sentiment['topic'] = topics
    df_sentiment['topic_probability'] = probs

    return df_sentiment, topic_model