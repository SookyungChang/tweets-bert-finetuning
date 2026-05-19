# src/analysis/topic_modeling.py
import hdbscan
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
import pandas as pd
import numpy as np


class BERTopicLangchainEmbedder(BaseEmbedder):
    def __init__(self, lanchain_model):
        self.model = lanchain_model

    def embed(self, documents, verbose=False):
        return np.array(self.model.embed_documents(documents))


def get_topics(
    df: pd.DataFrame,
    sentiment_label: int,
    embedding_model,
    min_cluster_size: int = 10,  # ← add with default value
    hdbscan_model=None,
):
    df_sentiment = df[df["sentiment_label"] == sentiment_label].copy()
    # ↑ add .copy() to avoid SettingWithCopyWarning

    # Guard: need minimum comments
    if len(df_sentiment) < min_cluster_size:
        return df_sentiment, None

    if hdbscan_model is None:
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True
        )

    bertopic_embedding = BERTopicLangchainEmbedder(embedding_model)

    topic_model = BERTopic(
        embedding_model=bertopic_embedding,
        hdbscan_model=hdbscan_model,
        # calculate_probabilities=False,
    )
    topics, probs = topic_model.fit_transform(df_sentiment["text"].tolist())
    # ↑ add .tolist() — BERTopic works better with lists than pandas Series

    if -1 in topics:
        new_topics = topic_model.reduce_outliers(
            df_sentiment["text"].tolist(), topics, strategy="c-tf-idf"
        )
        topic_model.update_topics(df_sentiment["text"].tolist(), topics=new_topics)
    else:
        new_topics = topics

    df_sentiment["topic"] = new_topics
    df_sentiment["topic_probability"] = probs

    return df_sentiment, topic_model
