import sqlite3
import pandas as pd
from datasets import Dataset, DatasetDict
from src.config import PathConfig

paths = PathConfig()

def to_list_data(train_df, dev_df, test_df):

    dfs = [train_df, test_df, dev_df]  
    
    X = [df["text"].values.tolist() for df in dfs]
    y = [df["label"].values.tolist() for df in dfs]
    
    return X, y


def build_dataset_db(type):
    conn = sqlite3.connect(paths.DB_PATH / "tweets.db")

    train_df = pd.read_sql("SELECT * FROM train ORDER BY RANDOM()", conn)
    dev_df = pd.read_sql("SELECT * FROM val ORDER BY RANDOM()", conn)
    test_df = pd.read_sql("SELECT * FROM test ORDER BY RANDOM()", conn)

    conn.close()

    print(f"Data is loaded (Train: {len(train_df):,}, Val: {len(dev_df):,}, Test: {len(test_df):,})")

    if type == "base":
        data_dict = {
            "train": train_df,
            "dev": dev_df,
            "test": test_df,
            }
        X, y = to_list_data(train_df, dev_df, test_df)
        return X, y
    else:
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "dev": Dataset.from_pandas(dev_df),
            "test": Dataset.from_pandas(test_df)
        })
        return dataset
