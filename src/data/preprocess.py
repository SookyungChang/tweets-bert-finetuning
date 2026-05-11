import pandas as pd
from datasets import Dataset, DatasetDict
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
import re


def load_data(path):
    return pd.read_parquet(path)


def get_names(df):
    return df.columns[0], df.columns[2]


def clean_labels(df):
    df["label"] = df["label"].replace(4, 1)
    return df


def split_by_fold(df, label_name, text_name):
    return {
        "label_name": label_name,
        "text_name": text_name,
        "train": df[df["fold"] == "train"],
        "test": df[df["fold"] == "test"],
        "dev": df[df["fold"] == "dev"],
    }


def analyze_label_distribution(data_dict, label_name):
    splits = ["train", "test", "dev"]
    dist_list = []

    for split in splits:
        dist = data_dict[split][label_name].value_counts(normalize=True) * 100
        dist.name = f"{split.upper()} (%)"
        dist_list.append(dist)

    return pd.concat(dist_list, axis=1).fillna(0).sort_index()


def to_list_data(data_dict):
    t_name, l_name = data_dict["text_name"], data_dict["label_name"]
    splits = ["train", "test", "dev"]

    X = [data_dict[s][t_name].str.replace("<p>", " ").values.tolist() for s in splits]

    y = [data_dict[s][l_name].values.tolist() for s in splits]

    return (X, y)


def build_dataset(path, type="bert", sample_size=None):
    df = load_data(path)
    label_name, text_name = get_names(df)
    df = clean_labels(df)
    data_dict = split_by_fold(df, label_name, text_name)
    # Check Imbalance
    print(analyze_label_distribution(data_dict, data_dict["label_name"]))

    if sample_size is not None:
        for split in ["train", "dev", "test"]:
            split_df = data_dict[split]
            if len(split_df) > sample_size:
                data_dict[split] = split_df.sample(n=sample_size, random_state=42)

    if type == "base":
        X, y = to_list_data(data_dict)
        return X, y
    else:
        dataset = DatasetDict(
            {
                split: Dataset.from_pandas(data_dict[split])
                for split in ["train", "dev", "test"]
            }
        )
        return dataset


def remove_urls(text):
    """Remove any URLs from a comment string."""
    # Regex pattern that matches http/https links and bare www. addresses
    url_pattern = r"https?://\S+|www\.\S+"
    # Replace any matched URLs with an empty string, then strip leftover whitespace
    return re.sub(url_pattern, "", str(text)).strip()


def safe_detect(text):
    """Try to detect the language of a comment. Returns result or None if uncertain."""

    # Skip empty or very short texts — too little content to detect reliably
    if not text or len(str(text)) <= 5:
        return None

    try:
        # detect_langs() returns a list of language guesses with probabilities
        # We take the top guess [0]
        res = detect_langs(text)[0]

        # Only accept if it's English AND the model is highly confident (>90%)
        if res.lang == "en" and res.prob > 0.9:
            return res

    except LangDetectException:
        # If detection fails entirely (e.g. unrecognizable characters), just skip it
        pass

    return None  # Return None for non-English or low-confidence results
