import pandas as pd
from datasets import Dataset, DatasetDict


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


def build_dataset(path, type="bert"):
    df = load_data(path)
    label_name, text_name = get_names(df)
    df = clean_labels(df)
    data_dict = split_by_fold(df, label_name, text_name)
    # Check Imbalance
    print(analyze_label_distribution(data_dict, data_dict["label_name"]))

    if type == "base":
        X, y = to_list_data(data_dict)
        return X, y
    else:
        dataset = DatasetDict(
            {
                split: Dataset.from_pandas(df)
                for split, df in data_dict.items()
                if split in ["train", "test", "dev"]
            }
        )
        return dataset
