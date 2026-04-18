from src.data.preprocess import build_dataset


def test_all():
    data_path = "data/tweets640k.parquet"
    dataset = build_dataset(data_path, type="bert")

    print(dataset)
    print(dataset["train"][0])


if __name__ == "__main__":
    test_all()
