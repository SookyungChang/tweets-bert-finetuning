import os
from src.training.train_baseline import train
from src.data.preprocess import build_dataset
from src.inference.predictor_base import predict
from src.config_base import PathConfig, ModelConfig
from src.inference import predictor_base


def test_train():
    data_path = "data/tweets640k.parquet"
    train(data_path)


def test_inf():
    model = predictor_base.load_model()
    print(model)
    texts = ["I love this!", "This is terrible", "I'm not sure how I feel"]

    for t in texts:
        print(predictor_base.predict(model, t))


if __name__ == "__main__":
    # test_train()
    test_inf()
