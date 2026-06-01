import os
from src.training.train_baseline import train
from src.inference.predictor_base import predict
from src.config import PathConfig
from src.inference import predictor_base

from src.data.build_datasets import build_dataset_db


def test_inf():
    model = predictor_base.load_model()
    print(model)
    texts = ["I love this!", "This is terrible", "I'm not sure how I feel"]

    for t in texts:
        print(predictor_base.predict(model, t))


if __name__ == "__main__":
    train()
    # test_inf()
