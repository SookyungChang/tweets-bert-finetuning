from src.data.preprocess import build_dataset
from src.training.train_bert import train
from src.config_bert import PathConfig


def test_all():
    train()


if __name__ == "__main__":
    test_all()
