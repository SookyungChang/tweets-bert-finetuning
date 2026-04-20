import os
import pickle
from pathlib import Path
from src.training.train_baseline import train
from src.data.preprocess import build_dataset
from src.inference.predictor_base import predict
from src.config_base import SearchConfig, SearchSpace, ModelConfig
from src.inference import predictor_base


def test_all():
    model = predictor_base.load_model()
    texts = ["I love this!", "This is terrible", "I'm not sure how I feel"]

    for t in texts:
        print(predictor_base.predict(model, t))


if __name__ == "__main__":
    test_all()

# data_path = "data/tweets640k.parquet"
# train(data_path)

# X, y = build_dataset(data_path, type="base")

# _, X_test, _, _, y_test, _ = (
#     *X,
#     *y,
# )

# model = ModelConfig()

# BASE_DIR = Path(__file__).resolve().parent
# SAVE_MODEL_PATH = BASE_DIR / "saved_models"
# pipe_path = SAVE_MODEL_PATH / f"tfidf_logreg_{model.version}.pkl"
# predict(pipe_path, X_test, y_test)
