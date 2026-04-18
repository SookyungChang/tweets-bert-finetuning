import os
import pickle
from pathlib import Path
from sklearn.metrics import f1_score

from src.data.preprocess import build_dataset
from src.models.baseline import IT_IDF
from src.experiments.optuna_search import get_best_para
from src.config_base import SearchConfig, SearchSpace, ModelConfig


def train(data_path):
    X, y = build_dataset(data_path)

    itidf = IT_IDF(X, y)

    config = SearchConfig()
    space = SearchSpace()
    model = ModelConfig()

    best_params = get_best_para(
        itidf.X_train, itidf.y_train, itidf.X_dev, itidf.y_dev, config, space, model
    )

    vectorizer, _, _, _ = itidf.get_vectors(
        best_params["ngram_range"],
        best_params["min_df"],
        best_params["max_features"],
    )

    pipe = itidf.get_pipe(
        vectorizer,
        best_params["C"],
        model.seed,
        max_iter=model.max_iter,
    )

    preds = pipe.predict(itidf.X_test)
    f1 = f1_score(itidf.y_test, preds, average=model.f1_avg)

    print(f"F1 score: {f1}")

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SAVE_MODEL_PATH = BASE_DIR / "saved_models"
    SAVE_MODEL_PATH.mkdir(exist_ok=True, parents=True)

    model_path = SAVE_MODEL_PATH / f"tfidf_logreg_{model.version}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    print(model_path, f"tfidf_logreg_{model.version}.pkl", "saved")


if __name__ == "__main__":
    train()
