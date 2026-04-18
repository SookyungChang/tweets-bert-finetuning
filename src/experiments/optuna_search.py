import json
import os
from pathlib import Path
import optuna  # https://optuna.readthedocs.io/en/stable/tutorial/index.html
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config_base import SearchConfig, SearchSpace, ModelConfig


def get_best_para(
    X_train,
    y_train,
    X_dev,
    y_dev,
    config: SearchConfig,
    space: SearchSpace,
    model: ModelConfig,
):
    X_train_sub, y_train_sub = (
        X_train[: config.sample_size],
        y_train[: config.sample_size],
    )

    X_dev_sub, y_dev_sub = X_dev[: config.sample_size], y_dev[: config.sample_size]

    ngram_dict = {"1-1": (1, 1), "1-2": (1, 2), "1-3": (1, 3)}

    def objective(trial):
        ngram_choice = trial.suggest_categorical("ngram_range", ["1-1", "1-2", "1-3"])

        current_ngram = ngram_dict[ngram_choice]

        hyp_dict = {
            "min_df": space.min_df,
            "max_features": space.max_features,
            "C": space.C,
            "max_iter": model.max_iter,
            "random_state": model.seed,
        }

        min_df = trial.suggest_int(
            "min_df", hyp_dict["min_df"][0], hyp_dict["min_df"][1]
        )
        max_features = trial.suggest_int(
            "max_features",
            hyp_dict["max_features"][0],
            hyp_dict["max_features"][1],
        )
        C = trial.suggest_float("C", hyp_dict["C"][0], hyp_dict["C"][1], log=True)

        current_pipe = Pipeline(
            [
                (
                    "vec",
                    TfidfVectorizer(
                        ngram_range=current_ngram,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        solver="saga",
                        max_iter=hyp_dict["max_iter"],
                        random_state=hyp_dict["random_state"],
                    ),
                ),
            ]
        )

        current_pipe.fit(X_train_sub, y_train_sub)
        return f1_score(y_dev_sub, current_pipe.predict(X_dev_sub), average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials)

    best_params = study.best_params

    best_params["ngram_range"] = ngram_dict[study.best_params["ngram_range"]]

    # Generate a unique key based on experiment conditions

    new_data = {
        "best_params": best_params,
        "f1_score": study.best_value,
        "n_trials": config.n_trials,
        "sample_size": config.sample_size,
    }

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SAVE_EXP_PATH = BASE_DIR / "saved_experiments"
    SAVE_EXP_PATH.mkdir(parents=True, exist_ok=True)
    exp_key = f"trials{config.n_trials}_sample{config.sample_size}"

    # Load existing history if the file exists
    # Update the record (Overwrites if the key exists, adds if it doesn't)
    file_path = SAVE_EXP_PATH / "tfidf_history.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)

            except json.JSONDecodeError:
                history = {}

    else:
        history = {}

    history[exp_key] = new_data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"Best F1 Score: {study.best_value}")
    print(f"Best Params: {best_params}")
    return best_params
