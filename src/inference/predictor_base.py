import os
import pickle
from src.config_base import ModelConfig
from sklearn.metrics import f1_score


def predict(pipe_path, X_test, y_test):
    model = ModelConfig()
    with open(pipe_path, "rb") as f:
        loaded_pipe = pickle.load(f)
    pred_labels_loaded = loaded_pipe.predict(X_test)
    f1_loaded = f1_score(y_test, pred_labels_loaded, average=model.f1_avg)

    print(f"F1 score with {model.f1_avg}-averaging is {f1_loaded}")
