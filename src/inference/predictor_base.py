import pickle
from pathlib import Path
from src.config_base import PathConfig, ModelConfig


def load_model():
    paths = PathConfig()
    model = ModelConfig()
    model_path = Path(paths.SAVE_MODEL_PATH / f"/tfidf_logreg_{model.version}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(model, text):
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    return {"text": text, "prediction": int(pred), "confidence": float(probs.max())}
