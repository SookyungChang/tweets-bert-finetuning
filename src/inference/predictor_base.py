import pickle
from pathlib import Path
from src.config_bert import PathConfig


def load_model():
    paths = PathConfig()
    SAVED_MODELS_PATH = paths.SAVED_MODELS_PATH
    model_path = Path(SAVED_MODELS_PATH / "tfidf_logreg_0.1.0.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(model, text):
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    return {"text": text, "prediction": int(pred), "confidence": float(probs.max())}
