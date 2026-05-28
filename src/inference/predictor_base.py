import pickle
from pathlib import Path
from src.config_base import PathConfig, ModelConfig


def load_model():
    paths = PathConfig()
    model = ModelConfig()
    model_path = paths.SAVE_MODEL_PATH / f"tfidf_logreg_{model.version}.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(model, text, threshold = 0.1):
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    margins = abs(probs[0]-probs[1])
    if abs(probs[0]-probs[1]) < threshold:
        pred = -1
    # return {"text": text, "prediction": int(pred), "confidence": float(probs.max())}
    return {"text": text, "prediction": int(pred), "confidence": margins / threshold}
