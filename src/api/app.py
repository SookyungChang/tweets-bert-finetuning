from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predictor_bert import Predictor

app = FastAPI()

predictor = Predictor("saved_models/bert")


class Request(BaseModel):
    text: str


@app.post("/predict")
def predict(req: Request):
    pred, conf = predictor.predict(req.text)
    return {"text": req.text, "prediction": pred, "confidence": conf}
