from fastapi import FastAPI
from pydantic import BaseModel
from src.config_bert import PathConfig
from src.inference import predictor_base, predictor_bert

app = FastAPI()

# Global variables
base_model = None
bert_model = None


# one time run as server starts
@app.on_event("startup")
def load_models():
    global base_model, bert_model

    base_model = predictor_base.load_model()
    paths = PathConfig()
    modelpath = paths.SAVED_MODELS_PATH / "bert-0.1.0/checkpoint-20000"
    bert_model = predictor_bert.Predictor(modelpath)


# Request: text
class TextRequest(BaseModel):
    text: str


# health check
@app.get("/")
def root():
    return {"message": "API is running"}


# endpoint
@app.post("/predict_all")
def predict_all(request: TextRequest):
    base_result = predictor_base.predict(base_model, request.text)
    bert_result = bert_model.predict(request.text)

    return {"text": request.text, "baseline": base_result, "bert": bert_result}
