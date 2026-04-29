from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from src.training.train_baseline import train
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

    data_path = "data/tweets640k.parquet"
    train(data_path)

    base_model = predictor_base.load_model()
    paths = PathConfig()
    bert_path = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
    bert_model = predictor_bert.Predictor(bert_path)


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
