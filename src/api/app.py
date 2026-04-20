from fastapi import FastAPI
from pydantic import BaseModel
from src.config_bert import PathConfig

from src.inference import predictor_base, predictor_bert

app = FastAPI()

# 🔥 서버 시작 시 모델 1번만 로드 (중요)
base_model = predictor_base.load_model()
paths = PathConfig()
modelpath = paths.SAVED_MODELS_PATH / "bert-0.1.0/checkpoint-20000"
bert = predictor_bert.Predictor(modelpath)


# 입력 형식 정의
class TextRequest(BaseModel):
    text: str


# health check
@app.get("/")
def root():
    return {"message": "API is running"}


# 🔥 핵심 endpoint
@app.post("/predict_all")
def predict_all(request: TextRequest):
    base_result = predictor_base.predict(base_model, request.text)
    bert_result = bert.predict(request.text)

    return {"text": request.text, "baseline": base_result, "bert": bert_result}
