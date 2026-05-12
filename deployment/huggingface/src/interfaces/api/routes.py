from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import main as state
from service import predict_text_service, analyze_youtube_service
from web_ui import (
    home_page,
    text_home_page,
    text_results_page,
    youtube_home_page,
    youtube_results_page,
    error_page
)
from src.utils.youtube_analyzer import extract_video_id

# ── Routers ─────────────────────────────────────────
main_router    = APIRouter(tags=["Home"])
text_router    = APIRouter(prefix="/text",    tags=["Text Comparison"])
youtube_router = APIRouter(prefix="/youtube", tags=["YouTube Analyzer"])
api_router     = APIRouter(prefix="/api",     tags=["API"])


# ── Landing page ─────────────────────────────────────
@main_router.get("/", response_class=HTMLResponse)
def home():
    return home_page()


# ── Text comparison ──────────────────────────────────
@text_router.get("/", response_class=HTMLResponse)
def text_home():
    return text_home_page()

@text_router.post("/predict", response_class=HTMLResponse)
def text_predict(text: str = Form(...)):
    try:
        base_result, bert_result = predict_text_service(
            text,
            state.base_model,
            state.bert_model
        )
        return text_results_page(text, base_result, bert_result)
    except Exception as e:
        return error_page(str(e), "/text")


# ── YouTube analyzer ─────────────────────────────────
@youtube_router.get("/", response_class=HTMLResponse)
def youtube_home():
    return youtube_home_page()

@youtube_router.post("/analyze", response_class=HTMLResponse)
async def youtube_analyze(video_id: str = Form(...)):
    try:
        video_id = extract_video_id(video_id)
        result = await analyze_youtube_service(
            video_id,
            state.bert_model
        )
        return youtube_results_page(**result)
    except Exception as e:
        return error_page(str(e), "/youtube")


# ── API (JSON) ────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

@api_router.post("/predict_all")
def predict_all(request: TextRequest):
    try:
        base_result, bert_result = predict_text_service(
            request.text,
            state.base_model,
            state.bert_model
        )
        return {
            "text":     request.text,
            "baseline": base_result,
            "bert":     bert_result
        }
    except Exception as e:
        return {"error": str(e)}