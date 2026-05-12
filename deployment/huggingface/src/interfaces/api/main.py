from pathlib import Path
import sys
import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent.parent
sys.path.append(str(parent_dir))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from huggingface_hub import snapshot_download
from src.inference import predictor_base, predictor_bert
import multiprocessing

# --- Global models ---
base_model = None
bert_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global base_model, bert_model
    base_model = predictor_base.load_model()
    bert_path  = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
    bert_model = predictor_bert.Predictor(bert_path)
    print("✅ Models loaded!")

    yield

    print("🛑 Shutting down...")
    for child in multiprocessing.active_children():
        child.terminate()
        child.join()

app = FastAPI(lifespan=lifespan)

# ── Register all routers ─────────────────────────────
from routes import main_router, text_router, youtube_router, api_router

app.include_router(main_router)
app.include_router(text_router)
app.include_router(youtube_router)
app.include_router(api_router)