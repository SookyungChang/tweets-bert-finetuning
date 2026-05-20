# 🧠 Production-style Sentiment Analysis: BERT Fine-tuning + RAG YouTube Comment QA

**0.85 F1 score** on 640K tweets using DistilBERT fine-tuning, significantly outperforming the TF-IDF baseline (0.80 F1). This repository combines a modular ML pipeline with a live **RAG-powered YouTube comment analyzer** deployed via HuggingFace Spaces.

---

## 📊 Quick Overview

| Aspect | Details |
|--------|---------|
| **Task** | Binary sentiment classification (positive vs negative) + comment-level RAG query support |
| **Dataset** | 640,000 tweets + YouTube comments |
| **Baseline** | TF-IDF + Logistic Regression |
| **Production Model** | DistilBERT (fine-tuned) |
| **Best Model F1** | **0.8506** (BERT) vs 0.8056 (TF-IDF) |

---

## ✨ New Feature: RAG-powered YouTube Comment QA

A new Gradio app at `deployment/huggingface/gradio/rag_app.py` enables:
- YouTube comment scraping from a video ID or URL
- BERT sentiment analysis on each comment
- BERTopic topic modeling for positive and negative comment groups
- Vector DB creation + semantic retrieval using `langchain_huggingface`
- Question answering over retrieved comments with RAG

This makes the project not just a sentiment classifier, but also an interactive comment analysis assistant.

---

## 📁 Dataset Details

- **Total samples**: 640,000 tweets
- **Split**: Train (320K) / Validation (160K) / Test (160K)
- **Class balance**: 50% positive / 50% negative
- **Additional data**: YouTube comments (from video ID: SbNDmAJBtyU)

**Preprocessing**:
- Text cleaning (via `clean_text` function)
- Language detection (English only)
- Label normalization
- Train/Dev/Test stratified split

---

## ⚙️ Methods & Models

### 1️⃣ TF-IDF + Logistic Regression (Baseline)

**Hyperparameter Tuning** (Optuna):
- 150 trials
- Subset: 100K samples
- Best parameters:
  - `ngram_range = (1, 3)`
  - `min_df = 4`
  - `max_features = 28,056`
  - `C = 1.02`

**Performance**:
- Validation F1 (macro): **0.7967**
- Test F1 (macro): **0.8056**

### 2️⃣ DistilBERT Fine-Tuning

**Model**: `distilbert-base-uncased-finetuned-sst-2-english`  
**Framework**: HuggingFace Transformers + Accelerate

**Training Configuration**:
- Epochs: 1
- Batch size: 16
- Learning rate: 5e-5
- Weight decay: 0.01
- Mixed precision training (fp16)
- Optimizer: AdamW

**Results**:
- Pre-fine-tuned: F1 = **0.7194**
- Post-fine-tuned: F1 = **0.8506** ✨
- Improvement: +13% F1 score

**Published**:
- Uploaded to HuggingFace Hub @ [sweetguma/bert-sentiment-model](https://huggingface.co/sweetguma/bert-sentiment-model)
- Try the live app on HuggingFace Spaces: [sweetguma/sentiment-app](https://huggingface.co/spaces/sweetguma/sentiment-app)

---

## 📈 Model Comparison

| Model | F1 Score | Key Insight |
|-------|----------|------------|
| TF-IDF + LogReg | 0.8056 | Keyword-based, fast, interpretable |
| DistilBERT | **0.8506** | Context-aware, captures nuance |

**Example Prediction Difference**:
```
Text: "It's okay, not great but not bad."
Baseline (TF-IDF) → pred: 0 (negative)  [conf: 0.5635]
BERT              → pred: 1 (positive)  [conf: 0.8851]
```

---

## 📂 Project Structure

```
tweets-bert-finetuning/
├── src/
│   ├── __init__.py
│   ├── config.py                      # Path configuration
│   ├── config_base.py                 # Baseline model config
│   ├── config_bert.py                 # BERT model config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── get_comments.py            # YouTube comment fetcher
│   │   └── preprocess.py              # Text preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                # TF-IDF + LogReg model
│   │   └── bert.py                    # DistilBERT model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline.py          # Baseline training pipeline
│   │   └── train_bert.py              # BERT fine-tuning pipeline
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor_base.py          # Baseline inference
│   │   └── predictor_bert.py          # BERT inference
│   ├── analysis/
│   │   ├── rag.py                     # RAG query integration
│   │   └── topic_modeling.py          # BERTopic pipeline
│   └── database/
│       └── storage.py                 # Persist comment + vector stores
├── deployment/
│   ├── aws/
│   │   ├── app.py                     # AWS EC2 handler
│   │   └── Dockerfile
│   ├── huggingface/
│   │   ├── app.py                     # HF Spaces FastAPI deployment
│   │   ├── Dockerfile                 # HF Spaces deployment
│   │   └── gradio/
│   │       ├── rag_app.py             # RAG Gradio app for comment QA
│   │       ├── Dockerfile.rag         # Gradio Docker config
│   │       └── requirements_rag.txt   # RAG service dependencies
├── data/
│   ├── tweets/                        # Tweet dataset
│   ├── youtube_comments/              # Comment dataset
│   └── data_source.txt                # Data source references
├── saved_models/
│   ├── base/                          # Baseline model artifacts
│   └── bert/                          # BERT model checkpoints
├── notebooks/
│   ├── exploration.ipynb              # EDA & analysis
├── saved_experiments/
│   └── tfidf_history.json             # Optuna trial history
├── test_run_base.py                   # Baseline test script
├── test_run_bert.py                   # BERT test script
├── test.py                            # General test runner
├── hf_upload.py                       # HF Hub uploader
├── requirementsfull.txt               # Full dependencies
└── README.md                          # This file
```

---

## 🚀 FastAPI & Gradio Inference Services

### Local Setup

1. Install dependencies:
```bash
pip install -r requirementsfull.txt
```

2. Run the FastAPI server:
```bash
cd deployment/aws
python app.py
# or:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

3. Access Swagger UI:
```
http://localhost:8000/docs
```

### Local RAG Gradio App

Run the new comment analyzer locally from the Gradio app folder:
```bash
cd deployment/huggingface/gradio
python rag_app.py
```

This launches a UI where you can:
- enter a YouTube video URL or ID
- process comments with BERT sentiment and BERTopic
- ask natural language questions using RAG

### Option 1: AWS EC2 (ECR)

```bash
docker build -t sentiment-api:latest -f deployment/aws/Dockerfile .

docker run -p 8000:8000 sentiment-api:latest
```

### Option 2: HuggingFace Spaces

```bash
docker build -t sentiment-api:latest -f deployment/huggingface/Dockerfile .
```

Deploy the RAG-enabled Gradio app with `deployment/huggingface/gradio/Dockerfile.rag` for HuggingFace Spaces or local container testing.

Try the live app on HuggingFace Spaces:
- https://huggingface.co/spaces/sweetguma/sentiment-app

---

## 📦 Model Distribution

Models are versioned and hosted on **HuggingFace Hub**:
- 🤗 [sweetguma/bert-sentiment-model](https://huggingface.co/sweetguma/bert-sentiment-model)

**Automatic download** on first inference (cached locally).

```python
# Upload new model version
python hf_upload.py
```

---

## 🧠 Why this is "MLOps-like"

This project includes key production ML concepts:
- Model training pipeline separate from serving
- BERT inference service with FastAPI
- RAG-enabled comment analysis via Gradio
- Containerized deployment for AWS and HuggingFace Spaces
- Reproducibility with requirements and fixed pipeline

---

## 📚 Usage Examples

### 1. Train a Model from Scratch

**Baseline**:
```bash
python -m src.training.train_baseline
```

**BERT**:
```bash
python -m src.training.train_bert
```

### 2. Run Inference Locally

**Quick test**:
```bash
python test_run_bert.py
python test_run_base.py
```

**Compare both models**:
```python
from src.inference import predictor_base, predictor_bert

base = predictor_base.load_model()
bert = predictor_bert.Predictor("path/to/bert/model")
text = "This product is amazing!"
print(f"Baseline: {base.predict(text)}")
print(f"BERT: {bert.predict(text)}")
```

### 3. Hyperparameter Tuning

```bash
python -m src.experiments.optuna_search
```

### 4. Fetch YouTube Comments

```bash
python -c "from src.data.get_comments import get_comments; get_comments()"
```

---

## 🏗️ Architecture Decisions

### Why This Design?

✅ **Separation of Concerns**
- Training pipeline separate from serving
- Models loaded once at startup
- Stateless API and interactive Gradio app

✅ **Reproducibility**
- Fixed requirements + versions
- Containerization
- Model versioning via HuggingFace

✅ **Production-Ready**
- FastAPI for async performance
- Gradio RAG app for exploratory QA
- Graceful error handling and validation
- Multi-cloud support

---

## 🧪 Experiment Tracking

**W&B Integration** (`notebooks/wandb/`):
- Track hyperparameters, metrics, loss curves
- Compare model runs
- Version experiments

---

## 📋 Core Dependencies

```txt
torch==2.9.1
transformers==5.5.4
accelerate==1.13.0
fastapi==latest
scikit-learn==1.8.0
pandas==3.0.1
optuna==latest
huggingface_hub==1.11.0
gradio==latest
langchain_huggingface==latest
hdbscan==latest
```

Install all:
```bash
pip install -r requirementsfull.txt
pip install -r requirements/analysis-cpu.txt
```

---

## 🔥 What's Production-Ready?

✅ Modular code structure
✅ Type hints & configuration management
✅ Docker + FastAPI + Gradio
✅ Model versioning (HuggingFace Hub)
✅ RAG-enabled interactive analytics
✅ Multi-deployment (AWS, HF Spaces)
⚠️ Unit test coverage (to improve)

---

## 🚀 Future Improvements

- [ ] GitHub Actions CI/CD
- [ ] Batch inference endpoint
- [ ] Model A/B testing
- [ ] API rate limiting & auth
- [ ] Advanced logging & tracing
- [ ] Performance benchmarking

---

## 💡 Key Insight

**Production ML isn't just about accuracy.** This project demonstrates:
- Data pipeline → Model training → Evaluation → API serving → Containerization
- Classical vs modern NLP modeling
- Interactive RAG-based analysis for real-world comments

---

## 📄 License & Attribution

Dataset: Twitter sentiment data (public)
Models: DistilBERT (HuggingFace), HuggingFace Hub for distribution

---

**Last Updated**: May 2026  
**Status**: ✅ Production-ready for sentiment analysis and RAG comment QA
