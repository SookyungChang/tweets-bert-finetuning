# 🧠 Production-style Sentiment Analysis: BERT Fine-tuning vs TF-IDF Baseline

**0.85 F1 score** on 640K tweets using DistilBERT fine-tuning, significantly outperforming the TF-IDF baseline (0.80 F1).  
Production-ready **modular ML pipeline** with FastAPI inference service, HuggingFace Hub integration, and multi-cloud deployment (AWS Lambda, HuggingFace Spaces).

---

## 📊 Quick Overview

| Aspect | Details |
|--------|---------|
| **Task** | Binary sentiment classification (positive vs negative) |
| **Dataset** | 640,000 tweets + YouTube comments |
| **Baseline** | TF-IDF + Logistic Regression |
| **Production Model** | DistilBERT (fine-tuned) |
| **Best Model F1** | **0.8506** (BERT) vs 0.8056 (TF-IDF) |

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

---

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


---

## 📈 Model Comparison

| Model | F1 Score | Key Insight |
|-------|----------|------------|
| TF-IDF + LogReg | 0.8056 | Keyword-based, fast, interpretable |
| DistilBERT | **0.8506** | Context-aware, captures nuance |

**Example Prediction Difference**:
```
Text: "It's okay, not great but not bad."

Baseline (TF-IDF) → pred: 0 (negative)  [conf: 0.57]
BERT              → pred: 1 (positive)  [conf: 0.88]

Why? BERT understands the overall positive sentiment despite negative words.
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
│   └── experiments/
│       ├── __init__.py
│       └── optuna_search.py           # Hyperparameter tuning
│
├── deployment/
│   ├── aws/
│   │   ├── app.py                     # AWS Lambda handler
│   │   └── Dockerfile
│   └── huggingface/
│       └── Dockerfile                 # HF Spaces deployment
│
├── data/
│   ├── tweets/                        # Tweet dataset
│   ├── youtube_comments/              # Comment dataset
│   └── data_source.txt                # Data source references
│
├── saved_models/
│   ├── base/                          # Baseline model artifacts
│   └── bert/                          # BERT model checkpoints
│
├── notebooks/
│   ├── exploration.ipynb              # EDA & analysis
│   └── wandb/                         # W&B experiment logs
│
├── saved_experiments/
│   └── tfidf_history.json             # Optuna trial history
│
├── test_run_base.py                   # Baseline test script
├── test_run_bert.py                   # BERT test script
├── test_app.py                        # API test script
├── hf_upload.py                       # HF Hub uploader
├── requirements.txt                   # Core dependencies
├── requirements_full.txt               # Full environment
└── README.md                          # This file
```
---

## 🚀 FastAPI Inference Service

Both models are exposed via **REST API endpoints** for real-time inference.

### Local Setup

**1. Install dependencies**:
```bash
pip install -r requirements.txt
```

**2. Run the FastAPI server** (AWS flavor):
```bash
cd deployment/aws
python app.py
# or:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**3. Access Swagger UI**:
```
http://localhost:8000/docs
```

### API Endpoints

**GET `/`** - Web UI for interactive predictions

**POST `/predict`** - Single text prediction
```json
{
  "text": "I love this product!"
}
```

**Response**:
```json
{
  "prediction": 1,
  "confidence": 0.99,
  "model": "bert"
}
```

---

## 🐳 Containerized Deployment

### Option 1: AWS Lambda (ECR)

```bash
# Build
docker build -t sentiment-api:latest -f deployment/aws/Dockerfile .

# Run locally
docker run -p 8000:8000 sentiment-api:latest

# Push to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_URI>
docker tag sentiment-api:latest <ECR_URI>/sentiment-api:latest
docker push <ECR_URI>/sentiment-api:latest
```

### Option 2: HuggingFace Spaces

```bash
# Build & push to HF
docker build -t sentiment-api:latest -f deployment/huggingface/Dockerfile .
# Push to HF (requires repo setup)
```

---

## 📦 Model Distribution

Models are versioned and hosted on **HuggingFace Hub**:
- 🤗 [sweetguma/bert-sentiment-model](https://huggingface.co/sweetguma/bert-sentiment-model)

**Automatic download** on first inference (cached locally).

```python
# Upload new model version
python hf_upload.py
```
## 🧠 Why this is "MLOps-like"

This project includes key production ML concepts:
✔ Model training pipeline (offline)
✔ Model inference service (FastAPI)
✔ Containerized deployment (Docker)
✔ Reproducibility (requirements + fixed pipeline)
✔ Separation of training vs serving

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
- Stateless API (horizontal scalability)

✅ **Reproducibility**
- Fixed requirements + versions
- Containerization
- W&B experiment tracking

✅ **Production-Ready**
- FastAPI for async performance
- Graceful error handling
- HuggingFace Hub integration
- Multi-cloud support

### Training vs Serving

Training runs offline (may use GPU); API servers are stateless and portable.

---

## 🧪 Experiment Tracking

**W&B Integration** (`notebooks/wandb/`):
- Track hyperparameters, metrics, loss curves
- Compare model runs
- Version experiments
- Auto-logged during training

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
```

Install all:
```bash
pip install -r requirements.txt
pip install -r requirements_full.txt  # Includes visualization & APIs
```

---

## 🔥 What's Production-Ready?

✅ Modular code structure  
✅ Type hints & configuration management  
✅ Docker + FastAPI  
✅ Model versioning (HuggingFace Hub)  
✅ Experiment tracking (W&B)  
✅ Error handling & validation  
✅ Multi-deployment (AWS, HF Spaces)  
⚠️ CI/CD pipeline (planned)  
⚠️ Comprehensive monitoring (partial)  
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

**Production ML isn't just about model accuracy.** This project demonstrates:
- Data pipeline → Model training → Evaluation → API serving → Containerization
- Comparison between classical (TF-IDF) and modern (BERT) approaches
- Real-world deployment considerations (versioning, scalability, reproducibility)

---

## 📄 License & Attribution

Dataset: Twitter sentiment data (public)
Models: DistilBERT (HuggingFace), HuggingFace Hub for distribution

---

**Last Updated**: May 2026  
**Status**: ✅ Production-ready for sentiment analysis inference