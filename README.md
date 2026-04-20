🧠 Sentiment Analysis: TF-IDF vs BERT (Production-Ready)

Achieved 0.85 F1 score on 640K tweets using BERT fine-tuning, outperforming a TF-IDF baseline.
Built as a modular ML pipeline + API service for real-world usage.

---

📊 Overview

This project compares a classical machine learning baseline with a modern transformer-based approach for sentiment classification.

Task

- Binary sentiment classification (positive vs negative)

Dataset

- 640,000 tweets

Models

- TF-IDF + Logistic Regression (baseline)
- DistilBERT (fine-tuned)

Key Extension (Production)

- Model inference API using FastAPI
- Modular ML pipeline (data → training → inference → API)
- Baseline vs BERT comparison analysis

---

📁 Dataset

- Total samples: 640,000
- Train: 320,000
- Validation: 160,000
- Test: 160,000
- Balanced classes (50% / 50%)

Preprocessing

- Text cleaning ("clean_text")
- Label normalization (4 → 1)
- Train / Dev / Test split

---

⚙️ Methods

1. TF-IDF + Logistic Regression

- n-grams: (1, 3)
- max_features: ~30K
- Hyperparameter tuning (Optuna)
  - 150 trials
  - 100K subset

Best Parameters

ngram_range = (1, 3)
min_df = 5
max_features = 29482
C = 1.01

Performance

- Validation F1 (macro): 0.7966
- Test F1 (macro): 0.8056

---

2. DistilBERT Fine-Tuning

- Model: distilbert-base-uncased-finetuned-sst-2-english
- Framework: HuggingFace Transformers

Training Setup

- Epochs: 1
- Batch size: 16
- Learning rate: 5e-5
- Weight decay: 0.01
- Mixed precision (fp16)

Performance

- Before fine-tuning: F1 = 0.719
- After fine-tuning: F1 = 0.8496

---

📈 Results

Model| F1 Score
TF-IDF + Logistic Regression| 0.8056
DistilBERT (fine-tuned)| 0.8496

---

🔍 Model Comparison (Key Insight)

Example:

Text: "It's okay, not great but not bad."

Baseline → pred: 0 (conf: 0.57)
BERT     → pred: 1 (conf: 0.88)

Interpretation

- Baseline relies on keywords ("not", "bad") → predicts negative
- BERT understands context → predicts slightly positive

👉 Conclusion

«Baseline is keyword-based, while BERT captures contextual semantics.»

---

🏗️ Project Structure (Production-Oriented)

bert-sentiment-project/
│
├── data/
│   └── tweets640k.parquet
│
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI service
│   │
│   ├── data/
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   └── bert.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   └── train_bert.py
│   │
│   ├── inference/
│   │   ├── predictor_base.py
│   │   └── predictor_bert.py
│   │
│   └── config/
│       ├── config_base.py
│       └── config_bert.py
│
├── saved_models/
├── saved_experiments/
├── notebooks/
├── test_run_base.py
├── test_run_bert.py
└── README.md

---

🚀 API (Model Serving)

This project includes a production-style inference API.

Run Server

uvicorn src.api.app:app --reload

---

Open API Docs (Swagger UI)

http://127.0.0.1:8000/docs

---

Example Request

POST /predict

{
  "text": "I feel so happy today!"
}

---

Example Response

{
  "prediction": 1,
  "confidence": 0.99
}

---

Compare Models (Optional Endpoint)

POST /predict_all

{
  "text": "It's okay, not great but not bad."
}

{
  "baseline": {...},
  "bert": {...}
}

---

🧪 Local Testing

Run Baseline Test

python test_run_base.py

Run BERT Test

python test_run_bert.py

---

🧠 ML Engineering Highlights

- Modular architecture (data / model / training / inference / API)
- Separation of concerns
- Reproducible experiments (config-driven)
- Hyperparameter tuning with Optuna
- Model comparison & error analysis
- Production-ready API

---

💾 Model Saving

- TF-IDF pipeline → "pickle"
- BERT → HuggingFace "Trainer.save_model()"

---

🚀 Future Work

- Improve BERT tuning (epochs, scheduling)
- Try larger models (RoBERTa, BERT-base)
- Add explainability (SHAP, attention visualization)
- Deploy API (Render / Docker)
- Add CI/CD pipeline

---

🧪 Reproducibility

- Seed: 42
- Libraries:
  - PyTorch
  - Transformers
  - Scikit-learn
  - Pandas / NumPy
  - FastAPI

---

🇩🇪 Deutsche Zusammenfassung

Dieses Projekt vergleicht klassische Machine-Learning-Methoden (TF-IDF + Logistic Regression) mit modernen Transformer-Modellen (DistilBERT) für die Sentimentanalyse auf einem großen Twitter-Datensatz (640K Samples).

Das feinabgestimmte BERT-Modell erreicht eine F1-Score von 0.85 und übertrifft die Baseline deutlich. Zusätzlich wurde eine API entwickelt, um das Modell in einer realen Umgebung bereitzustellen.

---

💡 Key Takeaway

«This project goes beyond model training — it demonstrates how to design, evaluate, and deploy machine learning systems in a production-like environment.»