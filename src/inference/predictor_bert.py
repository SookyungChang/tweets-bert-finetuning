import os

# Only make GPU 0 visible to this process when running inference.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Predictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs).item()

        return {"text": text, "prediction": pred, "confidence": float(torch.max(probs))}

    def predict_df(self, df, text_column="text"):
        texts = df[text_column].dropna().tolist()
        print(f"Number of comments: {len(texts)}")

        batch_size = 16
        all_logits = []
        all_labels, all_scores = [], []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            all_logits.append(logits)
            probs = torch.softmax(logits, dim=1)
            labels = probs.argmax(dim=1).tolist()
            all_labels.extend(labels)
            all_scores.extend(probs.max(dim=1).values.tolist())

        df["sentiment_label"] = all_labels
        df["sentiment_score"] = all_scores
        return df