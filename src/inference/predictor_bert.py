import os
import torch

if torch.cuda.is_available():
    # Only make GPU 0 visible to this process when running inference.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Predictor:
    def __init__(self, model_path: str, device: str = None, threshold: int = 0.1):
        self.threshold = threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.abspath("onnx_models"), provider="CPUExecutionProvider"
        )
        if self.device == "cuda":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(self.device)
        else:
            self.model = ORTModelForSequenceClassification.from_pretrained(
                os.path.abspath("onnx_models"),
                export=False,
                provider="CPUExecutionProvider",  # ONNX form
            )

    def predict_text(self, text):
        if self.device == "cuda":
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs).item()
                margins = abs(probs[0][0]-probs[0][1])
                if margins < self.threshold:
                    pred = -1
                
            return {
                "text": text,
                "prediction": pred,
                # "confidence": float(torch.max(probs)),
                "confidence": margins / self.threshold,
            }

        else:
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()

            exp_logits = np.exp(
                logits - np.max(logits, axis=-1, keepdims=True)
            )  # Softmax (numpy version)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            pred = int(np.argmax(probs))
            # confidence = float(np.max(probs))
            margins = abs(probs[0][0]-probs[0][1])
            if abs(probs[0][0]-probs[0][1]) < self.threshold:
                pred = -1

            return {"text": text, "prediction": pred, "confidence": margins / self.threshold}

    def predict_df(self, df, text_column="text"):
        texts = df[text_column].dropna().tolist()
        print(f"Number of comments: {len(texts)}")

        if self.device == "cuda":
            batch_size = 16
            all_logits = []
            all_labels, all_scores = [], []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                all_logits.append(logits)
                probs = torch.softmax(logits, dim=1)

                margins = torch.abs(probs[:, 0] - probs[:, 1])
                labels = torch.argmax(probs, dim=-1)
                labels[margins < self.threshold] = -1
                all_labels.extend(labels.tolist())
                # all_scores.extend(probs.max(dim=1).values.tolist())
                all_scores.extend(margins / self.threshold)
        else:  # batch is no needed for CPU
            all_labels = []
            all_scores = []

            for text in texts:
                inputs = self.tokenizer(
                    [str(text)], return_tensors="pt", truncation=True, max_length=128
                )
                outputs = self.model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                if hasattr(logits, "detach"):
                    logits = logits.detach().cpu().numpy()
                if len(logits.shape) > 1 and logits.shape[0] == 1:
                    logits = logits[0]
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                margins = abs(probs[0]-probs[1])
                if margins < self.threshold:
                    all_labels.append(-1)
                else:
                    all_labels.append(int(np.argmax(probs)))
                    # all_scores.append(float(np.max(probs)))
                all_scores.append(margins/self.threshold)

        df["sentiment_label"] = all_labels
        df["sentiment_score"] = all_scores

        return df
