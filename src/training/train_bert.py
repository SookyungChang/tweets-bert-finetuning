import os
from pathlib import Path
from transformers import TrainingArguments
from src.data.preprocess import build_dataset
from src.models.bert import BERTfinetuning


def train():
    dataset = build_dataset()

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    VERSION = "0.1.0"
    SAVED_MODELS_PATH = "../saved_models"
    Path(SAVED_MODELS_PATH).mkdir(parents=True, exist_ok=True)
    output_dir_path = os.path.join(SAVED_MODELS_PATH, f"bert-{VERSION}")

    # Training Pipeline: Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_path,  # Directory for saving model checkpoints
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir_path, "runs"),
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",
        learning_rate=5e-5,  # *Start with a small learning rate
        per_device_train_batch_size=16,  # *Batch size per GPU
        per_device_eval_batch_size=16,
        num_train_epochs=1,  # Number of epochs
        weight_decay=0.01,  # Regularization
        save_total_limit=2,  # Limit checkpoints to save space
        load_best_model_at_end=True,  # Automatically load the best checkpoint        # Directory for logs
        logging_steps=100,  # Log every 100 steps
        fp16=True,  # Enable mixed precision for faster training
    )

    model = BERTfinetuning(model_name, dataset, training_args)
    model.train()


if __name__ == "__main__":
    train()
