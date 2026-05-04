import os

# Only make GPU 0 visible to this process. This prevents the internal/bad GPU 1
# from being selected while still allowing a CPU fallback if CUDA is unavailable.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import TrainingArguments
from src.data.preprocess import build_dataset
from src.models.bert import BERTfinetuning
from src.config_bert import ModelConfig, PathConfig


def train(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), sample_size=None):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    paths = PathConfig()
    model = ModelConfig()

    # Use a smaller subset for faster debugging. Set sample_size=10000 to load up to 10k rows per split.
    dataset = build_dataset(paths.DATA_PATH, type="bert", sample_size=sample_size)
    model_name = model.model_name
    paths.SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    output_dir_path = os.path.join(paths.SAVED_MODELS_PATH, f"bert-{model.version}")

    # Training Pipeline: Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_path,  # Directory for saving model checkpoints
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir_path, "runs"),  # TENSORBOARD_LOGGING_DIR
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",
        learning_rate=5e-5,  # *Start with a small learning rate
        per_device_train_batch_size=16,  # Batch size per device
        per_device_eval_batch_size=16,
        num_train_epochs=1,  # Number of epochs
        weight_decay=0.01,  # Regularization
        save_total_limit=2,  # Limit checkpoints to save space
        load_best_model_at_end=True,  # Automatically load the best checkpoint        # Directory for logs
        logging_steps=100,  # Log every 100 steps
        fp16=torch.cuda.is_available(),  # Mixed precision only on GPU
    )
    bert = BERTfinetuning(model_name, dataset, training_args)
    print("Before train:", bert.test())
    bert.train()
    print("After train:", bert.test())
    return bert


if __name__ == "__main__":
    train(sample_size=10000)
