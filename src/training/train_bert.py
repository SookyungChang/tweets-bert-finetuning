import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import TrainingArguments
from src.data.preprocess import build_dataset
from src.models.bert import BERTfinetuning
from src.config_bert import ModelConfig, PathConfig


def train():

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is ROCm/CUDA available? : {torch.cuda.is_available()}")
    print(
        f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
    )

    paths = PathConfig()
    model = ModelConfig()

    dataset = build_dataset(paths.DATA_PATH, type="bert")
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
        per_device_train_batch_size=16,  # *Batch size per GPU
        per_device_eval_batch_size=16,
        num_train_epochs=1,  # Number of epochs
        weight_decay=0.01,  # Regularization
        save_total_limit=2,  # Limit checkpoints to save space
        load_best_model_at_end=True,  # Automatically load the best checkpoint        # Directory for logs
        logging_steps=100,  # Log every 100 steps
        fp16=True,  # Enable mixed precision for faster training
    )
    bert = BERTfinetuning(model_name, dataset, training_args)
    print("Before train:", bert.test())
    bert.train()
    print("After train:", bert.test())
    return bert


if __name__ == "__main__":
    train()
