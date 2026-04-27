from huggingface_hub import upload_folder
from src.config_bert import PathConfig


def upload(folder_name):
    paths = PathConfig()
    folder_path = paths.SAVED_MODELS_PATH / folder_name
    upload_folder(
        folder_path=folder_path,
        repo_id="sweetguma/bert-sentiment-model",
        repo_type="model",
    )


if __name__ == "__main__":
    folder_name = "bert-0.1.0/checkpoint-20000"
    upload(folder_name)
