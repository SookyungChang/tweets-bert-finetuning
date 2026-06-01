from huggingface_hub import upload_folder, create_repo
from src.config import PathConfig
from src.config_bert import ModelConfig
paths = PathConfig()
models = ModelConfig()

def upload(folder_name):
    folder_path = paths.SAVED_MODELS_PATH / folder_name
    repo_id=f"sweetguma/bert-sentiment-model-v{models.version}"
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    folder_name = f"bert-{models.version}/checkpoint-20000"
    upload(folder_name)
