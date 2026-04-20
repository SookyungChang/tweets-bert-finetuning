from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    seed: int = 42
    version: str = "0.1.0"


@dataclass
class PathConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    SAVED_MODELS_PATH: Path = BASE_DIR / "saved_models"
    DATA_PATH: Path = BASE_DIR / "data/tweets640k.parquet"


# @dataclass
# class SearchSpace:
#     min_df: tuple = (1, 100)
#     max_features: tuple = (5000, 30000)
#     C: tuple = (1e-3, 50.0)


# @dataclass
# class ModelConfig:
#     seed: int = 42
#     max_iter: int = 2000
#     version: str = "0.1.0"
#     f1_avg: str = "macro"
