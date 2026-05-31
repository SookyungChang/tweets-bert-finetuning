from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / "data"
    TWEETS_PATH: Path = DATA_PATH / "tweets"
    YOUTUBE_COMMENTS_PATH: Path = DATA_PATH / "youtube_comments"
    SAVED_MODELS_PATH: Path = BASE_DIR / "saved_models"
    VECTORSTORE_PATH: Path = BASE_DIR / "DB"
    DB_PATH: Path = BASE_DIR / "DBs"


@dataclass
class urlConfig:
    trained_model_id = "sweetguma/bert-sentiment-model"


@dataclass
class YoutubeCommentsConfig:
    VIDEO_ID: str = "SbNDmAJBtyU"  # https://youtu.be/SbNDmAJBtyU?si=9Fqc1fK7c1vYLkpS
    MAX_RESULTS: int = 100
    MAX_PAGES: int = 3
