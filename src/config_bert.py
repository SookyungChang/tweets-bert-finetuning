from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    seed: int = 42
    version: str = "0.2.1"
    num_epochs: int = 1
    f1_avg: str = "macro"
    dropout: float = 0.1
    fine_tuned_model: str = field(init=False)
    def __post_init__(self):
        self.fine_tuned_model = f"sweetguma/bert-sentiment-model-v{self.version}"

