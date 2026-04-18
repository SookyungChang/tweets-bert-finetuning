from dataclasses import dataclass


@dataclass
class SearchConfig:
    n_trials: int = 150
    sample_size: int = 100000


@dataclass
class SearchSpace:
    min_df: tuple = (1, 100)
    max_features: tuple = (5000, 30000)
    C: tuple = (1e-3, 50.0)


@dataclass
class ModelConfig:
    seed: int = 42
    max_iter: int = 2000
    version: str = "0.1.0"
    f1_avg: str = "macro"
