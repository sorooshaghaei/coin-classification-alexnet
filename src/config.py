import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
#20% of the training set
DEFAULT_VALIDATION_SIZE = 0.2
#changed epoch from 2 to 4
DEFAULT_HEAD_EPOCHS =4
# changed finetune from 5 to 10
DEFAULT_FINETUNE_EPOCHS =10
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_FINETUNE_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
#changed label smoothing from 0.1 to 0.05 
DEFAULT_LABEL_SMOOTHING = 0.05
# chnaged patience from 2 to 4
DEFAULT_EARLY_STOPPING_PATIENCE =4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except RuntimeError as exc:
            warnings.warn(
                f"MPS is available but could not be initialized ({exc}). Falling back to CPU.",
                stacklevel=2,
            )

    return torch.device("cpu")


DEVICE = resolve_device()


@dataclass(frozen=True)
class TrainingStage:
    name: str
    epochs: int
    learning_rate: float
    freeze_features: bool


def set_seed(seed: int = SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
