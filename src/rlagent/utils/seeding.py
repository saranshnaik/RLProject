"""Global seeding helpers."""

import random

import numpy as np
import torch

EVAL_SEED_OFFSET = 10_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
