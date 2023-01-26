import random
from typing import Union
from pathlib import Path


import numpy as np

import torch


def set_seeds(seed):
    if seed is None:
        return
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkdir(directory: Union[str, Path]):
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory
