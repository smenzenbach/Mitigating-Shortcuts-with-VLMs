import os
import torch
import numpy as np
import random


def seed_all(seed=60):
    """Seed all for reproducability."""
    print(f"[Using Seed= {seed}]")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
