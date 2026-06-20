"""General-purpose helpers for model analysis and reproducibility."""

import random
from typing import Tuple

import numpy as np
import torch
from torch import nn

from cvnets.utils.logger import double_dash_line, info, singe_dash_line


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model.

    Returns
    -------
    (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model: nn.Module, name: str = "Model") -> None:
    """Print a formatted summary of model architecture and parameters."""
    total, trainable = count_parameters(model)
    double_dash_line()
    info(f"{name} summary:")
    singe_dash_line()
    info(f"  Total params:     {total:,}")
    info(f"  Trainable params: {trainable:,}")
    info(f"  Non-trainable:    {total - trainable:,}")
    info(f"  Layers:           {len(list(model.modules())):,}")
    double_dash_line()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across torch, numpy, and Python.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool
        If ``True``, also set ``torch.backends.cudnn.deterministic = True``
        and ``torch.backends.cudnn.benchmark = False`` (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    info(f"Random seed set to {seed} (deterministic={deterministic})")
