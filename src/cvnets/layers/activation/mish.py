import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import register_act_fn


@register_act_fn("mish")
class Mish(nn.Module):
    """
    Mish activation function: x * tanh(softplus(x)).

    Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation
    Function" (Diganta Misra, 2019).
    """

    def __init__(self, inplace: bool = False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))

    def __repr__(self):
        return f"Mish : inplace {self.inplace}"
