from typing import Optional
from torch import Tensor, nn

from . import register_act_fn


@register_act_fn("elu")
class ELU(nn.ELU):
    def __init__(self, alpha: float = 1.0, inplace: bool = False, *args, **kwargs):
        super().__init__(alpha=alpha, inplace=inplace)
        self.alpha = alpha
        self.inplace = inplace

    def __repr__(self):
        return f"ELU(alpha={self.alpha}, inplace={self.inplace})"
