from typing import Optional
from torch import Tensor, nn

from . import register_act_fn


@register_act_fn("gelu")
class GELU(nn.GELU):
    def __init__(self, inplace: bool = False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def __repr__(self):
        return f"GELU : inplace {self.inplace}"
