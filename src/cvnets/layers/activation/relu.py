from typing import Optional
from torch import Tensor, nn

from . import register_act_fn


@register_act_fn("relu")
class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, *args, **kwargs):
        super().__init__(inplace=inplace)
        self.inplace = inplace

    def __repr__(self):
        return f"ReLU : inplace {self.inplace}"
