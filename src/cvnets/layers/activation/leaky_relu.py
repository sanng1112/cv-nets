from typing import Optional
from torch import Tensor, nn

from . import register_act_fn


@register_act_fn("leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(negative_slope=negative_slope, inplace=inplace)
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope}, inplace={self.inplace})"
