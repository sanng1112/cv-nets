from typing import Optional
from torch import Tensor, nn

from . import register_act_fn


@register_act_fn("prelu")
class PReLU(nn.PReLU):
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        *args,
        **kwargs,
    ):
        super().__init__(num_parameters=num_parameters, init=init)
        self.num_parameters = num_parameters
        self.init = init

    def __repr__(self):
        return f"PReLU(num_parameters={self.num_parameters}, init={self.init})"
