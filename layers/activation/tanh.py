from . import register_act_fn
from torch import nn


r"""
\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]
"""

@register_act_fn(name="tanh")
class Tanh(nn.Tanh):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "Tanh()"
