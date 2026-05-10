from . import register_act_fn
from torch import nn


r"""
\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]
"""

@register_act_fn(name="sigmoid")
class Sigmoid(nn.Sigmoid):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "Sigmoid()"
