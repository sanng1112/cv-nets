from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{Softsign}(x)=\frac{x}{1+|x|}
\]
"""

@register_act_fn(name="softsign")
class Softsign(nn.Softsign):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "Softsign()"
