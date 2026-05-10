from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{Softplus}(x)=\log(1+e^x)
\]
"""

@register_act_fn(name="softplus")
class Softplus(nn.Softplus):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__(beta=beta, threshold=threshold)

    def __repr__(self) -> str:
        return f"Softplus(beta={self.beta}, threshold={self.threshold})"
