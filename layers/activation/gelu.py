from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{GELU}(x)=x\Phi(x)
\]
"""

@register_act_fn(name="gelu")
class GELU(nn.GELU):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__(approximate=approximate)

    def __repr__(self) -> str:
        return f"GELU(approximate={self.approximate})"
