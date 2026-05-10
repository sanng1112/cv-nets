from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{ReLU6}(x)=\min(\max(0,x),6)
\]
"""

@register_act_fn(name="relu6")
class ReLU6(nn.ReLU6):
    def __init__(self, inplace: bool = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)

    def __repr__(self) -> str:
        return f"ReLU6(inplace={self.inplace})"
