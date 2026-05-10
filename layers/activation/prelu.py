from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{PReLU}(x)=
\begin{cases}
x, & x>0 \\
ax, & x\le 0
\end{cases}
\]
"""

@register_act_fn(name="prelu")
class PReLU(nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25, *args, **kwargs) -> None:
        super().__init__(num_parameters=num_parameters, init=init)

    def __repr__(self) -> str:
        init = self.weight.data.flatten()[0].item() if self.weight.numel() else "n/a"
        return f"PReLU(num_parameters={self.num_parameters}, init={init})"
