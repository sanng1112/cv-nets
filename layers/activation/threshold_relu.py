from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
f(x)=
\begin{cases}
x, & x>\theta \\
0, & \text{otherwise}
\end{cases}
\]
"""

@register_act_fn(name="threshold_relu")
class ThresholdReLU(nn.Module):
    def __init__(self, threshold: float = 1.0, value: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return F.threshold(x, self.threshold, self.value)

    def __repr__(self) -> str:
        return f"ThresholdReLU(threshold={self.threshold}, value={self.value})"
