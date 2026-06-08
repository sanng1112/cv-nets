from typing import Optional, Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LinearLayer(nn.Module):
    """A linear (fully-connected) layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool
        Whether to include a bias term (default ``False``).
    """

    def __init__(
        self,
        in_features: int = None,
        out_features: int = None,
        bias: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if in_features is None or out_features is None:
            raise ValueError(
                "in_features and out_features must be provided."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_params()

    def reset_params(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"LinearLayer(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )
