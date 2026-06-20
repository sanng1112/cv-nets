"""
Dropout layers — registered in the activation registry for factory compatibility.

Provides ``Dropout`` and ``Dropout2d`` wrappers around the standard PyTorch
implementations, each registered with ``@register_act_fn`` so that the
``build_activation_layer`` factory can instantiate them on demand.
"""

from typing import Optional

from torch import Tensor, nn

from cvnets.layers.activation import register_act_fn


@register_act_fn("dropout")
class Dropout(nn.Dropout):
    """
    Randomly zeroes elements of the input tensor with probability *p*
    during training, scaling the remainder by ``1 / (1 - p)``.

    Parameters
    ----------
    p : float
        Probability of an element to be zeroed (default ``0.5``).
    inplace : bool
        If ``True``, performs the operation in-place (default ``False``).
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, *args, **kwargs) -> None:
        super().__init__(p=p, inplace=inplace)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p}, inplace={self.inplace})"


@register_act_fn("dropout2d")
class Dropout2d(nn.Dropout2d):
    """
    Randomly zeroes entire channels of the input tensor with probability *p*
    during training, scaling the remainder by ``1 / (1 - p)``.

    Parameters
    ----------
    p : float
        Probability of a channel to be zeroed (default ``0.5``).
    inplace : bool
        If ``True``, performs the operation in-place (default ``False``).
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, *args, **kwargs) -> None:
        super().__init__(p=p, inplace=inplace)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)

    def __repr__(self) -> str:
        return f"Dropout2d(p={self.p}, inplace={self.inplace})"
