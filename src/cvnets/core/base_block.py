import abc
from torch import Tensor, nn


class BaseBlock(nn.Module, abc.ABC):
    """Abstract base for composable building blocks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        ...
