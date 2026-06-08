import abc
from torch import Tensor, nn


class BaseLayer(nn.Module, abc.ABC):
    """Abstract base for all neural network layers in cv-nets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        ...

    def init_weights(self) -> None:
        pass
