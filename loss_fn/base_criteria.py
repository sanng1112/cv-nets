import abc
import argparse
from typing import Any

from torch import nn

from utils import logger

class BaseCriteria(nn.Module, abc.ABC):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super(BaseCriteria, self).__init__()
        self.opts = opts
        self.eps = 1e-7
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseCriteria:
            return parser
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--loss.category",
            type=str,
            default=None,
            help="Loss function category (e.g., classification). Defaults to None.",
        )
        return parser
    
    @abc.abstractmethod
    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Any:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "{}({}\n)".format(self.__class__.__name__, self.extra_repr())
