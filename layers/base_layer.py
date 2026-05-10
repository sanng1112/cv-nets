import argparse

from typing import Any, Dict, Optional, List, Literal, Tuple
from torch import Tensor, nn


class BaseLayer(nn.Module):
    ''''
    Base class for all layers in the model. All layers should inherit from this class and implement the forward method.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add layer specific arguments"""
        return parser

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    # Khởi tạo trọng số cho lớp
    def int_weight(self) -> None:
        pass
    