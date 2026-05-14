import argparse

from typing import Any, Dict, Optional, List, Literal, Tuple
from torch import Tensor, nn


class BaseLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    # Khởi tạo trọng số cho lớp
    def int_weight(self) -> None:
        pass
    