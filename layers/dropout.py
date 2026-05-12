import argparse
from typing import Optional, Any
from torch import Tensor, nn
from torch.nn import functional as F

from layers.base_layer import BaseLayer

class Dropout(BaseLayer):
    """
    Dropout layer hỗ trợ lấy tham số từ code hoặc từ đối tượng opts.
    """
    def __init__(
        self, 
        p: Optional[float] = None, 
        inplace: Optional[bool] = None, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        opts = getattr(self, "opts", kwargs.get("opts", None))

        # Ưu tiên tham số truyền trực tiếp, nếu không có mới tìm trong opts
        self.p = p if p is not None else getattr(opts, "p", 0.0)
        self.inplace = inplace if inplace is not None else getattr(opts, "inplace", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if not any(arg.dest == 'dropout' for arg in parser._actions):
            parser.add_argument(
                "--dropout", 
                type=float, 
                default=0.0, 
                help="Xác suất Dropout (p)"
            )
        return parser

    def forward(self, x: Tensor) -> Tensor:
        if self.p <= 0.0:
            return x
        
        return F.dropout(
            input=x, 
            p=self.p, 
            training=self.training, 
            inplace=self.inplace
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, inplace={self.inplace})"


class Dropout2D(BaseLayer):
    """
    Dropout2D (Spatial Dropout) cho dữ liệu 4D (Tensor: N, C, H, W).
    """
    def __init__(
        self, 
        p: Optional[float] = None, 
        inplace: Optional[bool] = None, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        opts = getattr(self, "opts", kwargs.get("opts", None))

        # Tương tự như Dropout
        self.p = p if p is not None else getattr(opts, "p", 0.0)
        self.inplace = inplace if inplace is not None else getattr(opts, "inplace", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Dùng chung flag --dropout hoặc tạo flag riêng tùy ý bạn
        if not any(arg.dest == 'dropout' for arg in parser._actions):
            parser.add_argument(
                "--dropout", 
                type=float, 
                default=0.0, 
                help="Xác suất Dropout (p)"
            )
        return parser

    def forward(self, x: Tensor) -> Tensor:
        if self.p <= 0.0:
            return x
        
        return F.dropout2d(
            input=x, 
            p=self.p, 
            training=self.training, 
            inplace=self.inplace
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, inplace={self.inplace})"