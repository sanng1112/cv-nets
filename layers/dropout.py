import argparse
from typing import Optional, Any
from torch import Tensor, nn
from torch.nn import functional as F

from layers.base_layer import BaseLayer
from utils.config_helper import get_param


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
        self.p = get_param(opts, p, 'p', 0.0)
        self.inplace = get_param(opts, inplace, "inplace", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--p", 
            type=float, 
            default=0.0, 
            help="Xác suất Dropout (p)"
        )
        parser.add_argument(
            "--inplace",
            action = "store_true",
            help="Tiết kiệm bộ nhớ",
            default = False
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
        self.p = get_param(opts, p , 'p', 0.0)
        self.inplace = inplace if inplace is not None else getattr(opts, "inplace", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--p", 
            type=float, 
            default=0.0, 
            help="Xác suất Dropout (p)"
        )
        parser.add_argument(
            "--inplace",
            action = "store_true",
            help="Tiết kiệm bộ nhớ",
            default = False
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
    
    
if __name__ == "__main__":
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=240)
    parser = argparse.ArgumentParser(
        description="Kiểm tra thông số của lớp Dropout",
        formatter_class = formatter)
    Dropout.add_arguments(parser)
    args = parser.parse_args()