import argparse
from typing import Optional, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from layers.base_layer import BaseLayer
from utils import logger




class LinearLayer(BaseLayer):
    def __init__(
        self,
        in_features: int = None,
        out_features: int = None,
        bias: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        opts = getattr(self, "opts", kwargs.get("opts", None))

        self.in_features = in_features if in_features is not None else getattr(opts, "in_features", None)
        self.out_features = out_features if out_features is not None else getattr(opts, "out_features", None)

        if self.in_features is None or self.out_features is None:
            raise ValueError("in_features và out_features không được để trống (cần truyền trực tiếp hoặc qua opts)")

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        
        use_bias = bias if bias else getattr(opts, "bias", False)
        
        if use_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_params(opts=opts)
        
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--linear-init",
            type=str,
            default="xavier_uniform",
            choices=["xavier_uniform", "xavier_normal", "normal", "kaiming_uniform", "kaiming_normal"],
            help="Kiểu khởi tạo trọng số cho Linear layers",
        )
        parser.add_argument(
            "--linear-init-std-dev",
            type=float,
            default=0.01,
            help="Độ lệch chuẩn cho khởi tạo 'normal'",
        )
        parser.add_argument(
            "--bias",
            action="store_true",
            help="Sử dụng bias trong Linear layers",
        )   
        return parser


    def reset_params(self, opts: Optional[Any] = None) -> None:
        init_type = getattr(opts, "linear_init", "xavier_uniform")
        std_dev = getattr(opts, "linear_init_std_dev", 0.01)

        if self.weight is not None:
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(self.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(self.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
            elif init_type == "normal":
                nn.init.normal_(self.weight, mean=0.0, std=std_dev)
            else:
                logger.warning(f"Init type {init_type} không hỗ trợ. Sử dụng xavier_uniform.")
                nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
            
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f")"
        )